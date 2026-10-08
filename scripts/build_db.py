"""
Build the drug knowledge database data/polysafe.db from the CSV files in data/ (always rebuilt from scratch).

    python scripts/build_db.py

Validates the data before writing: every drug / class referenced by an interaction or rule must exist,
every evidence_doc must exist in data/evidence/, severities must be valid, and no pair may be listed twice.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from db.drug_db import DB_PATH, SCHEMA, SEVERITY_RANK  # noqa: E402
from src.config import get_data_dir  # noqa: E402

TABLES = [("drug_classes", "drug_classes.csv"), ("drugs", "drugs.csv"), ("drug_synonyms", "synonyms.csv"),
          ("interactions", "interactions.csv"), ("combination_rules", "combination_rules.csv")]


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for table, fname in TABLES:
        df = pd.read_csv(get_data_dir(fname), dtype=str, keep_default_na=False)
        out[table] = df.apply(lambda col: col.str.strip())
    return out


def validate(d: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    classes, drugs = set(d["drug_classes"].class_code), set(d["drugs"].drug_id)
    evidence = set(os.listdir(get_data_dir("evidence")))

    def ref_ok(ref: str) -> bool:
        return ref[6:] in classes if ref.startswith("class:") else ref in drugs

    for name, df, key in [("drug_classes", d["drug_classes"], "class_code"), ("drugs", d["drugs"], "drug_id"),
                          ("synonyms", d["drug_synonyms"], "synonym"), ("interactions", d["interactions"], "interaction_id"),
                          ("combination_rules", d["combination_rules"], "rule_id")]:
        dup = df[key][df[key].duplicated()].tolist()
        if dup:
            errors.append(f"{name}: duplicate {key} {dup}")
    for r in d["drugs"].itertuples():
        if r.class_code not in classes:
            errors.append(f"drug {r.drug_id}: unknown class {r.class_code}")
    for r in d["drug_classes"].itertuples():
        if r.duplication_group and r.duplication_severity not in SEVERITY_RANK:
            errors.append(f"class {r.class_code}: duplication group without valid severity")
    for r in d["drug_synonyms"].itertuples():
        if r.drug_id not in drugs:
            errors.append(f"synonym {r.synonym!r}: unknown drug {r.drug_id}")
        if r.synonym != r.synonym.lower():
            errors.append(f"synonym {r.synonym!r}: must be lower case")
    seen_pairs = set()
    for r in d["interactions"].itertuples():
        for ref in (r.drug_a, r.drug_b):
            if not ref_ok(ref):
                errors.append(f"{r.interaction_id}: unknown drug/class {ref}")
        if r.severity not in SEVERITY_RANK:
            errors.append(f"{r.interaction_id}: invalid severity {r.severity}")
        if r.evidence_doc not in evidence:
            errors.append(f"{r.interaction_id}: missing evidence doc {r.evidence_doc}")
        pair = frozenset((r.drug_a, r.drug_b))
        if pair in seen_pairs:
            errors.append(f"{r.interaction_id}: pair listed twice {sorted(pair)}")
        seen_pairs.add(pair)
    for r in d["combination_rules"].itertuples():
        if r.rule_type == "classes_all":
            for c in (c for grp in r.criteria.split(";") for c in grp.split("|")):
                if c not in classes:
                    errors.append(f"{r.rule_id}: unknown class {c}")
        elif r.criteria not in ("cns_active", "strong_anticholinergic"):
            errors.append(f"{r.rule_id}: unknown flag {r.criteria}")
        if r.evidence_doc not in evidence:
            errors.append(f"{r.rule_id}: missing evidence doc {r.evidence_doc}")
    return errors


def main():
    d = load()
    errors = validate(d)
    if errors:
        print("Data validation failed:\n  " + "\n  ".join(errors))
        sys.exit(1)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        for table, _ in TABLES:
            df = d[table].replace("", None)
            df.to_sql(table, conn, if_exists="append", index=False)
    print(f"Built {DB_PATH}")
    for table, _ in TABLES:
        print(f"  {table:<18} {len(d[table]):>4} rows")
    sev = d["interactions"].severity.value_counts()
    print("  interactions by severity: " + ", ".join(f"{s}={sev.get(s, 0)}" for s in SEVERITY_RANK))


if __name__ == "__main__":
    main()
