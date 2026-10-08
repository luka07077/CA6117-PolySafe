"""
Whole-regimen checks that pairwise interaction lookups cannot see:

  check_duplication        two or more drugs from the same duplication group (e.g. two NSAIDs, two
                           anticoagulants), or the same drug entered twice under different names
  check_combination_rules  multi-drug patterns from data/combination_rules.csv
                           classes_all: one drug from every listed class group (e.g. "triple whammy")
                           flag_count : at least N drugs carrying a flag (e.g. >= 3 CNS-active drugs)
  check_older_adult_risks  potentially inappropriate medications for patients aged 65+ (Beers-style flags)

All functions take normalized drug ids (or exact names) and return JSON-serialisable dicts.
"""
from tools.knowledge import get_kb

OLDER_ADULT_AGE = 65


def _ids(drugs: list[str]) -> tuple[list[str], list[str]]:
    kb = get_kb()
    ids, unknown = [], []
    for d in drugs:
        r = kb.resolve_exact(d)
        (ids if r else unknown).append(r or d)
    return ids, unknown


def check_duplication(drugs: list[str]) -> dict:
    kb = get_kb()
    ids, unknown = _ids(drugs)
    findings = []
    # same drug entered twice ("Advil" + "ibuprofen")
    for drug_id in sorted({i for i in ids if ids.count(i) > 1}):
        findings.append({"type": "same_drug", "group": drug_id, "drugs": [drug_id], "severity": "high",
                         "note": f"{kb.drug(drug_id)['generic_name']} appears more than once in the list "
                                 "(possibly under a brand and a generic name): risk of double dosing",
                         "evidence_doc": "13_therapeutic_duplication.md"})
    groups: dict[str, list[str]] = {}
    for drug_id in dict.fromkeys(ids):
        cls = kb.classes[kb.drug(drug_id)["class_code"]]
        if cls["duplication_group"]:
            groups.setdefault(cls["duplication_group"], []).append(drug_id)
    for group, members in groups.items():
        if len(members) > 1:
            cls = kb.classes[kb.drug(members[0])["class_code"]]
            findings.append({"type": "same_class", "group": group, "drugs": members,
                             "severity": cls["duplication_severity"], "note": cls["duplication_note"],
                             "evidence_doc": "13_therapeutic_duplication.md"})
    return {"status": "ok", "checked": list(dict.fromkeys(ids)), "unknown": unknown,
            "found": bool(findings), "duplications": findings}


def check_combination_rules(drugs: list[str]) -> dict:
    kb = get_kb()
    ids, unknown = _ids(drugs)
    ids = list(dict.fromkeys(ids))
    findings = []
    for rule in kb.rules:
        if rule["rule_type"] == "classes_all":
            matched = []
            for group in rule["criteria"].split(";"):
                classes = set(group.split("|"))
                hit = [i for i in ids if kb.drug(i)["class_code"] in classes]
                if not hit:
                    break
                matched += hit
            else:
                findings.append(_rule_finding(rule, matched))
        else:   # flag_count
            hit = [i for i in ids if kb.drug(i)[rule["criteria"]]]
            if len(hit) >= rule["min_count"]:
                findings.append(_rule_finding(rule, hit))
    return {"status": "ok", "checked": ids, "unknown": unknown, "found": bool(findings), "combinations": findings}


def _rule_finding(rule: dict, drugs: list[str]) -> dict:
    return {"rule_id": rule["rule_id"], "name": rule["name"], "drugs": list(dict.fromkeys(drugs)),
            "severity": rule["severity"], "effect": rule["effect"], "recommendation": rule["recommendation"],
            "source": rule["source"], "evidence_doc": rule["evidence_doc"]}


def check_older_adult_risks(drugs: list[str], age: int | None) -> dict:
    kb = get_kb()
    ids, unknown = _ids(drugs)
    if age is None:
        return {"status": "skipped", "reason": "age unknown", "found": False, "flags": []}
    if age < OLDER_ADULT_AGE:
        return {"status": "skipped", "reason": f"age {age} < {OLDER_ADULT_AGE}", "found": False, "flags": []}
    flags = [{"drug_id": i, "generic_name": kb.drug(i)["generic_name"], "severity": "low",
              "note": kb.drug(i)["beers_note"], "source": "AGS Beers Criteria 2023",
              "evidence_doc": "12_older_adults_beers.md"}
             for i in dict.fromkeys(ids) if kb.drug(i)["beers_flag"]]
    return {"status": "ok", "age": age, "unknown": unknown, "found": bool(flags), "flags": flags}
