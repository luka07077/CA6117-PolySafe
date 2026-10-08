"""
Read-only, in-memory view of the drug knowledge database (data/polysafe.db), shared by all tools.

The database is small (~75 drugs, ~75 interactions), so it is loaded once per process; every tool reads from
this snapshot and none can write to it.
"""
from dataclasses import dataclass
from functools import lru_cache

from db.drug_db import SEVERITY_RANK, connect


@dataclass(frozen=True)
class KnowledgeBase:
    drugs: dict            # drug_id -> row dict
    classes: dict          # class_code -> row dict
    synonyms: dict         # synonym (lower case) -> drug_id
    interactions: list     # row dicts
    rules: list            # combination-rule row dicts

    def drug(self, drug_id: str) -> dict:
        return self.drugs[drug_id]

    def refs(self, drug_id: str) -> set[str]:
        """Keys an interaction row can use to refer to this drug: its id and its class."""
        return {drug_id, f"class:{self.drugs[drug_id]['class_code']}"}

    def resolve_exact(self, name: str) -> str | None:
        """drug_id for an exact generic name, drug id or synonym (case-insensitive); None if not found."""
        key = " ".join(str(name).lower().replace("_", " ").split())
        if key.replace(" ", "_") in self.drugs:
            return key.replace(" ", "_")
        for d in self.drugs.values():
            if d["generic_name"].lower() == key:
                return d["drug_id"]
        return self.synonyms.get(key)


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    with connect() as c:
        drugs = {r["drug_id"]: dict(r) for r in c.execute("SELECT * FROM drugs")}
        classes = {r["class_code"]: dict(r) for r in c.execute("SELECT * FROM drug_classes")}
        synonyms = {r["synonym"]: r["drug_id"] for r in c.execute("SELECT * FROM drug_synonyms")}
        interactions = [dict(r) for r in c.execute("SELECT * FROM interactions")]
        rules = [dict(r) for r in c.execute("SELECT * FROM combination_rules")]
    return KnowledgeBase(drugs, classes, synonyms, interactions, rules)


def max_severity(severities) -> str | None:
    severities = [s for s in severities if s]
    return max(severities, key=SEVERITY_RANK.get) if severities else None
