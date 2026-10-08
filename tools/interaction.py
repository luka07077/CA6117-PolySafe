"""
check_interaction: look up one drug pair in the interaction table.

A row matches when each side refers to one of the two drugs, either by drug id ("warfarin") or by class
("class:NSAID"), in either order. A pair can match several rows (a drug-specific and a class-level one);
all are returned and `severity` is the highest. Severity always comes from the database, never from the LLM.
"""
from tools.knowledge import get_kb, max_severity


def _resolve(name: str) -> str | None:
    kb = get_kb()
    return kb.resolve_exact(name) or (name if name in kb.drugs else None)


def check_interaction(drug_a: str, drug_b: str) -> dict:
    kb = get_kb()
    a, b = _resolve(drug_a), _resolve(drug_b)
    unknown = [n for n, r in ((drug_a, a), (drug_b, b)) if r is None]
    if unknown:
        return {"pair": [drug_a, drug_b], "status": "error",
                "message": f"not in the drug knowledge base: {unknown}; normalize the names first"}
    if a == b:
        return {"pair": [a, b], "status": "error", "message": "same drug on both sides"}
    refs_a, refs_b = kb.refs(a), kb.refs(b)
    hits = []
    for row in kb.interactions:
        if (row["drug_a"] in refs_a and row["drug_b"] in refs_b) or (row["drug_a"] in refs_b and row["drug_b"] in refs_a):
            hits.append({k: row[k] for k in ("interaction_id", "drug_a", "drug_b", "severity", "mechanism", "effect",
                                               "recommendation", "source", "evidence_doc")}
                        | {"match_level": "class" if row["drug_a"].startswith("class:") or row["drug_b"].startswith("class:")
                           else "drug"})
    return {"pair": [a, b],
            "names": [kb.drug(a)["generic_name"], kb.drug(b)["generic_name"]],
            "status": "ok",
            "found": bool(hits),
            "severity": max_severity(h["severity"] for h in hits),
            "interactions": hits}
