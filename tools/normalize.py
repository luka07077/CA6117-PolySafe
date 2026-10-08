"""
normalize_drug: map a medication name as written (brand name, misspelling, with dose text) to a drug in the
knowledge base.

    1. clean      lower case; drop dose ("5 mg"), route/frequency words ("daily", "prn") and brackets
    2. exact      generic name, drug id, brand name or alias              -> confidence 1.0
    3. fuzzy      closest vocabulary entry (rapidfuzz, edit-distance ratio)
                  score >= ACCEPT  -> recognised (likely misspelling) but needs_confirmation=true
                  REVIEW <= score < ACCEPT -> NOT recognised, suggestions returned (agent asks to clarify)
                  score < REVIEW   -> unknown medication

An unrecognised drug is never guessed: it could hide a dangerous interaction, so the workflow asks a human.
"""
import re

from rapidfuzz import fuzz, process

from tools.knowledge import get_kb

ACCEPT, REVIEW = 88, 75

_DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|units?|iu|%|mmol|meq)\b|\b\d+(?:[.,]\d+)?\b", re.I)
_NOISE = re.compile(
    r"\b(?:tab(?:let)?s?|caps?(?:ule)?s?|oral|po|iv|sc|im|inj(?:ection)?|daily|once|twice|three|times|a|per|day|"
    r"bid|tid|qid|qd|od|bd|prn|as|needed|required|at|night|bedtime|morning|evening|nightly|weekly|every|hours?|"
    r"h|x|sr|xr|er|cr|dr|la|mr|otc|dose|doses|with|food|for|pain|sleep|take|takes|taking|of|and|the|in)\b", re.I)


def clean_name(text: str) -> str:
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", str(text).lower())
    t = _DOSE.sub(" ", t)
    t = re.sub(r"[^a-z\-\s]", " ", t)
    t = _NOISE.sub(" ", t)
    return " ".join(t.replace(" - ", " ").split())


def _vocabulary() -> dict[str, str]:
    kb = get_kb()
    vocab = dict(kb.synonyms)
    for d in kb.drugs.values():
        vocab[d["drug_id"].replace("_", " ")] = d["drug_id"]
        vocab[d["generic_name"].lower()] = d["drug_id"]
    return vocab


def _result(raw: str, cleaned: str, drug_id: str | None, match_type: str, confidence: float,
            suggestions: list | None = None) -> dict:
    kb = get_kb()
    # a fuzzy match may be a different real drug (look-alike / sound-alike names, e.g. prednisolone vs prednisone):
    # the review continues with it, but a human must confirm the interpretation
    out = {"input": raw, "cleaned": cleaned, "recognized": drug_id is not None, "match_type": match_type,
           "confidence": round(confidence, 2), "needs_confirmation": match_type == "fuzzy"}
    if drug_id:
        d, cls = kb.drug(drug_id), kb.classes[kb.drug(drug_id)["class_code"]]
        out.update(drug_id=drug_id, generic_name=d["generic_name"], class_code=d["class_code"],
                   class_name=cls["class_name"], atc_code=d["atc_code"])
    if suggestions:
        out["suggestions"] = suggestions
    return out


def normalize_drug(name: str) -> dict:
    kb = get_kb()
    cleaned = clean_name(name)
    if not cleaned:
        return _result(name, cleaned, None, "empty", 0.0)
    for text in (" ".join(str(name).lower().split()), cleaned):   # as written first, then without dose/noise words
        exact = kb.resolve_exact(text)
        if exact:
            generic = kb.drug(exact)["generic_name"].lower()
            match_type = "exact" if text in (exact.replace("_", " "), generic) else "synonym"
            return _result(name, cleaned, exact, match_type, 1.0)
    vocab = _vocabulary()
    # also try each word alone ("coumadin pill" -> "coumadin"), keeping the best score
    candidates = [cleaned] + [w for w in cleaned.split() if len(w) > 3]
    best = None
    for cand in candidates:
        if cand in vocab:
            return _result(name, cleaned, vocab[cand], "synonym", 1.0)
        hit = process.extractOne(cand, vocab.keys(), scorer=fuzz.ratio)
        if hit and (best is None or hit[1] > best[1]):
            best = hit
    if best and best[1] >= ACCEPT:
        return _result(name, cleaned, vocab[best[0]], "fuzzy", best[1] / 100)
    top = process.extract(cleaned, vocab.keys(), scorer=fuzz.ratio, limit=3)
    suggestions = list(dict.fromkeys(kb.drug(vocab[t[0]])["generic_name"] for t in top if t[1] >= REVIEW))
    return _result(name, cleaned, None, "uncertain" if suggestions else "unknown",
                   (best[1] / 100) if best else 0.0, suggestions)


def normalize_medications(names: list[str]) -> dict:
    """Normalize a whole list; also reports the same drug entered twice (e.g. "Advil" and "ibuprofen")."""
    results = [normalize_drug(n) for n in names]
    seen, repeated = {}, []
    for r in results:
        if r["recognized"]:
            if r["drug_id"] in seen:
                repeated.append({"drug_id": r["drug_id"], "inputs": [seen[r["drug_id"]], r["input"]]})
            seen.setdefault(r["drug_id"], r["input"])
    return {"medications": results,
            "recognized": [r["drug_id"] for r in results if r["recognized"]],
            "unresolved": [r["input"] for r in results if not r["recognized"]],
            "repeated_entries": repeated}
