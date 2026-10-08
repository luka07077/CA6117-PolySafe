"""
Unit tests for the deterministic tools (no LLM, no network). Run from the project root:

    pytest -q tests/test_tools.py
"""
import pytest

from tools.interaction import check_interaction
from tools.knowledge import max_severity
from tools.normalize import clean_name, normalize_drug, normalize_medications
from tools.regimen import check_combination_rules, check_duplication, check_older_adult_risks


# ---------------------------------------------------------------- normalize_drug
@pytest.mark.parametrize("name, drug_id, match_type", [
    ("Warfarin", "warfarin", "exact"),
    ("warfarin", "warfarin", "exact"),
    ("Coumadin 5 mg daily", "warfarin", "synonym"),
    ("Advil as needed", "ibuprofen", "synonym"),
    ("ACETAMINOPHEN 500 MG PRN", "paracetamol", "synonym"),
    ("Metoprolol succinate ER 50mg", "metoprolol", "synonym"),
    ("Lantus 20 units at night", "insulin_glargine", "synonym"),
    ("KCl 20 mEq", "potassium_chloride", "synonym"),
    ("co-trimoxazole", "trimethoprim_sulfamethoxazole", "synonym"),
    ("Bactrim DS", "trimethoprim_sulfamethoxazole", "synonym"),
    ("Tylenol with codeine", "codeine", "synonym"),
    ("warfrin", "warfarin", "synonym"),            # listed misspelling
    ("prednisolone", "prednisolone", "exact"),     # must NOT become prednisone
])
def test_normalize_recognised(name, drug_id, match_type):
    r = normalize_drug(name)
    assert r["recognized"] and r["drug_id"] == drug_id and r["match_type"] == match_type
    assert r["confidence"] == 1.0 and not r["needs_confirmation"]


@pytest.mark.parametrize("name, drug_id", [("asprin", "aspirin"), ("simvastin", "simvastatin"),
                                           ("lisinoprel", "lisinopril"), ("furosimide", "furosemide")])
def test_normalize_fuzzy_needs_confirmation(name, drug_id):
    r = normalize_drug(name)
    assert r["recognized"] and r["drug_id"] == drug_id and r["match_type"] == "fuzzy"
    assert r["needs_confirmation"] and r["confidence"] < 1.0


def test_normalize_uncertain_gives_suggestions_but_not_recognised():
    r = normalize_drug("diazapam")
    assert not r["recognized"] and r["match_type"] == "uncertain" and "Diazepam" in r["suggestions"]


@pytest.mark.parametrize("name", ["abcdefg", "vitamin D", "glyburide"])
def test_normalize_unknown(name):
    r = normalize_drug(name)
    assert not r["recognized"] and r["match_type"] == "unknown"


def test_normalize_empty():
    assert normalize_drug("10 mg daily")["match_type"] == "empty"


def test_clean_name_strips_dose_and_frequency():
    assert clean_name("Coumadin 5 mg tablet once daily (blue)") == "coumadin"


def test_normalize_medications_reports_repeated_entry():
    out = normalize_medications(["Advil", "ibuprofen 400mg", "warfarin", "abcdefg"])
    assert out["recognized"] == ["ibuprofen", "ibuprofen", "warfarin"]
    assert out["unresolved"] == ["abcdefg"]
    assert out["repeated_entries"][0]["drug_id"] == "ibuprofen"


# ---------------------------------------------------------------- check_interaction
@pytest.mark.parametrize("a, b, severity, ids", [
    ("warfarin", "ibuprofen", "high", {"DDI001"}),               # drug + class rule
    ("ibuprofen", "warfarin", "high", {"DDI001"}),               # order does not matter
    ("warfarin", "aspirin", "high", {"DDI002"}),
    ("simvastatin", "clarithromycin", "critical", {"DDI023"}),
    ("oxycodone", "lorazepam", "critical", {"DDI061"}),          # class + class rule
    ("sildenafil", "nitroglycerin", "critical", {"DDI057"}),
    ("clopidogrel", "omeprazole", "moderate", {"DDI018"}),
    ("lisinopril", "losartan", "high", {"DDI035"}),
    ("prednisolone", "naproxen", "moderate", {"DDI075"}),
])
def test_interaction_found(a, b, severity, ids):
    r = check_interaction(a, b)
    assert r["status"] == "ok" and r["found"] and r["severity"] == severity
    assert {h["interaction_id"] for h in r["interactions"]} == ids
    assert all(h["source"] and h["evidence_doc"] for h in r["interactions"])


def test_max_severity_picks_highest():
    # a pair matching several rows (drug-level + class-level) reports the highest severity
    assert max_severity(["moderate", "critical", "high"]) == "critical"
    assert max_severity(["low", None]) == "low"
    assert max_severity([]) is None


@pytest.mark.parametrize("a, b", [("amlodipine", "omeprazole"), ("metformin", "atorvastatin"),
                                  ("levothyroxine", "amlodipine"), ("paracetamol", "warfarin")])
def test_interaction_not_found(a, b):
    r = check_interaction(a, b)
    assert r["status"] == "ok" and not r["found"] and r["severity"] is None and r["interactions"] == []


def test_interaction_accepts_brand_names():
    assert check_interaction("coumadin", "advil")["severity"] == "high"


def test_interaction_unknown_drug_is_error_not_guess():
    r = check_interaction("warfarin", "abcdefg")
    assert r["status"] == "error" and "abcdefg" in r["message"]


def test_interaction_same_drug_is_error():
    assert check_interaction("warfarin", "warfarin")["status"] == "error"


# ---------------------------------------------------------------- check_duplication
def test_duplication_two_nsaids():
    r = check_duplication(["ibuprofen", "naproxen", "omeprazole"])
    d = r["duplications"]
    assert len(d) == 1 and d[0]["group"] == "NSAID" and d[0]["severity"] == "high"


def test_duplication_two_anticoagulants_is_critical():
    r = check_duplication(["warfarin", "apixaban"])
    assert r["duplications"][0]["group"] == "ANTICOAGULANT" and r["duplications"][0]["severity"] == "critical"


def test_duplication_same_drug_twice():
    r = check_duplication(["ibuprofen", "ibuprofen"])
    assert [d["type"] for d in r["duplications"]] == ["same_drug"]


def test_duplication_dual_antiplatelet_is_not_duplication():
    assert not check_duplication(["aspirin", "clopidogrel"])["found"]


def test_duplication_none():
    assert not check_duplication(["warfarin", "amlodipine", "metformin"])["found"]


# ---------------------------------------------------------------- check_combination_rules
def test_triple_whammy():
    r = check_combination_rules(["lisinopril", "furosemide", "ibuprofen", "metformin"])
    rules = {c["rule_id"]: c for c in r["combinations"]}
    assert set(rules) == {"CMB001"} and set(rules["CMB001"]["drugs"]) == {"lisinopril", "furosemide", "ibuprofen"}


def test_triple_whammy_needs_all_three():
    assert not check_combination_rules(["lisinopril", "ibuprofen"])["found"]


def test_anticoagulant_antiplatelet_nsaid_is_critical():
    r = check_combination_rules(["warfarin", "aspirin", "ibuprofen"])
    assert r["combinations"][0]["rule_id"] == "CMB002" and r["combinations"][0]["severity"] == "critical"


def test_three_cns_active_drugs():
    r = check_combination_rules(["sertraline", "zolpidem", "gabapentin"])
    assert [c["rule_id"] for c in r["combinations"]] == ["CMB003"]


def test_two_anticholinergics():
    r = check_combination_rules(["oxybutynin", "diphenhydramine"])
    assert [c["rule_id"] for c in r["combinations"]] == ["CMB004"]


# ---------------------------------------------------------------- check_older_adult_risks
def test_older_adult_flags_only_from_65():
    meds = ["diazepam", "amlodipine", "ibuprofen"]
    r = check_older_adult_risks(meds, 72)
    assert {f["drug_id"] for f in r["flags"]} == {"diazepam", "ibuprofen"}
    assert check_older_adult_risks(meds, 50)["status"] == "skipped"
    assert check_older_adult_risks(meds, None)["status"] == "skipped"
