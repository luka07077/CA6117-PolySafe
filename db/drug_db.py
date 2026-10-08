"""
Drug knowledge database (SQLite, data/polysafe.db), built from the CSV files in data/ by scripts/build_db.py.

    drugs(drug_id, generic_name, class_code, atc_code, cns_active, strong_anticholinergic, beers_flag, beers_note)
    drug_synonyms(synonym, drug_id, type)                   brand names, aliases, common misspellings
    drug_classes(class_code, class_name, duplication_group, duplication_severity, duplication_note)
    interactions(interaction_id, drug_a, drug_b, severity, mechanism, effect, recommendation, source, evidence_doc)
                 drug_a / drug_b are a drug_id or "class:<class_code>"
    combination_rules(rule_id, name, rule_type, criteria, min_count, severity, effect, recommendation, source, evidence_doc)

This simulates the drug-interaction knowledge base a hospital would license; the agent can only read it
through the tools in tools/ (no write access).
"""
import os
import sqlite3

from src.config import get_data_dir

DB_PATH = get_data_dir("polysafe.db")

SCHEMA = """
CREATE TABLE drugs (
    drug_id TEXT PRIMARY KEY, generic_name TEXT NOT NULL, class_code TEXT NOT NULL REFERENCES drug_classes(class_code),
    atc_code TEXT, cns_active INTEGER DEFAULT 0, strong_anticholinergic INTEGER DEFAULT 0,
    beers_flag INTEGER DEFAULT 0, beers_note TEXT);
CREATE TABLE drug_synonyms (
    synonym TEXT PRIMARY KEY, drug_id TEXT NOT NULL REFERENCES drugs(drug_id), type TEXT);
CREATE TABLE drug_classes (
    class_code TEXT PRIMARY KEY, class_name TEXT NOT NULL, duplication_group TEXT,
    duplication_severity TEXT, duplication_note TEXT);
CREATE TABLE interactions (
    interaction_id TEXT PRIMARY KEY, drug_a TEXT NOT NULL, drug_b TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'moderate', 'low')),
    mechanism TEXT, effect TEXT, recommendation TEXT, source TEXT, evidence_doc TEXT);
CREATE INDEX idx_interactions_a ON interactions(drug_a);
CREATE INDEX idx_interactions_b ON interactions(drug_b);
CREATE TABLE combination_rules (
    rule_id TEXT PRIMARY KEY, name TEXT NOT NULL, rule_type TEXT NOT NULL CHECK (rule_type IN ('classes_all', 'flag_count')),
    criteria TEXT NOT NULL, min_count INTEGER NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'moderate', 'low')),
    effect TEXT, recommendation TEXT, source TEXT, evidence_doc TEXT);
"""

SEVERITY_RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}


def connect(db_path: str | None = None, readonly: bool = True) -> sqlite3.Connection:
    path = db_path or DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run: python scripts/build_db.py")
    conn = sqlite3.connect(f"file:{path}?mode=ro" if readonly else path, uri=readonly, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn
