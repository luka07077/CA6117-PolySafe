"""
Runtime store (SQLite, data/runtime/polysafe_runtime.db), separate from the read-only drug knowledge base.

    reviews(review_id, created_at, updated_at, status, route, overall_severity, pending_type,
            patient_json, report_json, report_md, decisions_json)
        one row per medication review: the queue the pharmacist works from
    audit_log(id, review_id, ts, actor, event, node, tool, detail_json, source, status)
        append-only trail of everything that happened in a review:
          input_received · node · tool_call · finding (with its database source) · evidence (with document source)
          · report · interrupt (waiting for a human) · human_decision · status

The LangGraph checkpoints (the paused workflow state itself) live next to it in checkpoints.sqlite.
"""
import json
import os
import sqlite3
import threading
import time

from src.config import get_data_dir

RUNTIME_DIR = get_data_dir("runtime")
DB_PATH = os.path.join(RUNTIME_DIR, "polysafe_runtime.db")
CHECKPOINT_PATH = os.path.join(RUNTIME_DIR, "checkpoints.sqlite")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, status TEXT, route TEXT,
    overall_severity TEXT, pending_type TEXT, patient_json TEXT, report_json TEXT, report_md TEXT,
    decisions_json TEXT);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL, ts REAL NOT NULL, actor TEXT NOT NULL,
    event TEXT NOT NULL, node TEXT, tool TEXT, detail_json TEXT, source TEXT, status TEXT);
CREATE INDEX IF NOT EXISTS idx_audit_review ON audit_log(review_id, id);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status, updated_at);
"""


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class ReviewStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------- audit (append-only)
    def log(self, review_id: str, actor: str, event: str, node: str | None = None, tool: str | None = None,
            detail=None, source: str | None = None, status: str | None = None):
        with _lock, self._conn() as c:
            c.execute("INSERT INTO audit_log (review_id, ts, actor, event, node, tool, detail_json, source, status) "
                      "VALUES (?,?,?,?,?,?,?,?,?)",
                      (review_id, time.time(), actor, event, node, tool, _dumps(detail) if detail is not None else None,
                       source, status))

    def audit(self, review_id: str | None = None, limit: int = 5000) -> list[dict]:
        q, args = "SELECT * FROM audit_log", []
        if review_id:
            q, args = q + " WHERE review_id=?", [review_id]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY id LIMIT ?", args + [limit]).fetchall()
        return [dict(r) | {"detail": json.loads(r["detail_json"]) if r["detail_json"] else None} for r in rows]

    # ---------------- reviews
    def upsert_review(self, review_id: str, **fields):
        cols = {"status", "route", "overall_severity", "pending_type", "patient_json", "report_json", "report_md",
                "decisions_json"}
        fields = {k: v for k, v in fields.items() if k in cols}
        now = time.time()
        with _lock, self._conn() as c:
            c.execute("INSERT OR IGNORE INTO reviews (review_id, created_at, updated_at) VALUES (?,?,?)",
                      (review_id, now, now))
            if fields:
                sets = ", ".join(f"{k}=?" for k in fields)
                c.execute(f"UPDATE reviews SET {sets}, updated_at=? WHERE review_id=?", [*fields.values(), now, review_id])

    def get_review(self, review_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        return self._row(r) if r else None

    def list_reviews(self, statuses: list[str] | None = None, limit: int = 200) -> list[dict]:
        q, args = "SELECT * FROM reviews", []
        if statuses:
            q += f" WHERE status IN ({','.join('?' * len(statuses))})"
            args = list(statuses)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY updated_at DESC LIMIT ?", args + [limit]).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        for k in ("patient_json", "report_json", "decisions_json"):
            d[k.removesuffix("_json")] = json.loads(d.pop(k)) if d.get(k) else None
        return d
