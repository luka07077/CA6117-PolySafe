"""
Entry point used by the CLI, the demo UI and the evaluation scripts.

    agent = await PolySafeAgent.create()                    # role "agent" (dev/demo) or "eval" (pinned eval model)
    out = await agent.review({"age": 72, "medications": ["warfarin", "Advil"]})
    out["status"]   # completed | pending_review | escalated | needs_clarification | refused
    out["pending"]  # the open human checkpoint (type, allowed actions, question), if any
    out = await agent.resume(out["review_id"], {"action": "approve", "reviewer": "Pharmacist A"})

A case is a dict with any of: age, sex, medications (list of names as written), conditions, allergies,
text (free-text note, parsed by the LLM), note, acknowledged_unknown (names a reviewer allows to skip).

State across waits: every review is a LangGraph thread (thread_id = review_id) saved by an SQLite checkpointer
(data/runtime/checkpoints.sqlite), so a paused review can be resumed by another process later.
Audit: every step is appended to data/runtime/polysafe_runtime.db (db/review_store.py) while it runs.
For sync callers (Streamlit) use SyncPolySafeAgent, which runs a private event loop in a thread.
"""
import asyncio
import os
import queue
import threading
import time
import uuid

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agent.graph import PolySafeGraph
from agent.guardrails import redact_pii
from agent.models.cloud_model import CloudChatModel, UsageTracker
from db.review_store import CHECKPOINT_PATH, ReviewStore
from tools.mcp_client import load_tools

CHECKPOINT_NODES = {"human_review", "escalate", "await_clarification"}
REQUIRE_COMMENT = {"reject", "escalate", "acknowledge", "edit", "cancel"}


class DecisionError(ValueError):
    """The human decision does not fit the open checkpoint."""


def _redact(obj):
    if isinstance(obj, str):
        return redact_pii(obj)
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    return obj


class PolySafeAgent:
    def __init__(self, graph: PolySafeGraph, tools: dict, model_name: str, store: ReviewStore, conn=None):
        self.graph, self.tools, self.model_name, self.store, self._conn = graph, tools, model_name, store, conn

    @classmethod
    async def create(cls, role: str = "agent", pipeline_overrides: dict | None = None, persist: bool = True,
                     store_path: str | None = None, checkpoint_path: str | None = None) -> "PolySafeAgent":
        """persist=False keeps checkpoints in memory (evaluation); store and checkpoint paths can be redirected (tests)."""
        tools = await load_tools()
        model = CloudChatModel(role)
        helper = CloudChatModel("helper").llm
        conn = None
        if persist:
            path = checkpoint_path or CHECKPOINT_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            conn = await aiosqlite.connect(path)
            saver = AsyncSqliteSaver(conn)
        else:
            saver = InMemorySaver()
        graph = PolySafeGraph(tools, model.llm, helper, pipeline_overrides, model.fallback_llms, checkpointer=saver)
        return cls(graph, tools, model.model_name, ReviewStore(store_path), conn)

    async def close(self):
        if self._conn is not None:
            await self._conn.close()

    @staticmethod
    def _config(review_id: str, usage: UsageTracker) -> dict:
        return {"configurable": {"thread_id": review_id}, "callbacks": [usage], "recursion_limit": 60}

    # ------------------------------------------------------------ public API
    async def review(self, case: dict, review_id: str | None = None, on_update=None) -> dict:
        """on_update(node, update) is called after every graph step (live progress in the UI)."""
        review_id = review_id or f"R-{uuid.uuid4().hex[:8]}"
        self.store.upsert_review(review_id, status="running")
        self.store.log(review_id, "user", "input_received", detail=_redact(case))
        return await self._run(review_id, {"review_id": review_id, "case": case, "trace": [], "tool_usage": []}, on_update)

    async def state(self, review_id: str) -> dict:
        """Current workflow state of a review (trace, plan, findings, report…) from the checkpointer."""
        snap = await self.graph.graph.aget_state({"configurable": {"thread_id": review_id}})
        return dict(snap.values or {})

    async def pending(self, review_id: str) -> dict | None:
        snap = await self.graph.graph.aget_state({"configurable": {"thread_id": review_id}})
        tasks = [i.value for t in snap.tasks for i in t.interrupts]
        return tasks[0] if tasks else None

    async def resume(self, review_id: str, decision: dict, on_update=None) -> dict:
        """Continue a paused review with a human decision; raises DecisionError if it does not fit the checkpoint."""
        pending = await self.pending(review_id)
        if not pending:
            raise DecisionError(f"review {review_id} is not waiting for a human decision")
        action = decision.get("action")
        if action not in pending["allowed"]:
            raise DecisionError(f"action {action!r} not allowed at {pending['type']}; choose one of {pending['allowed']}")
        if not (decision.get("reviewer") or "").strip():
            raise DecisionError("reviewer name is required (accountability)")
        if action in REQUIRE_COMMENT and not (decision.get("comment") or "").strip():
            raise DecisionError(f"a comment is required for {action!r}")
        if action == "edit" and not (decision.get("summary") or decision.get("finding_notes")):
            raise DecisionError("an edit needs a new summary or at least one finding note")
        if action == "resolve" and not (decision.get("corrections") or decision.get("skip")):
            raise DecisionError("resolve needs corrections {name: new name} and/or skip [names]")
        self.store.log(review_id, f"human:{decision['reviewer']}", "resume", detail={"checkpoint": pending["type"],
                                                                                    "action": action})
        return await self._run(review_id, Command(resume=decision), on_update)

    # ------------------------------------------------------------ run + audit
    async def _run(self, review_id: str, payload, on_update=None) -> dict:
        t0, usage = time.perf_counter(), UsageTracker()
        config = self._config(review_id, usage)
        tool_tokens = []
        async for chunk in self.graph.graph.astream(payload, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    for itr in update:
                        self.store.log(review_id, "system", "interrupt", detail=itr.value, status="waiting_for_human")
                    continue
                self._audit_update(review_id, node, update or {})
                tool_tokens += (update or {}).get("tool_usage", [])
                if on_update:
                    on_update(node, update or {})
        snap = await self.graph.graph.aget_state({"configurable": {"thread_id": review_id}})
        values = snap.values
        pending = next((i.value for t in snap.tasks for i in t.interrupts), None)
        tok = usage.as_dict()
        for u in tool_tokens:   # helper-LLM calls made inside tools (RAG reflection)
            tok["llm_calls"] += u.get("llm_calls", 0)
            tok["total_tokens"] += u.get("total_tokens", 0)
            for m, v in (u.get("by_model") or {}).items():
                b = tok["by_model"].setdefault(m, {"calls": 0, "tokens": 0})
                b["calls"] += v["calls"]
                b["tokens"] += v["tokens"]
        report = values.get("report") or {}
        self.store.upsert_review(review_id, status=values.get("status"), route=values.get("route"),
                                 overall_severity=report.get("overall_severity"),
                                 pending_type=pending["type"] if pending else None,
                                 patient_json=_dumps(values.get("patient")), report_json=_dumps(report),
                                 report_md=values.get("report_markdown", ""), decisions_json=_dumps(values.get("decisions", [])))
        self.store.log(review_id, "system", "usage", detail=tok | {"latency_s": round(time.perf_counter() - t0, 2)},
                       status=values.get("status"))
        return {"review_id": review_id, "status": values.get("status"), "route": values.get("route"),
                "pending": pending, "report": report, "report_markdown": values.get("report_markdown", ""),
                "findings": values.get("findings", []), "medications": values.get("medications", []),
                "patient": values.get("patient"), "plan": values.get("plan"), "guard": values.get("guard"),
                "review": values.get("review"), "revisions": values.get("revision_count", 0),
                "decisions": values.get("decisions", []), "draft_source": values.get("draft_source"),
                "trace": values.get("trace", []), "usage": tok, "model": self.model_name,
                "latency_s": round(time.perf_counter() - t0, 2)}

    def _audit_update(self, review_id: str, node: str, update: dict):
        log = self.store.log
        for d in update.get("decisions", []):
            log(review_id, f"human:{d.get('reviewer', '?')}", "human_decision", node=node, detail=d, status=d["action"])
        if node not in CHECKPOINT_NODES:
            for step in update.get("trace", []):
                extra = {k: v for k, v in step.items() if k not in ("node", "tools", "status")}
                log(review_id, "agent", "node", node=step["node"], detail=extra, status=step["status"])
                for t in step.get("tools", []):
                    log(review_id, "agent", "tool_call", node=step["node"], tool=t["tool"],
                        detail={k: t[k] for k in ("args", "ms", "ok", "preview")}, status="ok" if t["ok"] else "failed")
        if node == "run_checks":
            for f in update.get("findings", []):
                log(review_id, "agent", "finding", node=node, source=f"{f['source']} [{', '.join(f['db_refs'])}]",
                    detail={k: f[k] for k in ("id", "type", "title", "severity", "effect")}, status=f["severity"])
        if node == "retrieve_evidence":
            for eid, e in (update.get("evidence") or {}).items():
                log(review_id, "agent", "evidence", node=node, source=f"{e['source']} · {e['section']}",
                    detail={"id": eid, "passage": e["passage"][:300]})
        if node in ("compose_report", "finalize", "refuse", "clarify"):
            r = update.get("report") or {}
            log(review_id, "agent", "report", node=node, status=update.get("status"),
                detail={"overall_severity": r.get("overall_severity"), "route": update.get("route") or r.get("route"),
                        "findings": len(r.get("findings") or []), "summary": r.get("summary", "")[:300]})


def _dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False, default=str) if obj is not None else None


class SyncPolySafeAgent:
    """Thread-safe synchronous facade (one background event loop) for Streamlit and scripts."""

    def __init__(self, **kwargs):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self.agent: PolySafeAgent = self._run(PolySafeAgent.create(**kwargs))

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def review(self, case: dict, review_id: str | None = None) -> dict:
        return self._run(self.agent.review(case, review_id))

    def resume(self, review_id: str, decision: dict) -> dict:
        return self._run(self.agent.resume(review_id, decision))

    def stream(self, method: str, *args):
        """
        Run review(...) or resume(...) in the background loop and yield ("step", node, update) as each graph step
        finishes, then ("done", result) — or ("error", exception). Lets the UI show the workflow live.
        """
        q: queue.Queue = queue.Queue()
        fut = asyncio.run_coroutine_threadsafe(
            getattr(self.agent, method)(*args, on_update=lambda n, u: q.put(("step", n, u))), self._loop)
        while True:
            try:
                yield q.get(timeout=0.1)
            except queue.Empty:
                if fut.done():
                    while not q.empty():
                        yield q.get()
                    exc = fut.exception()
                    yield ("error", exc) if exc else ("done", fut.result())
                    return

    def state(self, review_id: str) -> dict:
        return self._run(self.agent.state(review_id))

    def pending(self, review_id: str) -> dict | None:
        return self._run(self.agent.pending(review_id))

    @property
    def store(self) -> ReviewStore:
        return self.agent.store
