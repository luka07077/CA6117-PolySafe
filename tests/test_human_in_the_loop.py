"""
Integration tests for the human checkpoints (LangGraph interrupts + SQLite checkpointer) and the audit log.
Each test runs real reviews (LLM calls), so the file is skipped without an API key. Stores go to a temp dir.

    pytest -q tests/test_human_in_the_loop.py
"""
import asyncio
import json
import os

import pytest

from src.config import get_data_dir

pytestmark = pytest.mark.skipif(not os.environ.get("DASHSCOPE_API_KEY"), reason="needs DASHSCOPE_API_KEY in .env")
DEMOS = json.load(open(get_data_dir("demo_cases.json")))


@pytest.fixture
def paths(tmp_path):
    return {"store_path": str(tmp_path / "runtime.db"), "checkpoint_path": str(tmp_path / "checkpoints.sqlite")}


def run(coro):
    return asyncio.run(coro)


async def _agent(paths):
    from agent.runtime import PolySafeAgent
    return await PolySafeAgent.create(**paths)


def test_pharmacist_edit_then_resume_in_new_process(paths):
    """pause at pharmacist review -> invalid decisions refused -> edit applied by a NEW agent instance."""
    from agent.runtime import DecisionError

    async def first():
        a = await _agent(paths)
        out = await a.review(DEMOS["demo_out_of_scope"]["case"], review_id="t-edit")
        await a.close()
        return out

    out = run(first())
    assert out["status"] == "pending_review" and out["pending"]["type"] == "pharmacist_review"
    assert "does not decide" in out["report_markdown"]          # scope notice for the "which drug to stop" request

    async def second():   # simulates a different process: new agent, same checkpoint + store files
        a = await _agent(paths)
        for bad in ({"action": "approve"},                                     # no reviewer
                    {"action": "reject", "reviewer": "A"},                     # reject without comment
                    {"action": "acknowledge", "reviewer": "A", "comment": "x"},  # not allowed at this checkpoint
                    {"action": "edit", "reviewer": "A", "comment": "x"}):        # edit without content
            with pytest.raises(DecisionError):
                await a.resume("t-edit", bad)
        out = await a.resume("t-edit", {"action": "edit", "reviewer": "Pharmacist A", "comment": "added GP plan",
                                        "finding_notes": {"F1": "Discussed with GP: paracetamol trial planned."}})
        audit = a.store.audit("t-edit")
        await a.close()
        return out, audit

    out, audit = run(second())
    assert out["status"] == "approved_with_edits" and out["pending"] is None
    assert "Discussed with GP" in out["report_markdown"] and "Human review log" in out["report_markdown"]
    events = [r["event"] for r in audit]
    for e in ("input_received", "node", "tool_call", "finding", "evidence", "report", "interrupt", "human_decision"):
        assert e in events, e
    decision = next(r for r in audit if r["event"] == "human_decision")
    assert decision["actor"] == "human:Pharmacist A" and decision["ts"] > 0
    assert any(r["event"] == "finding" and r["source"] for r in audit)


def test_critical_escalation_acknowledged(paths):
    async def go():
        a = await _agent(paths)
        out = await a.review(DEMOS["demo_critical"]["case"], review_id="t-crit")
        assert out["status"] == "escalated" and out["pending"]["type"] == "prescriber_escalation"
        out = await a.resume("t-crit", {"action": "acknowledge", "reviewer": "Dr B",
                                        "comment": "Benzodiazepine taper started; naloxone prescribed"})
        await a.close()
        return out
    out = run(go())
    assert out["status"] == "acknowledged" and "Review status: ACKNOWLEDGED" in out["report_markdown"]


def test_pharmacist_escalates_then_prescriber_rejects(paths):
    async def go():
        a = await _agent(paths)
        await a.review(DEMOS["demo_out_of_scope"]["case"], review_id="t-esc")
        out = await a.resume("t-esc", {"action": "escalate", "reviewer": "Pharmacist A", "comment": "needs prescriber"})
        assert out["status"] == "escalated" and out["pending"]["type"] == "prescriber_escalation"
        out = await a.resume("t-esc", {"action": "reject", "reviewer": "Dr B", "comment": "wrong patient list"})
        await a.close()
        return out
    out = run(go())
    assert out["status"] == "rejected" and [d["action"] for d in out["decisions"]] == ["escalate", "reject"]


def test_clarification_correction_reruns_checks(paths):
    """unknown drug -> pause -> reviewer corrects the name -> workflow re-normalises and finds the duplication."""
    async def go():
        a = await _agent(paths)
        out = await a.review(DEMOS["demo_unknown"]["case"], review_id="t-clar")
        assert out["status"] == "needs_clarification" and out["pending"]["type"] == "clarification"
        out = await a.resume("t-clar", {"action": "resolve", "reviewer": "Pharmacist A",
                                        "corrections": {"abcdefg 10mg": "Eliquis 5mg"}})
        await a.close()
        return out
    out = run(go())
    # warfarin + apixaban = duplicate anticoagulation (critical) -> escalated to the prescriber
    assert out["status"] == "escalated" and out["report"]["overall_severity"] == "critical"


def test_clarification_skip_unknown(paths):
    async def go():
        a = await _agent(paths)
        await a.review({"age": 60, "medications": ["warfarin", "Advil", "vitamin D 1000 IU"]}, review_id="t-skip")
        out = await a.resume("t-skip", {"action": "resolve", "reviewer": "Pharmacist A", "skip": ["vitamin D 1000 IU"]})
        await a.close()
        return out
    out = run(go())
    assert out["status"] == "pending_review"                     # warfarin + ibuprofen (high)
    assert "vitamin D 1000 IU" in out["report_markdown"]          # listed as "not checked"
