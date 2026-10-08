"""
Run one medication review from the command line and print the agent trace and the report.

    python scripts/run_review.py --case demo_high                  # a case from data/demo_cases.json
    python scripts/run_review.py --all                             # every demo case (summary table)
    python scripts/run_review.py --age 72 --meds "warfarin" "Advil 200mg"
    python scripts/run_review.py --text "80yo on Zocor 40mg and clarithromycin 500mg bid"
    python scripts/run_review.py --case demo_unknown --ack "abcdefg 10mg"     # reviewer: skip an unknown drug
    python scripts/run_review.py --case demo_high --role eval      # pinned evaluation model, no fallback

Human checkpoints (a paused review is resumed later, from a new process — state is in the checkpointer):
    python scripts/run_review.py --list                            # reviews waiting for a human
    python scripts/run_review.py --resume R-1a2b3c4d --action approve --reviewer "Pharmacist A"
    python scripts/run_review.py --resume R-1a2b3c4d --action escalate --reviewer "Pharmacist A" --comment "needs GP"
    python scripts/run_review.py --resume R-1a2b3c4d --action acknowledge --reviewer "Dr B" --comment "NSAID stopped"
    python scripts/run_review.py --resume R-1a2b3c4d --action resolve --reviewer "A" --skip "vitamin D"
    python scripts/run_review.py --resume R-1a2b3c4d --action resolve --reviewer "A" --correct "abcdefg 10mg=apixaban"
    python scripts/run_review.py --audit R-1a2b3c4d                # the audit trail of one review
"""
import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.runtime import DecisionError, PolySafeAgent  # noqa: E402
from db.review_store import ReviewStore  # noqa: E402
from src.config import get_data_dir  # noqa: E402

DEMOS = json.load(open(get_data_dir("demo_cases.json")))


def print_pending(out: dict):
    p = out.get("pending")
    if not p:
        return
    print(f"\n⏸  Waiting for a human ({p['type']}): {p['question']}")
    for u in p.get("unresolved", []):
        print(f"     unresolved: {u['input']!r}" + (f"  suggestions: {u['suggestions']}" if u["suggestions"] else ""))
    if p.get("needs_confirmation"):
        print(f"     confirm names: {p['needs_confirmation']}")
    print(f"   allowed: {p['allowed']}\n   resume:  python scripts/run_review.py --resume {out['review_id']} "
          f"--action {p['allowed'][0]} --reviewer \"<name>\" [--comment ...]")


def print_audit(review_id: str):
    rows = ReviewStore().audit(review_id)
    print(f"\n── Audit trail {review_id} ({len(rows)} entries) " + "─" * 30)
    for r in rows:
        ts = datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
        what = r["tool"] or r["node"] or ""
        det = r["detail"] or {}
        short = det.get("summary") or det.get("title") or det.get("action") or det.get("question") or ""
        print(f"  {ts} {r['actor']:<22} {r['event']:<15} {what:<22} {str(r['status'] or ''):<18} "
              f"{(r['source'] or '')[:45]:<45} {str(short)[:60]}")


def print_trace(out: dict):
    print("\n── Agent trace " + "─" * 60)
    for s in out["trace"]:
        print(f"  {s['node']:<18} {s.get('ms', 0):>6} ms  [{s['status']}] {s['summary']}")
        for t in s.get("tools", []):
            args = ", ".join(f"{k}={v}" for k, v in t["args"].items())
            print(f"      ↳ {t['tool']}({args[:90]})  {t['ms']} ms {'ok' if t['ok'] else 'FAILED'}")
    u = out["usage"]
    print(f"\n  status={out['status']} route={out['route']} revisions={out['revisions']} draft={out['draft_source']} "
          f"· {out['latency_s']} s · {u['llm_calls']} LLM calls · {u['total_tokens']:,} tokens {u['by_model']}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(DEMOS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--age", type=int)
    ap.add_argument("--meds", nargs="+")
    ap.add_argument("--text")
    ap.add_argument("--ack", nargs="+", default=[], help="unknown medication names the reviewer allows to skip")
    ap.add_argument("--role", default="agent", choices=["agent", "eval"])
    ap.add_argument("--quiet", action="store_true", help="no trace, report only")
    ap.add_argument("--list", action="store_true", help="reviews waiting for a human")
    ap.add_argument("--audit", metavar="REVIEW_ID")
    ap.add_argument("--resume", metavar="REVIEW_ID")
    ap.add_argument("--action")
    ap.add_argument("--reviewer")
    ap.add_argument("--comment")
    ap.add_argument("--summary", help="edit: replacement summary")
    ap.add_argument("--note", nargs=2, action="append", metavar=("FINDING_ID", "TEXT"), help="edit: reviewer note")
    ap.add_argument("--skip", nargs="+", default=[], help="resolve: unknown names to skip")
    ap.add_argument("--correct", nargs="+", default=[], metavar="OLD=NEW", help="resolve: name corrections")
    args = ap.parse_args()

    if args.audit:
        print_audit(args.audit)
        return
    if args.list:
        rows = ReviewStore().list_reviews(["pending_review", "escalated", "needs_clarification"])
        print(f"{'review_id':<26}{'status':<22}{'severity':<10}{'updated':<20}")
        for r in rows:
            print(f"{r['review_id']:<26}{r['status']:<22}{str(r['overall_severity']):<10}"
                  f"{datetime.fromtimestamp(r['updated_at']).strftime('%Y-%m-%d %H:%M'):<20}")
        return

    agent = await PolySafeAgent.create(role=args.role)
    if args.resume:
        decision = {k: v for k, v in {"action": args.action, "reviewer": args.reviewer, "comment": args.comment,
                                      "summary": args.summary, "finding_notes": dict(args.note or []),
                                      "skip": args.skip, "corrections": dict(c.split("=", 1) for c in args.correct)}.items() if v}
        try:
            out = await agent.resume(args.resume, decision)
        except DecisionError as e:
            print(f"Decision rejected: {e}")
            pending = await agent.pending(args.resume)
            if pending:
                print(f"Open checkpoint: {pending['type']}, allowed {pending['allowed']}")
            await agent.close()
            return
        if not args.quiet:
            print_trace(out)
        print("\n── Report " + "─" * 65 + "\n" + out["report_markdown"])
        print_pending(out)
        await agent.close()
        return
    if args.all:
        rows = []
        for key, demo in DEMOS.items():
            out = await agent.review(demo["case"], review_id=f"{key}-{uuid.uuid4().hex[:4]}")
            sev = (out.get("report") or {}).get("overall_severity")
            rows.append((key, out["status"], out["route"], sev, len(out["findings"]), out["latency_s"],
                         out["usage"]["total_tokens"]))
        print(f"\n{'case':<18}{'status':<22}{'route':<14}{'severity':<10}{'findings':>9}{'sec':>7}{'tokens':>9}")
        for r in rows:
            print(f"{r[0]:<18}{r[1]:<22}{str(r[2]):<14}{str(r[3]):<10}{r[4]:>9}{r[5]:>7}{r[6]:>9,}")
        await agent.close()
        return
    if args.case:
        case = dict(DEMOS[args.case]["case"])
    else:
        case = {k: v for k, v in {"age": args.age, "medications": args.meds, "text": args.text}.items() if v}
        if not case:
            ap.error("give --case, --all, --meds or --text")
    if args.ack:
        case["acknowledged_unknown"] = args.ack
    rid = f"{args.case}-{uuid.uuid4().hex[:4]}" if args.case else None
    out = await agent.review(case, review_id=rid)
    if not args.quiet:
        print_trace(out)
    print("\n── Report " + "─" * 65 + "\n" + out["report_markdown"])
    print_pending(out)
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
