"""
Medication Safety Review report, assembled deterministically from the workflow state.

Severity, effect, recommendation and source always come from the database; the LLM contributes only the
summary and the per-finding explanation (already checked by the safety reviewer). Evidence is quoted verbatim.
"""
from db.drug_db import SEVERITY_RANK

SEVERITY_LABEL = {"critical": "CRITICAL", "high": "HIGH", "moderate": "MODERATE", "low": "LOW"}
ROUTE_ACTION = {
    "escalate": "Critical risk: autonomous workflow stopped — escalated for prescriber review before the next dose.",
    "human_review": "Pharmacist review required before this review is released.",
    "finalize": "Released with warnings for routine clinician awareness.",
}
DISCLAIMER = ("PolySafe is a clinical decision-support prototype. It does not decide to stop, start, switch or "
              "re-dose any medication; all actions are for the reviewing clinician or pharmacist to decide.")


def overall_severity(findings: list[dict]) -> str | None:
    return max((f["severity"] for f in findings), key=SEVERITY_RANK.get, default=None)


def build_report(state: dict) -> dict:
    patient, findings = state.get("patient", {}), state.get("findings", [])
    draft = state.get("draft") or {}
    expl = {i["finding_id"]: i for i in draft.get("items", [])}
    evidence = state.get("evidence", {})
    items = []
    for f in findings:
        e = expl.get(f["id"], {})
        cited = set(e.get("evidence_ids", []))
        # every passage retrieved for the finding is shown (grounding is visible); the ones the explanation cites are marked
        shown = [evidence[x] | {"id": x, "cited": x in cited} for x in f.get("evidence_ids", []) if x in evidence]
        items.append({**f, "explanation": e.get("explanation", ""), "evidence": shown})
    meds = state.get("medications", [])
    return {
        "review_id": state.get("review_id"),
        "patient": patient,
        "medications_checked": [m for m in meds if m.get("recognized")],
        "not_checked": state.get("acknowledged_unknown", []),
        "needs_confirmation": [m for m in meds if m.get("needs_confirmation")],
        "pairs_checked": len(state.get("plan", {}).get("pairs", [])),
        "overall_severity": overall_severity(findings),
        "route": state.get("route"),
        "summary": draft.get("summary", ""),
        "draft_source": state.get("draft_source", "llm"),
        "scope_notice": state.get("scope_notice", ""),
        "findings": items,
        "decisions": state.get("decisions", []),
        "reviewer_notes": {},
        "disclaimer": DISCLAIMER,
    }


def render_markdown(r: dict) -> str:
    p = r["patient"]
    who = ", ".join(x for x in [f"{p['age']}y" if p.get("age") is not None else "age unknown", p.get("sex") or ""] if x)
    lines = [f"# Medication Safety Review — {r.get('review_id') or ''}".rstrip(" —"),
             f"Patient: {who} · {len(r['medications_checked'])} medications checked · {r['pairs_checked']} pairs",
             f"**Overall: {SEVERITY_LABEL.get(r['overall_severity'], 'NO ISSUES FOUND')}** — "
             + (ROUTE_ACTION.get(r["route"], "") if r["overall_severity"] or r["route"] != "finalize"
                else "Released; nothing to escalate."), ""]
    if r.get("release_status"):
        last = (r.get("decisions") or [{}])[-1]
        lines += [f"**Review status: {r['release_status'].replace('_', ' ').upper()}**"
                  + (f" — {last.get('reviewer', '')}, {last.get('ts', '')}" if last else ""), ""]
    if r.get("scope_notice"):
        lines += [f"> {r['scope_notice']}", ""]
    if r.get("summary"):
        lines += [r["summary"], ""]
    if r["needs_confirmation"]:
        lines.append("**Please confirm medication names:** " + "; ".join(
            f"'{m['input']}' was read as {m['generic_name']} ({int(m['confidence'] * 100)}% match)"
            for m in r["needs_confirmation"]))
    if r["not_checked"]:
        lines.append("**Not checked (not in the drug knowledge base, confirmed by reviewer):** " + ", ".join(r["not_checked"]))
    if r["needs_confirmation"] or r["not_checked"]:
        lines.append("")
    main = [f for f in r["findings"] if f["severity"] != "low"]
    notes = [f for f in r["findings"] if f["severity"] == "low"]
    if not main:
        lines += ["No interactions, duplications or high-risk combinations of moderate or higher severity were found in "
                  "the knowledge base for this list. Absence of a finding is not proof of safety.", ""]
    for f in main:
        lines.append(f"## [{SEVERITY_LABEL[f['severity']]}] {f['title']}")
        lines.append(f"- **Risk:** {f['effect']}")
        if f.get("mechanism"):
            lines.append(f"- **Mechanism:** {f['mechanism']}")
        if f.get("explanation"):
            lines.append(f"- **Assessment:** {f['explanation']}")
        lines.append(f"- **Suggested review action:** {f['recommendation']}")
        if r.get("reviewer_notes", {}).get(f["id"]):
            lines.append(f"- **Reviewer note:** {r['reviewer_notes'][f['id']]}")
        for ev in f["evidence"]:
            passage = " ".join(ev["passage"].split("\n", 1)[-1].split())   # drop the repeated section heading
            lines.append(f"- **Evidence [{ev['id']}]{' (cited)' if ev['cited'] else ''}** ({ev['source']} · "
                         f"{ev['section']}): \"{passage[:350]}{'…' if len(passage) > 350 else ''}\"")
        if f.get("evidence_status") == "insufficient":
            lines.append("- **Evidence:** insufficient evidence in the knowledge base — supported by the interaction database only")
        lines.append(f"- **Source:** {f['source']} ({', '.join(f['db_refs'])})")
        lines.append("")
    if notes:
        lines.append("## Lower-priority notes")
        for f in notes:
            lines.append(f"- [LOW] **{f['title']}** — {f['effect']}. _{f['recommendation']}._ ({f['source']})")
        lines.append("")
    if r.get("decisions"):
        lines.append("## Human review log")
        for d in r["decisions"]:
            lines.append(f"- {d['ts']} · {d['checkpoint']} · **{d['action']}** by {d.get('reviewer', '?')}"
                         + (f" — {d['comment']}" if d.get("comment") else ""))
        lines.append("")
    lines.append(f"_{r['disclaimer']}_")
    return "\n".join(lines)
