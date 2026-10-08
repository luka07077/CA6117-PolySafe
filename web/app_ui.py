"""
PolySafe demo UI (Streamlit).

    streamlit run web/app_ui.py

Pages: New review (input + live workflow) · Agent workflow (plan, state transitions, tool calls, graph) ·
Safety report · Human review (checkpoint queue + decisions) · Audit log.
"""
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent.runtime import DecisionError, SyncPolySafeAgent  # noqa: E402
from db.review_store import ReviewStore  # noqa: E402
from src.config import get_agent_config, get_data_dir  # noqa: E402

st.set_page_config(page_title="PolySafe", page_icon="💊", layout="wide")
ss = st.session_state
for _k, _v in {"f_age": None, "f_sex": "—", "f_conditions": "", "f_allergies": "", "f_meds": "", "f_text": ""}.items():
    ss.setdefault(_k, _v)   # form fields live in session state so a demo case can prefill them

SEV_COLOR = {"critical": "red", "high": "orange", "moderate": "yellow", "low": "gray"}
STATUS = {
    "completed": ("Released", "green"), "approved": ("Approved by pharmacist", "green"),
    "approved_with_edits": ("Approved with edits", "green"), "acknowledged": ("Acknowledged by prescriber", "green"),
    "pending_review": ("Waiting for pharmacist", "orange"), "escalated": ("Escalated to prescriber", "red"),
    "needs_clarification": ("Needs clarification", "orange"), "refused": ("Refused (safety)", "red"),
    "rejected": ("Rejected", "gray"), "cancelled": ("Cancelled", "gray"), "running": ("Running", "blue"),
}
PENDING = ["pending_review", "escalated", "needs_clarification"]
PHASE = {"guard_input": "Perceive", "parse_input": "Perceive", "normalize_meds": "Perceive",
         "plan_checks": "Reason", "assess_risk": "Reason", "safety_reviewer": "Reason", "revise": "Reason",
         "run_checks": "Act", "retrieve_evidence": "Act", "compose_report": "Act", "finalize": "Act",
         "fallback": "Safety", "refuse": "Safety stop", "clarify": "Safety stop",
         "human_review": "Human", "escalate": "Human", "await_clarification": "Human"}
NODE_ICON = {"Perceive": "👁️", "Reason": "🧠", "Act": "🛠️", "Safety": "🛡️", "Safety stop": "🛑", "Human": "🧑‍⚕️"}


# ---------------------------------------------------------------- shared resources
@st.cache_resource(show_spinner="Starting PolySafe agent (MCP tool server, models)…")
def get_agent() -> SyncPolySafeAgent:
    return SyncPolySafeAgent()


@st.cache_resource
def get_store() -> ReviewStore:
    return ReviewStore()


@st.cache_data
def demos() -> dict:
    return json.load(open(get_data_dir("demo_cases.json")))


def badge(status: str | None) -> str:
    label, color = STATUS.get(status or "", (status or "—", "gray"))
    return f":{color}-badge[{label}]"


def sev_badge(sev: str | None, status: str | None = None) -> str:
    if status in ("needs_clarification", "refused", "cancelled", "running"):
        return ":gray-badge[NOT ASSESSED]"   # no screening result yet: never show "no issues"
    return f":{SEV_COLOR[sev]}-badge[{sev.upper()}]" if sev else ":green-badge[NO ISSUES]"


def ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def pick_review(label: str, statuses: list[str] | None = None, key: str = "pick") -> str | None:
    rows = get_store().list_reviews(statuses, limit=100)
    if not rows:
        st.info("No reviews yet — start one on **New review**.")
        return None
    ids = [r["review_id"] for r in rows]
    names = {r["review_id"]: f"{r['review_id']} · {STATUS.get(r['status'], (r['status'],))[0]}"
                                + (f" · {r['overall_severity'].upper()}" if r["overall_severity"] else "") for r in rows}
    idx = ids.index(ss.current) if ss.get("current") in ids else 0
    rid = st.selectbox(label, ids, index=idx, format_func=names.get, key=key)
    ss.current = rid
    return rid


def step_line(step: dict) -> str:
    phase = PHASE.get(step["node"], "")
    tools = f" · {len(step['tools'])} tool call(s)" if step.get("tools") else ""
    ms = f" · {step['ms']} ms" if step.get("ms") else ""
    return f"{NODE_ICON.get(phase, '•')} **{phase} · `{step['node']}`** — {step['summary']}{tools}{ms}"


def run_live(method: str, *args) -> dict | None:
    """Run review/resume and show each workflow step as it completes."""
    with st.status("Agent working…", expanded=True) as box:
        for item in get_agent().stream(method, *args):
            if item[0] == "step":
                for step in item[2].get("trace", []):
                    st.markdown(step_line(step))
            elif item[0] == "error":
                if isinstance(item[1], DecisionError):   # validation refused the decision: nothing was changed
                    box.update(label=f"Decision not accepted — {item[1]}", state="error", expanded=True)
                    st.markdown("The review is still waiting; correct the decision and submit again.")
                else:
                    box.update(label=f"Failed — {type(item[1]).__name__}", state="error", expanded=True)
                    st.exception(item[1])
                return None
            else:
                out = item[1]
                label, _ = STATUS.get(out["status"], (out["status"], ""))
                box.update(label=f"{label} · {out['latency_s']} s · {out['usage']['llm_calls']} LLM calls · "
                                 f"{out['usage']['total_tokens']:,} tokens",
                           state="complete" if out["status"] not in ("refused",) else "error", expanded=False)
                ss.current = out["review_id"]
                return out


# ---------------------------------------------------------------- page: new review
def _load_demo():
    key = ss.demo_choice
    case = demos()[key]["case"] if key in demos() else {}
    ss.f_age = case.get("age")
    ss.f_sex = case.get("sex") or "—"
    ss.f_conditions = ", ".join(case.get("conditions", []))
    ss.f_allergies = ", ".join(case.get("allergies", []))
    ss.f_meds = "\n".join(case.get("medications", []))
    ss.f_text = case.get("text") or case.get("note") or ""


def page_new():
    st.title("New medication review")
    st.caption("Enter the patient's medication list (one per line — brand names, misspellings and dose text are fine) "
               "and/or paste a free-text note. PolySafe screens it and prepares a review for the pharmacist.")
    opts = ["blank"] + list(demos())
    st.selectbox("Load a demo case", opts, key="demo_choice", on_change=_load_demo,
                 format_func=lambda k: "— blank form —" if k == "blank" else demos()[k]["label"])
    with st.form("case"):
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.number_input("Age", min_value=0, max_value=120, key="f_age", placeholder="unknown")
        c2.selectbox("Sex", ["—", "female", "male", "other"], key="f_sex")
        c3.text_input("Conditions (comma-separated)", key="f_conditions")
        m1, m2 = st.columns(2)
        m1.text_area("Current medications (one per line)", key="f_meds", height=180,
                     placeholder="Warfarin 5 mg daily\nAdvil as needed\n…")
        m2.text_area("Free-text note (optional, parsed by the LLM)", key="f_text", height=180,
                     placeholder="81 year old woman with heart failure. Current meds: Zestril 20mg od, Lasix 40mg…")
        st.text_input("Drug allergies (comma-separated)", key="f_allergies")
        submitted = st.form_submit_button("Run medication review", type="primary", icon=":material/play_arrow:")
    if not submitted:
        last = get_store().get_review(ss["last_run"]) if ss.get("last_run") else None
        if last:   # keep the latest result visible across reruns (e.g. after typing the reviewer name)
            result_card(last | {"pending": get_agent().pending(last["review_id"])})
        return
    split = lambda s: [x.strip() for x in (s or "").split(",") if x.strip()]  # noqa: E731
    case = {"age": ss.f_age, "sex": None if ss.f_sex == "—" else ss.f_sex, "conditions": split(ss.f_conditions),
            "allergies": split(ss.f_allergies),
            "medications": [m.strip() for m in (ss.f_meds or "").splitlines() if m.strip()]}
    if (ss.f_text or "").strip():
        case["text"] = ss.f_text.strip()
    case = {k: v for k, v in case.items() if v not in (None, [], "")}
    if not case.get("medications") and not case.get("text"):
        st.warning("Enter at least one medication or a free-text note.")
        return
    out = run_live("review", case)
    if out:
        ss.last_run = out["review_id"]
        result_card(out)


def result_card(out: dict):
    r = out.get("report") or {}
    with st.container(border=True):
        a, b, c = st.columns([2, 2, 3])
        a.markdown(f"**{out['review_id']}**  \n{badge(out['status'])}")
        b.markdown(f"Overall risk  \n{sev_badge(r.get('overall_severity'), out['status'])}")
        if out.get("pending"):
            c.warning(out["pending"]["question"], icon=":material/pan_tool:")
        l1, l2, l3 = st.columns(3)
        l1.page_link(PAGES["report"], label="Open safety report", icon=":material/description:")
        l2.page_link(PAGES["workflow"], label="See agent workflow", icon=":material/account_tree:")
        if out.get("pending"):
            l3.page_link(PAGES["review"], label="Go to human review", icon=":material/how_to_reg:")


# ---------------------------------------------------------------- page: workflow
def graph_dot(visited: list[str], current: str | None) -> str:
    g = get_agent().agent.graph.graph.get_graph()
    lines = ['digraph G { rankdir=TB; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" '
             'fontsize=10 fillcolor="#f2f2f2" color="#bbbbbb"]; edge [color="#999999" arrowsize=0.6];']
    for nid in g.nodes:
        if nid in ("__start__", "__end__"):
            lines.append(f'"{nid}" [label="{"START" if nid == "__start__" else "END"}" shape=circle fontsize=8];')
            continue
        fill = "#ffd59e" if nid == current else ("#b7e4c7" if nid in visited else "#f2f2f2")
        lines.append(f'"{nid}" [label="{nid}\\n({PHASE.get(nid, "")})" fillcolor="{fill}"];')
    for e in g.edges:
        style = "dashed" if e.conditional else "solid"
        lines.append(f'"{e.source}" -> "{e.target}" [style={style}];')
    return "\n".join(lines + ["}"])


def page_workflow():
    st.title("Agent workflow")
    st.caption("Perceive → Reason → Act, step by step: what the agent planned, every state transition and tool call, "
               "and where it is now.")
    rid = pick_review("Review", key="wf_pick")
    if not rid:
        return
    state = get_agent().state(rid)
    if not state:
        st.warning("No workflow state stored for this review.")
        return
    trace = state.get("trace", [])
    pending = get_agent().pending(rid)
    usage = [r for r in get_store().audit(rid) if r["event"] == "usage"]
    m = st.columns(5)
    m[0].markdown(f"Status  \n{badge(state.get('status'))}")
    m[1].markdown(f"Overall risk  \n{sev_badge((state.get('report') or {}).get('overall_severity'), state.get('status'))}")
    m[2].metric("Steps", len(trace))
    m[3].metric("Tool calls", sum(len(s.get("tools", [])) for s in trace))
    m[4].metric("LLM tokens", f"{sum(u['detail'].get('total_tokens', 0) for u in usage):,}")

    left, right = st.columns([3, 2])
    with left:
        plan = state.get("plan")
        if plan:
            st.subheader("Plan (Reason)")
            n = len(dict.fromkeys(plan["drugs"]))
            st.markdown(f"{n} recognised drugs → **C({n},2) = {len(plan['pairs'])} pair checks** + "
                        + ", ".join(f"`{c}`" for c in plan["regimen_checks"]))
            st.dataframe(pd.DataFrame(plan["pairs"], columns=["drug A", "drug B"]), hide_index=True, height=180)
        st.subheader("State transitions")
        for i, s in enumerate(trace, 1):
            with st.expander(f"{i}. {PHASE.get(s['node'], '')} · {s['node']} — [{s['status']}] {s['summary'][:90]}",
                             expanded=False):
                st.markdown(step_line(s))
                if s.get("tools"):
                    st.dataframe(pd.DataFrame([{"tool": t["tool"], "args": json.dumps(t["args"], ensure_ascii=False)[:120],
                                                "ms": t["ms"], "ok": t["ok"], "result": t["preview"]} for t in s["tools"]]),
                                 hide_index=True)
                extra = {k: v for k, v in s.items() if k not in ("node", "ms", "status", "summary", "tools")}
                if extra:
                    st.json(extra, expanded=False)
        if pending:
            st.warning(f"Paused at **{pending['type']}** — {pending['question']}", icon=":material/pause_circle:")
    with right:
        st.subheader("Workflow graph")
        st.caption("green = executed · orange = waiting for a human · dashed = conditional route")
        visited = [s["node"] for s in trace]
        current = {"pharmacist_review": "human_review", "prescriber_escalation": "escalate",
                   "clarification": "await_clarification"}.get(pending["type"]) if pending else None
        st.graphviz_chart(graph_dot(visited, current), width="stretch")
        with st.expander("Medications (normalisation)"):
            meds = state.get("medications", [])
            if meds:
                st.dataframe(pd.DataFrame([{"as written": m["input"], "drug": m.get("generic_name", "—"),
                                            "match": m["match_type"], "confidence": m["confidence"],
                                            "confirm?": m.get("needs_confirmation", False)} for m in meds]), hide_index=True)
        with st.expander("Input guard & safety reviewer"):
            st.json({"guard": state.get("guard"), "safety_reviewer": state.get("review"),
                     "revisions": state.get("revision_count", 0), "draft_source": state.get("draft_source")})


# ---------------------------------------------------------------- page: report
def page_report():
    st.title("Medication safety report")
    rid = pick_review("Review", key="rep_pick")
    if not rid:
        return
    rev = get_store().get_review(rid)
    r = rev.get("report") or {}
    if not r.get("findings") and r.get("findings") != []:   # refusal / clarification stub
        st.markdown(f"{badge(rev['status'])}")
        st.markdown(rev.get("report_md") or "")
        if rev["status"] == "needs_clarification":
            st.page_link(PAGES["review"], label="Resolve in Human review", icon=":material/how_to_reg:")
        return
    p = r.get("patient", {})
    m = st.columns(5)
    m[0].markdown(f"Status  \n{badge(rev['status'])}")
    m[1].markdown(f"Overall risk  \n{sev_badge(r.get('overall_severity'))}")
    m[2].metric("Patient", f"{p.get('age', '?')}y {(p.get('sex') or '')[:1].upper()}")
    m[3].metric("Medications checked", len(r.get("medications_checked", [])))
    m[4].metric("Pairs checked", r.get("pairs_checked", 0))
    if r.get("scope_notice"):
        st.info(r["scope_notice"], icon=":material/policy:")
    if rev["status"] in PENDING:
        st.warning("Not released yet — waiting for a human decision.", icon=":material/pending_actions:")
        st.page_link(PAGES["review"], label="Open in Human review", icon=":material/how_to_reg:")
    if r.get("summary"):
        st.markdown(f"> {r['summary']}")
    for m_ in r.get("needs_confirmation", []):
        st.warning(f"Please confirm: '{m_['input']}' was read as **{m_['generic_name']}** "
                   f"({int(m_['confidence'] * 100)}% match)", icon=":material/spellcheck:")
    if r.get("not_checked"):
        st.warning("Not checked (not in the knowledge base, confirmed by reviewer): " + ", ".join(r["not_checked"]))

    main = [f for f in r["findings"] if f["severity"] != "low"]
    notes = [f for f in r["findings"] if f["severity"] == "low"]
    if not main:
        st.success("No interactions, duplications or high-risk combinations of moderate or higher severity found. "
                   "Absence of a finding is not proof of safety.")
    for f in main:
        with st.container(border=True):
            st.markdown(f"{sev_badge(f['severity'])} **{f['title']}** &nbsp; :gray[{f['type']} · {f['id']}]")
            st.markdown(f"**Risk:** {f['effect']}" + (f"  \n**Mechanism:** {f['mechanism']}" if f.get("mechanism") else ""))
            if f.get("explanation"):
                st.markdown(f"**Assessment:** {f['explanation']}")
            st.markdown(f"**Suggested review action:** {f['recommendation']}")
            note = (r.get("reviewer_notes") or {}).get(f["id"])
            if note:
                st.markdown(f":blue-background[**Reviewer note:** {note}]")
            ev = f.get("evidence", [])
            label = f"Evidence ({len(ev)} passage{'s' if len(ev) != 1 else ''})" if ev else "Evidence: insufficient — database only"
            with st.expander(label):
                for e in ev:
                    body = " ".join(e["passage"].split("\n", 1)[-1].split())
                    st.markdown(f"**[{e['id']}]**{' · cited' if e['cited'] else ''} — *{e['source']} · {e['section']}*")
                    st.caption(body)
                st.caption(f"Source: {f['source']} ({', '.join(f['db_refs'])})")
    if notes:
        st.subheader("Lower-priority notes")
        for f in notes:
            st.markdown(f"{sev_badge('low')} **{f['title']}** — {f['effect']}. *{f['recommendation']}.*")
    if r.get("decisions"):
        st.subheader("Human review log")
        st.dataframe(pd.DataFrame(r["decisions"]), hide_index=True)
    st.caption(r.get("disclaimer", ""))
    st.download_button("Download report (Markdown)", rev.get("report_md") or "", file_name=f"{rid}.md",
                       icon=":material/download:")


# ---------------------------------------------------------------- page: human review
def page_review():
    st.title("Human review")
    st.caption("Reviews paused at a human checkpoint. The workflow resumes from where it stopped once you decide; "
               "every decision is recorded with your name in the audit log.")
    queue = get_store().list_reviews(PENDING)
    if queue:
        st.dataframe(pd.DataFrame([{"review": q["review_id"], "status": STATUS[q["status"]][0],
                                    "risk": (q["overall_severity"] or "—").upper(), "waiting since": ts(q["updated_at"])}
                                   for q in queue]), hide_index=True)
    rid = pick_review("Review to decide", PENDING, key="hr_pick")
    if not rid:
        return
    pending = get_agent().pending(rid)
    if not pending:
        st.success("This review is not waiting for a decision.")
        return
    rev = get_store().get_review(rid)
    r = rev.get("report") or {}
    st.warning(pending["question"], icon=":material/pan_tool:")
    if not (ss.get("reviewer") or "").strip():
        st.error("Enter your name in the sidebar first (accountability: every decision is signed).")

    if pending["type"] == "clarification":
        with st.form("clarify"):
            corrections, skip = {}, []
            for u in pending["unresolved"]:
                st.markdown(f"**'{u['input']}'**" + (f" — suggestions: {', '.join(u['suggestions'])}" if u["suggestions"] else ""))
                c1, c2 = st.columns([1, 2])
                how = c1.radio("Action", ["Correct name", "Skip (not in knowledge base)"], key=f"how_{u['input']}",
                               horizontal=False, label_visibility="collapsed")
                fix = c2.text_input("Correct name", value=(u["suggestions"] or [""])[0], key=f"fix_{u['input']}")
                if how.startswith("Skip"):
                    skip.append(u["input"])
                elif fix.strip():
                    corrections[u["input"]] = fix.strip()
            comment = st.text_input("Comment (required to cancel)")
            b1, b2 = st.columns(2)
            go = b1.form_submit_button("Continue review", type="primary")
            cancel = b2.form_submit_button("Cancel review")
        if go or cancel:
            decision = {"action": "cancel" if cancel else "resolve", "reviewer": ss.get("reviewer", ""),
                        "comment": comment, "corrections": corrections, "skip": skip}
            out = run_live("resume", rid, decision)
            if out:
                result_card(out)
        return

    main = [f for f in r.get("findings", []) if f["severity"] != "low"]
    with st.container(border=True):
        st.markdown(f"Overall risk {sev_badge(r.get('overall_severity'))} &nbsp; {r.get('summary', '')}")
        for f in main:
            st.markdown(f"- {sev_badge(f['severity'])} **{f['title']}** — {f['effect']}")
        for c in pending.get("needs_confirmation", []):
            st.markdown(f"- :orange-badge[CONFIRM NAME] {c}")
        st.page_link(PAGES["report"], label="Full report with evidence", icon=":material/description:")

    labels = {"approve": "Approve and release", "edit": "Edit, then approve", "reject": "Reject (do not release)",
              "escalate": "Escalate to prescriber", "acknowledge": "Acknowledge (record action taken)"}
    action = st.radio("Decision", pending["allowed"], format_func=labels.get, horizontal=True)
    with st.form("decide"):
        comment = st.text_area("Comment / rationale" + ("" if action == "approve" else " (required)"), height=80)
        summary, notes = None, {}
        if action == "edit":
            summary = st.text_area("Summary shown to the clinician", value=r.get("summary", ""), height=120)
            for f in main:
                n = st.text_input(f"Reviewer note for {f['id']} · {f['title']}", key=f"note_{f['id']}")
                if n.strip():
                    notes[f["id"]] = n.strip()
        ok = st.form_submit_button(f"Submit: {labels[action]}", type="primary")
    if ok:
        decision = {"action": action, "reviewer": ss.get("reviewer", ""), "comment": comment}
        if action == "edit":
            decision |= {"summary": summary if summary != r.get("summary") else None, "finding_notes": notes}
        out = run_live("resume", rid, {k: v for k, v in decision.items() if v})
        if out:
            result_card(out)


# ---------------------------------------------------------------- page: audit
def page_audit():
    st.title("Audit log")
    st.caption("Append-only record of every review: input, agent steps, tool calls, sources, outputs, human decisions, "
               "timestamps and token usage.")
    rows = get_store().audit(limit=20000)
    if not rows:
        st.info("No audit entries yet.")
        return
    df = pd.DataFrame([{"time": ts(r["ts"]), "review": r["review_id"], "actor": r["actor"], "event": r["event"],
                        "node / tool": r["tool"] or r["node"] or "", "status": r["status"] or "",
                        "source": r["source"] or "",
                        "detail": json.dumps(r["detail"], ensure_ascii=False)[:300] if r["detail"] else ""} for r in rows])
    c1, c2, c3 = st.columns([2, 2, 2])
    reviews = ["All"] + list(dict.fromkeys(df["review"][::-1]))
    default = reviews.index(ss.current) if ss.get("current") in reviews else 0
    pick = c1.selectbox("Review", reviews, index=default)
    events = c2.multiselect("Events", sorted(df["event"].unique()))
    actor = c3.text_input("Actor contains", placeholder="human")
    if pick != "All":
        df = df[df["review"] == pick]
    if events:
        df = df[df["event"].isin(events)]
    if actor:
        df = df[df["actor"].str.contains(actor, case=False)]
    m = st.columns(4)
    m[0].metric("Entries", len(df))
    m[1].metric("Tool calls", int((df["event"] == "tool_call").sum()))
    m[2].metric("Human decisions", int((df["event"] == "human_decision").sum()))
    m[3].metric("Reviews", df["review"].nunique())
    st.dataframe(df, hide_index=True, width="stretch", height=520)
    st.download_button("Export CSV", df.to_csv(index=False), file_name="polysafe_audit.csv", icon=":material/download:")


# ---------------------------------------------------------------- navigation + sidebar
PAGES = {
    "new": st.Page(page_new, title="New review", icon=":material/add_circle:", default=True),
    "workflow": st.Page(page_workflow, title="Agent workflow", icon=":material/account_tree:"),
    "report": st.Page(page_report, title="Safety report", icon=":material/description:"),
    "review": st.Page(page_review, title="Human review", icon=":material/how_to_reg:"),
    "audit": st.Page(page_audit, title="Audit log", icon=":material/receipt_long:"),
}
nav = st.navigation(list(PAGES.values()))
with st.sidebar:
    st.markdown("### 💊 PolySafe")
    st.caption("Agentic polypharmacy medication-safety review · decision support for clinicians and pharmacists")
    st.text_input("Reviewer name", key="reviewer", placeholder="e.g. Pharmacist A")
    pending_slot = st.empty()   # filled after the page runs, so a decision made on this run is counted
    cfg = get_agent_config()["llm"]
    st.caption(f"Model: `{cfg['agent_model']}` (fallback `{cfg['agent_fallback_models'][0]}`)  \n"
               f"Embeddings: `{cfg['embedding_model']}`")
    st.caption("Prototype for CA6117 — not for clinical use. PolySafe never stops, starts, switches or re-doses a "
               "medication; humans decide.")
get_agent()   # start the agent (and MCP server) once per process
nav.run()
pending_slot.metric("Waiting for a human", len(get_store().list_reviews(PENDING)))
