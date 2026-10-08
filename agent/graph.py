"""
PolySafe review workflow (LangGraph, fixed order; the plan adapts to the input, no LLM router):

  PERCEIVE   guard_input ──injection──► refuse ─► END                       (safe stop: refusal)
                 │
             parse_input            free text -> PatientInput (LLM); structured form -> no LLM call
                 │
             normalize_meds ──unknown / uncertain name──► clarify ─► END   (safe stop: ask a human)
                 │
  REASON     plan_checks            C(n,2) drug pairs + regimen checks (+ older-adult check if age >= 65)
                 │
  ACT        run_checks             check_interaction × pairs, check_duplication, check_combination_rules, ...
                 │──no findings──────────────────────────────┐
             retrieve_evidence      RAG per finding (severity >= moderate), verbatim passages
                 │                                           │
  REASON     assess_risk            LLM explains each finding (severity fixed by the database)
                 │                                           │
             safety_reviewer ──fail, revisions left──► revise ─► assess_risk
                 │      └──fail after max revisions──► fallback (database facts only)
                 ▼                                           ▼
             compose_report ◄────────────────────────────────┘
                 │  route by overall severity (configs/agent_config.yaml `routing`)
                 ├─ low / moderate ─► finalize ─► END           status completed
                 ├─ high (or a name needs confirmation) ─► human_review ─► END   status pending_review
                 └─ critical ─► escalate ─► END                  status escalated (autonomous workflow stops)

Stop conditions: every planned check executed AND every finding has evidence or is marked insufficient AND
severity assigned; or a safe stop (refuse / clarify); or a critical finding (escalate).
Human checkpoints are LangGraph interrupts: the run pauses, its state is saved by the checkpointer (thread_id =
review_id) and resumes — possibly in another process, hours later — with Command(resume=<decision>):
    human_review         pharmacist: approve | edit | reject | escalate     -> finalize (or escalate)
    escalate             prescriber: acknowledge | edit | reject             -> finalize
    await_clarification  reviewer:   resolve (correct names / skip unknown)  -> normalize_meds again
                                     cancel                                  -> finalize

Switches (configs/agent_config.yaml `pipeline`): rag, safety_reviewer, reviewer_llm_check.
"""
import asyncio
import itertools
import json
import operator
import re
import time
from typing import Annotated, TypedDict

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.guardrails import check_input, redact_pii
from agent.report import build_report, overall_severity, render_markdown
from agent.schemas import PatientInput, ReviewDraft, ReviewResult
from db.drug_db import SEVERITY_RANK
from src.config import load_config
from src.utils.logger import get_logger
from tools.mcp_client import tool_text

logger = get_logger("polysafe.graph")

# imperative medication-change instructions (allowed: "consider whether to stop…", "review…", "monitor…")
_DIRECTIVE = re.compile(
    r"(?:^|[.;:!]\s+)(?:stop|discontinue|cease|withdraw|switch|replace|start|increase|decrease|reduce|double|halve)\b|"
    r"\b(?:should|must|needs? to|has to|have to)\s+(?:be\s+)?(?:stop|stopped|discontinue|discontinued|cease|ceased|"
    r"withdrawn?|switch|switched|replaced?|increased?|decreased?|reduced?|doubled|halved)\b", re.I)


class ReviewState(TypedDict, total=False):
    review_id: str
    case: dict                     # original input (form fields and/or free text)
    guard: dict                    # input-guard verdict
    patient: dict                  # age, sex, conditions, allergies
    raw_medications: list          # medications as written
    medications: list              # normalize_drug results
    acknowledged_unknown: list     # names a human confirmed may be skipped (not in the knowledge base)
    unresolved: list               # names that stopped the run (clarification needed)
    plan: dict                     # pairs + regimen checks to run
    findings: list                 # one per interaction pair / duplication / combination / older-adult flag
    evidence: dict                 # evidence id -> {source, section, passage}
    draft: dict                    # ReviewDraft (summary + explanations)
    draft_source: str              # llm | fallback | none
    review: dict                   # safety reviewer verdict
    revision_count: int
    scope_notice: str
    report: dict
    report_markdown: str
    route: str                     # finalize | human_review | escalate
    status: str
    skip_names: list               # unknown names a reviewer chose to skip at a clarification checkpoint
    decisions: Annotated[list, operator.add]    # human decisions at checkpoints (who, what, when, why)
    tool_usage: Annotated[list, operator.add]   # token usage reported by tools (RAG helper LLM)
    trace: Annotated[list, operator.add]


_UNICODE_ESC = re.compile(r"\\u([0-9a-fA-F]{4})")
_CITE = re.compile(r"\bE\d+\b")


def _clean(text: str) -> str:
    """Decode literal \\uXXXX sequences that some models emit inside function-call JSON strings."""
    return _UNICODE_ESC.sub(lambda m: chr(int(m.group(1), 16)), text or "")


def _fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace(f"<<{k}>>", v)
    return template


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


class PolySafeGraph:
    def __init__(self, tools: dict, llm, helper_llm=None, cfg: dict | None = None, fallback_llms: list | None = None,
                 checkpointer=None):
        conf = load_config("agent_config")
        self.cfg = {**conf["pipeline"], **(cfg or {})}
        self.max_revisions = conf["review"]["max_revisions"]
        self.evidence_min = SEVERITY_RANK[conf["review"]["evidence_min_severity"]]
        self.routing = conf["routing"]
        self.prompts = load_config("prompts")
        self.tools, self.llm, self.helper = tools, llm, helper_llm
        fallback_llms = fallback_llms or []

        def structured(schema):
            r = llm.with_structured_output(schema, method="function_calling")
            return r.with_fallbacks([f.with_structured_output(schema, method="function_calling")
                                     for f in fallback_llms]) if fallback_llms else r
        self.parser, self.assessor, self.reviewer = structured(PatientInput), structured(ReviewDraft), structured(ReviewResult)
        self.graph = self._build(checkpointer)

    # ------------------------------------------------------------ helpers
    async def _tool(self, calls: list, tool_name: str, **args) -> dict:
        """Call an MCP tool and record it in the node's trace."""
        t0 = time.perf_counter()
        try:
            out = json.loads(tool_text(await self.tools[tool_name].ainvoke(args)))
            ok = out.get("status") != "error"
            return out
        except Exception as e:
            out, ok = {"status": "error", "message": f"{type(e).__name__}: {e}"}, False
            return out
        finally:
            calls.append({"tool": tool_name, "args": args, "ms": round((time.perf_counter() - t0) * 1000), "ok": ok,
                          "preview": json.dumps(out, ensure_ascii=False)[:160]})

    @staticmethod
    def _step(node: str, t0: float, status: str, summary: str, tools: list | None = None, **extra) -> dict:
        return {"node": node, "ms": round((time.perf_counter() - t0) * 1000), "status": status, "summary": summary,
                "tools": tools or [], **extra}

    def _patient_context(self, state: ReviewState) -> str:
        p = dict(state.get("patient", {}))
        p["medications"] = [m["generic_name"] for m in state.get("medications", []) if m.get("recognized")]
        conf = [f"'{m['input']}' read as {m['generic_name']}" for m in state.get("medications", []) if m.get("needs_confirmation")]
        if conf:
            p["names_needing_confirmation"] = conf
        if state.get("acknowledged_unknown"):
            p["not_checked_not_in_knowledge_base"] = state["acknowledged_unknown"]
        return json.dumps(p, ensure_ascii=False)

    def _to_explain(self, state: ReviewState) -> list[dict]:
        """Findings the LLM explains: moderate and above. Low findings (minor interactions, older-adult notes) are
        listed in the report as database facts only, to limit alert fatigue and save tokens."""
        return [f for f in state.get("findings", []) if SEVERITY_RANK[f["severity"]] >= self.evidence_min]

    def _findings_for_prompt(self, state: ReviewState) -> str:
        keep = ("id", "type", "title", "severity", "effect", "mechanism", "recommendation", "evidence_ids", "evidence_status")
        return _dumps([{k: f.get(k) for k in keep if f.get(k) not in (None, "", [])} for f in self._to_explain(state)])

    def _evidence_for_prompt(self, state: ReviewState) -> str:
        ev = state.get("evidence", {})
        return "\n".join(f"[{i}] ({e['source']} · {e['section']}) {e['passage']}" for i, e in ev.items()) or "(none)"

    # ------------------------------------------------------------ PERCEIVE
    async def guard_input(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        case = state["case"]
        text = "\n".join([case.get("text", ""), *case.get("medications", []), case.get("note", "")]).strip()
        g = await check_input(text, self.helper)
        notice = ""
        if g["verdict"] == "out_of_scope":
            notice = ("Scope: PolySafe does not decide which medication to stop, start, switch or re-dose. The findings "
                      "below are for the reviewing clinician or pharmacist to act on.")
        return {"guard": g, "scope_notice": notice, "revision_count": 0, "status": "guarded",
                "trace": [self._step("guard_input", t0, g["verdict"], f"{g['verdict']} ({g['method']}) {g['reason']}".strip())]}

    def route_after_guard(self, state: ReviewState) -> str:
        return "refuse" if state["guard"]["verdict"] == "injection" else "parse_input"

    async def refuse(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        md = ("# Request refused\nThe input appears to contain instructions that try to change PolySafe's safety rules "
              f"({state['guard']['reason']}). No review was performed. Please resubmit the patient's medication list only.")
        return {"status": "refused", "report_markdown": md, "report": {"refused": True, "reason": state["guard"]["reason"]},
                "trace": [self._step("refuse", t0, "refused", "safe stop: input refused")]}

    async def parse_input(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        case = state["case"]
        patient = {k: case.get(k) for k in ("age", "sex")} | {k: case.get(k) or [] for k in ("conditions", "allergies")}
        meds = list(case.get("medications") or [])
        detail = "structured form (no LLM call)"
        if case.get("text"):
            parsed: PatientInput = await self.parser.ainvoke([
                ("system", self.prompts["parse_input"]),
                ("user", f"<patient_text>\n{redact_pii(case['text'])}\n</patient_text>")])
            for k in ("age", "sex"):
                patient[k] = patient[k] if patient[k] is not None else getattr(parsed, k)
            for k in ("conditions", "allergies"):
                patient[k] = list(dict.fromkeys(patient[k] + getattr(parsed, k)))
            meds += [m for m in parsed.medications if m not in meds]
            detail = f"free text parsed by LLM: {len(parsed.medications)} medications"
        return {"patient": patient, "raw_medications": meds, "status": "parsed",
                "trace": [self._step("parse_input", t0, "parsed", detail, patient=patient, medications=meds)]}

    async def normalize_meds(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        calls, results = [], []
        for name in state["raw_medications"]:
            results.append(await self._tool(calls, "normalize_drug", name=name))
        ack = {a.lower() for a in state["case"].get("acknowledged_unknown", []) + state.get("skip_names", [])}
        unresolved = [r["input"] for r in results if not r.get("recognized") and r["input"].lower() not in ack]
        acknowledged = [r["input"] for r in results if not r.get("recognized") and r["input"].lower() in ack]
        n_ok = sum(1 for r in results if r.get("recognized"))
        conf = [r["input"] for r in results if r.get("needs_confirmation")]
        summary = f"{n_ok}/{len(results)} recognised" + (f"; unresolved: {unresolved}" if unresolved else "") + \
                  (f"; needs confirmation: {conf}" if conf else "") + (f"; skipped by reviewer: {acknowledged}" if acknowledged else "")
        return {"medications": results, "unresolved": unresolved, "acknowledged_unknown": acknowledged,
                "status": "normalized", "trace": [self._step("normalize_meds", t0, "normalized", summary, calls)]}

    def route_after_normalize(self, state: ReviewState) -> str:
        recognised = [m for m in state["medications"] if m.get("recognized")]
        return "clarify" if state["unresolved"] or not recognised else "plan_checks"

    async def clarify(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        lines = []
        for m in state["medications"]:
            if m["input"] in state["unresolved"]:
                hint = f" — did you mean {' or '.join(m['suggestions'])}?" if m.get("suggestions") else \
                       " — not in the drug knowledge base"
                lines.append(f"- '{m['input']}'{hint}")
        if not any(m.get("recognized") for m in state["medications"]):
            lines.append("- no recognisable medication in the list")
        md = ("# Clarification needed\nThe review was paused because these medications could not be identified. "
              "An unidentified drug could hide a dangerous interaction, so PolySafe does not guess:\n" + "\n".join(lines) +
              "\n\nCorrect the name(s), or confirm that a drug is not in the knowledge base to continue without checking it.")
        return {"status": "needs_clarification", "report_markdown": md,
                "report": {"clarification": True, "unresolved": state["unresolved"]},
                "trace": [self._step("clarify", t0, "needs_clarification", f"safe stop: {len(lines)} item(s) to clarify")]}

    # ------------------------------------------------------------ REASON: plan
    async def plan_checks(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        ids = [m["drug_id"] for m in state["medications"] if m.get("recognized")]
        unique = list(dict.fromkeys(ids))
        age = state["patient"].get("age")
        plan = {"drugs": ids, "pairs": [list(p) for p in itertools.combinations(unique, 2)],
                "regimen_checks": ["check_duplication", "check_combination_rules"]
                                  + (["check_older_adult_risks"] if age is not None and age >= 65 else [])}
        n = len(unique)
        summary = (f"{n} drugs -> C({n},2) = {len(plan['pairs'])} pair checks + " + ", ".join(plan["regimen_checks"]))
        return {"plan": plan, "status": "planned", "trace": [self._step("plan_checks", t0, "planned", summary, plan=plan)]}

    # ------------------------------------------------------------ ACT: checks
    async def run_checks(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        calls, findings, plan = [], [], state["plan"]
        names = {m["drug_id"]: m["generic_name"] for m in state["medications"] if m.get("recognized")}
        for a, b in plan["pairs"]:
            r = await self._tool(calls, "check_interaction", drug_a=a, drug_b=b)
            if r.get("found"):
                main = max(r["interactions"], key=lambda h: SEVERITY_RANK[h["severity"]])
                findings.append({"type": "interaction", "title": f"{names[a]} + {names[b]}", "drugs": [a, b],
                                 "severity": r["severity"], "effect": main["effect"], "mechanism": main["mechanism"],
                                 "recommendation": main["recommendation"], "source": main["source"],
                                 "evidence_doc": main["evidence_doc"], "db_refs": [h["interaction_id"] for h in r["interactions"]],
                                 "query": f"{names[a]} {names[b]} interaction {main['effect']}"})
        dup = await self._tool(calls, "check_duplication", drugs=plan["drugs"])
        for d in dup.get("duplications", []):
            label = ", ".join(names[x] for x in d["drugs"])
            findings.append({"type": "duplication", "title": f"Therapeutic duplication: {label}", "drugs": d["drugs"],
                             "severity": d["severity"], "effect": d["note"], "mechanism": "",
                             "recommendation": "Pharmacist/clinician review: confirm whether more than one drug from this "
                                               "group is intended", "source": "STOPP/START v3 (duplicate drug class)",
                             "evidence_doc": d["evidence_doc"], "db_refs": [f"DUP:{d['group']}"],
                             "query": f"therapeutic duplication two {label}"})
        comb = await self._tool(calls, "check_combination_rules", drugs=plan["drugs"])
        for c in comb.get("combinations", []):
            label = ", ".join(names[x] for x in c["drugs"])
            findings.append({"type": "combination", "title": f"{c['name']}: {label}", "drugs": c["drugs"],
                             "severity": c["severity"], "effect": c["effect"], "mechanism": "",
                             "recommendation": c["recommendation"], "source": c["source"],
                             "evidence_doc": c["evidence_doc"], "db_refs": [c["rule_id"]], "query": f"{c['name']} {label}"})
        if "check_older_adult_risks" in plan["regimen_checks"]:
            old = await self._tool(calls, "check_older_adult_risks", drugs=plan["drugs"], age=state["patient"]["age"])
            for f in old.get("flags", []):
                findings.append({"type": "older_adult", "title": f"Older adult (65+): {f['generic_name']}",
                                 "drugs": [f["drug_id"]], "severity": f["severity"], "effect": f["note"], "mechanism": "",
                                 "recommendation": "Review whether this medication is still appropriate for an older adult",
                                 "source": f["source"], "evidence_doc": f["evidence_doc"], "db_refs": ["BEERS-2023"],
                                 "query": f"{f['generic_name']} older adults Beers"})
        findings.sort(key=lambda f: -SEVERITY_RANK[f["severity"]])
        for i, f in enumerate(findings, 1):
            f["id"] = f"F{i}"
        failed = [c for c in calls if not c["ok"]]
        counts = {s: sum(f["severity"] == s for f in findings) for s in SEVERITY_RANK if any(f["severity"] == s for f in findings)}
        summary = f"{len(calls)} tool calls, {len(findings)} findings {counts or ''}" + (f", {len(failed)} FAILED" if failed else "")
        return {"findings": findings, "status": "checks_done", "trace": [self._step("run_checks", t0, "checks_done", summary, calls)]}

    def route_after_checks(self, state: ReviewState) -> str:
        return "retrieve_evidence" if state["findings"] else "compose_report"

    async def retrieve_evidence(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        calls, findings, evidence, usage = [], [dict(f) for f in state["findings"]], {}, []
        if not self.cfg.get("rag", True):
            for f in findings:
                f.update(evidence_ids=[], evidence_status="not_retrieved")
            return {"findings": findings, "evidence": {}, "status": "evidence_retrieved",
                    "trace": [self._step("retrieve_evidence", t0, "evidence_retrieved", "RAG disabled (ablation)")]}
        targets = [f for f in findings if SEVERITY_RANK[f["severity"]] >= self.evidence_min]
        queries = list(dict.fromkeys(f["query"] for f in targets))
        results = await asyncio.gather(*(self._tool(calls, "search_drug_evidence", query=q) for q in queries))
        by_query = dict(zip(queries, results))
        seen: dict[str, str] = {}   # passage text -> evidence id (shared passages get one id)
        for f in findings:
            if f not in targets:
                f.update(evidence_ids=[], evidence_status="database_only")
                continue
            res = by_query[f["query"]]
            usage.append(res.get("usage", {}))
            ids = []
            for p in res.get("passages", []):
                if p["passage"] not in seen:
                    seen[p["passage"]] = f"E{len(seen) + 1}"
                    evidence[seen[p["passage"]]] = p
                ids.append(seen[p["passage"]])
            f["evidence_ids"] = ids
            sources = {evidence[i]["source"] for i in ids}
            f["evidence_status"] = ("insufficient" if not ids else "matched" if f["evidence_doc"] in sources else "other_source")
        st = {s: sum(f.get("evidence_status") == s for f in findings) for s in ("matched", "other_source", "insufficient")}
        summary = f"{len(queries)} RAG queries, {len(evidence)} passages; " + ", ".join(f"{k}={v}" for k, v in st.items() if v)
        return {"findings": findings, "evidence": evidence, "status": "evidence_retrieved", "tool_usage": usage,
                "trace": [self._step("retrieve_evidence", t0, "evidence_retrieved", summary, calls)]}

    def route_after_evidence(self, state: ReviewState) -> str:
        return "assess_risk" if self._to_explain(state) else "compose_report"

    # ------------------------------------------------------------ REASON: assess + review
    async def assess_risk(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        review_txt = ""
        rv = state.get("review")
        if rv and not rv.get("passed", True):
            review_txt = ("\nYour previous draft was REJECTED by the safety reviewer. Fix these issues:\n- "
                          + "\n- ".join(rv["issues"]) + f"\nPrevious draft:\n{_dumps(state.get('draft', {}))}\n")
        prompt = _fill(self.prompts["assess_risk"], patient=self._patient_context(state),
                       findings=self._findings_for_prompt(state), evidence=self._evidence_for_prompt(state), review=review_txt)
        draft: ReviewDraft | None = await self.assessor.ainvoke(prompt)
        if draft is None:   # the model returned no (parsable) function call: retry once
            draft = await self.assessor.ainvoke(prompt)
        if draft is None:   # still nothing: conservative database-facts draft instead of failing the review
            fb = self._facts_draft(state, "the explanation model returned no usable output")
            return fb | {"status": "assessed", "trace": [self._step("assess_risk", t0, "fallback",
                                                                    "no structured output after retry -> database facts draft")]}
        d = draft.model_dump()
        d["summary"] = _clean(d["summary"])
        for it in d["items"]:
            it["explanation"] = _clean(it["explanation"])
            # citations written inline ("(E1, E2)") count as cited evidence too
            it["evidence_ids"] = list(dict.fromkeys([e.strip("[]") for e in it["evidence_ids"]] + _CITE.findall(it["explanation"])))
        return {"draft": d, "draft_source": "llm", "status": "assessed",
                "trace": [self._step("assess_risk", t0, "assessed", f"{len(draft.items)} explanations drafted",
                                     revision=state.get("revision_count", 0))]}

    def _rule_review(self, state: ReviewState) -> list[str]:
        draft, issues = state.get("draft", {}), []
        fids = {f["id"]: f for f in self._to_explain(state)}
        items = {i["finding_id"]: i for i in draft.get("items", [])}
        missing, extra = set(fids) - set(items), set(items) - set(fids)
        if missing:
            issues.append(f"Explain every finding: missing {sorted(missing)}.")
        if extra:
            issues.append(f"Remove explanations for findings that do not exist: {sorted(extra)}.")
        for fid, it in items.items():
            if fid in fids:
                bad = set(it.get("evidence_ids", [])) - set(fids[fid].get("evidence_ids", []))
                if bad:
                    issues.append(f"{fid}: cites evidence {sorted(bad)} that was not retrieved for this finding.")
                if fids[fid].get("evidence_ids") and not it.get("evidence_ids"):
                    issues.append(f"{fid}: cite at least one of its evidence passages {fids[fid]['evidence_ids']}.")
        for where, text in [("summary", draft.get("summary", ""))] + [(i["finding_id"], i["explanation"]) for i in draft.get("items", [])]:
            m = _DIRECTIVE.search(text or "")
            if m:
                issues.append(f"{where}: medication-change instruction {m.group(0).strip()!r}; rephrase as a consideration "
                              "for the clinician (review / monitor / consider).")
        return issues

    async def safety_reviewer(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        issues = self._rule_review(state)
        rule_issues, llm_passed = list(issues), None
        if self.cfg.get("reviewer_llm_check", True):
            prompt = _fill(self.prompts["safety_reviewer"], patient=self._patient_context(state),
                           findings=self._findings_for_prompt(state),
                           evidence=self._evidence_for_prompt(state), draft=_dumps(state.get("draft", {})))
            try:
                verdict: ReviewResult = await self.reviewer.ainvoke(prompt)
                llm_passed = verdict.passed
                if not verdict.passed:
                    issues += verdict.issues or ["Reviewer flagged the draft as unfaithful or unsafe."]
            except Exception as e:   # the rule check still applies
                logger.warning(f"[reviewer] LLM check failed: {e}")
        review = {"passed": not issues, "issues": issues, "rule_issues": rule_issues, "llm_passed": llm_passed}
        summary = "passed" if not issues else f"rejected: {' | '.join(issues)[:200]}"
        return {"review": review, "status": "reviewed",
                "trace": [self._step("safety_reviewer", t0, "passed" if not issues else "rejected", summary,
                                     review=review, revision=state.get("revision_count", 0))]}

    def route_after_review(self, state: ReviewState) -> str:
        if state["review"]["passed"]:
            return "compose_report"
        return "revise" if state.get("revision_count", 0) < self.max_revisions else "fallback"

    async def revise(self, state: ReviewState) -> dict:
        return {"revision_count": state.get("revision_count", 0) + 1,
                "trace": [{"node": "revise", "ms": 0, "status": "revising", "summary": f"revision {state.get('revision_count', 0) + 1}", "tools": []}]}

    def _facts_draft(self, state: ReviewState, reason: str) -> dict:
        """Conservative draft from database facts only (no LLM text)."""
        items = [{"finding_id": f["id"], "evidence_ids": f.get("evidence_ids", []),
                  "explanation": f"{f['effect']}." + (f" Mechanism: {f['mechanism']}." if f.get("mechanism") else "")}
                 for f in self._to_explain(state)]
        sev = overall_severity(state["findings"])
        return {"draft": {"summary": f"{len(items)} finding(s); highest severity {sev}. Explanations are database facts "
                                     f"only because {reason}.", "items": items}, "draft_source": "fallback"}

    async def fallback(self, state: ReviewState) -> dict:
        """Conservative draft after repeated reviewer rejections."""
        t0 = time.perf_counter()
        return self._facts_draft(state, "the generated explanation did not pass the safety review") | {
            "trace": [self._step("fallback", t0, "fallback", "draft replaced by database facts (reviewer rejected twice)")]}

    # ------------------------------------------------------------ report + routing
    async def compose_report(self, state: ReviewState) -> dict:
        t0 = time.perf_counter()
        sev = overall_severity(state.get("findings", []))
        route = self.routing.get(sev, "finalize") if sev else "finalize"
        if route == "finalize" and any(m.get("needs_confirmation") for m in state.get("medications", [])):
            route = "human_review"   # a fuzzy-matched name must be confirmed by a human
        st = {**state, "route": route}
        if not state.get("draft"):
            st["draft"], st["draft_source"] = {"summary": "", "items": []}, "none"
        report = build_report(st)
        md = redact_pii(render_markdown(report))
        status = {"finalize": "report_ready", "human_review": "pending_review", "escalate": "escalated"}[route]
        return {"report": report, "report_markdown": md, "route": route, "status": status,
                "trace": [self._step("compose_report", t0, "report_ready", f"overall {sev or 'none'} -> {route}")]}

    def route_by_severity(self, state: ReviewState) -> str:
        return state["route"]

    # ------------------------------------------------------------ human checkpoints (interrupts)
    # interrupt() pauses the run; on resume the node re-runs from the top and interrupt() returns the decision,
    # so nothing before interrupt() may have side effects.
    @staticmethod
    def _decision(checkpoint: str, decision: dict) -> dict:
        keep = ("action", "reviewer", "comment", "summary", "finding_notes", "corrections", "skip")
        return {"checkpoint": checkpoint, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **{k: decision[k] for k in keep if decision.get(k) not in (None, "", [], {})}}

    @staticmethod
    def _decision_step(node: str, d: dict) -> dict:
        return {"node": node, "ms": 0, "status": d["action"], "tools": [],
                "summary": f"{d['action']} by {d.get('reviewer', '?')}" + (f": {d['comment']}" if d.get("comment") else "")}

    async def human_review(self, state: ReviewState) -> Command:
        report = state.get("report", {})
        decision = interrupt({
            "type": "pharmacist_review", "review_id": state.get("review_id"),
            "allowed": ["approve", "edit", "reject", "escalate"],
            "overall_severity": report.get("overall_severity"),
            "needs_confirmation": [f"'{m['input']}' read as {m['generic_name']}" for m in report.get("needs_confirmation", [])],
            "question": "Pharmacist review required before release: approve, edit (summary / notes), reject, or escalate "
                        "to the prescriber."})
        d = self._decision("pharmacist_review", decision)
        update = {"decisions": [d], "trace": [self._decision_step("human_review", d)]}
        if d["action"] == "escalate":
            return Command(goto="escalate", update=update | {"status": "escalated"})
        return Command(goto="finalize", update=update)

    async def escalate(self, state: ReviewState) -> Command:
        decision = interrupt({
            "type": "prescriber_escalation", "review_id": state.get("review_id"),
            "allowed": ["acknowledge", "edit", "reject"],
            "overall_severity": state.get("report", {}).get("overall_severity"),
            "question": "Critical / escalated risk: the autonomous workflow has stopped. Prescriber: acknowledge with "
                        "the action taken, edit, or reject the review."})
        d = self._decision("prescriber_escalation", decision)
        return Command(goto="finalize", update={"decisions": [d], "trace": [self._decision_step("escalate", d)]})

    async def await_clarification(self, state: ReviewState) -> Command:
        by_input = {m["input"]: m for m in state.get("medications", [])}
        decision = interrupt({
            "type": "clarification", "review_id": state.get("review_id"), "allowed": ["resolve", "cancel"],
            "unresolved": [{"input": u, "suggestions": by_input.get(u, {}).get("suggestions", [])} for u in state["unresolved"]],
            "question": "Some medications could not be identified. Correct each name, or skip a medication that is not "
                        "in the knowledge base to continue without checking it."})
        d = self._decision("clarification", decision)
        update = {"decisions": [d], "trace": [self._decision_step("await_clarification", d)]}
        if d["action"] == "cancel":
            return Command(goto="finalize", update=update)
        fixes = d.get("corrections", {})
        raw = [fixes.get(m, m) for m in state["raw_medications"]]
        return Command(goto="normalize_meds", update=update | {
            "raw_medications": raw, "skip_names": state.get("skip_names", []) + d.get("skip", [])})

    async def finalize(self, state: ReviewState) -> dict:
        """Release / close the review and apply the human decision to the report."""
        decisions = state.get("decisions", [])
        last = decisions[-1]["action"] if decisions else None
        status = {None: "completed", "approve": "approved", "edit": "approved_with_edits", "acknowledge": "acknowledged",
                  "reject": "rejected", "cancel": "cancelled", "resolve": "completed"}[last]
        report, md = dict(state.get("report") or {}), state.get("report_markdown", "")
        if report.get("findings") is not None:   # a real review (not a refusal / clarification stub)
            for d in decisions:
                if d["action"] == "edit":
                    report["summary"] = d.get("summary") or report.get("summary", "")
                    report["reviewer_notes"] = {**report.get("reviewer_notes", {}), **d.get("finding_notes", {})}
            report["decisions"], report["release_status"] = decisions, status
            md = redact_pii(render_markdown(report))
        summary = {"completed": "released automatically (no high-risk findings)"}.get(status, f"closed: {status}")
        return {"status": status, "report": report, "report_markdown": md,
                "trace": [{"node": "finalize", "ms": 0, "status": status, "summary": summary, "tools": []}]}

    # ------------------------------------------------------------ wiring
    def _build(self, checkpointer=None):
        g = StateGraph(ReviewState)
        nodes = ["guard_input", "refuse", "parse_input", "normalize_meds", "clarify", "plan_checks", "run_checks",
                 "retrieve_evidence", "assess_risk", "safety_reviewer", "revise", "fallback", "compose_report", "finalize"]
        for n in nodes:
            g.add_node(n, getattr(self, n))
        # checkpoint nodes route with Command(goto=...); destinations are declared for the graph diagram
        g.add_node("human_review", self.human_review, destinations=("finalize", "escalate"))
        g.add_node("escalate", self.escalate, destinations=("finalize",))
        g.add_node("await_clarification", self.await_clarification, destinations=("normalize_meds", "finalize"))
        g.add_edge(START, "guard_input")
        g.add_conditional_edges("guard_input", self.route_after_guard, {"refuse": "refuse", "parse_input": "parse_input"})
        g.add_edge("parse_input", "normalize_meds")
        g.add_conditional_edges("normalize_meds", self.route_after_normalize, {"clarify": "clarify", "plan_checks": "plan_checks"})
        g.add_edge("plan_checks", "run_checks")
        g.add_conditional_edges("run_checks", self.route_after_checks,
                                {"retrieve_evidence": "retrieve_evidence", "compose_report": "compose_report"})
        g.add_conditional_edges("retrieve_evidence", self.route_after_evidence,
                                {"assess_risk": "assess_risk", "compose_report": "compose_report"})
        if self.cfg.get("safety_reviewer", True):
            g.add_edge("assess_risk", "safety_reviewer")
            g.add_conditional_edges("safety_reviewer", self.route_after_review,
                                    {"compose_report": "compose_report", "revise": "revise", "fallback": "fallback"})
            g.add_edge("revise", "assess_risk")
            g.add_edge("fallback", "compose_report")
        else:
            g.add_edge("assess_risk", "compose_report")
        g.add_conditional_edges("compose_report", self.route_by_severity,
                                {"finalize": "finalize", "human_review": "human_review", "escalate": "escalate"})
        g.add_edge("clarify", "await_clarification")
        g.add_edge("refuse", END)
        g.add_edge("finalize", END)
        return g.compile(checkpointer=checkpointer)
