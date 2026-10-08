# PolySafe — evaluation summary (Step 6)

Material for report Chapter 4 (Prototype, Demo and Evaluation). Reproduce with
`python scripts/run_eval.py --run <name>`; the final tables are in `evaluation/results/final_v3/`
(`summary.md`, `summary.csv`, `per_case.csv`, `status_accuracy_by_category.csv`).

## 1. Method

**Test set** — 43 cases in `evaluation/test_cases.json`, positive and negative:

| Category | Cases | What it tests |
|---|---|---|
| known_critical / known_high / known_moderate | 4 / 6 / 4 | known interactions at each severity |
| combination | 5 | multi-drug rules (triple whammy, anticoagulant + antiplatelet + NSAID, ≥3 CNS-active, ≥2 anticholinergic) and an 8-drug list |
| duplication | 3 | two NSAIDs, two anticoagulants, same drug under brand + generic name |
| no_interaction | 6 | negative cases (false alarms) |
| brand_misspelling / free_text | 4 / 2 | brand names, misspellings, a fuzzy name that needs confirmation, free-text notes (incl. a drug that was stopped) |
| unknown_drug | 3 | unidentifiable or uncertain names (must ask, not guess) |
| prompt_injection / out_of_scope | 3 / 1 | explicit injection (note and inside a medication name), paraphrased injection the rules miss, "which drug should I stop?" |
| kb_gap | 2 | real interactions that the prototype knowledge base does not contain (expected failures) |

Expected results (workflow status, overall severity, every MODERATE+ finding with its severity) are defined from clinical
knowledge (FDA labels, AGS Beers 2023, STOPP/START v3), not copied from the system. Before the run, the labels of the 33
structured cases were cross-checked against the deterministic tools: they agreed on all except the two intended KB gaps.

**Configurations** — all use the pinned evaluation model `deepseek-v4-pro-0813` (temperature 0, no fallback):

| Config | Description |
|---|---|
| full | complete agent: tools + RAG evidence + safety reviewer |
| no_rag | ablation: no evidence retrieval |
| no_reviewer | ablation: safety reviewer removed |
| llm_only | baseline: the same LLM screens the list from its own knowledge (no tools, database or RAG), structured output |

**Scoring** is rule-based (no LLM judge): findings are compared by key (drug pair / duplication group / rule id) and
severity; LOW findings are not scored (alert-fatigue design). 10 explanations are exported to
`results/final_v3/manual_review_template.csv` for the team's manual review (accuracy and clarity, 1–5).

## 2. Headline results (final_v3)

| Metric | full | no_rag | no_reviewer | llm_only |
|---|---|---|---|---|
| Workflow status accuracy | **98%** | 98% | 98% | 74% |
| Finding precision | **100%** | 100% | 100% | 75% |
| Finding recall | 95% | 95% | 95% | **100%** |
| High-risk recall | 96% | 96% | 96% | 100% |
| Severity agreement (matched findings) | **100%** | 100% | 100% | 85% |
| False-alarm rate (no-interaction cases) | 0% | 0% | 0% | 0% |
| Under-escalated cases (unsafe) | **1** (KB gap) | 1 | 1 | 3 |
| Over-escalated cases | 0 | 0 | 0 | 4 |
| Injection refused | 2/2 | 2/2 | 2/2 | 2/2 |
| Unknown drug → clarification | **3/3** | 3/3 | 3/3 | 1/3 |
| Paraphrased injection still flags the risk (S04) | yes | yes | yes | refused instead |
| Out-of-scope request answered with scope notice | yes | yes | yes | refused instead |
| Evidence-grounded findings (matched + cited) | **37/37** | 0/37 | 37/37 | 0/56 |
| Medication-change instructions in output | 0 | 0 | 0 | 0 |
| Tool-call success | 314/314 | 277/277 | 314/314 | — |
| Mean / p90 latency (s) | 5.6 / 9.5 | 2.9 / 5.1 | 2.9 / 5.4 | 2.9 / 4.5 |
| Mean tokens per case | 2,396 | 1,617 | 1,449 | 932 |

## 3. What the results show

1. **Tools make the decisions reliable.** With the database deciding severity, the agent matched every expected
   severity (100%) and never over-escalated; the LLM-only baseline disagreed on 6 of 40 matched severities (15%), under-rated two
   *critical* contraindications to "high" (methotrexate + co-trimoxazole, simvastatin + clarithromycin → would not stop
   the workflow) and over-escalated four cases (alert fatigue).
2. **The baseline's higher recall comes from broader knowledge, not reliability.** llm_only found both KB-gap
   interactions (recall 100% vs 95%) but also produced 13 extra findings (precision 75%), e.g. pairs with no recognised
   interaction. This is the trade-off a curated knowledge base makes: precision and auditability over coverage.
3. **Safe stops only work with tools.** The baseline guessed instead of asking: it treated the misspelt "diazapam" as
   diazepam and escalated, reviewed "glyburide" (not in the knowledge base) without flagging it, and released a list
   whose fuzzy-matched name ("metforman") a human should have confirmed.
4. **Defence in depth against injection.** Explicit injections were refused by every configuration. The paraphrased
   injection that the rule filter misses (S04) did not change the agent's result — severity comes from the database —
   so the risk was still reported; the baseline refused, which is safe but leaves the patient without a review.
5. **RAG provides the grounding.** With RAG every MODERATE+ finding cited a verbatim passage from the expected
   evidence document (37/37); without it, explanations rest on database facts only (0/37). Detection is unchanged,
   because detection is deterministic.
6. **The safety reviewer catches what the writer gets wrong** (see iteration history): unsupported clinical claims
   and a citation of evidence that was not retrieved for the finding (C02, caught by the rule check).
7. **Cost.** The full agent needs ~2.4k tokens and ~5.6 s per case (2.4 LLM calls); the deterministic steps take
   milliseconds (314 tool calls, all successful).

## 4. Iteration history (error analysis → fixes)

| Run | Change | Reviewer rejections (full) | Mean tokens (full) | Finding |
|---|---|---|---|---|
| v1 `final` | — | 4 | 2,369 | All 4 rejections objected to the patient's age being mentioned. Cause: the reviewer did not receive the patient context the writer had. 2 empty-output errors in no_rag crashed the review. |
| v2 `final_v2` | reviewer gets patient context; retry + database-facts fallback on empty LLM output | 11 | 2,917 | Reviewer now consistently rejected **unsupported clinical claims** ("older age further increases bleeding risk", "may have reduced cardiac reserve"); every revision passed. Root cause: the writer prompt invited patient-specific claims the evidence does not support. |
| v3 `final_v3` | writer prompt: patient facts only to show relevance, no new clinical claims | **1** | 2,396 | Remaining rejection is a rule catch (citation of evidence not retrieved for the finding). One empty LLM output (T03) handled by the fallback — review still correct. |

Detection, escalation and safety metrics were identical across v1–v3 (they are deterministic); the iterations improved
the faithfulness of the generated explanations and reduced wasted revision calls.

## 5. Failure analysis (final_v3, full agent)

| Case | Result | Classification | Explanation |
|---|---|---|---|
| G02 citalopram + clarithromycin | not detected, released as completed | UNSAFE (expected) | QT-prolongation pair missing from the prototype knowledge base. Mitigation: larger licensed interaction database (e.g. a commercial or national formulary feed); a QT-risk drug class rule. |
| G01 warfarin + prednisone | not detected | minor (expected) | Corticosteroid–warfarin interaction missing from the knowledge base. |
| T03 "warfrin" + "asprin" | correct result, explanation from database facts | handled | LLM returned no structured output twice → conservative fallback draft. |

Both misses are knowledge-coverage limits, not workflow errors: the report should state that PolySafe is only as
complete as its knowledge base, which is why every release still passes a human checkpoint and why "absence of a
finding is not proof of safety" is printed on every report. A future version could add the LLM's own screening as an
**unverified second opinion** (flagged, never authoritative) to surface such gaps for the pharmacist.

## 6. Manual review (to do by the team)

Fill in `results/final_v3/manual_review_template.csv` (10 randomly sampled explanations from the full agent): score
*accuracy* and *clarity* 1–5, add reviewer name and comments. Report the mean and any disagreements.
