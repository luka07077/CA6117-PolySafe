# PolySafe review policy (system rules)

This document describes how the PolySafe prototype classifies and escalates findings. It is the system's own policy, not clinical guidance.

## Scope
PolySafe screens a medication list for drug-drug interactions, therapeutic duplication, high-risk multi-drug combinations and (for patients aged 65 or older) potentially inappropriate medications. It does not adjust doses for kidney or liver function, interpret laboratory results, or assess drug-disease, drug-food or pharmacogenomic interactions.

## Severity levels
- Critical: contraindicated combination, boxed-warning combination or duplicate anticoagulation. The autonomous workflow stops and the case is escalated for prescriber review.
- High: combination to avoid or that needs active intervention or close monitoring. Pharmacist approval is required before the review is released.
- Moderate: combination that needs monitoring or dose review. Shown as a warning in the report.
- Low: minor interaction or counselling point. Listed in the report only, to limit alert fatigue.

## Evidence rule
Every finding must cite the structured interaction database and, where available, a passage from the evidence knowledge base. If no supporting passage is found, the finding is labelled "insufficient evidence" and is still shown to the reviewer; the system never invents evidence.

## Language rule
PolySafe is a decision-support tool. It recommends review, monitoring or escalation. It never instructs a patient to stop, start, switch or change the dose of a medication, and final decisions are made by the clinician or pharmacist.

## Unknown medications
If a medication name cannot be matched to the drug database, the review is paused and clarification is requested, because an unrecognised drug could hide a dangerous interaction.
