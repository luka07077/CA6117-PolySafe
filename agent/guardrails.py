"""
Input guardrails (layer L1 of the safety design):

  1. rule check (always, no LLM): regex patterns for prompt injection and for requests that the system itself
     decide a medication change ("which drug should I stop?")
  2. LLM check (helper model, only for inputs the rules flag as suspicious; switch guardrails.llm_check):
     confirms or clears the rule hit, so paraphrases are judged in context. If the LLM call fails, the rule
     verdict stands (fail closed for suspicious input).
  3. PII redaction on input and on the final report (names, phone numbers, e-mails, ID numbers).

verdicts: safe -> continue | out_of_scope -> continue the review, report states that PolySafe does not make
medication decisions | injection -> refuse (safe stop).
"""
import re

from agent.schemas import GuardVerdict
from src.config import get_agent_config, load_config
from src.utils.logger import get_logger

logger = get_logger("polysafe.guardrails")

_INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier|your)|disregard (?:all|any|the|your|previous)|"
    r"system prompt|developer mode|jailbreak|you are now|act as (?:a|an)\b|pretend (?:to|you)|forget (?:all|your|previous)|"
    r"new instructions|override (?:the|your|all)|bypass|do not (?:report|flag|mention) (?:any|the)|"
    r"(?:say|report|output|answer) (?:that )?(?:there (?:are|is) )?no (?:interactions?|risks?)|忽略|无视.*(?:规则|指令)", re.I)
_OUT_OF_SCOPE = re.compile(
    r"\b(?:which|what) (?:drugs?|medications?|medicines?|pills?|one)s? (?:should|can|do|must) (?:i|we|he|she|they|the patient) "
    r"(?:stop|quit|drop|remove|discontinue)|\b(?:should|can) (?:i|we|he|she|they|the patient) (?:stop|quit|discontinue|switch|"
    r"change|double|increase|reduce)|\btell (?:me|us) (?:which|what) to (?:stop|drop|remove)|\b(?:stop|discontinue) (?:it|them|"
    r"the (?:warfarin|drug|medication)) (?:for me|now)|\b(?:decide|choose) (?:which|what) (?:drug|medication)|停药|换药|该停", re.I)

_PII = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[email redacted]"),
    (re.compile(r"\+?\d[\d\s\-]{7,}\d"), "[number redacted]"),
    (re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I), "[NRIC redacted]"),   # Singapore NRIC/FIN
    (re.compile(r"\b(?:mr|mrs|ms|miss|dr)\.? [A-Z][a-z]+(?: [A-Z][a-z]+)?", re.I), "[name redacted]"),
]


def redact_pii(text: str) -> str:
    for pattern, repl in _PII:
        text = pattern.sub(repl, text)
    return text


def rule_check(text: str) -> tuple[str, str]:
    if _INJECTION.search(text):
        return "injection", f"matched injection pattern: {_INJECTION.search(text).group(0)!r}"
    if _OUT_OF_SCOPE.search(text):
        return "out_of_scope", f"matched medication-decision request: {_OUT_OF_SCOPE.search(text).group(0)!r}"
    return "safe", ""


async def check_input(text: str, llm=None) -> dict:
    """Returns {"verdict", "reason", "method"}; method = rules | rules+llm."""
    verdict, reason = rule_check(text)
    if verdict == "safe":
        return {"verdict": "safe", "reason": "", "method": "rules"}
    if llm is None or not get_agent_config().get("guardrails", {}).get("llm_check", True):
        return {"verdict": verdict, "reason": reason, "method": "rules"}
    try:
        judge = llm.with_structured_output(GuardVerdict, method="function_calling")
        out: GuardVerdict = await judge.ainvoke([("system", load_config("prompts")["input_guard"]),
                                                 ("user", text[:2000])])
        return {"verdict": out.verdict, "reason": out.reason or reason, "method": "rules+llm"}
    except Exception as e:   # fail closed: keep the rule verdict for suspicious input
        logger.warning(f"[guardrails] LLM check failed, keeping rule verdict {verdict}: {e}")
        return {"verdict": verdict, "reason": reason, "method": "rules"}
