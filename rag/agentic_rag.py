"""
Agentic RAG over the drug-safety evidence corpus (3 steps):
  1. query rewriting      (helper LLM; switch rag.query_rewrite — off by default to save tokens)
  2. multi-path retrieval (original + rewritten queries, de-duplicated)
  3. self-reflection      (helper LLM picks the passages that actually support the question; rag.self_reflection)

Unlike HealthRec, reflection SELECTS passages by index instead of rewriting them, so the evidence
returned is always verbatim text from the corpus and can be cited and checked by the safety reviewer.
Returns a list of {"source", "section", "passage"}; an empty list means "insufficient evidence".
"""
import json
import re

from langchain_core.prompts import ChatPromptTemplate

from agent.models.cloud_model import CloudChatModel
from rag.vector_stores import search
from src.config import get_agent_config, load_config
from src.utils.logger import get_logger

logger = get_logger("polysafe.agentic_rag")
_llm = None


def _helper_llm():
    global _llm
    if _llm is None:
        _llm = CloudChatModel("helper").llm_safe
    return _llm


def _to_passage(doc) -> dict:
    m = doc.metadata
    return {"source": m.get("source", "?"), "section": m.get("section") or m.get("title") or "",
            "passage": doc.page_content.strip()}


def search_evidence(query: str, callbacks: list | None = None) -> list[dict]:
    cfg, prompts = get_agent_config()["rag"], load_config("prompts")
    run_cfg = {"callbacks": callbacks or []}
    queries = [query]
    if cfg.get("query_rewrite", False):
        try:
            out = (ChatPromptTemplate.from_template(prompts["rag_query_rewrite"]) | _helper_llm()).invoke({"query": query}, config=run_cfg)
            queries += [q.strip("-• ").strip() for q in out.content.splitlines() if q.strip()][:3]
        except Exception as e:   # retrieval still works with the original query
            logger.warning(f"[rag] rewrite failed: {e}")
    seen, docs = set(), []
    for q in queries:
        for d in search(q):
            key = d.page_content[:120]
            if key not in seen:
                seen.add(key)
                docs.append(d)
    passages = [_to_passage(d) for d in docs]
    if not passages or not cfg.get("self_reflection", True):
        return passages
    context = "\n\n".join(f"[{i}] ({p['source']} · {p['section']})\n{p['passage']}" for i, p in enumerate(passages))
    try:
        out = (ChatPromptTemplate.from_template(prompts["rag_reflection"]) | _helper_llm()).invoke(
            {"query": query, "context": context}, config=run_cfg)
        text = out.content.strip()
        if "NOT_FOUND" in text:
            return []
        listed = re.search(r"\[[\d,\s]*\]", text)
        if not listed:   # unparseable answer: keep the raw retrieval rather than drop evidence
            return passages
        return [passages[i] for i in dict.fromkeys(json.loads(listed.group(0))) if 0 <= i < len(passages)]
    except Exception as e:   # reflection is a filter: on failure keep the raw retrieval
        logger.warning(f"[rag] reflection failed, returning unfiltered passages: {e}")
        return passages


def format_evidence(passages: list[dict]) -> str:
    return json.dumps(passages, ensure_ascii=False)
