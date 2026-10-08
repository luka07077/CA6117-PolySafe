"""
Step-1 smoke test: checks that every external piece the agent depends on works.

    python scripts/smoke_test.py              # all checks
    python scripts/smoke_test.py --skip-llm   # only database + MCP (no API key needed)

Checks:
  1. drug database          data/polysafe.db exists and is readable
  2. MCP server             starts, lists tools, answers a tool call
  3. LLMs (pro + flash)     plain reply, tool calling, structured output; latency of each call
  4. embeddings             one query embedded with llm.embedding_model
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field  # noqa: E402

from src.config import get_agent_config  # noqa: E402

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def check_db():
    from db.drug_db import connect
    try:
        with connect() as c:
            n = c.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        record("drug database", True, f"{n} interactions")
    except Exception as e:
        record("drug database", False, str(e))


async def check_mcp():
    from tools.mcp_client import load_tools, tool_text
    try:
        tools = await load_tools()
        out = json.loads(tool_text(await tools["get_knowledge_base_info"].ainvoke({})))
        record("MCP server", out.get("status") == "ok", f"tools={list(tools)} counts={out.get('counts')}")
    except Exception as e:
        record("MCP server", False, f"{type(e).__name__}: {e}")


class MedList(BaseModel):
    """Medications mentioned in the text."""
    age: int | None = Field(None, description="patient age in years")
    medications: list[str] = Field(default_factory=list, description="medication names as written")


def check_llm(model_name: str):
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from agent.models.cloud_model import _credentials

    key, url = _credentials()
    llm = ChatOpenAI(model=model_name, api_key=key, base_url=url, temperature=0, timeout=120, max_retries=0)

    t0 = time.perf_counter()
    try:
        r = llm.invoke("Reply with exactly: OK")
        record(f"{model_name} · plain reply", "OK" in r.content, f"{time.perf_counter() - t0:.1f}s, reply={r.content[:40]!r}")
    except Exception as e:
        record(f"{model_name} · plain reply", False, f"{type(e).__name__}: {str(e)[:200]}")
        return

    @tool
    def check_interaction(drug_a: str, drug_b: str) -> str:
        """Look up the interaction between two drugs (generic names)."""
        return "{}"

    t0 = time.perf_counter()
    try:
        r = llm.bind_tools([check_interaction]).invoke("Check the interaction between warfarin and ibuprofen.")
        calls = r.tool_calls
        ok = bool(calls) and calls[0]["name"] == "check_interaction"
        record(f"{model_name} · tool calling", ok, f"{time.perf_counter() - t0:.1f}s, calls={calls}")
    except Exception as e:
        record(f"{model_name} · tool calling", False, f"{type(e).__name__}: {str(e)[:200]}")

    t0 = time.perf_counter()
    try:
        r = llm.with_structured_output(MedList, method="function_calling").invoke(
            "72-year-old man taking Coumadin 5 mg daily, aspirin 81 mg and Advil as needed.")
        ok = isinstance(r, MedList) and r.age == 72 and len(r.medications) == 3
        record(f"{model_name} · structured output", ok, f"{time.perf_counter() - t0:.1f}s, {r!r}")
    except Exception as e:
        record(f"{model_name} · structured output", False, f"{type(e).__name__}: {str(e)[:200]}")


def check_embeddings():
    from agent.models.cloud_model import get_embeddings
    name = get_agent_config()["llm"]["embedding_model"]
    t0 = time.perf_counter()
    try:
        emb = get_embeddings()
        if emb is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            vec = DefaultEmbeddingFunction()(["warfarin bleeding risk"])[0]
        else:
            vec = emb.embed_query("warfarin bleeding risk")
        record(f"embeddings ({name})", len(vec) > 0, f"dim={len(vec)}, {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        record(f"embeddings ({name})", False, f"{type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()
    cfg = get_agent_config()["llm"]
    print("1. drug database");  check_db()
    print("2. MCP server");     asyncio.run(check_mcp())
    if not args.skip_llm:
        print("3. LLMs")
        for m in dict.fromkeys([cfg["agent_model"], *cfg["agent_fallback_models"], cfg["helper_model"]]):
            check_llm(m)
        print("4. embeddings");  check_embeddings()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
