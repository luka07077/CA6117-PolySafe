"""
PolySafe MCP tool server: exposes the drug knowledge database and the evidence RAG as read-only MCP tools.

Runs as a long-lived HTTP service (streamable-http):
    python -m tools.mcp_server            # http://127.0.0.1:8775/mcp  (host/port in configs/agent_config.yaml)
tools/mcp_client.py starts it automatically when it is not running.

Least privilege: every tool is read-only. The agent has no tool that can write a prescription or
change a medication record.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from db.drug_db import connect  # noqa: E402
from src.config import get_agent_config  # noqa: E402
from src.utils.logger import get_logger, log_async_tool_call, log_tool_call  # noqa: E402
from tools import interaction, normalize, regimen  # noqa: E402

logger = get_logger("polysafe.mcp_server")
_cfg = get_agent_config()["mcp"]
_port = int(os.environ.get("POLYSAFE_MCP_PORT", _cfg["port"]))
if os.environ.get("POLYSAFE_EVAL_MODE") == "1":   # evaluation: no extra LLM calls inside RAG (planned token saving)
    get_agent_config()["rag"].update(query_rewrite=False, self_reflection=False)
mcp = FastMCP("PolySafe", host=_cfg["host"], port=_port, stateless_http=True, json_response=True,
              log_level="WARNING")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@mcp.tool()
@log_tool_call
def get_knowledge_base_info() -> str:
    """
    Size of the drug knowledge base this server answers from (drugs, synonyms, interactions,
    combination rules). Used by the UI and health checks.
    """
    with connect() as c:
        counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("drugs", "drug_synonyms", "drug_classes", "interactions", "combination_rules")}
    return _dump({"status": "ok", "counts": counts})


@mcp.tool()
@log_tool_call
def normalize_drug(name: str) -> str:
    """
    Map one medication name as written (brand name, misspelling, may include dose and frequency text) to a
    drug in the knowledge base. Returns drug_id, generic name, class, match_type (exact / synonym / fuzzy /
    uncertain / unknown) and confidence. recognized=false means the drug must be clarified by a human;
    "uncertain" results include suggestions.

    Args:
        name: the medication as written, e.g. "Coumadin 5 mg daily".
    """
    return _dump(normalize.normalize_drug(name))


@mcp.tool()
@log_tool_call
def check_interaction(drug_a: str, drug_b: str) -> str:
    """
    Look up the interaction between two drugs in the interaction database (drug-specific and drug-class
    rules). Returns found, the highest severity (critical / high / moderate / low) and every matching
    interaction with mechanism, effect, recommendation, source and evidence document.

    Args:
        drug_a: normalized drug_id (from normalize_drug), e.g. "warfarin".
        drug_b: normalized drug_id, e.g. "ibuprofen".
    """
    return _dump(interaction.check_interaction(drug_a, drug_b))


@mcp.tool()
@log_tool_call
def check_duplication(drugs: list[str]) -> str:
    """
    Check the whole medication list for therapeutic duplication: two or more drugs from the same group
    (e.g. two NSAIDs, two anticoagulants, two SSRIs) or the same drug entered twice under different names.

    Args:
        drugs: normalized drug_ids of ALL medications, including repeats.
    """
    return _dump(regimen.check_duplication(drugs))


@mcp.tool()
@log_tool_call
def check_combination_rules(drugs: list[str]) -> str:
    """
    Check the whole medication list against multi-drug high-risk patterns that pairwise checks miss:
    ACE inhibitor/ARB + diuretic + NSAID ("triple whammy"), anticoagulant + antiplatelet + NSAID,
    three or more CNS-active drugs, two or more strongly anticholinergic drugs.

    Args:
        drugs: normalized drug_ids of all medications.
    """
    return _dump(regimen.check_combination_rules(drugs))


@mcp.tool()
@log_tool_call
def check_older_adult_risks(drugs: list[str], age: int | None = None) -> str:
    """
    Flag potentially inappropriate medications for patients aged 65 or older (AGS Beers Criteria 2023
    summary). Skipped when the age is unknown or under 65.

    Args:
        drugs: normalized drug_ids of all medications.
        age: patient age in years.
    """
    return _dump(regimen.check_older_adult_risks(drugs, age))


@mcp.tool()
@log_async_tool_call
async def search_drug_evidence(query: str) -> str:
    """
    Search the drug-safety evidence knowledge base (interaction monographs and medication-safety guidance
    summarised from FDA labels, AGS Beers Criteria 2023, STOPP/START v3). Returns verbatim passages with
    source document and section; an empty list means no supporting evidence was found.

    Args:
        query: e.g. "warfarin ibuprofen interaction bleeding".
    """
    from agent.models.cloud_model import UsageTracker
    from rag.agentic_rag import search_evidence
    usage = UsageTracker()   # helper-LLM tokens used inside this tool, reported back to the agent
    # async tool + worker thread: the server keeps serving other calls, so the agent's parallel queries overlap
    passages = await asyncio.to_thread(search_evidence, query, [usage])
    return _dump({"status": "ok", "query": query, "found": bool(passages), "passages": passages,
                  "usage": usage.as_dict()})


if __name__ == "__main__":
    from rag.vector_stores import get_vector_store
    get_vector_store()   # open the evidence store before accepting requests
    logger.info(f"PolySafe MCP server on http://{_cfg['host']}:{_port}/mcp")
    mcp.run(transport="streamable-http")
