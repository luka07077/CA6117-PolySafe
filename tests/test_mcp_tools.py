"""
Integration test: every tool called through the MCP server (starts it if needed).
search_drug_evidence uses the embedding model and the helper LLM, so this file is skipped without an API key.

    pytest -q tests/test_mcp_tools.py
"""
import asyncio
import json
import os

import pytest

from src.config import get_project_root  # noqa: F401  (loads .env)

pytestmark = pytest.mark.skipif(not os.environ.get("DASHSCOPE_API_KEY"), reason="needs DASHSCOPE_API_KEY in .env")

EXPECTED_TOOLS = {"get_knowledge_base_info", "normalize_drug", "check_interaction", "check_duplication",
                  "check_combination_rules", "check_older_adult_risks", "search_drug_evidence"}


@pytest.fixture(scope="module")
def call():
    from tools.mcp_client import load_tools, tool_text
    tools = asyncio.run(load_tools())

    def _call(tool_name: str, **args) -> dict:
        return json.loads(tool_text(asyncio.run(tools[tool_name].ainvoke(args))))
    _call.names = set(tools)
    return _call


def test_all_tools_registered(call):
    assert call.names == EXPECTED_TOOLS


def test_mcp_normalize_and_interaction(call):
    a, b = call("normalize_drug", name="Coumadin 5mg"), call("normalize_drug", name="Advil")
    r = call("check_interaction", drug_a=a["drug_id"], drug_b=b["drug_id"])
    assert r["found"] and r["severity"] == "high"


def test_mcp_regimen_checks(call):
    meds = ["warfarin", "aspirin", "ibuprofen", "naproxen"]
    assert call("check_duplication", drugs=meds)["duplications"][0]["group"] == "NSAID"
    assert call("check_combination_rules", drugs=meds)["combinations"][0]["rule_id"] == "CMB002"
    assert call("check_older_adult_risks", drugs=meds, age=78)["found"]


def test_mcp_evidence_found_and_not_found(call):
    hit = call("search_drug_evidence", query="warfarin ibuprofen interaction bleeding")
    assert hit["found"] and any(p["source"] == "01_anticoagulants_bleeding.md" for p in hit["passages"])
    miss = call("search_drug_evidence", query="amlodipine omeprazole interaction")
    assert not miss["found"]
