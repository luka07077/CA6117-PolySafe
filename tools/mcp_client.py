"""
MCP client: connects to the PolySafe MCP server (streamable HTTP) and returns LangChain tools.
If the server is not running, it is started in the background (logs: outputs/logs/mcp_server.log).
"""
import os
import socket
import subprocess
import sys
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import get_agent_config, get_outputs_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger("polysafe.mcp_client")


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def ensure_server() -> str:
    cfg = get_agent_config()["mcp"]
    host, port = cfg["host"], int(os.environ.get("POLYSAFE_MCP_PORT", cfg["port"]))   # evaluation uses its own server
    url = f"http://{host}:{port}/mcp"
    if _port_open(host, port):
        return url
    os.makedirs(get_outputs_dir("logs"), exist_ok=True)
    log = open(get_outputs_dir("logs", "mcp_server.log"), "a")
    logger.info("[mcp] starting PolySafe MCP server...")
    subprocess.Popen([sys.executable, "-m", "tools.mcp_server"], cwd=get_project_root(), stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + cfg.get("startup_timeout", 60)
    while time.time() < deadline:
        if _port_open(host, port):
            logger.info(f"[mcp] server ready at {url}")
            return url
        time.sleep(0.5)
    raise RuntimeError(f"MCP server did not start within {cfg.get('startup_timeout')}s; see outputs/logs/mcp_server.log")


async def load_tools() -> dict:
    """Return {tool_name: tool} for all PolySafe MCP tools."""
    url = ensure_server()
    client = MultiServerMCPClient({"polysafe": {"transport": "streamable_http", "url": url}})
    tools = await client.get_tools()
    logger.info(f"[mcp] loaded tools: {[t.name for t in tools]}")
    return {t.name: t for t in tools}


def tool_text(result) -> str:
    """MCP tool results arrive as a string or a list of content blocks; return the plain text."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in result)
    return getattr(result, "content", None) and tool_text(result.content) or str(result)
