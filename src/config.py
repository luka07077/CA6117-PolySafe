import os
import yaml

"""
Config loader: reads YAML config files under configs/ and makes them available to other modules.
The .env file in the project root (API key, base URL) is loaded automatically.

    POLYSAFE_OUTPUTS   results directory instead of outputs/ (relative to the project root or absolute)
"""

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONF_DIR = os.path.join(_PROJECT_ROOT, "configs")

# Autoload .env file from project root if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed, just use system env vars

# Streamlit Community Cloud: copy app secrets into env vars (also inherited by the MCP server subprocess)
if not os.environ.get("DASHSCOPE_API_KEY"):
    try:
        import streamlit as st
        for _k in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"):
            if _k in st.secrets:
                os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass  # no secrets file (local runs use .env)

# Cache to avoid re-reading files every time
_configs: dict = {}


def get_project_root() -> str:
    return _PROJECT_ROOT


def get_data_dir(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, "data", *parts)


def get_outputs_dir(*parts: str) -> str:
    """Results directory (outputs/ by default, or $POLYSAFE_OUTPUTS), joined with `parts`."""
    base = os.environ.get("POLYSAFE_OUTPUTS", "outputs")
    return os.path.join(base if os.path.isabs(base) else os.path.join(_PROJECT_ROOT, base), *parts)


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (cached)."""
    if name not in _configs:
        path = os.path.join(_CONF_DIR, f"{name}.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _configs[name] = yaml.safe_load(f)
    return _configs[name]


def get_agent_config() -> dict:
    return load_config("agent_config")


def get_prompt_config() -> dict:
    return load_config("prompts")
