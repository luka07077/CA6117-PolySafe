"""
The single cloud LLM wrapper (DashScope OpenAI-compatible endpoint).

All roles share one class; the model name comes from configs/agent_config.yaml (llm section):
    agent     parse_input / assess_risk / safety-reviewer nodes
              (with agent_fallback_models tried in order when a call fails)
    helper    input guard and RAG helper calls (cheap flash model)
    eval      the agent under test in formal evaluation (pinned, temperature 0, no fallback)
Environment: DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL (see .env.example).
Override the agent model without editing YAML: POLYSAFE_AGENT_MODEL=<name>.
"""
import os
import threading
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import get_agent_config
from src.utils.logger import get_logger

logger = get_logger("polysafe.llm")

_ROLE_KEYS = {
    "agent": ("agent_model", "agent_temperature"),
    "helper": ("helper_model", "helper_temperature"),
    "eval": ("eval_agent_model", "eval_temperature"),
}


def _credentials() -> tuple[str, str]:
    api_key, base_url = os.environ.get("DASHSCOPE_API_KEY"), os.environ.get("DASHSCOPE_BASE_URL")
    if not api_key or not base_url or api_key.startswith("sk-your"):
        raise ValueError("DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL missing — copy .env.example to .env and fill in the key")
    return api_key, base_url


class CloudChatModel:
    """
    Factory for chat models of one role. `.llm` is a LangChain ChatOpenAI pointed at DashScope.
    For the "agent" role, `.fallback_llms` are the fallback models in order (empty otherwise: evaluation stays
    pinned) and `.llm_safe` is `.llm` with those fallbacks attached.
    """

    def __init__(self, role: str = "agent", temperature: float | None = None, max_tokens: int | None = None):
        cfg = get_agent_config()["llm"]
        model_key, temp_key = _ROLE_KEYS[role]
        model = cfg[model_key]
        if role == "agent" and os.environ.get("POLYSAFE_AGENT_MODEL"):
            model = os.environ["POLYSAFE_AGENT_MODEL"]
        if temperature is None:
            temperature = cfg.get(temp_key, 0.0)
        api_key, base_url = _credentials()
        self.model_name = model

        # DashScope serves DeepSeek (and Qwen) with thinking mode ON by default; hidden reasoning tokens made long
        # prompts take 20-70 s. enable_thinking is a DashScope parameter, sent explicitly for every model.
        extra = {"enable_thinking": bool(cfg.get("enable_thinking", False))}

        def make(name: str) -> ChatOpenAI:
            return ChatOpenAI(model=name, api_key=api_key, base_url=base_url, temperature=temperature,
                              max_tokens=max_tokens, timeout=cfg.get("request_timeout", 120), max_retries=2,
                              extra_body=extra)

        self.llm = make(model)
        pinned = role != "agent" or os.environ.get("POLYSAFE_AGENT_MODEL")
        fallbacks = [] if pinned else (cfg.get("agent_fallback_models") or [])
        self.fallback_names = [f for f in dict.fromkeys(fallbacks) if f != model]
        self.fallback_llms = [make(f) for f in self.fallback_names]
        logger.info(f"[LLM] role={role} model={model} fallbacks={self.fallback_names} temperature={temperature}")

    @property
    def llm_safe(self):
        """The model with the fallbacks attached (for plain invoke calls outside create_agent)."""
        return self.llm.with_fallbacks(self.fallback_llms) if self.fallback_llms else self.llm


def get_embeddings():
    """
    Embedding function for the evidence vector store.
    embedding_model "local" -> None (Chroma then uses its built-in all-MiniLM-L6-v2 ONNX model, no key needed).
    """
    cfg = get_agent_config()["llm"]
    if cfg["embedding_model"] == "local":
        return None
    api_key, base_url = _credentials()
    return OpenAIEmbeddings(model=cfg["embedding_model"], api_key=api_key, base_url=base_url,
                            check_embedding_ctx_length=False, chunk_size=10)


@dataclass(eq=False)   # identity hash: LangChain keeps callback handlers in sets
class UsageTracker(BaseCallbackHandler):
    """Callback that counts LLM calls and tokens for one request (pass via config={'callbacks': [tracker]})."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_llm_end(self, response, **kwargs):
        usage = (response.llm_output or {}).get("token_usage") or {}
        model = (response.llm_output or {}).get("model_name")
        for gen in response.generations:   # streamed calls carry usage/model only on the message
            for g in gen:
                msg = getattr(g, "message", None)
                if not usage:
                    meta = getattr(msg, "usage_metadata", None) or {}
                    usage = {"prompt_tokens": meta.get("input_tokens", 0), "completion_tokens": meta.get("output_tokens", 0)}
                if not model:
                    model = (getattr(msg, "response_metadata", None) or {}).get("model_name")
        model = model or "?"
        with self._lock:
            self.calls += 1
            p, c = int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
            self.prompt_tokens += p
            self.completion_tokens += c
            m = self.by_model.setdefault(model, {"calls": 0, "tokens": 0})
            m["calls"] += 1
            m["tokens"] += p + c

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens, "by_model": self.by_model}
