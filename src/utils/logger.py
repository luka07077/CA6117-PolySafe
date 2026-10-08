import logging
import time
import functools
from typing import Callable, Any

"""
Logger module for the project.
Provides a shared logger to track tool calls, timing, and errors.
"""

# Singleton logger for the whole project
_logger = None


def get_logger(name: str = "polysafe") -> logging.Logger:
    """Get (or create) the shared project logger."""
    global _logger
    if _logger is not None and name == "polysafe":
        return _logger

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%m/%d/%y %H:%M:%S"
        ))
        logger.addHandler(handler)

    if name == "polysafe":
        _logger = logger
    return logger


def log_tool_call(func: Callable) -> Callable:
    """
    Decorator that automatically logs tool input args, output, and time taken.

    Usage:
        @log_tool_call
        def check_interaction(drug_a: str, drug_b: str) -> str:
            ...
    """
    logger = get_logger("polysafe.tools")

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        call_args = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        logger.info(f"CALL  -> {func_name}({call_args})")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            result_preview = str(result)[:200]
            logger.info(f"OK    <- {func_name} ({elapsed:.2f}s) => {result_preview}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"ERROR <- {func_name} ({elapsed:.2f}s) => {type(e).__name__}: {e}")
            raise

    return wrapper


def log_async_tool_call(func: Callable) -> Callable:
    """Decorator: async version of log_tool_call."""
    logger = get_logger("polysafe.tools")

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        call_args = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        logger.info(f"CALL  -> {func_name}({call_args})")

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            result_preview = str(result)[:200]
            logger.info(f"OK    <- {func_name} ({elapsed:.2f}s) => {result_preview}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"ERROR <- {func_name} ({elapsed:.2f}s) => {type(e).__name__}: {e}")
            raise

    return wrapper
