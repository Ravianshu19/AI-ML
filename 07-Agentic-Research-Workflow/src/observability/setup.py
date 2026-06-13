"""Unified observability setup for LangSmith and Langfuse.

This module provides a single entry point to configure tracing and
callback handlers for both LangSmith and Langfuse, depending on
the chosen provider in the application config.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.config import config, ObservabilityProvider

logger = logging.getLogger(__name__)


def _setup_langsmith() -> list[BaseCallbackHandler]:
    """Configure LangSmith tracing.

    LangSmith is configured primarily through environment variables:
    - LANGCHAIN_TRACING_V2=true
    - LANGCHAIN_API_KEY
    - LANGCHAIN_PROJECT
    - LANGCHAIN_ENDPOINT

    Returns:
        List of callback handlers (empty for LangSmith since it uses env vars).
    """
    required_vars = ["LANGCHAIN_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.warning(
            "LangSmith: Missing environment variables: %s. "
            "Tracing will not be active.",
            ", ".join(missing),
        )
        return []

    # Ensure tracing is enabled
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "agentic-research-workflow")
    os.environ.setdefault(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    logger.info(
        "LangSmith tracing enabled — project: %s",
        os.getenv("LANGCHAIN_PROJECT"),
    )
    # LangSmith auto-instruments via env vars; no explicit callback needed.
    return []


def _setup_langfuse() -> list[BaseCallbackHandler]:
    """Configure Langfuse tracing with a callback handler.

    Returns:
        List containing the Langfuse callback handler.
    """
    required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.warning(
            "Langfuse: Missing environment variables: %s. "
            "Tracing will not be active.",
            ", ".join(missing),
        )
        return []

    try:
        from langfuse.callback import CallbackHandler as LangfuseHandler

        handler = LangfuseHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            # Attach metadata for filtering in the Langfuse dashboard
            tags=["research-agent", "langgraph"],
            metadata={"workflow": "multi-step-research"},
        )
        logger.info(
            "Langfuse tracing enabled — host: %s",
            os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return [handler]

    except ImportError:
        logger.error(
            "Langfuse package not installed. Run: pip install langfuse"
        )
        return []


def initialize_observability() -> list[BaseCallbackHandler]:
    """Initialize the configured observability provider(s).

    Reads from `config.observability` and sets up the appropriate
    tracing backends.

    Returns:
        A list of callback handlers to pass to LangChain/LangGraph.
    """
    callbacks: list[BaseCallbackHandler] = []
    provider = config.observability

    if provider in (ObservabilityProvider.LANGSMITH, ObservabilityProvider.BOTH):
        callbacks.extend(_setup_langsmith())

    if provider in (ObservabilityProvider.LANGFUSE, ObservabilityProvider.BOTH):
        callbacks.extend(_setup_langfuse())

    if provider == ObservabilityProvider.NONE:
        logger.info("Observability disabled.")

    return callbacks


def get_callbacks() -> dict[str, Any]:
    """Get a config dict with callbacks for use in LangChain invocations.

    Usage:
        llm.invoke(prompt, **get_callbacks())
        graph.invoke(state, config=get_callbacks())

    Returns:
        A dict with 'callbacks' key, suitable for unpacking.
    """
    callbacks = initialize_observability()
    if callbacks:
        return {"callbacks": callbacks}
    return {}
