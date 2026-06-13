"""Centralized configuration for the agentic research workflow."""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ObservabilityProvider(str, Enum):
    """Supported observability backends."""
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"
    BOTH = "both"
    NONE = "none"


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLM provider to use.",
    )
    model: str = Field(
        default="gpt-4o",
        description="Model name (e.g. gpt-4o, claude-sonnet-4-20250514).",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class SearchConfig(BaseModel):
    """Search tool configuration."""
    provider: Literal["tavily"] = "tavily"
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: Literal["basic", "advanced"] = "advanced"


class WorkflowConfig(BaseModel):
    """Workflow-level settings."""
    max_research_loops: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max iterations for the research loop.",
    )
    max_tool_calls_per_step: int = Field(default=5, ge=1)
    parallel_tool_calls: bool = True


class AppConfig(BaseModel):
    """Root application configuration."""
    llm: LLMConfig = LLMConfig()
    search: SearchConfig = SearchConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    observability: ObservabilityProvider = Field(
        default_factory=lambda: ObservabilityProvider(
            os.getenv("OBSERVABILITY_PROVIDER", "langsmith")
        )
    )


# Singleton config instance
config = AppConfig()


def get_llm():
    """Create and return the configured LLM instance."""
    if config.llm.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    elif config.llm.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
