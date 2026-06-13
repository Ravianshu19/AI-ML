# 🔬 Agentic Research Workflow

A production-ready multi-step research agent built with **LangGraph**, featuring tool calling, iterative research loops, and dual observability via **LangSmith** and **Langfuse**.

## Architecture

```
    START
      │
      ▼
  ┌──────────┐
  │   PLAN   │  ← Decompose query into targeted sub-queries
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ RESEARCH │  ← Search web + scrape pages (tool calling)
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ ANALYZE  │  ← Cross-reference sources, find gaps
  └────┬─────┘
       │
       ├── Need more? ──► YES ──► back to RESEARCH
       │
       ▼ NO
  ┌──────────┐
  │  REPORT  │  ← Generate comprehensive Markdown report
  └────┬─────┘
       │
       ▼
      END
```

## Features

- **Multi-step research pipeline** — Plan → Research → Analyze → Report
- **Iterative loops** — Agent autonomously decides if more research is needed
- **Tool calling** — Web search (Tavily), page scraping, fact extraction, source comparison
- **Dual observability** — LangSmith and/or Langfuse for full trace visibility
- **Checkpointing** — State persistence via LangGraph's MemorySaver
- **Beautiful CLI** — Rich terminal output with progress, panels, and Markdown rendering
- **CrewAI alternative** — Included as a reference implementation

## Quick Start

### 1. Install Dependencies

```bash
cd agentic-research-workflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required keys:**
| Key | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | LLM provider (or `ANTHROPIC_API_KEY`) |
| `TAVILY_API_KEY` | Web search tool |
| `LANGCHAIN_API_KEY` | LangSmith observability (optional) |
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | Langfuse observability (optional) |

### 3. Run Research

```bash
# Basic usage
python -m src.main research "What are the latest advances in quantum computing?"

# With options
python -m src.main research "Impact of AI on healthcare" \
  --max-iterations 5 \
  --output report.md \
  --observability both \
  --verbose
```

### 4. Other Commands

```bash
# View the workflow graph
python -m src.main visualize

# Check configuration
python -m src.main config-info
```

## Project Structure

```
agentic-research-workflow/
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized configuration (Pydantic)
│   ├── main.py               # CLI entry point (Typer + Rich)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py         # Tavily web search tools
│   │   ├── scraper.py        # Web page scraper
│   │   └── analyzer.py       # Fact extraction & source comparison
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py          # ResearchState dataclass
│   │   ├── nodes.py          # Graph node functions
│   │   └── workflow.py       # LangGraph assembly + runner
│   ├── observability/
│   │   ├── __init__.py
│   │   └── setup.py          # LangSmith + Langfuse setup
│   └── crewai_alternative/
│       ├── __init__.py
│       └── crew.py           # CrewAI implementation (alternative)
└── README.md
```

## Observability

### LangSmith

LangSmith provides automatic tracing for all LangChain/LangGraph operations via environment variables:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=agentic-research-workflow
```

View traces at: https://smith.langchain.com

### Langfuse

Langfuse provides an open-source alternative with a callback handler:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

View traces at: https://cloud.langfuse.com

### Switching Providers

```bash
# Use LangSmith only
OBSERVABILITY_PROVIDER=langsmith

# Use Langfuse only
OBSERVABILITY_PROVIDER=langfuse

# Use both simultaneously
OBSERVABILITY_PROVIDER=both

# Disable observability
OBSERVABILITY_PROVIDER=none
```

## CrewAI Alternative

An equivalent implementation using CrewAI is included in `src/crewai_alternative/crew.py`. 
To use it:

```bash
pip install crewai crewai-tools
python -m src.crewai_alternative.crew
```

## License

MIT


---

# Previous Repository Content


# AI-ML
# 🧠 Advanced LLM Projects 

Welcome to my collection of hands-on projects using cutting-edge Large Language Models (LLMs). This repo showcases how I’ve explored LLMs for generative AI, agentic tasks, code generation, chatbot development, and workflow automation.

---

## 🔍 Project Overview

### 1. `AI using Agentic AI.ipynb`
Builds an agent-style AI that can reason, plan, and execute tasks autonomously using tool integrations.

### 2. `Code_Generation_Model_with_LLMs.ipynb`
A demonstration of how LLMs can generate working code from natural language prompts. Includes best practices and prompt optimization.

### 3. `End_to_End_Chatbot.ipynb`
Complete pipeline for building an AI chatbot — from prompt design to deployment-ready interfaces.

### 4. `Generative AI Model.ipynb`
Explores use of LLMs for creative content generation — including stories, images (via prompts), and contextual expansion.

### 5. `LLM .ipynb`
General experiments and utilities using LLMs — API calls, embeddings, summarization, and interactive testing.

---

## 🛠️ Technologies Used

- OpenAI / Hugging Face Transformers
- LangChain (where applicable)
- Python (NumPy, pandas, matplotlib)
- Streamlit / Gradio (for deployment-ready UIs)
- VS Code + Jupyter Notebooks

---


