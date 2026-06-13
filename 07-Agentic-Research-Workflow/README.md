# 🔬 Agentic Research Workflow

Multi-step research agent built with **LangGraph** and **CrewAI**, featuring tool calling and dual observability via **LangSmith** and **Langfuse**.

## Architecture

```
START → PLAN → RESEARCH → ANALYZE → (loop?) → REPORT → END
```

## Setup

```bash
cd 07-Agentic-Research-Workflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Run

```bash
# LangGraph version
python -m src.main research "Your research topic" -n 3 -o report.md

# CrewAI version
pip install crewai crewai-tools
python -m src.crewai_alternative.crew "Your research topic"
```

See the [main project README](../README.md) for the full project index.
