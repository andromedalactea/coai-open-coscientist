# Academic MCP Server — Usage Guide

This guide explains how to use the **Academic Literature MCP Server** (Semantic Scholar + arXiv + Unpaywall) to run research with Open Coscientist, with a focus on astronomy and general academic domains.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Configure the Academic MCP Server](#step-1-configure-the-academic-mcp-server)
4. [Step 2: Start the Academic MCP Server](#step-2-start-the-academic-mcp-server)
5. [Step 3: Point Open Coscientist to the Academic MCP](#step-3-point-open-coscientist-to-the-academic-mcp)
6. [Step 4: Run a Research Workflow](#step-4-run-a-research-workflow)
7. [Astronomy-Specific Tips](#astronomy-specific-tips)
8. [Understanding the Pipeline](#understanding-the-pipeline)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Academic MCP server is a **drop-in replacement** for the PubMed MCP. It:

- **Discovers papers** via Semantic Scholar (natural language and keyword search)
- **Filters** to papers with both arXiv ID and DOI (required for content download)
- **Reranks** results with an AI model to pick the most relevant papers
- **Downloads content** from arXiv (LaTeX source preferred, PDF fallback) or Unpaywall (PDF)
- **Caches** papers in a shared pool for reuse across runs

The main Open Coscientist app does **not** need code changes — you switch literature engines by changing the YAML configuration.

---

## Prerequisites

Before starting, ensure you have:

| Requirement | Purpose |
|-------------|---------|
| **Python 3.12+** | Run the MCP server and Open Coscientist |
| **LLM API key** | For hypothesis generation (Gemini, OpenAI, Anthropic, etc.) |
| **LLM API key** | For the reranker inside the Academic MCP (e.g. `OPENAI_API_KEY` for `gpt-4.1-mini`) |
| **Valid email** | Required by Unpaywall API (e.g. `your_email@example.com`) |
| **Optional: Semantic Scholar API key** | Increases Semantic Scholar rate limit from 1 to 10 req/sec |

---

## Step 1: Configure the Academic MCP Server

### 1.1 Create the environment file

From the project root:

```bash
cp mcps/mcp_server_academic/.env.example mcps/mcp_server_academic/.env
```

### 1.2 Edit `mcps/mcp_server_academic/.env`

Set at least these variables:

```bash
# Required: Unpaywall needs a real email (not test@example.com)
UNPAYWALL_EMAIL=your_real_email@example.com

# Required: At least one LLM key for the reranker
OPENAI_API_KEY=sk-...
# or: ANTHROPIC_API_KEY=...
# or: GEMINI_API_KEY=...

# Optional: Semantic Scholar API key (higher rate limits)
SEMANTIC_SCHOLAR_API_KEY=your_s2_key

# Optional: Reranker model (default: gpt-4.1-mini)
RERANKER_MODEL=gpt-4.1-mini

# Optional: Paper cache directory
COSCIENTIST_LIT_REVIEW_DIR=./cache/literature_review
```

**Important:** Unpaywall rejects generic emails like `test@example.com`. Use your real institutional or personal email.

---

## Step 2: Start the Academic MCP Server

You can run it with **Docker** or **locally**.

### Option A: Docker (recommended)

From the project root:

```bash
# Build and start the academic MCP server
docker compose up -d mcp-server-academic

# Verify it's running
curl http://localhost:8889
```

You should see JSON with `"status": "running"` and `"service": "coscientist-academic-lit-review"`.

### Option B: Local development

From the project root:

```bash
# Activate your virtual environment
source .venv/bin/activate   # or: source venv/bin/activate

# Install the academic MCP package
pip install -e ./mcps/mcp_server_academic

# Start the server
uvicorn mcp_server_academic.server:app --host 0.0.0.0 --port 8889
```

The server listens on **port 8889** (PubMed MCP uses 8888).

---

## Step 3: Point Open Coscientist to the Academic MCP

Open Coscientist uses a YAML config to choose which MCP server and tools to use. You have two options.

### Option A: Use the example config directly

Pass the example config when creating the generator:

```python
from open_coscientist import HypothesisGenerator

generator = HypothesisGenerator(
    model_name="gemini/gemini-3-flash",
    tools_config="src/open_coscientist/config/examples/academic_semantic_scholar.yaml",
)
```

### Option B: Copy to user config (persistent)

```bash
mkdir -p ~/.coscientist
cp src/open_coscientist/config/examples/academic_semantic_scholar.yaml ~/.coscientist/tools.yaml
```

Open Coscientist will automatically load `~/.coscientist/tools.yaml` if it exists, so you don't need to pass `tools_config` in code.

### Option C: Override the server URL

If the Academic MCP runs on a different host or port:

```bash
export ACADEMIC_MCP_SERVER_URL=http://your-host:8889/mcp
```

The YAML uses `${ACADEMIC_MCP_SERVER_URL:-http://localhost:8889/mcp}`, so this env var overrides the default.

---

## Step 4: Run a Research Workflow

### 4.1 Set LLM API keys for the main app

The main app needs an API key for hypothesis generation:

```bash
export GEMINI_API_KEY="your-gemini-key"
# or: export OPENAI_API_KEY="..."
# or: export ANTHROPIC_API_KEY="..."
```

### 4.2 Run the example script

```bash
cd /path/to/coai-open-coscientist
source .venv/bin/activate

# If using Option A (explicit config), edit examples/run.py to add tools_config
python examples/run.py
```

### 4.3 Or use a minimal script with explicit config

```python
import asyncio
from open_coscientist import HypothesisGenerator
from open_coscientist.console import ConsoleReporter, default_progress_callback, run_console

async def main():
    generator = HypothesisGenerator(
        model_name="gemini/gemini-3-flash",
        max_iterations=2,
        initial_hypotheses_count=7,
        evolution_max_count=4,
        tools_config="src/open_coscientist/config/examples/academic_semantic_scholar.yaml",
    )

    reporter = ConsoleReporter()
    await reporter.run(
        event_stream=generator.generate_hypotheses(
            research_goal="Novel methods for detecting exoplanets in high-contrast imaging",
            progress_callback=default_progress_callback,
            opts={
                "enable_literature_review_node": True,
                "enable_tool_calling_generation": True,
            },
            stream=True,
        ),
        research_goal="Novel methods for detecting exoplanets in high-contrast imaging",
    )

if __name__ == "__main__":
    run_console(main())
```

### 4.4 What happens during a run

1. **Supervisor** — Plans the research strategy.
2. **Literature Review** — Generates 2–4 search queries from your research goal, calls `academic_search_with_fulltext` for each, and analyzes the papers.
3. **Generate** — Creates hypotheses informed by the literature (and optionally uses tools in real time).
4. **Reflection** — Compares hypotheses to the literature.
5. **Review, Rank, Tournament, Meta-Review, Evolve** — Refines and ranks hypotheses.

Papers are cached in `COSCIENTIST_LIT_REVIEW_DIR` (or `./cache/literature_review` by default) under `academic/<slug>/shared/`, so repeated runs with the same research goal reuse downloaded papers.

---

## Astronomy-Specific Tips

### Query style

Semantic Scholar accepts natural language and keyword queries. For astronomy:

- **Good:** `"exoplanet detection high-contrast imaging coronagraph"`
- **Good:** `"dark matter halo substructure Milky Way"`
- **Good:** `"gravitational wave cosmology standard sirens"`
- **Avoid:** PubMed-style boolean operators (`AND`, `OR`, `NOT`) — Semantic Scholar does not use them the same way.

### Paper coverage

- **arXiv** — Strong for astronomy, physics, and math. Most papers have LaTeX source.
- **Unpaywall** — Covers many published papers with open-access versions.
- **Filter** — Only papers with both arXiv ID and DOI are used. Some older or non-astro papers may be excluded.

### Recency

The `recency_years` parameter filters to recent papers. The default is 0 (no filter). For fast-moving topics, consider adding a recency filter in the workflow config if supported.

---

## Understanding the Pipeline

When you call `academic_search_with_fulltext`:

```
1. Semantic Scholar search
   → Returns up to 100 candidate papers

2. Filter
   → Keep only papers with BOTH arXiv ID and DOI

3. AI Reranker
   → LLM scores papers by relevance to the query
   → Returns top n × 1.5 (50% buffer for download failures)

4. Content download (in ranked order)
   → Try arXiv source (.tex, .bib, .bbl) first
   → Fallback: arXiv PDF
   → Fallback: Unpaywall PDF via DOI
   → Skip paper if all fail

5. Text extraction
   → LaTeX → markdown (preferred)
   → PDF → text (fallback)

6. Shared pool
   → Cache in slug/shared/, symlink to slug/runs/{run_id}/
```

---

## Troubleshooting

### "Academic search pipeline is unavailable"

- Ensure the Academic MCP server is running: `curl http://localhost:8889`
- Check `mcps/mcp_server_academic/.env` has `UNPAYWALL_EMAIL` set.
- Semantic Scholar may rate-limit; add `SEMANTIC_SCHOLAR_API_KEY` for higher limits.

### "No papers with both arXiv ID and DOI"

- Some domains have fewer papers with both identifiers. Try broader or different queries.
- Astronomy and physics typically have good coverage on arXiv with DOIs.

### Reranker errors

- Ensure `OPENAI_API_KEY` (or another LLM key) is set in `mcps/mcp_server_academic/.env`.
- Check `RERANKER_MODEL` is a valid LiteLLM model name.

### Unpaywall 422 error

- Unpaywall rejects generic emails. Use a real email in `UNPAYWALL_EMAIL`.

### Papers not downloading

- arXiv rate limit: 3 seconds between requests. Large batches take time.
- Some papers have no open-access PDF on Unpaywall.
- Check logs: `COSCIENTIST_MCP_LOG_LEVEL=DEBUG` in `mcps/mcp_server_academic/.env`.

### Wrong MCP server used

- Confirm `tools_config` points to `academic_semantic_scholar.yaml`.
- If using `~/.coscientist/tools.yaml`, ensure it’s the academic config, not the PubMed one.
- Default PubMed MCP is on port 8888; Academic MCP is on 8889.

### Both MCPs running

- You can run both. Use `tools_config` to choose which one Open Coscientist uses.
- Only one config is active per run.

---

## Quick Reference

| Item | Value |
|------|-------|
| Academic MCP port | 8889 |
| PubMed MCP port | 8888 |
| Config file | `src/open_coscientist/config/examples/academic_semantic_scholar.yaml` |
| Env file | `mcps/mcp_server_academic/.env` |
| Paper cache | `COSCIENTIST_LIT_REVIEW_DIR` (default: `./cache/literature_review`) |
| Docker service | `mcp-server-academic` |
