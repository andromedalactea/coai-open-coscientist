"""
Full academic search pipeline with content download.

Orchestrates: Semantic Scholar search -> filter (arXiv ID + DOI required) ->
AI rerank -> arXiv source/PDF download -> Unpaywall fallback -> text extraction.

Uses shared pool architecture for caching across runs.
Analogous to the PubMed MCP's pubmed_search_with_fulltext tool.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from mcp_server_academic.literature_review import AcademicSource, LiteratureReviewAgent

logger = logging.getLogger(__name__)


async def academic_search_with_fulltext(
    query: str,
    slug: str,
    max_papers: int = 10,
    recency_years: int = 0,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for academic papers and download their full content.

    Uses Semantic Scholar for discovery, AI reranking for relevance,
    and arXiv/Unpaywall for content retrieval.

    Uses shared pool architecture — papers stored in slug/shared/ and symlinked
    to slug/runs/{run_id}/ for per-run isolation.

    Args:
        query: Search query (natural language or keywords).
        slug: snake_case identifier for organizing results (research goal hash).
        max_papers: Target number of papers with fulltext to collect.
        recency_years: Filter to papers from last N years (0 = no filter).
        run_id: Unique run identifier for per-run tracking.

    Returns:
        Dict mapping paper_id to metadata dict including 'fulltext' field.
    """
    lit_review_dir = Path(
        os.getenv("COSCIENTIST_LIT_REVIEW_DIR", "./cache/literature_review")
    )
    lit_review_dir.mkdir(parents=True, exist_ok=True)

    agent = LiteratureReviewAgent(lit_review_dir)
    academic_source = AcademicSource()
    agent.add_source("academic", academic_source)

    logger.info(
        f"Academic fulltext search: query='{query}', slug={slug}, "
        f"max_papers={max_papers}, recency_years={recency_years}, run_id={run_id}"
    )

    results = await agent.fetch_for_query(
        "academic", query, slug, max_papers, recency_years, run_id
    )

    logger.info(f"Academic search complete — {len(results)} papers with fulltext")
    return results
