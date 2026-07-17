"""
Metadata-only academic paper search via Semantic Scholar.

Returns paper metadata (title, abstract, authors, DOI, etc.) without
downloading fulltext content. Analogous to the PubMed MCP's search_pubmed tool.
"""

import json
import logging
from typing import List

from mcp_server_academic.models import Article
from mcp_server_academic.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarRateLimitError,
    extract_paper_metadata,
)

logger = logging.getLogger(__name__)


async def search_academic(query: str, max_papers: int = 10) -> str:
    """
    Search Semantic Scholar for papers and return metadata.

    Args:
        query: Search query (natural language or keywords).
        max_papers: Maximum number of papers to retrieve.

    Returns:
        JSON string with list of articles (for LLM agent consumption).
    """
    logger.info(f"Searching Semantic Scholar: '{query}' (max {max_papers} papers)")

    try:
        client = SemanticScholarClient()
        try:
            raw_papers = await client.search_papers(query, limit=max_papers)
        except SemanticScholarRateLimitError as e:
            logger.warning("%s — falling back to arXiv API", e)
            raw_papers = []

        if not raw_papers:
            logger.warning(
                "No Semantic Scholar results for query: %s — trying arXiv API fallback",
                query[:80],
            )
            from mcp_server_academic.arxiv_downloader import search_arxiv_api

            raw_papers = await search_arxiv_api(query, max_papers=max_papers)

        if not raw_papers:
            logger.warning(f"No results found for query: {query}")
            return json.dumps({"results": [], "count": 0})

        articles: List[Article] = []
        for paper in raw_papers:
            meta = extract_paper_metadata(paper)
            ext_ids = paper.get("externalIds") or {}
            doi = ext_ids.get("DOI", "")
            arxiv_id = ext_ids.get("ArXiv", "")

            url = paper.get("url", "")
            if doi:
                url = f"https://doi.org/{doi}"
            elif arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"

            article = Article(
                title=meta.get("title", "Unknown"),
                url=url,
                authors=meta.get("authors", []),
                year=meta.get("year"),
                venue=meta.get("publication", ""),
                citations=meta.get("citation_count", 0),
                abstract=meta.get("abstract"),
                source_id=paper.get("paperId", ""),
                source="semantic_scholar",
            )
            articles.append(article)

        logger.info(f"Retrieved {len(articles)} papers from Semantic Scholar")

        articles_json = [a.to_dict() for a in articles]
        return json.dumps({"results": articles_json, "count": len(articles)})

    except Exception as e:
        logger.error(f"Error searching Semantic Scholar: {e}")
        return json.dumps({"error": str(e), "results": [], "count": 0})
