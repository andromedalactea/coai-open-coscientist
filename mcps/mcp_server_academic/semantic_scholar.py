"""
Semantic Scholar Graph API client using httpx.

Direct REST API access for robust JSON-based data extraction.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"

SEARCH_FIELDS = ",".join([
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "citationCount",
    "publicationDate",
    "journal",
    "openAccessPdf",
])

DETAIL_FIELDS = SEARCH_FIELDS


class SemanticScholarClient:
    """Async client for the Semantic Scholar Graph API."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        self._min_delay = 0.15 if self._api_key else 1.1
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "OpenCoscientist-Academic-MCP/0.1"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    async def _rate_limit(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_delay:
                await asyncio.sleep(self._min_delay - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

    async def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        await self._rate_limit()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=self._headers())
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "5"))
                logger.warning(f"Rate limited by Semantic Scholar, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self._get(url, params)
            response.raise_for_status()
            return response.json()

    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[str] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers by keyword query.

        Args:
            query: Natural language or keyword search query.
            limit: Max results (API max per call is 100).
            year_range: Year filter like "2020-" or "2018-2024".
            offset: Pagination offset.

        Returns:
            List of paper dicts with all requested fields.
        """
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": SEARCH_FIELDS,
        }
        if year_range:
            params["year"] = year_range

        logger.info(f"Searching Semantic Scholar: '{query}' (limit={limit}, year={year_range})")

        all_papers: List[Dict[str, Any]] = []
        remaining = limit
        url = f"{BASE_URL}/paper/search"

        while remaining > 0:
            params["limit"] = min(remaining, 100)
            params["offset"] = offset

            try:
                data = await self._get(url, params)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 500 and year_range and "year" in params:
                    logger.warning(
                        f"Semantic Scholar returned 500 with year filter, retrying without year"
                    )
                    del params["year"]
                    data = await self._get(url, params)
                else:
                    raise

            papers = data.get("data", [])
            if not papers:
                break

            all_papers.extend(papers)
            remaining -= len(papers)
            offset += len(papers)

            total = data.get("total", 0)
            if offset >= total:
                break

        logger.info(f"Semantic Scholar returned {len(all_papers)} papers for '{query}'")
        return all_papers

    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a single paper by Semantic Scholar ID or external ID."""
        try:
            data = await self._get(
                f"{BASE_URL}/paper/{paper_id}",
                {"fields": DETAIL_FIELDS},
            )
            return data
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to fetch paper {paper_id}: {e}")
            return None

    async def check_available(self) -> bool:
        """Test API connectivity with a minimal query."""
        try:
            data = await self._get(
                f"{BASE_URL}/paper/search",
                {"query": "test", "limit": 1, "fields": "paperId"},
            )
            return bool(data.get("data"))
        except Exception as e:
            logger.error(f"Semantic Scholar availability check failed: {e}")
            return False


def filter_papers_with_arxiv_and_doi(
    papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep only papers that have EITHER an arXiv ID OR a DOI in externalIds.
    
    Papers without either identifier cannot go through the arXiv download +
    Unpaywall fallback pipeline, so they are discarded.
    """
    filtered = []
    for paper in papers:
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        doi = ext_ids.get("DOI")
        if arxiv_id or doi:
            filtered.append(paper)
        else:
            logger.debug(
                f"Discarding paper '{paper.get('title', '?')[:60]}' "
                f"(ArXiv={'yes' if arxiv_id else 'no'}, DOI={'yes' if doi else 'no'})"
            )
    logger.info(
        f"Filtered {len(papers)} -> {len(filtered)} papers (require ArXiv ID OR DOI)"
    )
    return filtered


def extract_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Semantic Scholar API paper dict to our internal metadata format.

    Mirrors the structure returned by the PubMed MCP so the main app can
    consume it via the same YAML field mappings.
    """
    ext_ids = paper.get("externalIds") or {}
    authors_raw = paper.get("authors") or []
    author_names = [a.get("name", "") for a in authors_raw if a.get("name")]

    journal = paper.get("journal") or {}
    venue = journal.get("name") or paper.get("venue") or ""

    pub_date = paper.get("publicationDate") or ""
    year = paper.get("year")
    date_revised = ""
    if pub_date:
        date_revised = pub_date.replace("-", "/")
    elif year:
        date_revised = f"{year}/01/01"

    oa_pdf = paper.get("openAccessPdf") or {}
    pdf_url = oa_pdf.get("url", "")

    doi = ext_ids.get("DOI", "")
    arxiv_id = ext_ids.get("ArXiv", "")

    return {
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "authors": author_names,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "date_revised": date_revised,
        "publication": venue,
        "year": year,
        "citation_count": paper.get("citationCount", 0),
        "semantic_scholar_id": paper.get("paperId", ""),
        "open_access_pdf_url": pdf_url,
        "external_ids": ext_ids,
    }
