"""
Academic literature review engine.

Orchestrates the full pipeline: Semantic Scholar search -> filter ->
AI rerank -> arXiv/Unpaywall download -> text extraction.

Uses a shared pool architecture identical to the PubMed MCP:
- Papers stored in slug/shared/ (accumulated across runs)
- Per-run view in slug/runs/{run_id}/ (symlinks to shared)
"""

import asyncio
import json
import logging
import os
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp_server_academic import arxiv_downloader, unpaywall
from mcp_server_academic.semantic_scholar import (
    SemanticScholarClient,
    extract_paper_metadata,
    filter_papers_with_arxiv_and_doi,
)
from mcp_server_academic.reranker import rerank_papers
from mcp_server_academic.text_extraction import (
    extract_text_from_latex,
    extract_text_from_pdf,
)
from mcp_server_academic.arxiv_downloader import (
    download_source,
    download_pdf,
    extract_source_files,
    find_main_tex,
    find_bib_content,
    find_bbl_content,
)

logger = logging.getLogger(__name__)


class DocumentSource(ABC):
    """Abstract base for document sources (mirrors mcp_server.literature_review)."""

    data_dir: str
    qualified_path: Optional[Path]

    async def fetch_for_query(
        self, query: str, slug: str = "", max_papers: int = 10, **kwargs
    ):
        ...


class LiteratureReviewAgent:
    """Manages multiple document sources with a shared root directory."""

    def __init__(self, source_root: Path):
        self.source_root = source_root
        self.sources: Dict[str, DocumentSource] = {}
        logger.info(f"Initialized LiteratureReviewAgent with source root: {source_root}")

    def add_source(self, name: str, source: DocumentSource):
        source.qualified_path = self.source_root / source.data_dir
        self.sources[name] = source

    async def fetch_for_query(
        self,
        source_name: str,
        query: str,
        slug: str,
        max_papers: int = 10,
        recency_years: int = 0,
        run_id: Optional[str] = None,
    ):
        return await self.sources[source_name].fetch_for_query(
            query, slug, max_papers, recency_years=recency_years, run_id=run_id
        )


class AcademicSource(DocumentSource):
    """
    Semantic Scholar + arXiv + Unpaywall document source.

    Pipeline per query:
    1. Search Semantic Scholar for candidate papers
    2. Filter: require both arXiv ID and DOI
    3. AI rerank by relevance to query
    4. Download content (arXiv source -> arXiv PDF -> Unpaywall PDF)
    5. Extract text (LaTeX preferred, PDF fallback)
    6. Cache in shared pool
    """

    def __init__(self, qualified_path: Optional[Path] = None):
        self.data_dir = "academic"
        self.qualified_path = qualified_path
        self._s2_client = SemanticScholarClient()

    def _assert_qualified_path(self) -> Path:
        if self.qualified_path is None:
            raise ValueError("qualified_path must be set")
        return self.qualified_path

    async def fetch_for_query(
        self,
        query: str,
        slug: str,
        max_papers: int = 10,
        recency_years: int = 0,
        run_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run the full search-filter-rerank-download pipeline."""

        base_dir = self._assert_qualified_path() / slug
        shared_dir = base_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)

        run_dir = None
        if run_id:
            run_dir = base_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using shared pool with per-run tracking: run_id={run_id}")

        # --- 1. Search Semantic Scholar ---
        year_range = None
        if recency_years > 0:
            from datetime import datetime
            current_year = datetime.now().year
            min_year = current_year - recency_years
            # Use closed range "2019-2025" - open-ended "2019-" can cause 500 on relevance search
            year_range = f"{min_year}-{current_year}"

        search_limit = max(max_papers * 10, 100)
        raw_papers = await self._s2_client.search_papers(
            query, limit=search_limit, year_range=year_range
        )

        if not raw_papers:
            logger.warning(f"No results from Semantic Scholar for: {query}")
            return self._supplement_from_pool(
                {}, shared_dir, run_dir, max_papers, set()
            )

        # We removed strict arXiv/DOI filtering to get more candidates
        # --- 3. AI Rerank ---
        reranked = await rerank_papers(query, raw_papers, max_papers)

        # --- 4 & 5. Download + Extract ---
        results: Dict[str, Dict[str, Any]] = {}
        current_run_papers: List[str] = []

        semaphore = asyncio.Semaphore(3)

        async def process_paper(paper: Dict[str, Any]) -> Optional[str]:
            """Download and extract content for a single paper. Returns paper_id on success."""
            ext_ids = paper.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv", "")
            doi = ext_ids.get("DOI", "")
            paper_id = paper.get("paperId", arxiv_id)

            metadata = extract_paper_metadata(paper)

            # Check shared pool cache
            metadata_file = shared_dir / f"{paper_id}.metadata.json"
            content_file = shared_dir / f"{paper_id}.content.md"

            if content_file.exists() and metadata_file.exists():
                logger.info(f"Paper {paper_id} found in shared pool, reusing")
                cached_meta = json.loads(metadata_file.read_text())
                cached_meta["fulltext"] = content_file.read_text()
                results[paper_id] = cached_meta
                _create_symlinks(shared_dir, run_dir, paper_id)
                current_run_papers.append(paper_id)
                return paper_id

            # Try downloading content
            async with semaphore:
                fulltext = await self._download_and_extract(arxiv_id, doi, shared_dir, paper_id)

            if not fulltext:
                logger.debug(f"Could not obtain content for {paper_id}, skipping")
                return None

            # Save to shared pool
            metadata_file.write_text(json.dumps(metadata, indent=2))
            content_file.write_text(fulltext)
            logger.info(f"Saved paper {paper_id} to shared pool ({len(fulltext)} chars)")

            metadata["fulltext"] = fulltext
            results[paper_id] = metadata
            _create_symlinks(shared_dir, run_dir, paper_id)
            current_run_papers.append(paper_id)
            return paper_id

        # Process papers in ranked order, stop when we have enough
        success_count = 0
        for paper in reranked:
            if success_count >= max_papers:
                break
            pid = await process_paper(paper)
            if pid:
                success_count += 1

        logger.info(
            f"Downloaded {success_count}/{max_papers} papers from search results"
        )

        # Supplement from pool if short
        if success_count < max_papers and run_dir:
            results = self._supplement_from_pool(
                results, shared_dir, run_dir, max_papers, set(results.keys())
            )

        # Save manifest
        if run_id and run_dir:
            manifest = {
                "run_id": run_id,
                "paper_ids": list(results.keys()),
                "query": query,
            }
            (run_dir / ".manifest.json").write_text(json.dumps(manifest, indent=2))
            logger.info(f"Saved manifest for run {run_id}: {len(results)} papers")

        logger.info(f"Returning {len(results)} papers with fulltext (target was {max_papers})")
        return results

    async def _download_and_extract(
        self,
        arxiv_id: str,
        doi: str,
        shared_dir: Path,
        paper_id: str,
    ) -> Optional[str]:
        """
        Try to download and extract paper content.

        Order: arXiv source (LaTeX) -> arXiv PDF -> Unpaywall PDF.
        """
        # --- Try arXiv source (tar.gz with .tex/.bib/.bbl) ---
        if arxiv_id:
            source_bytes = await download_source(arxiv_id)
            if source_bytes:
                extracted_files = extract_source_files(source_bytes)
                if extracted_files:
                    tex = find_main_tex(extracted_files)
                    if tex:
                        bib = find_bib_content(extracted_files)
                        bbl = find_bbl_content(extracted_files)
                        fulltext = extract_text_from_latex(tex, bib, bbl)
                        if fulltext and len(fulltext.strip()) > 100:
                            # Save source files for reference
                            src_dir = shared_dir / f"{paper_id}_source"
                            src_dir.mkdir(exist_ok=True)
                            for fname, content in extracted_files.items():
                                (src_dir / Path(fname).name).write_text(content)
                            logger.info(
                                f"Extracted LaTeX content for {paper_id} "
                                f"({len(fulltext)} chars)"
                            )
                            return fulltext

            # --- Fallback: arXiv PDF ---
            pdf_bytes = await download_pdf(arxiv_id)
            if pdf_bytes:
                fulltext = extract_text_from_pdf(pdf_bytes)
                if fulltext and len(fulltext.strip()) > 100:
                    (shared_dir / f"{paper_id}.pdf").write_bytes(pdf_bytes)
                    logger.info(
                        f"Extracted arXiv PDF content for {paper_id} "
                        f"({len(fulltext)} chars)"
                    )
                    return fulltext

        # --- Fallback: Unpaywall PDF ---
        if doi:
            pdf_url = await unpaywall.get_pdf_url(doi)
            if pdf_url:
                pdf_bytes = await unpaywall.download_pdf(pdf_url)
                if pdf_bytes:
                    fulltext = extract_text_from_pdf(pdf_bytes)
                    if fulltext and len(fulltext.strip()) > 100:
                        (shared_dir / f"{paper_id}.pdf").write_bytes(pdf_bytes)
                        logger.info(
                            f"Extracted Unpaywall PDF content for {paper_id} "
                            f"({len(fulltext)} chars)"
                        )
                        return fulltext

        return None

    def _supplement_from_pool(
        self,
        current_results: Dict[str, Dict[str, Any]],
        shared_dir: Path,
        run_dir: Optional[Path],
        max_papers: int,
        exclude_ids: set,
    ) -> Dict[str, Dict[str, Any]]:
        """Fill up to max_papers from the shared pool if we're short."""
        shortfall = max_papers - len(current_results)
        if shortfall <= 0 or not run_dir:
            return current_results

        logger.info(f"Attempting to supplement {shortfall} papers from shared pool")

        candidates = []
        for meta_file in shared_dir.glob("*.metadata.json"):
            paper_id = meta_file.stem.replace(".metadata", "")
            if paper_id in exclude_ids:
                continue
            content_file = shared_dir / f"{paper_id}.content.md"
            if content_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    candidates.append((paper_id, meta))
                except Exception:
                    continue

        candidates.sort(
            key=lambda x: x[1].get("year") or 0, reverse=True
        )

        supplemented = 0
        for paper_id, meta in candidates[:shortfall]:
            content_file = shared_dir / f"{paper_id}.content.md"
            meta["fulltext"] = content_file.read_text()
            current_results[paper_id] = meta
            _create_symlinks(shared_dir, run_dir, paper_id)
            supplemented += 1

        if supplemented:
            logger.info(
                f"Supplemented {supplemented} papers from shared pool "
                f"(total: {len(current_results)}/{max_papers})"
            )
        else:
            logger.warning("No suitable papers found in shared pool for supplementation")

        return current_results


def _create_symlinks(shared_dir: Path, run_dir: Optional[Path], paper_id: str):
    """Create symlinks from run directory to shared pool files."""
    if not run_dir:
        return

    for suffix in [".metadata.json", ".content.md", ".pdf"]:
        shared_file = shared_dir / f"{paper_id}{suffix}"
        if shared_file.exists():
            link = run_dir / f"{paper_id}{suffix}"
            if not link.exists():
                try:
                    link.symlink_to(f"../../shared/{paper_id}{suffix}")
                except OSError as e:
                    logger.debug(f"Failed to create symlink for {paper_id}{suffix}: {e}")
