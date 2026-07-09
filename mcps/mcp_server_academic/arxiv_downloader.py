"""
arXiv content downloader.

Downloads paper source archives (tar.gz with .tex/.bib/.bbl) and PDFs from arXiv.
Respects arXiv's rate limiting guidelines (>= 3s between requests).
"""

import asyncio
import gzip
import io
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
REQUEST_DELAY = 3.0

_last_request_time = 0.0
_lock = asyncio.Lock()


async def _rate_limit():
    global _last_request_time
    async with _lock:
        now = asyncio.get_event_loop().time()
        elapsed = now - _last_request_time
        if elapsed < REQUEST_DELAY:
            await asyncio.sleep(REQUEST_DELAY - elapsed)
        _last_request_time = asyncio.get_event_loop().time()


async def download_source(arxiv_id: str) -> Optional[bytes]:
    """
    Download the source archive for an arXiv paper.

    Returns raw bytes of the archive (tar.gz, gz, or raw tex),
    or None on failure.
    """
    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
    logger.info(f"Downloading arXiv source: {url}")

    await _rate_limit()
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(
                url, headers={"User-Agent": "OpenCoscientist-Academic-MCP/0.1"}
            )
            if response.status_code == 200:
                logger.info(
                    f"Downloaded arXiv source for {arxiv_id} "
                    f"({len(response.content)} bytes, "
                    f"type={response.headers.get('content-type', '?')})"
                )
                return response.content
            else:
                logger.warning(
                    f"arXiv source download failed for {arxiv_id}: "
                    f"HTTP {response.status_code}"
                )
                return None
    except Exception as e:
        logger.error(f"arXiv source download error for {arxiv_id}: {e}")
        return None


async def download_pdf(arxiv_id: str) -> Optional[bytes]:
    """Download the PDF for an arXiv paper. Returns raw PDF bytes or None."""
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    logger.info(f"Downloading arXiv PDF: {url}")

    await _rate_limit()
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(
                url, headers={"User-Agent": "OpenCoscientist-Academic-MCP/0.1"}
            )
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "pdf" in content_type or len(response.content) > 1000:
                    logger.info(
                        f"Downloaded arXiv PDF for {arxiv_id} "
                        f"({len(response.content)} bytes)"
                    )
                    return response.content
            logger.warning(
                f"arXiv PDF download failed for {arxiv_id}: HTTP {response.status_code}"
            )
            return None
    except Exception as e:
        logger.error(f"arXiv PDF download error for {arxiv_id}: {e}")
        return None


def extract_source_files(
    archive_bytes: bytes,
) -> Dict[str, str]:
    """
    Extract .tex, .bib, and .bbl files from an arXiv source archive.

    Handles tar.gz, gzipped single files, and raw tex.

    Returns:
        Dict mapping filename -> content string for relevant files.
    """
    extracted: Dict[str, str] = {}

    # Try tar.gz first
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                lower = member.name.lower()
                if lower.endswith((".tex", ".bib", ".bbl")):
                    f = tar.extractfile(member)
                    if f:
                        try:
                            content = f.read().decode("utf-8", errors="replace")
                            extracted[member.name] = content
                        except Exception as e:
                            logger.debug(f"Failed to read {member.name}: {e}")
            if extracted:
                logger.info(
                    f"Extracted {len(extracted)} files from tar.gz: "
                    f"{list(extracted.keys())}"
                )
                return extracted
    except (tarfile.TarError, Exception):
        pass

    # Try plain gzip (single file)
    try:
        decompressed = gzip.decompress(archive_bytes)
        text = decompressed.decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            extracted["main.tex"] = text
            logger.info("Extracted single gzipped .tex file")
            return extracted
    except Exception:
        pass

    # Try raw tex
    try:
        text = archive_bytes.decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            extracted["main.tex"] = text
            logger.info("Extracted raw .tex content")
            return extracted
    except Exception:
        pass

    logger.warning("Could not extract any .tex/.bib/.bbl files from archive")
    return extracted


def save_source_files(
    extracted: Dict[str, str], dest_dir: Path
) -> List[Path]:
    """Save extracted source files to a directory. Returns list of paths written."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in extracted.items():
        safe_name = Path(name).name
        path = dest_dir / safe_name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def find_main_tex(extracted: Dict[str, str]) -> Optional[str]:
    """
    Find the main .tex file content from extracted source files.

    Heuristic: the file containing \\documentclass or \\begin{document}.
    Falls back to the largest .tex file.
    """
    tex_files = {k: v for k, v in extracted.items() if k.lower().endswith(".tex")}
    if not tex_files:
        return None

    for name, content in tex_files.items():
        if "\\documentclass" in content or "\\begin{document}" in content:
            return content

    return max(tex_files.values(), key=len)


def find_bib_content(extracted: Dict[str, str]) -> Optional[str]:
    """Get combined .bib file content if available."""
    bib_files = {k: v for k, v in extracted.items() if k.lower().endswith(".bib")}
    if not bib_files:
        return None
    return "\n\n".join(bib_files.values())


def find_bbl_content(extracted: Dict[str, str]) -> Optional[str]:
    """Get combined .bbl file content if available."""
    bbl_files = {k: v for k, v in extracted.items() if k.lower().endswith(".bbl")}
    if not bbl_files:
        return None
    return "\n\n".join(bbl_files.values())
