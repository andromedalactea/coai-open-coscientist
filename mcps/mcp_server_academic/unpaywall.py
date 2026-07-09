"""
Unpaywall API client for open-access PDF discovery.

Uses the Unpaywall REST API to find freely accessible PDF versions of papers
identified by DOI. Serves as a fallback when arXiv source/PDF is unavailable.
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"


async def get_pdf_url(doi: str, email: Optional[str] = None) -> Optional[str]:
    """
    Look up the open-access PDF URL for a DOI via Unpaywall.

    Args:
        doi: The DOI of the paper.
        email: Contact email (required by Unpaywall TOS).

    Returns:
        Direct PDF URL string, or None if not available.
    """
    email = email or os.environ.get("UNPAYWALL_EMAIL", "")
    if not email:
        logger.warning("UNPAYWALL_EMAIL not set — Unpaywall requires an email")
        return None

    url = f"{UNPAYWALL_BASE_URL}/{doi}"
    params = {"email": email}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                logger.debug(f"Unpaywall: no record for DOI {doi}")
                return None
            response.raise_for_status()
            data = response.json()

        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")

        if not pdf_url:
            for location in data.get("oa_locations", []):
                pdf_url = location.get("url_for_pdf")
                if pdf_url:
                    break

        if pdf_url:
            logger.info(f"Unpaywall found PDF for DOI {doi}: {pdf_url[:80]}")
        else:
            logger.debug(f"Unpaywall: no PDF URL for DOI {doi}")

        return pdf_url

    except httpx.HTTPStatusError as e:
        logger.warning(f"Unpaywall HTTP error for DOI {doi}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unpaywall error for DOI {doi}: {e}")
        return None


async def download_pdf(pdf_url: str) -> Optional[bytes]:
    """
    Download a PDF from a URL.

    Args:
        pdf_url: Direct URL to the PDF file.

    Returns:
        Raw PDF bytes, or None on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(
                pdf_url,
                headers={"User-Agent": "OpenCoscientist-Academic-MCP/0.1"},
            )
            if response.status_code == 200 and len(response.content) > 500:
                logger.info(
                    f"Downloaded PDF ({len(response.content)} bytes) from {pdf_url[:80]}"
                )
                return response.content
            logger.warning(
                f"PDF download returned status {response.status_code} "
                f"or too small ({len(response.content)} bytes) from {pdf_url[:80]}"
            )
            return None
    except Exception as e:
        logger.error(f"PDF download failed from {pdf_url[:80]}: {e}")
        return None
