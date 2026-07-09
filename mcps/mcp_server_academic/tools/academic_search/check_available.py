"""
Availability check tool for the academic literature search pipeline.

Tests connectivity to Semantic Scholar API.
"""

import logging

from mcp_server_academic.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


async def check_academic_available() -> str:
    """
    Check if the academic search pipeline is available.

    Tests Semantic Scholar API connectivity with a minimal query.

    Returns:
        "true" if the API is reachable, "false" otherwise.
    """
    try:
        client = SemanticScholarClient()
        available = await client.check_available()
        if available:
            logger.info("Academic search pipeline is available")
            return "true"
        else:
            logger.warning("Semantic Scholar API check returned no results")
            return "false"
    except Exception as e:
        logger.error(f"Academic search availability check failed: {e}")
        return "false"
