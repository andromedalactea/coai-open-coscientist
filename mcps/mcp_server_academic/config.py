import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / '.env'

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env file not found at {env_path} - using system environment only")

LOG_LEVEL = os.environ.get('COSCIENTIST_MCP_LOG_LEVEL') or os.environ.get('LOG_LEVEL', 'INFO')
LOG_LEVEL = LOG_LEVEL.upper()

# Denario docs historically used SEMANTIC_SCHOLAR_KEY; Academic MCP used
# SEMANTIC_SCHOLAR_API_KEY. Accept either so a key set for Denario also unlocks
# the Academic MCP rate limit.
SEMANTIC_SCHOLAR_API_KEY = (
    os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    or os.environ.get("SEMANTIC_SCHOLAR_KEY", "").strip()
)
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "gpt-4.1-mini")
LIT_REVIEW_DIR = os.environ.get("COSCIENTIST_LIT_REVIEW_DIR", "./cache/literature_review")
MCP_PORT = int(os.environ.get("COSCIENTIST_ACADEMIC_MCP_PORT", "8889"))

if not SEMANTIC_SCHOLAR_API_KEY:
    logger.warning(
        "No SEMANTIC_SCHOLAR_API_KEY / SEMANTIC_SCHOLAR_KEY set — Semantic Scholar "
        "will use the unauthenticated quota (~1 req/s) and often returns HTTP 429."
    )
else:
    # Ensure the client (which reads the env directly) sees a canonical name.
    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = SEMANTIC_SCHOLAR_API_KEY
    logger.info("Semantic Scholar API key configured")
