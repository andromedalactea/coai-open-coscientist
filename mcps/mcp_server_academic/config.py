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

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "gpt-4.1-mini")
LIT_REVIEW_DIR = os.environ.get("COSCIENTIST_LIT_REVIEW_DIR", "./cache/literature_review")
MCP_PORT = int(os.environ.get("COSCIENTIST_ACADEMIC_MCP_PORT", "8889"))
