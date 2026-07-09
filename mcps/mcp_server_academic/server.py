"""
Open Coscientist academic literature review MCP server.

Provides Semantic Scholar + arXiv + Unpaywall literature search tools
via the MCP protocol, using the same FastMCP + FastAPI architecture
as the PubMed MCP server.
"""

import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastmcp import FastMCP

import fastmcp
fastmcp.settings.stateless_http = True

from mcp_server_academic import config

log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('mcp_server_academic').setLevel(log_level)

from mcp_server_academic.tools.academic_search import (
    check_academic_available,
    search_academic,
    academic_search_with_fulltext,
)

logger = logging.getLogger(__name__)

s2_key_present = bool(config.SEMANTIC_SCHOLAR_API_KEY)
unpaywall_email_present = bool(config.UNPAYWALL_EMAIL)

logger.info("Academic MCP server starting")
logger.debug(
    f"Config: S2_API_KEY={'yes' if s2_key_present else 'no'}, "
    f"UNPAYWALL_EMAIL={'yes' if unpaywall_email_present else 'no'}, "
    f"RERANKER_MODEL={config.RERANKER_MODEL}"
)

mcp = FastMCP("open-coscientist-academic-lit-review")

mcp.tool(check_academic_available, name="check_academic_available")
mcp.tool(search_academic, name="search_academic")
mcp.tool(academic_search_with_fulltext, name="academic_search_with_fulltext")

mcp_http_app = mcp.http_app()
app = FastAPI(lifespan=mcp_http_app.lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API status endpoint."""
    return JSONResponse({
        "status": "running",
        "service": "coscientist-academic-lit-review",
        "version": "0.1.0",
        "mcp_tools": [
            "check_academic_available",
            "search_academic",
            "academic_search_with_fulltext",
        ],
        "api_keys_configured": {
            "SEMANTIC_SCHOLAR_API_KEY": s2_key_present,
            "UNPAYWALL_EMAIL": unpaywall_email_present,
        },
        "reranker_model": config.RERANKER_MODEL,
    })


app.mount("/", mcp_http_app)

if __name__ == "__main__":
    port = config.MCP_PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
