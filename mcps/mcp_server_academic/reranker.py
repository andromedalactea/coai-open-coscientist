"""
AI-based paper reranker using LLM to score relevance to a research query.

Given a search query (research context) and candidate paper metadata,
uses an LLM to rank papers by relevance and return the top N + buffer.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import litellm

logger = logging.getLogger(__name__)

RERANKER_SYSTEM_PROMPT = """\
You are an expert academic research assistant specializing in evaluating \
the relevance of scientific papers to a given research query.

You will receive a research query and a list of candidate papers with their \
metadata (title, abstract, year, citation count). Your task is to rank these \
papers by their relevance and potential contribution to the research query.

Evaluation criteria (in order of importance):
1. **Relevance**: How directly does the paper address the research query?
2. **Methodological contribution**: Does the paper introduce methods, \
frameworks, or data that could advance the research?
3. **Recency**: More recent papers are generally preferred, but seminal \
older works should still rank highly.
4. **Impact**: Citation count as a rough proxy for community recognition.

Return a JSON object with a single key "ranked_papers" containing an array \
of objects. Each object must have:
- "index": the 0-based index of the paper from the input list
- "score": a relevance score from 0.0 to 1.0
- "reason": a brief (1 sentence) justification

Sort the array by score descending. Include ALL papers from the input list \
in your ranking."""

RERANKER_USER_TEMPLATE = """\
Research Query:
{query}

Candidate Papers ({count} total):
{papers_text}

Rank ALL {count} papers by relevance to the research query. \
Return JSON with "ranked_papers" array sorted by score descending."""

BUFFER_RATIO = 1.5


async def rerank_papers(
    query: str,
    papers: List[Dict[str, Any]],
    max_papers: int,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank papers using an LLM and return the top candidates.

    Args:
        query: The search/research query for context.
        papers: List of paper metadata dicts (must have title, abstract).
        max_papers: Desired number of final papers.
        model: LLM model name (defaults to RERANKER_MODEL env var).

    Returns:
        Ordered list of paper dicts (best first), sized to
        ceil(max_papers * BUFFER_RATIO) to provide download-failure buffer.
    """
    if not papers:
        return []

    model = model or os.environ.get("RERANKER_MODEL", os.environ.get("DEFAULT_MODEL", "gpt-4.1-mini"))
    buffer_count = math.ceil(max_papers * BUFFER_RATIO)
    target = min(buffer_count, len(papers))

    if len(papers) <= target:
        logger.info(
            f"Reranker: only {len(papers)} candidates <= target {target}, "
            f"returning all without LLM call"
        )
        return papers

    papers_text = _format_papers_for_prompt(papers)

    user_message = RERANKER_USER_TEMPLATE.format(
        query=query,
        count=len(papers),
        papers_text=papers_text,
    )

    logger.info(
        f"Reranking {len(papers)} papers with {model} "
        f"(target: {max_papers}, buffer: {target})"
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4096,
        )

        content = response.choices[0].message.content
        ranking = json.loads(content)
        ranked_list = ranking.get("ranked_papers", [])

        reranked = []
        seen_indices = set()
        for entry in ranked_list:
            idx = entry.get("index")
            if idx is not None and 0 <= idx < len(papers) and idx not in seen_indices:
                seen_indices.add(idx)
                paper = papers[idx].copy()
                paper["_rerank_score"] = entry.get("score", 0.0)
                paper["_rerank_reason"] = entry.get("reason", "")
                reranked.append(paper)

        if len(reranked) < len(papers):
            for i, p in enumerate(papers):
                if i not in seen_indices:
                    reranked.append(p)

        result = reranked[:target]
        logger.info(
            f"Reranker returned {len(result)} papers "
            f"(top score: {result[0].get('_rerank_score', '?') if result else 'N/A'})"
        )
        return result

    except Exception as e:
        logger.error(f"Reranker LLM call failed: {e}. Falling back to citation-count order.")
        fallback = sorted(papers, key=lambda p: p.get("citationCount", 0), reverse=True)
        return fallback[:target]


def _format_papers_for_prompt(papers: List[Dict[str, Any]]) -> str:
    """Format paper list as numbered text for the LLM prompt."""
    lines = []
    for i, p in enumerate(papers):
        title = p.get("title", "Untitled")
        abstract = p.get("abstract") or "No abstract available."
        if len(abstract) > 400:
            abstract = abstract[:400] + "..."
        year = p.get("year", "?")
        citations = p.get("citationCount", 0)
        lines.append(
            f"[{i}] Title: {title}\n"
            f"    Year: {year} | Citations: {citations}\n"
            f"    Abstract: {abstract}\n"
        )
    return "\n".join(lines)
