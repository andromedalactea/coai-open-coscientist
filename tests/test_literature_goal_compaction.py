"""Unit tests for literature-search goal compaction and query fallbacks."""

from __future__ import annotations

from open_coscientist.nodes.literature_review_helpers import (
    compact_research_goal_for_literature,
    default_literature_fallback_queries,
)


def test_compact_strips_iau404_transcript():
    prompt = (
        "You are an astrobiologist studying technosignatures.\n"
        "Use POSEIDON and PandExo.\n\n"
        "## IAU Symposium 404 Inspiration Corpus\n\n"
        "### IAU404 transcript chunk 1/4\n\n"
        "---- BEGIN IAU404 TRANSCRIPT CHUNK ----\n\n"
        + ("word " * 50000)
        + "\n---- END IAU404 TRANSCRIPT CHUNK ----\n"
    )
    compact = compact_research_goal_for_literature(prompt)
    assert "BEGIN IAU404" not in compact
    assert "technosignatures" in compact.lower()
    assert len(compact) < 8000
    assert len(compact) < len(prompt) // 10


def test_fallback_queries_never_use_full_goal():
    huge = "x" * 250_000
    queries = default_literature_fallback_queries(huge)
    assert queries
    assert all(len(q) < 200 for q in queries)
    assert all(q != huge for q in queries)


def test_fallback_prefers_domain_tokens():
    goal = "Research bio-technosignature exoplanet atmosphere JWST SETI detection"
    queries = default_literature_fallback_queries(goal)
    joined = " ".join(queries).lower()
    assert "technosignature" in joined or "exoplanet" in joined or "jwst" in joined


if __name__ == "__main__":
    test_compact_strips_iau404_transcript()
    print("PASS: test_compact_strips_iau404_transcript")
    test_fallback_queries_never_use_full_goal()
    print("PASS: test_fallback_queries_never_use_full_goal")
    test_fallback_prefers_domain_tokens()
    print("PASS: test_fallback_prefers_domain_tokens")
    print("All literature goal compaction tests passed.")
