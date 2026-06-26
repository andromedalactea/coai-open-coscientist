"""
Per-run output management.

Creates a folder per run with:
- run_info.json: run metadata (research goal, config, timestamps)
- research_trace.log: detailed trace of the entire research process
- results/summary.json: all hypotheses with full data
- results/hypothesis_N.md: individual hypothesis markdown files
- literature/references.json: all referenced papers with metadata
- literature/synthesis.md: literature review synthesis text
- literature/papers/: paper content files (when available)
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = "./coscientist_runs"


def _get_runs_dir() -> Path:
    return Path(os.getenv("COSCIENTIST_RUNS_DIR", DEFAULT_RUNS_DIR))


def _sanitize_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_')
    return name[:max_len] if name else "unnamed"


def _authors_to_str(authors: Any, max_count: int = 5) -> str:
    """Convert authors (list of str or dict) to a comma-separated string."""
    if not authors:
        return ""
    if not isinstance(authors, list):
        return str(authors)
    strs = []
    for a in authors[:max_count]:
        if isinstance(a, str):
            strs.append(a)
        elif isinstance(a, dict):
            strs.append(a.get("name", "") or a.get("author", "") or str(a))
        else:
            strs.append(str(a))
    result = ", ".join(s for s in strs if s)
    if len(authors) > max_count:
        result += " et al."
    return result


class RunFolder:
    """Manages per-run output folder and file creation."""

    def __init__(self, run_id: str, research_goal: str, config: Dict[str, Any]):
        self.run_id = run_id
        self.research_goal = research_goal
        self.config = config
        self.start_time = datetime.now(timezone.utc)

        goal_slug = _sanitize_filename(research_goal[:40])
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{ts}_{goal_slug}_{run_id[:8]}"

        self.run_dir = _get_runs_dir() / folder_name
        self.results_dir = self.run_dir / "results"
        self.literature_dir = self.run_dir / "literature"
        self.papers_dir = self.literature_dir / "papers"

        self._create_dirs()
        self._write_run_info()

    @classmethod
    def from_existing(cls, run_dir: str) -> "RunFolder":
        """Reconstruct a RunFolder from an existing directory path."""
        obj = cls.__new__(cls)
        obj.run_dir = Path(run_dir)
        obj.results_dir = obj.run_dir / "results"
        obj.literature_dir = obj.run_dir / "literature"
        obj.papers_dir = obj.literature_dir / "papers"
        obj.start_time = datetime.now(timezone.utc)
        obj.research_goal = ""
        obj.config = {}

        info_path = obj.run_dir / "run_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            obj.run_id = info.get("run_id", "")
            obj.research_goal = info.get("research_goal", "")
            obj.config = info.get("config", {})
            if info.get("start_time"):
                try:
                    obj.start_time = datetime.fromisoformat(info["start_time"])
                except (ValueError, TypeError):
                    pass
        else:
            obj.run_id = obj.run_dir.name.split("_")[-1] if "_" in obj.run_dir.name else ""

        for d in [obj.results_dir, obj.literature_dir, obj.papers_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return obj

    def _create_dirs(self):
        for d in [self.run_dir, self.results_dir, self.literature_dir, self.papers_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _write_run_info(self):
        info = {
            "run_id": self.run_id,
            "research_goal": self.research_goal,
            "start_time": self.start_time.isoformat(),
            "config": self.config,
            "status": "running",
        }
        self._write_json(self.run_dir / "run_info.json", info)

    def get_trace_path(self) -> Path:
        return self.run_dir / "research_trace.log"

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def save_results(self, state: Dict[str, Any], execution_time: float):
        """Save final results: summary JSON + per-hypothesis markdown files."""
        hypotheses = state.get("hypotheses", [])
        end_time = datetime.now(timezone.utc)

        summary = {
            "run_id": self.run_id,
            "research_goal": self.research_goal,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "hypothesis_count": len(hypotheses),
            "metrics": state.get("metrics", {}),
            "meta_review": state.get("meta_review", {}),
            "research_plan": state.get("research_plan", {}),
            "hypotheses": hypotheses,
        }
        self._write_json(self.results_dir / "summary.json", summary)

        for i, hyp in enumerate(hypotheses, 1):
            try:
                self._write_hypothesis_md(i, hyp)
            except Exception as e:
                logger.warning(f"Failed to write hypothesis_{i:02d}.md: {e}", exc_info=True)

        self._update_run_status("completed", execution_time)
        logger.info(f"Results saved to {self.results_dir} ({len(hypotheses)} hypotheses)")

    def regenerate_hypothesis_files(self) -> int:
        """
        Regenerate hypothesis_*.md files from existing summary.json.
        Use after fixing a save bug to recover missing hypothesis files.
        Returns the number of files written.
        """
        summary_path = self.results_dir / "summary.json"
        if not summary_path.exists():
            logger.warning("No summary.json found")
            return 0
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        hypotheses = data.get("hypotheses", [])
        count = 0
        for i, hyp in enumerate(hypotheses, 1):
            try:
                self._write_hypothesis_md(i, hyp)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to write hypothesis_{i:02d}.md: {e}")
        return count

    def _write_hypothesis_md(self, index: int, hyp: Dict[str, Any]):
        """Write a single hypothesis as a markdown file."""
        score = hyp.get("score", 0)
        elo = hyp.get("elo_rating", 0)
        method = hyp.get("generation_method", "unknown")

        def _str(v: Any, default: str = "") -> str:
            return v if isinstance(v, str) else (str(v) if v is not None else default)

        lines = [
            f"# Hypothesis {index}",
            "",
            f"**Score:** {score:.1f} | **Elo:** {elo} | **Method:** {method}",
            f"**Win rate:** {hyp.get('win_rate', 0):.0f}% "
            f"({hyp.get('win_count', 0)}W/{hyp.get('loss_count', 0)}L)",
            "",
            "## Hypothesis",
            "",
            _str(hyp.get("text")),
            "",
        ]

        if hyp.get("explanation"):
            lines += ["## Explanation", "", _str(hyp["explanation"]), ""]

        if hyp.get("literature_grounding"):
            lines += ["## Literature Grounding", "", _str(hyp["literature_grounding"]), ""]

        if hyp.get("experiment"):
            lines += ["## Experiment Design", "", _str(hyp["experiment"]), ""]

        if hyp.get("reflection_notes"):
            lines += ["## Reflection Notes", "", _str(hyp["reflection_notes"]), ""]

        if hyp.get("novelty_validation"):
            lines += ["## Novelty Validation", "", _str(hyp["novelty_validation"]), ""]

        if hyp.get("reviews"):
            lines += ["## Reviews", ""]
            for ri, rev in enumerate(hyp["reviews"], 1):
                lines.append(f"### Review {ri} (Score: {rev.get('overall_score', 'N/A')})")
                lines.append("")
                summary = rev.get("review_summary", "")
                lines.append(_str(summary))
                fb = rev.get("constructive_feedback", "")
                if fb:
                    lines += ["", f"**Feedback:** {_str(fb)}"]
                lines.append("")

        if hyp.get("citation_map"):
            lines += ["## References", ""]
            for key, ref in hyp["citation_map"].items():
                ref_type = ref.get("type", "paper")
                title = _str(ref.get("title") or ref.get("display"), "Unknown")
                url = _str(ref.get("url"))
                year = _str(ref.get("year"))
                author_str = _authors_to_str(ref.get("authors", []), max_count=3)

                if ref_type == "paper":
                    lines.append(f"- **{key}**: {author_str} ({year}). *{title}*. {url}")
                else:
                    lines.append(f"- **{key}** [{ref_type}]: {title}")
                lines.append("")

        path = self.results_dir / f"hypothesis_{index:02d}.md"
        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Literature / Papers
    # ------------------------------------------------------------------

    def save_literature(self, state: Dict[str, Any]):
        """Save literature synthesis, article metadata, and paper content."""
        synthesis = state.get("articles_with_reasoning")
        if synthesis:
            (self.literature_dir / "synthesis.md").write_text(
                f"# Literature Review Synthesis\n\n{synthesis}", encoding="utf-8"
            )

        raw_refs = self._collect_all_references(state)

        clean_refs = [{k: v for k, v in ref.items() if k != "_content"} for ref in raw_refs]
        self._write_json(self.literature_dir / "references.json", clean_refs)

        for ref in raw_refs:
            self._save_paper_content(ref)

        logger.info(
            f"Literature saved: {len(raw_refs)} references, "
            f"synthesis={'yes' if synthesis else 'no'}"
        )

    def _collect_all_references(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect unique paper references from articles + hypothesis citation_maps."""
        refs_by_id: Dict[str, Dict[str, Any]] = {}

        for art in state.get("articles", []):
            a = art if isinstance(art, dict) else art.to_dict() if hasattr(art, 'to_dict') else {}
            ref_id = a.get("source_id") or a.get("title", "")
            if ref_id and ref_id not in refs_by_id:
                refs_by_id[ref_id] = {
                    "id": ref_id,
                    "title": a.get("title", ""),
                    "authors": a.get("authors", []),
                    "year": a.get("year"),
                    "url": a.get("url"),
                    "abstract": a.get("abstract"),
                    "source": a.get("source", ""),
                    "venue": a.get("venue"),
                    "pdf_links": a.get("pdf_links", []),
                    "used_in_analysis": a.get("used_in_analysis", False),
                    "has_content": bool(a.get("content")),
                    "_content": a.get("content"),
                }

        for hyp in state.get("hypotheses", []):
            h = hyp if isinstance(hyp, dict) else hyp.to_dict() if hasattr(hyp, 'to_dict') else {}
            for cite_key, cite_data in h.get("citation_map", {}).items():
                if cite_data.get("type") != "paper":
                    continue
                title = cite_data.get("title", "")
                url = cite_data.get("url", "")
                ref_id = url or title
                if ref_id and ref_id not in refs_by_id:
                    refs_by_id[ref_id] = {
                        "id": ref_id,
                        "title": title,
                        "authors": cite_data.get("authors", []),
                        "year": cite_data.get("year"),
                        "url": url,
                        "abstract": None,
                        "source": "citation",
                        "venue": None,
                        "pdf_links": [],
                        "used_in_analysis": False,
                        "has_content": False,
                        "citation_keys": [cite_key],
                    }
                elif ref_id in refs_by_id:
                    existing = refs_by_id[ref_id]
                    keys = existing.get("citation_keys", [])
                    keys.append(cite_key)
                    existing["citation_keys"] = keys

        return list(refs_by_id.values())

    def _save_paper_content(self, ref: Dict[str, Any]):
        """Save paper content to file if available from articles."""
        ref_id = ref.get("id", "")
        if not ref_id:
            return

        content = ref.get("_content")
        if not content:
            return

        safe_name = _sanitize_filename(str(ref_id), 60)
        path = self.papers_dir / f"{safe_name}.md"

        title = ref.get("title", "Unknown")
        year = ref.get("year", "")
        url = ref.get("url", "")
        author_str = _authors_to_str(ref.get("authors", []), max_count=5)

        header = (
            f"# {title}\n\n"
            f"**Authors:** {author_str}\n"
            f"**Year:** {year}\n"
            f"**URL:** {url}\n"
            f"**Source ID:** {ref_id}\n\n"
            f"---\n\n"
        )

        path.write_text(header + content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_run_status(self, status: str, execution_time: float = 0):
        info_path = self.run_dir / "run_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
        else:
            info = {}
        info["status"] = status
        info["end_time"] = datetime.now(timezone.utc).isoformat()
        if execution_time:
            info["execution_time_seconds"] = round(execution_time, 2)
        self._write_json(info_path, info)

    def mark_failed(self, error: str):
        self._update_run_status("failed")
        (self.run_dir / "error.txt").write_text(error, encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: Any):
        path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


class RunTracer:
    """
    Writes a detailed trace log of the entire research process.

    Captures every node start/end, progress event, key decisions, and timing.
    More detailed than the console output - serves as a full audit trail.
    """

    def __init__(self, trace_path: Path, research_goal: str):
        self._path = trace_path
        self._start = time.time()
        self._file = open(trace_path, "w", encoding="utf-8")
        self._write_header(research_goal)

    def _write_header(self, goal: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._file.write(
            f"{'=' * 80}\n"
            f"RESEARCH TRACE LOG\n"
            f"{'=' * 80}\n"
            f"Started: {ts}\n"
            f"Research Goal:\n{goal}\n"
            f"{'=' * 80}\n\n"
        )
        self._file.flush()

    def log_node_start(self, node_name: str):
        elapsed = time.time() - self._start
        self._write(f"\n{'─' * 60}\n")
        self._write(f"[{elapsed:7.1f}s] ▶ NODE: {node_name}\n")
        self._write(f"{'─' * 60}\n")

    def log_node_end(self, node_name: str, state: Dict[str, Any]):
        elapsed = time.time() - self._start
        self._write(f"[{elapsed:7.1f}s] ✓ NODE COMPLETE: {node_name}\n")

        hyps = state.get("hypotheses", [])
        if hyps:
            self._write(f"  Hypotheses: {len(hyps)}\n")

        metrics = state.get("metrics", {})
        if isinstance(metrics, dict):
            llm_calls = metrics.get("llm_calls", 0)
            if llm_calls:
                self._write(f"  LLM calls so far: {llm_calls}\n")

        if node_name == "supervisor":
            plan = state.get("research_plan", {})
            if plan:
                self._write(f"  Research plan keys: {list(plan.keys())}\n")

        if node_name == "literature_review":
            articles = state.get("articles", [])
            synth = state.get("articles_with_reasoning")
            queries = state.get("literature_review_queries", [])
            self._write(f"  Articles found: {len(articles)}\n")
            self._write(f"  Queries used: {len(queries)}\n")
            for i, q in enumerate(queries, 1):
                self._write(f"    Q{i}: {q}\n")
            if synth:
                self._write(f"  Synthesis length: {len(synth)} chars\n")

        if node_name == "generate":
            for i, h in enumerate(hyps, 1):
                text = h.get("text", "")[:120]
                method = h.get("generation_method", "?")
                self._write(f"  H{i} [{method}]: {text}...\n")

        if node_name in ("review", "ranking"):
            for i, h in enumerate(hyps, 1):
                score = h.get("score", 0)
                elo = h.get("elo_rating", 0)
                text = h.get("text", "")[:80]
                self._write(f"  H{i}: score={score:.1f} elo={elo} | {text}...\n")

        if node_name == "evolve":
            details = state.get("evolution_details", [])
            self._write(f"  Evolutions performed: {len(details)}\n")

        if node_name == "meta_review":
            mr = state.get("meta_review", {})
            if isinstance(mr, dict):
                self._write(f"  Meta-review keys: {list(mr.keys())}\n")

        self._write("\n")

    def log_progress(self, event: str, data: Dict[str, Any]):
        elapsed = time.time() - self._start
        msg = data.get("message", "")
        progress = data.get("progress", "")
        self._write(f"[{elapsed:7.1f}s]   ● {event}: {msg}")
        if progress:
            self._write(f" ({progress})")
        self._write("\n")

    def log_event(self, message: str):
        elapsed = time.time() - self._start
        self._write(f"[{elapsed:7.1f}s]   {message}\n")

    def finalize(self, execution_time: float):
        self._write(f"\n{'=' * 80}\n")
        self._write(f"TRACE COMPLETE\n")
        self._write(f"Total time: {execution_time:.1f}s\n")
        self._write(f"Ended: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        self._write(f"{'=' * 80}\n")
        self._file.close()

    def _write(self, text: str):
        self._file.write(text)
        self._file.flush()
