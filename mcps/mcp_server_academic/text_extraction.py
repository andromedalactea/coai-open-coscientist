"""
Text extraction from LaTeX source files and PDFs.

Two extraction modes:
- LaTeX mode: Parse .tex/.bib/.bbl into structured markdown
- PDF mode: Extract text from PDF bytes using PyMuPDF

Both return markdown-formatted text suitable for LLM consumption.
"""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MAX_CHARS = 200_000


# ---------------------------------------------------------------------------
# LaTeX extraction
# ---------------------------------------------------------------------------

def extract_text_from_latex(
    tex_content: str,
    bib_content: Optional[str] = None,
    bbl_content: Optional[str] = None,
    max_chars: int = MAX_CHARS,
) -> str:
    """
    Convert LaTeX source into clean markdown.

    Extracts title, abstract, sections, and optionally appends
    bibliography information from .bib/.bbl files.
    """
    try:
        parts = []

        title = _extract_latex_field(tex_content, "title")
        if title:
            parts.append(f"# {_clean_latex(title)}")

        abstract = _extract_environment(tex_content, "abstract")
        if abstract:
            parts.append(f"## Abstract\n\n{_clean_latex(abstract)}")

        body = _extract_document_body(tex_content)
        if body:
            sections = _split_into_sections(body)
            for heading, content in sections:
                clean_heading = _clean_latex(heading)
                clean_content = _clean_latex(content)
                if clean_content.strip():
                    parts.append(f"## {clean_heading}\n\n{clean_content}")

        if bbl_content:
            refs = _extract_references_from_bbl(bbl_content)
            if refs:
                parts.append(f"## References\n\n{refs}")
        elif bib_content:
            refs = _extract_references_from_bib(bib_content)
            if refs:
                parts.append(f"## References\n\n{refs}")

        markdown = "\n\n".join(parts)

        if not markdown.strip():
            markdown = _clean_latex(tex_content)

        if len(markdown) > max_chars:
            logger.info(f"Truncating LaTeX extraction from {len(markdown)} to {max_chars} chars")
            markdown = markdown[:max_chars] + "\n\n[... truncated for length ...]"

        return markdown

    except Exception as e:
        logger.error(f"LaTeX extraction failed: {e}")
        fallback = _clean_latex(tex_content)
        if len(fallback) > max_chars:
            fallback = fallback[:max_chars] + "\n\n[... truncated for length ...]"
        return fallback


def _extract_latex_field(tex: str, field: str) -> Optional[str]:
    """Extract a top-level LaTeX command like \\title{...}."""
    pattern = rf"\\{field}\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}"
    m = re.search(pattern, tex, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_environment(tex: str, env_name: str) -> Optional[str]:
    """Extract content of a LaTeX environment."""
    pattern = rf"\\begin\{{{env_name}\}}(.*?)\\end\{{{env_name}\}}"
    m = re.search(pattern, tex, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_document_body(tex: str) -> Optional[str]:
    """Extract content between \\begin{document} and \\end{document}."""
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex, re.DOTALL)
    return m.group(1).strip() if m else None


def _split_into_sections(body: str) -> list:
    """Split document body into (heading, content) tuples by \\section commands."""
    pattern = r"\\(?:section|subsection|subsubsection)\*?\{((?:[^{}]|\{[^{}]*\})*)\}"
    splits = re.split(pattern, body)

    sections = []
    if splits[0].strip():
        sections.append(("Introduction", splits[0]))

    for i in range(1, len(splits) - 1, 2):
        heading = splits[i].strip()
        content = splits[i + 1] if i + 1 < len(splits) else ""
        sections.append((heading, content))

    return sections


def _clean_latex(text: str) -> str:
    """Remove LaTeX commands and convert to plain text."""
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        converter = LatexNodes2Text()
        cleaned = converter.latex_to_text(text)
    except Exception:
        cleaned = _regex_clean_latex(text)
    return cleaned.strip()


def _regex_clean_latex(text: str) -> str:
    """Fallback regex-based LaTeX cleaning when pylatexenc is unavailable."""
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|textrm|textsf)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:cite|ref|label|eqref|autoref|cref)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"%[^\n]*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_references_from_bbl(bbl_content: str) -> str:
    """Extract references from a .bbl file into readable text."""
    entries = re.split(r"\\bibitem", bbl_content)
    refs = []
    for entry in entries[1:]:
        entry = entry.strip()
        clean = _clean_latex(entry)
        if clean.strip():
            refs.append(f"- {clean.strip()}")
    return "\n".join(refs) if refs else ""


def _extract_references_from_bib(bib_content: str) -> str:
    """Extract references from a .bib file into readable text."""
    entries = re.findall(
        r"@\w+\{([^,]+),\s*(.*?)\n\}", bib_content, re.DOTALL
    )
    refs = []
    for key, body in entries:
        title_m = re.search(r"title\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        author_m = re.search(r"author\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        year_m = re.search(r"year\s*=\s*\{?(\d{4})\}?", body, re.IGNORECASE)

        parts = []
        if author_m:
            parts.append(author_m.group(1).strip())
        if year_m:
            parts.append(f"({year_m.group(1)})")
        if title_m:
            parts.append(f'"{title_m.group(1).strip()}"')

        if parts:
            refs.append(f"- [{key}] {' '.join(parts)}")
    return "\n".join(refs) if refs else ""


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    pdf_bytes: bytes,
    max_chars: int = MAX_CHARS,
) -> str:
    """
    Extract text from PDF bytes using PyMuPDF.

    Returns markdown-formatted text.
    """
    try:
        import fitz  # pymupdf

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                parts.append(text)
        doc.close()

        full_text = "\n\n".join(parts)

        if len(full_text) > max_chars:
            logger.info(f"Truncating PDF extraction from {len(full_text)} to {max_chars} chars")
            full_text = full_text[:max_chars] + "\n\n[... truncated for length ...]"

        return full_text

    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return "[error: could not extract text from PDF]"
