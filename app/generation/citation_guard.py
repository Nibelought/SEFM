from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.retrieval.types import Chunk, Citation

log = logging.getLogger(__name__)

REFUSAL = "Not found in the indexed documents."

# [source: <filename>, p. <page>]. src allows commas in filenames.
_FULL_CITATION_RE = re.compile(
    r"\[source:\s*(?P<src>[^\]]+?)\s*,\s*p\.\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)

# [N], [N, p. M], etc. Only the leading integer is trusted.
_NUMERIC_CITATION_RE = re.compile(r"\[(\d+)(?:[^\]]*)\]")


def expand_numeric_citations(answer: str, retrieved: Sequence[Chunk]) -> str:
    """Rewrite [N] -> [source: <filename>, p. <page>]. Out-of-range indices
    are left untouched for validate_answer to reject."""
    chunks = list(retrieved)

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            return f"[source: {c.source}, p. {c.page}]"
        return m.group(0)

    return _NUMERIC_CITATION_RE.sub(repl, answer)


def parse_citations(text: str) -> list[Citation]:
    return [
        Citation(source=m.group("src").strip(), page=int(m.group("page")))
        for m in _FULL_CITATION_RE.finditer(text)
    ]


def validate_answer(answer: str, retrieved: Sequence[Chunk]) -> str:
    """Expand numeric citations, then require every citation to map to a
    retrieved (source, page). Uncited or hallucinated -> refusal sentence."""
    if answer.strip() == REFUSAL:
        log.debug("validate_answer: model self-refused")
        return REFUSAL

    expanded = expand_numeric_citations(answer, retrieved)
    log.debug(
        "validate_answer: expanded answer (%d chars)\n%s\n--- end expanded ---",
        len(expanded),
        expanded,
    )

    cited = parse_citations(expanded)
    allowed = {(c.source, c.page) for c in retrieved}
    log.debug(
        "validate_answer: allowed sources (%d): %s",
        len(allowed),
        ", ".join(f"{src} p.{pg}" for src, pg in sorted(allowed)),
    )
    log.debug(
        "validate_answer: cited sources (%d): %s",
        len(cited),
        ", ".join(f"{c.source} p.{c.page}" for c in cited),
    )

    if not cited:
        log.warning("validate_answer: refused — no citations found in answer")
        return REFUSAL

    for c in cited:
        if (c.source, c.page) not in allowed:
            log.warning(
                "validate_answer: refused — hallucinated citation [source: %s, p. %d]"
                " (not in retrieved set)",
                c.source,
                c.page,
            )
            return REFUSAL

    log.debug("validate_answer: all %d citation(s) grounded — accepted", len(cited))
    return expanded
