from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.types import Chunk

PROMPT_TEMPLATE = """\
You are a technical assistant for industrial automation engineers.
Answer ONLY using the CONTEXT below. If the answer is not in the context,
reply exactly: "Not found in the indexed documents."

After each factual claim, add a citation in this exact format:
  [source: <filename>, p. <page>]
Copy <filename> and <page> verbatim from the matching CONTEXT block header
(e.g. "[3] source: manual.pdf, p. 47" → cite as [source: manual.pdf, p. 47]).
Do not invent filenames or page numbers.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_context(chunks: Sequence[Chunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] source: {c.source}, p. {c.page}\n{c.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    return PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
