from __future__ import annotations

from app.generation.prompts import PROMPT_TEMPLATE, build_prompt, format_context
from app.retrieval.types import Chunk


def test_format_context_includes_source_and_page() -> None:
    chunks = [
        Chunk(chunk_id="a", text="alpha", source="vfd.pdf", page=42),
        Chunk(chunk_id="b", text="beta", source="ev.pdf", page=7),
    ]
    ctx = format_context(chunks)
    assert "source: vfd.pdf, p. 42" in ctx
    assert "source: ev.pdf, p. 7" in ctx
    assert "alpha" in ctx and "beta" in ctx


def test_build_prompt_contains_question_and_template_rules() -> None:
    prompt = build_prompt("What is fault 0x21?", [])
    assert "What is fault 0x21?" in prompt
    assert "Not found in the indexed documents." in prompt
    assert "[source: <filename>, p. <page>]" in prompt


def test_template_is_a_string_template() -> None:
    # Guards against accidental f-string mutation that would break formatting.
    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE
