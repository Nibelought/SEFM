from __future__ import annotations

from app.generation.citation_guard import (
    REFUSAL,
    expand_numeric_citations,
    parse_citations,
    validate_answer,
)
from app.retrieval.types import Chunk


def _chunk(source: str, page: int) -> Chunk:
    return Chunk(chunk_id=f"{source}#{page}", text="...", source=source, page=page)


class TestParseCitations:
    def test_finds_single(self) -> None:
        cites = parse_citations("Overcurrent. [source: vfd.pdf, p. 42]")
        assert len(cites) == 1
        assert cites[0].source == "vfd.pdf"
        assert cites[0].page == 42

    def test_finds_multiple(self) -> None:
        text = "A [source: a.pdf, p. 1] and B [source: b.pdf, p. 9]."
        cites = parse_citations(text)
        assert {(c.source, c.page) for c in cites} == {("a.pdf", 1), ("b.pdf", 9)}

    def test_tolerates_whitespace(self) -> None:
        cites = parse_citations("x [source:  vfd.pdf ,  p.  42 ]")
        assert cites[0].source == "vfd.pdf"
        assert cites[0].page == 42

    def test_filename_with_commas(self) -> None:
        long_name = "Terminal Blocks, Fuse Holders, and Power Distribution Blocks (Class 9080) Catalog.pdf"
        cites = parse_citations(f"See [source: {long_name}, p. 17].")
        assert len(cites) == 1
        assert cites[0].source == long_name
        assert cites[0].page == 17


class TestValidateAnswer:
    def test_passes_when_citations_match(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        answer = "Overcurrent. [source: vfd.pdf, p. 42]"
        assert validate_answer(answer, retrieved) == answer

    def test_refuses_hallucinated_source(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        answer = "Overcurrent. [source: not-real.pdf, p. 42]"
        assert validate_answer(answer, retrieved) == REFUSAL

    def test_refuses_hallucinated_page(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        answer = "Overcurrent. [source: vfd.pdf, p. 99]"
        assert validate_answer(answer, retrieved) == REFUSAL

    def test_refuses_uncited(self) -> None:
        assert validate_answer("Overcurrent.", [_chunk("vfd.pdf", 42)]) == REFUSAL

    def test_passes_refusal_through(self) -> None:
        assert validate_answer(REFUSAL, []) == REFUSAL

    def test_rejects_when_any_citation_bad(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        answer = "A [source: vfd.pdf, p. 42]. B [source: vfd.pdf, p. 7]."
        assert validate_answer(answer, retrieved) == REFUSAL

    def test_passes_filename_with_commas(self) -> None:
        long_name = "Terminal Blocks, Fuse Holders, and Power Distribution Blocks (Class 9080) Catalog.pdf"
        retrieved = [_chunk(long_name, 17)]
        answer = f"Max voltage is 600V [source: {long_name}, p. 17]."
        assert validate_answer(answer, retrieved) == answer


class TestNumericCitationExpansion:
    def test_expands_simple_n(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42), _chunk("ev.pdf", 7)]
        out = expand_numeric_citations("Overcurrent [1]. Step [2].", retrieved)
        assert out == "Overcurrent [source: vfd.pdf, p. 42]. Step [source: ev.pdf, p. 7]."

    def test_expands_n_with_page_hint(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        out = expand_numeric_citations("Overcurrent [1, p. 1].", retrieved)
        assert out == "Overcurrent [source: vfd.pdf, p. 42]."

    def test_leaves_out_of_range_untouched(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        out = expand_numeric_citations("Bogus [9].", retrieved)
        assert out == "Bogus [9]."

    def test_validate_accepts_numeric_in_range(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        v = validate_answer("Overcurrent [1].", retrieved)
        assert v == "Overcurrent [source: vfd.pdf, p. 42]."

    def test_validate_refuses_numeric_out_of_range(self) -> None:
        retrieved = [_chunk("vfd.pdf", 42)]
        # [2] doesn't expand and parse_citations finds no full-form citation
        # -> uncited -> refusal.
        assert validate_answer("Overcurrent [2].", retrieved) == REFUSAL
