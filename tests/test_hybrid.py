from __future__ import annotations

from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.types import Chunk, Hit


def _hit(cid: str, score: float = 1.0) -> Hit:
    return Hit(chunk=Chunk(chunk_id=cid, text=cid, source="x.pdf", page=1), score=score)


def test_rrf_boosts_chunks_in_both_lists() -> None:
    dense = [_hit("A"), _hit("B"), _hit("C")]
    sparse = [_hit("B"), _hit("D"), _hit("A")]
    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=4)
    ids = [h.chunk.chunk_id for h in fused]
    # A and B appear in both lists, so they outrank C and D.
    assert set(ids[:2]) == {"A", "B"}


def test_rrf_respects_top_n() -> None:
    dense = [_hit(f"A{i}") for i in range(20)]
    fused = reciprocal_rank_fusion(dense, k=60, top_n=5)
    assert len(fused) == 5


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], [], top_n=5) == []


def test_rrf_single_list_preserves_order() -> None:
    dense = [_hit("X"), _hit("Y"), _hit("Z")]
    fused = reciprocal_rank_fusion(dense, k=60, top_n=3)
    assert [h.chunk.chunk_id for h in fused] == ["X", "Y", "Z"]
