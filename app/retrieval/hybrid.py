from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.types import Chunk, Hit


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[Hit],
    k: int = 60,
    top_n: int = 5,
) -> list[Hit]:
    """Reciprocal Rank Fusion. Each list contributes 1/(k + rank); returns the
    top_n chunks by fused score. k=60 is the standard default."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid] = hit.chunk
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [Hit(chunk=chunks[cid], score=score) for cid, score in fused[:top_n]]
