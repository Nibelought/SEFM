from __future__ import annotations

from app.ingestion.chunking import Chunker, ChunkerConfig
from app.retrieval.types import Chunk


def test_short_text_returned_as_is() -> None:
    chunker = Chunker(ChunkerConfig(max_chars=2000))
    pre = Chunk(chunk_id="a#1", text="short paragraph", source="x.pdf", page=1)
    assert chunker.split(pre) == [pre]


def test_table_is_kept_atomic_even_when_oversize() -> None:
    chunker = Chunker(ChunkerConfig(max_chars=10))
    big_table = "| reg | name |\n" + "\n".join(f"| 0x{i:04X} | x |" for i in range(50))
    pre = Chunk(chunk_id="t#1", text=big_table, source="x.pdf", page=1, kind="table")
    assert chunker.split(pre) == [pre]


def test_long_text_is_split() -> None:
    chunker = Chunker(ChunkerConfig(max_chars=100, overlap_chars=0))
    paras = [f"Paragraph {i} with some filler content to grow size." for i in range(10)]
    pre = Chunk(chunk_id="p#1", text="\n\n".join(paras), source="x.pdf", page=1)
    pieces = chunker.split(pre)
    assert len(pieces) > 1
    assert all(p.source == "x.pdf" and p.page == 1 for p in pieces)
    assert {p.chunk_id for p in pieces} == {f"p#1#{i}" for i in range(len(pieces))}
