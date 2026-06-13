from __future__ import annotations

import pytest

from app.retrieval.types import Chunk, Hit


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="a#1#0", text="Fault 0x21 indicates overcurrent.", source="vfd.pdf", page=42),
        Chunk(chunk_id="b#1#0", text="Register 0x2001 = output current.", source="vfd.pdf", page=12, kind="table"),
        Chunk(chunk_id="c#1#0", text="Commissioning step 1: power on.", source="ev.pdf", page=3),
    ]


@pytest.fixture
def hits(chunks: list[Chunk]) -> list[Hit]:
    return [Hit(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(chunks)]
