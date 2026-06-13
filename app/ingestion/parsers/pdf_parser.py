from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.errors import DocumentParseError
from app.ingestion.parsers.base import DocumentParser
from app.retrieval.types import Chunk

log = logging.getLogger(__name__)


class PdfParser(DocumentParser):
    """PyMuPDF for text/layout, pdfplumber for tables.

    Per page, yields one kind="text" chunk (body text with table regions
    removed) plus one kind="table" chunk per table (rows as Markdown).
    Page numbers are 1-based.
    """

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> Iterator[Chunk]:
        import pdfplumber
        import pymupdf

        source = path.name

        # Open both eagerly so encryption/corruption fails fast, not mid-generator.
        doc: Any
        try:
            doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise DocumentParseError(
                f"{source} could not be opened: {exc}",
                source=source,
                hint="The file may be corrupt or not a valid PDF.",
            ) from exc

        if doc.needs_pass or doc.is_encrypted:
            doc.close()
            raise DocumentParseError(
                f"{source} is password-protected.",
                source=source,
                hint=f"{source} is password-protected; decrypt it before ingesting.",
            )

        plumb: Any
        try:
            plumb = pdfplumber.open(path)
        except Exception as exc:
            doc.close()
            raise DocumentParseError(
                f"{source} could not be opened by pdfplumber: {exc}",
                source=source,
                hint="The file may be corrupt or not a valid PDF.",
            ) from exc

        try:
            yield from self._iter_pages(doc, plumb, source)
        finally:
            plumb.close()
            doc.close()

    def _iter_pages(
        self,
        doc: Any,
        plumb: Any,
        source: str,
    ) -> Iterator[Chunk]:
        for page_index in range(doc.page_count):
            page_num = page_index + 1
            try:
                fitz_page = doc[page_index]
                plumb_page = plumb.pages[page_index]

                tables = plumb_page.find_tables() or []

                # Table chunks first; collect bboxes to subtract from body text.
                table_bboxes: list[tuple[float, float, float, float]] = []
                for t_idx, table in enumerate(tables):
                    rows = table.extract() or []
                    if not _has_real_content(rows):
                        continue
                    md = _rows_to_markdown(rows)
                    if not md.strip():
                        continue
                    yield Chunk(
                        chunk_id=_chunk_id(source, page_num, f"t{t_idx}"),
                        text=md,
                        source=source,
                        page=page_num,
                        kind="table",
                    )
                    table_bboxes.append(tuple(table.bbox))

                body_text = _extract_text_excluding(fitz_page, table_bboxes)
                if body_text.strip():
                    yield Chunk(
                        chunk_id=_chunk_id(source, page_num, "text"),
                        text=body_text,
                        source=source,
                        page=page_num,
                        kind="text",
                    )
            except Exception:
                log.warning(
                    "skipping page %d of %s due to extraction error",
                    page_num,
                    source,
                    exc_info=True,
                )


def _chunk_id(source: str, page: int, suffix: str) -> str:
    return f"{source}#p{page}#{suffix}"


def _has_real_content(rows: list[list[str | None]]) -> bool:
    for row in rows:
        for cell in row:
            if cell and cell.strip():
                return True
    return False


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[_clean_cell(c) for c in row] for row in rows if row]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header = cleaned[0]
    sep = ["---"] * width
    body = cleaned[1:] if len(cleaned) > 1 else []
    lines = [_md_row(header), _md_row(sep), *(_md_row(r) for r in body)]
    return "\n".join(lines)


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return " ".join(cell.split())


def _extract_text_excluding(
    fitz_page: Any, exclude_bboxes: list[tuple[float, float, float, float]]
) -> str:
    """Page text with words inside any excluded bbox dropped. Lines are rebuilt
    in (block, line) order. get_text('words') -> (x0,y0,x1,y1,word,block,line,word_no)."""
    words = fitz_page.get_text("words")
    if not exclude_bboxes:
        return str(fitz_page.get_text("text")).strip()

    kept = [w for w in words if not _word_in_any_bbox(w, exclude_bboxes)]
    if not kept:
        return ""
    kept.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word

    lines: list[list[str]] = []
    cur_key: tuple[int, int] | None = None
    for w in kept:
        key = (w[5], w[6])
        if key != cur_key:
            lines.append([])
            cur_key = key
        lines[-1].append(w[4])
    return "\n".join(" ".join(line) for line in lines).strip()


_BBOX_PAD = 3.0  # PDF points; absorbs cell-text overflow that pdfplumber misses


def _word_in_any_bbox(
    word: tuple[Any, ...], bboxes: list[tuple[float, float, float, float]]
) -> bool:
    """True if the word's bbox intersects any table bbox (with padding).
    Intersection, not containment, so wrapped/clipped cell text isn't leaked
    into the body chunk."""
    wx0, wy0, wx1, wy1 = word[0], word[1], word[2], word[3]
    for bx0, by0, bx1, by1 in bboxes:
        if (
            wx0 <= bx1 + _BBOX_PAD
            and wx1 >= bx0 - _BBOX_PAD
            and wy0 <= by1 + _BBOX_PAD
            and wy1 >= by0 - _BBOX_PAD
        ):
            return True
    return False
