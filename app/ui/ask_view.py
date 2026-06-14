from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.errors import SefmError
from app.generation.citation_guard import REFUSAL, parse_citations, validate_answer
from app.generation.prompts import build_prompt
from app.retrieval.types import Chunk, Citation
from app.service import AppService
from app.ui.flow_layout import FlowContainer
from app.ui.pdf_viewer import PdfViewerDialog

log = logging.getLogger(__name__)

_STOP_WORDS = {
    "the", "and", "for", "this", "that", "with", "from", "are", "was", "has", "its",
}


def _search_phrase(chunk_text: str, max_chars: int = 60) -> str:
    """Short clean phrase from chunk text, for PDF search."""
    text = re.sub(r"[|#\-]{2,}", " ", chunk_text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 20 else truncated


class _StreamSignals(QObject):
    retrieved = Signal(list)
    token = Signal(str)
    done = Signal(str)
    failed = Signal(str)


class _StreamWorker(QRunnable):
    def __init__(self, service: AppService, question: str) -> None:
        super().__init__()
        self.service = service
        self.question = question
        self.signals = _StreamSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            log.info("ask (stream): %r", self.question[:120])
            hits = self.service.search(self.question)
            chunks = [h.chunk for h in hits]
            log.debug("ask (stream): %d chunks retrieved", len(chunks))
            for i, h in enumerate(hits, 1):
                log.debug("  chunk[%d] %s p.%d  rrf=%.4f", i, h.chunk.source, h.chunk.page, h.score)

            self.signals.retrieved.emit(chunks)
            if not chunks:
                log.warning("ask (stream): no chunks retrieved — empty response")
                self.signals.done.emit("")
                return

            assert self.service.llm is not None
            prompt = build_prompt(self.question, chunks)
            log.debug(
                "ask (stream): prompt (%d chars)\n%s\n--- end prompt ---",
                len(prompt),
                prompt,
            )

            buf: list[str] = []
            for delta in self.service.llm.stream(
                prompt,
                max_tokens=self.service.settings.llm_max_tokens,
                temperature=self.service.settings.llm_temperature,
            ):
                if self._cancelled:
                    log.info("ask (stream): cancelled by user")
                    break
                buf.append(delta)
                self.signals.token.emit(delta)

            raw = "".join(buf)
            log.debug(
                "ask (stream): raw output (%d chars)\n%s\n--- end raw output ---",
                len(raw),
                raw,
            )
            self.signals.done.emit(raw)
        except Exception as e:
            log.exception("ask (stream): generation failed")
            if isinstance(e, SefmError):
                self.signals.failed.emit(str(e))
            else:
                self.signals.failed.emit(f"{type(e).__name__}: {e}")


class _CitationChip(QFrame):
    """A rounded pill for one cited source/page, with Open and Verify actions."""

    def __init__(self, citation: Citation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("citationChip")
        self.citation = citation
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 6, 4)
        layout.setSpacing(4)

        label = QLabel(f"\U0001F4C4  {citation.source}  ·  p.{citation.page}")

        self.open_btn = QPushButton("Open ↗")
        self.open_btn.setObjectName("chipOpen")
        self.open_btn.setToolTip("Open the source PDF at this page")
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.verify_btn = QPushButton("Verify")
        self.verify_btn.setObjectName("chipVerify")
        self.verify_btn.setToolTip("Check this citation against the source PDF page")
        self.verify_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        layout.addWidget(label)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.verify_btn)


class _VerifyDialog(QDialog):
    """Side-by-side comparison of a retrieved chunk and its source PDF page text."""

    def __init__(
        self,
        citation: Citation,
        chunk_text: str,
        path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Verify — {citation.source} p.{citation.page}")
        self.resize(960, 640)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Retrieved chunk</b>"))
        chunk_view = QTextEdit()
        chunk_view.setReadOnly(True)
        chunk_view.setPlainText(chunk_text)
        ll.addWidget(chunk_view)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("<b>PDF page text</b> (chunk keywords highlighted)"))
        self._page_view = QTextEdit()
        self._page_view.setReadOnly(True)
        rl.addWidget(self._page_view)
        splitter.addWidget(right)

        layout.addWidget(splitter)

        self._status = QLabel()
        layout.addWidget(self._status)

        self._load_page_text(path, citation.page, chunk_text)

    def _load_page_text(self, path: Path, page: int, chunk_text: str) -> None:
        try:
            import pdfplumber  # lazy; requires sefm[ingest]
            with pdfplumber.open(str(path)) as pdf:
                page_text = pdf.pages[page - 1].extract_text() if page <= len(pdf.pages) else ""
        except ImportError:
            self._page_view.setPlainText("(pdfplumber not available — install sefm[ingest])")
            self._status.setText("Cannot extract page text without pdfplumber.")
            return
        except Exception as exc:
            self._page_view.setPlainText(f"Error: {exc}")
            return

        page_text = page_text or ""
        if not page_text:
            self._page_view.setPlainText(
                "(No extractable text on this page — may be image/OCR only.)"
            )
            self._status.setText("No text extracted from page.")
            return

        # Chunk words (>3 chars, non-stop) to highlight in the page text.
        chunk_words = {
            w.lower()
            for w in re.findall(r"\b\w{4,}\b", chunk_text)
            if w.lower() not in _STOP_WORDS
        }

        escaped = html.escape(page_text)
        for word in sorted(chunk_words, key=len, reverse=True):
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            escaped = pattern.sub(
                lambda m: f'<mark style="background:#ffe066;">{m.group()}</mark>',
                escaped,
            )
        self._page_view.setHtml(
            "<pre style='font-family:monospace; font-size:9pt; white-space:pre-wrap;'>"
            f"{escaped}</pre>"
        )

        page_words = {w.lower() for w in re.findall(r"\b\w{4,}\b", page_text)}
        if chunk_words:
            overlap = len(chunk_words & page_words) / len(chunk_words)
            quality = "High" if overlap > 0.7 else "Medium" if overlap > 0.4 else "Low"
            self._status.setText(
                f"Match quality: {quality}"
                f" ({overlap:.0%} of chunk keywords found on this page)"
            )
        else:
            self._status.setText("Chunk has no indexable keywords.")


class AskView(QWidget):
    """Streaming generation with strict citation enforcement."""

    def __init__(self, service: AppService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.pool = QThreadPool.globalInstance()
        self._worker: _StreamWorker | None = None
        self._retrieved: list[Chunk] = []
        # Response-speed tracking (all updated on the UI thread).
        self._t_start = 0.0
        self._t_first_token = 0.0
        self._token_count = 0
        self._build()
        self._refresh_llm_state()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.query = QLineEdit()
        self.query.setPlaceholderText(
            "Ask anything grounded in your indexed documents..."
        )
        self.query.setClearButtonEnabled(True)
        self.query.returnPressed.connect(self._ask)
        self.btn_ask = QPushButton("Ask")
        self.btn_ask.setProperty("accent", True)
        self.btn_ask.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ask.clicked.connect(self._ask)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.query, stretch=1)
        row.addWidget(self.btn_ask)
        row.addWidget(self.btn_stop)
        layout.addLayout(row)

        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText("The answer will stream here.")
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.answer.setFont(mono)
        self.answer.document().setDocumentMargin(12)
        layout.addWidget(self.answer, stretch=1)

        # Citations live in a panel that stays hidden until there's something
        # to show — no more empty gray box under every answer.
        self.citations_panel = QWidget()
        cp = QVBoxLayout(self.citations_panel)
        cp.setContentsMargins(0, 0, 0, 0)
        cp.setSpacing(6)
        self.cite_header = QLabel("Citations")
        self.cite_header.setProperty("heading", True)
        cp.addWidget(self.cite_header)
        self.citation_flow = FlowContainer(spacing=8)
        cp.addWidget(self.citation_flow)
        self.citations_panel.setVisible(False)
        layout.addWidget(self.citations_panel)

        status_row = QHBoxLayout()
        self.status = QLabel()
        self.status.setProperty("muted", True)
        self.speed = QLabel()
        self.speed.setProperty("muted", True)
        self.speed.setToolTip(
            "Generation speed. 'first token' is the latency before the model "
            "starts answering; tok/s is the streaming throughput."
        )
        status_row.addWidget(self.status, stretch=1)
        status_row.addWidget(self.speed)
        layout.addLayout(status_row)

    def refresh(self) -> None:
        self._refresh_llm_state()

    def _refresh_llm_state(self) -> None:
        ok = self.service.llm is not None
        self.btn_ask.setEnabled(ok)
        self.query.setEnabled(ok)
        if ok:
            self.status.setText("Ready.")
        else:
            self.status.setText(
                "LLM not configured — set SEFM_LLM_MODEL_PATH in .env and restart."
            )

    def _ask(self) -> None:
        if self.service.llm is None or self._worker is not None:
            return
        q = self.query.text().strip()
        if not q:
            return
        self._clear_citations()
        self.answer.clear()
        self.status.setText("Retrieving...")
        self.speed.clear()
        self._t_start = time.perf_counter()
        self._t_first_token = 0.0
        self._token_count = 0
        self.btn_ask.setEnabled(False)
        self.btn_stop.setEnabled(True)

        worker = _StreamWorker(self.service, q)
        worker.signals.retrieved.connect(self._on_retrieved)
        worker.signals.token.connect(self._on_token)
        worker.signals.done.connect(self._on_done)
        worker.signals.failed.connect(self._on_failed)
        self._worker = worker
        self.pool.start(worker)

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Stopping...")

    def _on_retrieved(self, chunks: list[Chunk]) -> None:
        self._retrieved = chunks
        if not chunks:
            self.status.setText("No matching content. Ingest documents first.")
            self._finish()
            return
        self.status.setText(f"Generating from {len(chunks)} chunk(s)...")

    def _on_token(self, delta: str) -> None:
        now = time.perf_counter()
        if self._token_count == 0:
            self._t_first_token = now
        self._token_count += 1

        cursor = self.answer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(delta)
        self.answer.setTextCursor(cursor)

        # Live throughput, refreshed every few tokens to avoid label churn.
        if self._token_count % 4 == 0:
            gen = now - self._t_first_token
            rate = self._token_count / gen if gen > 0 else 0.0
            self.speed.setText(f"⏱ first token {self._t_first_token - self._t_start:.1f}s · {rate:.1f} tok/s")

    def _set_final_speed(self) -> None:
        if self._token_count == 0:
            self.speed.clear()
            return
        ttft = self._t_first_token - self._t_start
        gen = time.perf_counter() - self._t_first_token
        rate = self._token_count / gen if gen > 0 else 0.0
        self.speed.setText(
            f"⏱ {self._token_count} tok in {gen:.1f}s · "
            f"{rate:.1f} tok/s · first token {ttft:.1f}s"
        )

    def _on_done(self, raw: str) -> None:
        self._set_final_speed()
        if raw:
            validated = validate_answer(raw, self._retrieved)
            log.debug(
                "ask (stream): validated output (%d chars)\n%s\n--- end validated output ---",
                len(validated),
                validated,
            )
            # validate_answer may rewrite [N] -> [source: ...]; the refusal
            # sentence is the only failure signal.
            self.answer.setPlainText(validated)
            if validated == REFUSAL:
                log.warning("ask (stream): answer refused by citation guard")
                self.status.setText(
                    "Refused: the model's citations could not be grounded "
                    "in the retrieved context."
                )
            else:
                log.info("ask (stream): answer accepted (%d chars)", len(validated))
                self.status.setText("Done.")
            self._render_citations(parse_citations(validated))
        self._finish()

    def _on_failed(self, msg: str) -> None:
        self.status.setText("Error.")
        QMessageBox.critical(self, "Generation failed", msg)
        self._finish()

    def _finish(self) -> None:
        self._worker = None
        self.btn_ask.setEnabled(self.service.llm is not None)
        self.btn_stop.setEnabled(False)

    def _clear_citations(self) -> None:
        flow = self.citation_flow.flow
        while flow.count():
            item = flow.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        self.citations_panel.setVisible(False)

    def _chunk_for_citation(self, citation: Citation) -> Chunk | None:
        for chunk in self._retrieved:
            if chunk.source == citation.source and chunk.page == citation.page:
                return chunk
        return None

    def _render_citations(self, citations: list[Citation]) -> None:
        self._clear_citations()
        seen: set[tuple[str, int]] = set()
        count = 0
        for cit in citations:
            key = (cit.source, cit.page)
            if key in seen:
                continue
            seen.add(key)
            chip = _CitationChip(cit)
            chip.open_btn.clicked.connect(lambda _=False, c=cit: self._open_citation(c))
            chip.verify_btn.clicked.connect(lambda _=False, c=cit: self._verify_citation(c))
            self.citation_flow.flow.addWidget(chip)
            count += 1
        if count:
            self.cite_header.setText(f"Citations ({count})")
            self.citations_panel.setVisible(True)

    def _open_citation(self, citation: Citation) -> None:
        path = self.service.resolve_source_path(citation.source)
        if path is None:
            QMessageBox.warning(
                self,
                "File not found",
                f"The original path for {citation.source} is unknown.",
            )
            return
        chunk = self._chunk_for_citation(citation)
        highlight = _search_phrase(chunk.text) if chunk else None
        dlg = PdfViewerDialog(path, citation.page, highlight_text=highlight, parent=self)
        dlg.show()

    def _verify_citation(self, citation: Citation) -> None:
        path = self.service.resolve_source_path(citation.source)
        if path is None:
            QMessageBox.warning(
                self,
                "File not found",
                f"Cannot verify — source path for {citation.source} is unknown.",
            )
            return
        chunk = self._chunk_for_citation(citation)
        dlg = _VerifyDialog(citation, chunk.text if chunk else "", path, parent=self)
        dlg.show()
