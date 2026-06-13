from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.retrieval.types import Hit
from app.service import AppService
from app.ui.pdf_viewer import PdfViewerDialog
from app.ui.workers import Worker


class SearchView(QWidget):
    def __init__(self, service: AppService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.pool = QThreadPool.globalInstance()
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText(
            "Ask anything — e.g. 'what does fault 0x21 mean?'"
        )
        self.query.returnPressed.connect(self._run)
        self.top_n = QSpinBox()
        self.top_n.setRange(1, 25)
        self.top_n.setValue(5)
        self.top_n.setPrefix("top ")
        self.btn = QPushButton("Search")
        self.btn.clicked.connect(self._run)
        row.addWidget(self.query, stretch=1)
        row.addWidget(self.top_n)
        row.addWidget(self.btn)
        layout.addLayout(row)

        self.results = QTreeWidget()
        self.results.setHeaderLabels(["#", "Source", "Page", "Kind", "Score", "Snippet"])
        self.results.setRootIsDecorated(False)
        self.results.setAlternatingRowColors(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.results.setFont(mono)
        self.results.itemDoubleClicked.connect(self._open_hit)
        layout.addWidget(self.results, stretch=1)

        actions = QHBoxLayout()
        self.btn_open = QPushButton("Open PDF at page")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open_selected)
        self.status = QLabel("Type a query and press Enter.")
        actions.addWidget(self.btn_open)
        actions.addStretch(1)
        actions.addWidget(self.status)
        layout.addLayout(actions)

        self.results.itemSelectionChanged.connect(
            lambda: self.btn_open.setEnabled(bool(self.results.selectedItems()))
        )

    def _run(self) -> None:
        q = self.query.text().strip()
        if not q:
            return
        self._busy(True)
        worker = Worker(self.service.search, q, self.top_n.value())
        worker.signals.finished.connect(self._show_hits)
        worker.signals.failed.connect(self._on_error)
        self.pool.start(worker)

    def _show_hits(self, hits: list[Hit]) -> None:
        self.results.clear()
        self.results.setUpdatesEnabled(False)
        for i, hit in enumerate(hits, 1):
            c = hit.chunk
            snippet = _snippet(c.text, 220)
            item = QTreeWidgetItem(
                [str(i), c.source, str(c.page), c.kind, f"{hit.score:.3f}", snippet]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, (c.source, c.page))
            self.results.addTopLevelItem(item)
        for i in range(6):
            self.results.resizeColumnToContents(i)
        self.results.setUpdatesEnabled(True)
        self._busy(False)
        self.status.setText(
            f"{len(hits)} result(s)." if hits else "No results — try ingesting a PDF first."
        )

    def _on_error(self, msg: str) -> None:
        self._busy(False)
        QMessageBox.critical(self, "Search failed", msg)

    def _open_selected(self) -> None:
        items = self.results.selectedItems()
        if not items:
            return
        self._open_hit(items[0], 0)

    def _open_hit(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        source, page = data
        path = self.service.resolve_source_path(source)
        if path is None:
            QMessageBox.warning(
                self,
                "File not found",
                f"The original path for {source} is unknown. Re-ingest the file from the Library tab.",
            )
            return
        try:
            dlg = PdfViewerDialog(path, page, parent=self)
            dlg.show()
        except Exception:
            # Fallback: hand off to the OS viewer with a #page= fragment.
            url = QUrl.fromLocalFile(str(path))
            url.setFragment(f"page={page}")
            QDesktopServices.openUrl(url)

    def _busy(self, on: bool) -> None:
        self.btn.setEnabled(not on)
        self.query.setEnabled(not on)
        self.status.setText("Searching..." if on else self.status.text())


def _snippet(text: str, max_len: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_len else flat[: max_len - 1] + "..."
