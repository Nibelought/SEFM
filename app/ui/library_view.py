from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.service import AppService
from app.ui.workers import Worker


class LibraryView(QWidget):
    """Add PDFs/folders, list what's indexed, ingest with progress."""

    documents_changed = Signal()

    def __init__(self, service: AppService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.pool = QThreadPool.globalInstance()
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add PDFs...")
        self.btn_add_folder = QPushButton("Add Folder...")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setEnabled(False)
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_add_folder)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_remove)
        layout.addLayout(btn_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Document", "Chunks", "Path"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree, stretch=1)

        self.status = QLabel("Ready.")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.tree.itemSelectionChanged.connect(self._on_selection)

    def refresh(self) -> None:
        self.tree.clear()
        docs = self.service.list_documents()
        for source, count in sorted(docs.items()):
            path = self.service.resolve_source_path(source)
            item = QTreeWidgetItem(
                [source, str(count), str(path) if path else "(path unknown)"]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, source)
            self.tree.addTopLevelItem(item)
        for i in range(3):
            self.tree.resizeColumnToContents(i)

    def _on_selection(self) -> None:
        self.btn_remove.setEnabled(bool(self.tree.selectedItems()))

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF files", "", "PDF Files (*.pdf)"
        )
        if files:
            self._ingest([Path(f) for f in files])

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of PDFs")
        if folder:
            self._ingest([Path(folder)])

    def _ingest(self, paths: list[Path]) -> None:
        self._busy(True, f"Indexing {len(paths)} item(s)...")

        def task() -> tuple[int, list[tuple[str, str]]]:
            n = 0
            skipped: list[tuple[str, str]] = []
            for p in paths:
                n += self.service.ingest_path(p)
                skipped.extend(getattr(self.service.ingestion, "skipped", []))
            return n, skipped

        worker = Worker(task)
        worker.signals.finished.connect(self._on_done)
        worker.signals.failed.connect(self._on_error)
        self.pool.start(worker)

    def _on_done(self, result: tuple[int, list[tuple[str, str]]]) -> None:
        n, skipped = result
        if skipped:
            self._busy(False, f"Indexed {n} chunks, {len(skipped)} file(s) skipped.")
        else:
            self._busy(False, f"Indexed {n} chunks.")
        self.refresh()
        self.documents_changed.emit()
        if skipped:
            lines = "\n".join(f"{name} — {reason}" for name, reason in skipped)
            QMessageBox.warning(
                self,
                "Some files were skipped",
                f"The following file(s) were skipped during ingestion:\n\n{lines}",
            )

    def _on_error(self, msg: str) -> None:
        self._busy(False, "Error.")
        QMessageBox.critical(self, "Ingest failed", msg)

    def _remove_selected(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        sources = [it.data(0, Qt.ItemDataRole.UserRole) for it in items]
        confirm = QMessageBox.question(
            self,
            "Remove from index",
            "Remove the selected document(s) from the index?\n\n"
            + "\n".join(sources),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        for s in sources:
            self.service.remove_document(s)
        self.refresh()
        self.documents_changed.emit()

    def _busy(self, on: bool, msg: str) -> None:
        self.progress.setVisible(on)
        self.btn_add_files.setEnabled(not on)
        self.btn_add_folder.setEnabled(not on)
        self.status.setText(msg)
