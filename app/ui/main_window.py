from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QToolButton,
)

from app.service import AppService
from app.ui.ask_view import AskView
from app.ui.library_view import LibraryView
from app.ui.search_view import SearchView
from app.ui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(self, service: AppService) -> None:
        super().__init__()
        self.service = service
        self._settings_dlg: SettingsDialog | None = None
        self.setWindowTitle("SEFM — Offline ICS/SCADA RAG Assistant")
        self.resize(1100, 740)
        self.setMinimumSize(820, 560)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.library = LibraryView(service)
        self.search = SearchView(service)
        self.ask = AskView(service)
        self.tabs.addTab(self.ask, "Ask")
        self.tabs.addTab(self.search, "Search")
        self.tabs.addTab(self.library, "Library")
        self.setCentralWidget(self.tabs)

        # Settings lives as a gear in the tab-bar corner — no near-empty menu bar.
        self.btn_settings = QToolButton(self)
        self.btn_settings.setText("⚙")
        self.btn_settings.setToolTip("Settings (Ctrl+,)")
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.clicked.connect(self._open_settings)
        self.tabs.setCornerWidget(self.btn_settings, Qt.Corner.TopRightCorner)

        self.library.documents_changed.connect(self._refresh_status)

        self.setStatusBar(QStatusBar(self))
        self._refresh_status()

        # Keyboard shortcuts without a menu bar to host them.
        act_settings = QAction(self)
        act_settings.setShortcut("Ctrl+,")
        act_settings.triggered.connect(self._open_settings)
        self.addAction(act_settings)
        act_quit = QAction(self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        self.addAction(act_quit)

    def _open_settings(self) -> None:
        if self._settings_dlg is None or not self._settings_dlg.isVisible():
            self._settings_dlg = SettingsDialog(self.service, parent=self)
        self._settings_dlg.show()
        self._settings_dlg.raise_()

    def _refresh_status(self) -> None:
        docs = self.service.list_documents()
        total = sum(docs.values())
        self.statusBar().showMessage(
            f"{len(docs)} document(s), {total} chunk(s) indexed | LLM: "
            f"{'loaded' if self.service.llm else 'not configured'}"
        )
