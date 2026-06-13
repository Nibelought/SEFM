from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
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
        self.resize(1100, 720)

        self.tabs = QTabWidget(self)
        self.library = LibraryView(service)
        self.search = SearchView(service)
        self.ask = AskView(service)
        self.tabs.addTab(self.ask, "Ask")
        self.tabs.addTab(self.search, "Search")
        self.tabs.addTab(self.library, "Library")
        self.setCentralWidget(self.tabs)

        self.library.documents_changed.connect(self._refresh_status)

        self.setStatusBar(QStatusBar(self))
        self._refresh_status()

        m_file = self.menuBar().addMenu("&File")
        act_settings = QAction("&Settings...", self)
        act_settings.setShortcut("Ctrl+,")
        act_settings.triggered.connect(self._open_settings)
        m_file.addAction(act_settings)
        m_file.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

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
