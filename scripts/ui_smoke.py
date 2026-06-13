"""Launch the UI, run a query against the indexed fixture, screenshot the
Library and Search tabs."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.factory import build_app_service
from app.ui.main_window import MainWindow

OUT_LIBRARY = Path("S:/SEFM/ui_library.png")
OUT_SEARCH = Path("S:/SEFM/ui_search.png")


def main() -> int:
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("SEFM")

    svc = build_app_service(with_llm=False)
    win = MainWindow(svc)
    win.resize(1200, 760)
    win.show()
    qapp.processEvents()

    def capture_library() -> None:
        win.tabs.setCurrentIndex(0)
        qapp.processEvents()
        win.grab().save(str(OUT_LIBRARY))
        print(f"saved {OUT_LIBRARY}")

    def run_query() -> None:
        win.tabs.setCurrentIndex(1)
        win.search.query.setText("what does fault 0x21 indicate")
        win.search._run()

    def capture_search() -> None:
        win.grab().save(str(OUT_SEARCH))
        print(f"saved {OUT_SEARCH}")
        qapp.quit()

    QTimer.singleShot(200, capture_library)
    QTimer.singleShot(700, run_query)
    QTimer.singleShot(12000, capture_search)
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
