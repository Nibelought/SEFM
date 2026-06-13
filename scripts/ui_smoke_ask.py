"""Drive the Ask tab end-to-end: type a question, wait for streaming, then
screenshot. One-shot check of the full RAG path through the UI."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.factory import build_app_service
from app.ui.main_window import MainWindow

OUT = Path("S:/SEFM/ui_ask.png")
QUESTION = "What amperage range does the JTN600 fuse holder support?"


def main() -> int:
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("SEFM")

    svc = build_app_service(with_llm=True)
    win = MainWindow(svc)
    win.resize(1200, 820)
    win.show()
    qapp.processEvents()

    win.tabs.setCurrentIndex(0)  # Ask tab
    ask = win.ask
    ask.query.setText(QUESTION)

    def finalize() -> None:
        qapp.processEvents()
        win.grab().save(str(OUT))
        print(f"saved {OUT}")
        QTimer.singleShot(50, qapp.quit)

    def watch_for_finish() -> None:
        # The Ask button re-enables when the worker finishes (hard cap below).
        if ask._worker is None and ask.btn_ask.isEnabled():
            QTimer.singleShot(300, finalize)
        else:
            QTimer.singleShot(500, watch_for_finish)

    QTimer.singleShot(400, ask._ask)
    QTimer.singleShot(2000, watch_for_finish)
    QTimer.singleShot(300_000, finalize)  # 5 min hard cap

    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
