"""Apply the theme and screenshot the Ask tab (with sample citations) plus the
other tabs, for visual review."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.factory import build_app_service
from app.retrieval.types import Citation
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

OUT = Path("S:/SEFM")


def main() -> int:
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("SEFM")
    apply_theme(qapp)

    svc = build_app_service(with_llm=False)
    win = MainWindow(svc)
    win.resize(1200, 760)
    win.show()

    # Simulate a finished answer so the citation chips are visible.
    win.ask.answer.setPlainText(
        "Fault 0x21 indicates a DC bus overvoltage condition "
        "[source: vfd_manual.pdf, p. 42]. Reduce the deceleration rate or add a "
        "braking resistor [source: vfd_manual.pdf, p. 43]."
    )
    win.ask._render_citations(
        [
            Citation(source="vfd_manual.pdf", page=42),
            Citation(source="vfd_manual.pdf", page=43),
            Citation(source="schneider_atv320_quickstart.pdf", page=7),
        ]
    )
    win.ask.status.setText("Done.")
    win.ask.speed.setText("⏱ 96 tok in 5.4s · 17.8 tok/s · first token 1.1s")
    qapp.processEvents()

    names = ["ask", "search", "library"]

    def shoot(i: int) -> None:
        win.tabs.setCurrentIndex(i)
        qapp.processEvents()
        path = OUT / f"theme_{names[i]}.png"
        win.grab().save(str(path))
        print(f"saved {path}")
        if i + 1 < len(names):
            QTimer.singleShot(300, lambda: shoot(i + 1))
        else:
            qapp.quit()

    QTimer.singleShot(300, lambda: shoot(0))
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
