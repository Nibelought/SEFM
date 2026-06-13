from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import Settings
from app.factory import build_app_service
from app.logging_config import install_excepthook, setup_logging
from app.ui.main_window import MainWindow


def main() -> int:
    _s = Settings()
    setup_logging(_s.log_level, _s.resolved_log_file)
    install_excepthook()

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("SEFM")
    qt_app.setOrganizationName("SEFM")

    service = build_app_service(with_llm=True)
    window = MainWindow(service)
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
