from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from app.errors import SefmError

log = logging.getLogger(__name__)


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Run a callable on QThreadPool, emitting finished(result) or
    failed(error_message)."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _Signals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            log.exception("worker task failed")
            if isinstance(e, SefmError):
                self.signals.failed.emit(str(e))
            else:
                self.signals.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.signals.finished.emit(result)
