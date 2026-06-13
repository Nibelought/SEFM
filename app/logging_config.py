from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger. Console at `level`; file always DEBUG,
    rotating 5 MB x 3. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter("%(levelname)-8s  %(name)s  %(message)s")
    fmt_file = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(fmt_console)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        root.addHandler(fh)

    # Quiet noisy third-party loggers.
    for lib in (
        "chromadb",
        "sentence_transformers",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "pymupdf",
        "pdfplumber",
        "pdfminer",
        "bm25s",
        "PIL",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


@contextmanager
def log_duration(logger: logging.Logger, label: str, level: int = logging.INFO) -> Iterator[None]:
    """Log how long the enclosed block took. Success -> "<label> ... done in
    X.XXXs" at *level*; exception -> "<label> failed after X.XXXs" at ERROR,
    re-raised. *label* is pre-formatted (no *args expansion)."""
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.error("%s failed after %.3fs", label, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - t0
        logger.log(level, "%s ... done in %.3fs", label, elapsed)


def install_excepthook() -> None:
    """Route uncaught exceptions through logging. Hooks both sys and threading
    excepthooks (so background-thread exceptions are captured);
    KeyboardInterrupt is forwarded to the original hook. Idempotent."""
    _app_logger = logging.getLogger("app")
    _original_excepthook = sys.__excepthook__

    def _excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: object) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            _original_excepthook(exc_type, exc_value, exc_tb)  # type: ignore[arg-type]
            return
        _app_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),  # type: ignore[arg-type]
        )

    def _threading_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is None or issubclass(args.exc_type, KeyboardInterrupt):
            return
        if args.exc_value is None:
            _app_logger.critical("Uncaught exception in thread %s", args.thread)
        else:
            _app_logger.critical(
                "Uncaught exception in thread %s",
                args.thread,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

    sys.excepthook = _excepthook
    threading.excepthook = _threading_excepthook
