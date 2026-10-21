"""
logging_setup.py — Centralized logging configuration
====================================================
Configures two handlers:
  1. StreamHandler (stderr)   — INFO level, visible in the harness console.
  2. RotatingFileHandler      — DEBUG level, written to logs/mcp_agent_chatterbox.log.
                                  Rotates at 5 MB, keeps 5 backups.

Third-party chatter is pinned to WARNING (see _NOISY_LOGGERS): leftover DEBUG
output from numba's JIT compiler alone was rotating the 5 MB file every couple
of minutes and crowding out this package's own records. Warnings and errors
from those libraries still land in the file.

stderr matters: an MCP server on stdio owns stdout for the protocol, so any
library that prints to stdout (Chatterbox's from_local() does) would corrupt
the JSON-RPC framing. The console handler is on stderr for that reason.

Naming mirrors the mail / docparser / transcriber siblings so the stack tools
look alike in logs.

Call setup_logging(logs_dir) exactly once, before any other module emits
records.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FILE_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_CONSOLE_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

_LOG_FILENAME = "mcp_agent_chatterbox.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 5  # keep mcp_agent_chatterbox.log through .log.5

#: Third-party loggers that emit tens of thousands of DEBUG lines per model
#: load. numba dumps every bytecode dispatch of Chatterbox's JIT compiler:
#: a single load measured ~460,000 DEBUG lines against ~5,300 from this
#: package — the 5 MB log rotated every couple of minutes, so the records
#: worth keeping were already gone by the time anything was diagnosed.
#: Their WARNING and above still reaches the file; only the chatter is
#: suppressed, and each logger's own children (numba.core.*) inherit it.
_NOISY_LOGGERS = (
    "numba",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "asyncio",
    "matplotlib",
    "PIL",
)


def _make_console_encoding_safe() -> None:
    """
    Make the console streams survive non-ASCII characters.

    A Windows console defaults to a legacy code page (cp1252 here), so any
    character outside it raises UnicodeEncodeError — or, with a
    bytes-oriented handler, comes out as mojibake. Neither is acceptable for a
    tool whose output a human reads: Chatterbox itself logs a checkmark
    ("EOS token detected!"), and this project's own messages use em-dashes.

    errors="replace" turns an unencodable character into '?' instead of
    crashing or garbling the rest of the line. The file handler is unaffected —
    it is opened as UTF-8, where everything round-trips.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # noqa: PERF203 — best effort only
                pass


def setup_logging(logs_dir: str) -> Path:
    """
    Configure the root logger with a console handler and a rotating file handler.

    Args:
        logs_dir: Path to the directory where log files are written.
                  Created if it does not exist.

    Returns:
        The resolved path to the log file.
    """
    _make_console_encoding_safe()

    log_dir = Path(logs_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / _LOG_FILENAME

    root_logger = logging.getLogger()

    # Avoid duplicate handlers if called more than once (e.g. in tests).
    if any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        return log_file

    root_logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "[logging] Log file: %s (rotating, max %d MB x %d backups)",
        log_file,
        _MAX_BYTES // (1024 * 1024),
        _BACKUP_COUNT,
    )

    return log_file


__all__ = ["setup_logging"]
