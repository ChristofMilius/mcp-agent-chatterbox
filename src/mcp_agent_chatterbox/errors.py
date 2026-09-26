"""
errors.py — safe error boundary
===============================
The single channel through which unexpected tool exceptions reach the model.

Full tracebacks contain file paths and internal state — none of which belongs
in the model's context window. tool_error() logs the full exception
server-side (exc_info=True) and returns only the tool name and exception
class name to the caller.

Known, reproducible conditions (no reference clip for a model that requires
one, unsupported language, text over the length budget, not enough free VRAM)
are raised as ToolFault instead: the domain layer decodes them into static,
path-free failure payloads the model can act on, because those are decisions
the model can make itself rather than bugs to report.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class ToolFault(Exception):
    """
    A condition the model can recover from, carrying a safe payload.

    `payload` is returned verbatim to the caller instead of a generic error
    string, so it must never contain absolute paths or secrets.
    """

    def __init__(self, reason: str, message: str, **extra: object) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.extra = extra

    def to_json(self) -> str:
        return json.dumps(
            {"status": "failed", "reason": self.reason, "message": self.message, **self.extra},
            indent=2,
            ensure_ascii=False,
        )


def tool_error(tool_name: str, exc: Exception) -> str:
    """
    Log the full exception server-side and return a safe, sanitized string.

    The model receives only the tool name and the exception class name.
    Exception messages are NOT included — they may contain file paths or
    internal state.
    """
    logger.error("[%s] unhandled exception: %s", tool_name, exc, exc_info=True)
    return f"Error in {tool_name}: {type(exc).__name__}. Check server logs for details."


__all__ = ["ToolFault", "tool_error"]
