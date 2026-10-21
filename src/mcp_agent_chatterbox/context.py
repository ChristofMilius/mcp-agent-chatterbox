"""
context.py — application object graph
=====================================
A single container holding the live instances the tool surface needs. Tools
receive this context at registration time (via closures) instead of reading
module globals — this keeps the tool surface testable without a full startup
bootstrap and makes the wiring explicit.

Only the configuration lives here. The synthesis worker is a separate process
(worker.py), so the MCP server holds no engine and no queue: the one thing the
tool surface needs from this process is where the worker listens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_agent_chatterbox.config import Config


@dataclass(frozen=True)
class AppContext:
    cfg: Config


__all__ = ["AppContext"]
