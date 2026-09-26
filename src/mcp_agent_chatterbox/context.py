"""
context.py — application object graph
=====================================
A single container holding the live instances the tool surface needs. Tools
receive this context at registration time (via closures) instead of reading
module globals — this keeps the tool surface testable without a full startup
bootstrap and makes the wiring explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.engine import ChatterboxEngine


@dataclass(frozen=True)
class AppContext:
    cfg: Config
    engine: ChatterboxEngine


__all__ = ["AppContext"]
