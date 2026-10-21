"""
server.py — MCP server assembly
===============================
Wires Config into an AppContext, builds an MCPServer, and registers the tool
surface. Uses mcp 2.x (`mcp.server.mcpserver.MCPServer`).

The server owns no engine and no queue. Synthesis belongs to the resident
worker process (worker.py); this process carries text to it. That split exists
because background CUDA work inside the SDK's process stalled after two
sentences every time, over both transports, while the same code outside a
server finished the whole reply.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from mcp.server.mcpserver import MCPServer

from mcp_agent_chatterbox import __version__
from mcp_agent_chatterbox.config import ENV_FILE, Config
from mcp_agent_chatterbox.context import AppContext
from mcp_agent_chatterbox.logging_setup import setup_logging
from mcp_agent_chatterbox.tool_surface import register_all

logger = logging.getLogger(__name__)

SERVER_NAME = "mcp-agent-chatterbox"

INSTRUCTIONS = """
Local text-to-speech with Chatterbox (Resemble AI), running on this machine's
GPU. Text in, spoken audio out.

Picking a model:
  * nano         110M, English. PREFERRED default: same architecture, tags and
                 built-in voice as turbo with competitive quality, at a third
                 of the size — tightest latency/memory budget (also runs on
                 CPU). Understands paralinguistic tags written inline in the
                 text: [laugh], [chuckle], [sigh], [cough], [whisper].
  * turbo        350M, English. nano's bigger sibling: same features, more
                 VRAM headroom. Has a built-in voice, so it works with no
                 reference clip.
  * multilingual 500M, 23 languages including German. REQUIRES a reference
                 clip (voice= or reference_clip=). Pass language="de" etc.
  * original     500M, English, the CFG/exaggeration tuning model. REQUIRES a
                 reference clip.

Reference clips: voice= / reference_clip= clone a voice on ANY model — the
stock voice of turbo/nano is only the fallback when no clip is given. The two
500M models require a clip because they ship no built-in voice.

Workflow:
  1. Make sure the resident worker is running (`mcp-agent-chatterbox worker`);
     speak() reports worker_not_running when it is not.
  2. tts_status() -> confirm the device, see free VRAM per GPU (a card with
     under ~4 GB free cannot load a model), and what the worker is doing.
     Loads nothing.
  3. list_voices() -> the reference clips you can clone from, if you need one.
  4. speak(text, model=..., voice=..., language=...) -> hands the text to the
     worker, which renders one sentence at a time and plays it as soon as it is
     written. The call returns when the whole reply has been spoken.
  5. tts_unload() -> hand the GPU back when another process needs it. Only one
     model stays resident at a time; asking for a different one evicts the
     previous automatically.

Notes:
  * Weights download from Hugging Face on the very first call to a fresh worker
    and stay loaded for the life of that worker.
  * Style controls (exaggeration, cfg_weight, temperature) are optional; leave
    them unset so each model keeps its own tuned defaults.
  * On a machine where a large language model already occupies a GPU, check
    free VRAM with tts_status() before a long run of speak() calls.
""".strip()


def build_context() -> AppContext:
    """Construct the application object graph."""
    cfg = Config()
    setup_logging(str(cfg.logs_dir))
    return AppContext(cfg=cfg)


def create_server(ctx: AppContext | None = None) -> MCPServer:
    """Build an MCPServer with the full tool surface registered."""
    if ctx is None:
        ctx = build_context()

    server = MCPServer(
        name=SERVER_NAME,
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    register_all(server, ctx)

    logger.info(
        "[server] %s v%s ready (pid=%d, ppid=%d, output=%s, voices=%s, model=%s, voice=%s, config=%s)",
        SERVER_NAME,
        __version__,
        os.getpid(),
        os.getppid(),
        ctx.cfg.output_dir,
        ctx.cfg.voices_dir,
        ctx.cfg.model,
        ctx.cfg.voice or "unset",
        ENV_FILE if ENV_FILE.is_file() else "no .env - defaults in use",
    )
    return server


#: The transports MCPServer.run() accepts, matching the SDK's own Literal
#: exactly. Declaring it here (instead of passing a bare str) means the type
#: checker validates the value handed to the SDK -- no cast required.
Transport = Literal["stdio", "sse", "streamable-http"]
TRANSPORTS: tuple[Transport, ...] = ("stdio", "sse", "streamable-http")


def run(transport: Transport = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Build and run the server.

    transport: "stdio" (default, what opencode uses), "sse" or
               "streamable-http". host/port apply to the network transports.

    Nothing is drained on the way out: this process no longer owns synthesis.
    A reply that is still being spoken when the harness ends the turn keeps
    playing, because the resident worker is a different process.
    """
    # Reachable only from an untyped caller (argparse hands us a str); the
    # membership test narrows to Transport for the calls below.
    if transport not in TRANSPORTS:
        raise SystemExit(f"unknown transport {transport!r}; choose one of {', '.join(TRANSPORTS)}")
    ctx = build_context()
    server = create_server(ctx)
    try:
        if transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport=transport, host=host, port=port)
    except KeyboardInterrupt:
        pass


__all__ = [
    "SERVER_NAME",
    "TRANSPORTS",
    "Transport",
    "build_context",
    "create_server",
    "run",
]
