"""
mcp_agent_chatterbox — MCP server for local GPU-accelerated text-to-speech
===========================================================================
Wraps Resemble AI's Chatterbox family (MIT) as an agent-facing tool surface.

Three models, chosen per call:

  * turbo        350M, English, native paralinguistic tags ([laugh],
                 [chuckle], ...). Lowest VRAM and lowest latency, and the
                 only model that ships a stock voice — it can speak with no
                 reference clip at all.
  * multilingual 500M, 23 languages including German. Voice cloning is
                 mandatory; there is no stock voice.
  * original     500M, English, the CFG/exaggeration-tuning model. Voice
                 cloning is mandatory.

Design notes worth knowing:

  * Weights are never loaded at server startup. A chatterbox server must boot
    in milliseconds so that status/voice listing stay responsive; the model
    downloads and loads on the first speak() call and stays resident.

  * Exactly one model is resident at a time (see MAX_RESIDENT). On a 12 GB
    card that is the difference between working and a CUDA OOM. Switching
    models evicts the previous one and empties the allocator cache.

  * The device string is handed straight to Chatterbox's from_pretrained()
    and never routed through its .to() method. That method is incomplete
    upstream: in ChatterboxTTS and ChatterboxMultilingualTTS it moves only
    `t3` and `gen`, leaving `ve`, `s3gen` and `conds` on the old device and
    never updating `self.device` (so generate() would later move tensors to
    the wrong place); and in ChatterboxTurboTTS it iterates a `gen`
    attribute that __init__ never sets, so it raises AttributeError. Loading
    straight onto the target device is the only correct path.

  * The CUDA wheel index is pinned in pyproject.toml, so "GPU accelerated"
    is a property of the environment rather than a platform accident.
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    """Console entry point (`mcp-agent-chatterbox`). Dispatches to the CLI."""
    from mcp_agent_chatterbox.cli import main as _cli_main

    return _cli_main()


__all__ = ["__version__", "main"]
