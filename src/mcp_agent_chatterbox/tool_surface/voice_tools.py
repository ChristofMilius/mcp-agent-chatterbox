"""
voice_tools — the reference-clip inventory and playback control
==============================================================
list_voices:  what reference clips exist, so the model can pick one by name
              instead of inventing a path.
stop_speech:  cut off playback of a long utterance — the audible equivalent of
              interrupting the agent mid-sentence.
"""

from __future__ import annotations

import json

from mcp_agent_chatterbox.errors import tool_error
from mcp_agent_chatterbox.playback import stop
from mcp_agent_chatterbox.registry import VOICE_SUFFIXES, describe_models

# Aliased: the MCP tool below is itself called list_voices, and an unaliased
# import would be shadowed by it.
from mcp_agent_chatterbox.voices import list_voices as _scan_voices


def register(server, ctx) -> None:
    voices_dir = ctx.cfg.voices_dir

    @server.tool()
    def list_voices() -> str:
        """
        List the reference clips available for voice cloning — the files in the
        voices directory. Returns each clip's name (what you pass as voice=),
        filename, size, duration, sample rate and channel count, plus which
        models can use it. The multilingual and original models require one of
        these; turbo does not.

        For good cloning use as much clean single-speaker audio as you have, not
        a short excerpt — clipping a good recording down measurably adds fine
        clicks and crackle. A mastered or compressed source is fine; do not
        normalise it. Sample rate and channel count need not match anything.
        Clips too short to be safe carry a quality_note.
        """
        try:
            voices = _scan_voices(voices_dir)
            return json.dumps(
                {
                    "status": "ok",
                    "voices_dir": str(voices_dir),
                    "count": len(voices),
                    "voices": voices,
                    "accepted_formats": list(VOICE_SUFFIXES),
                    "usable_with": [m["key"] for m in describe_models() if not m["stock_voice"]],
                    "note": (
                        "No reference clips yet — add a wav/mp3 to the voices "
                        "directory, or speak with model='turbo' which ships a "
                        "built-in voice."
                        if not voices
                        else "Pass the name as voice= to speak()."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:
            return tool_error("list_voices", e)

    @server.tool()
    def stop_speech() -> str:
        """
        Stop audio that is currently playing. Useful for cutting off a long
        utterance started with wait=false.
        """
        try:
            stopped, reason = stop()
            return json.dumps(
                {"status": "ok" if stopped else "failed", "stopped": stopped, "reason": reason},
                indent=2,
            )
        except Exception as e:
            return tool_error("stop_speech", e)


__all__ = ["register"]
