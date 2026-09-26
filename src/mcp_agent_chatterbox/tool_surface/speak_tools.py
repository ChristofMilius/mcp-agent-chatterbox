"""
speak_tools — the main MCP call
===============================
speak: render text to speech, save the WAV, optionally play it. This is the
       one tool that loads weights, and the only one whose first call on a fresh
       machine waits for a Hugging Face download.

The tool is a thin shell over speak.speak_once(), which the CLI shares. All it
       adds is JSON framing and the error boundary.

Model selection:
  * model="turbo" (default) speaks with no reference clip and understands
    paralinguistic tags inline in the text ([laugh], [chuckle], ...).
  * model="multilingual" needs voice=/reference_clip= and accepts language=
    as an ISO 639-1 code.
  * model="original" needs voice=/reference_clip= and is the English
    CFG/exaggeration-tuning model.
"""

from __future__ import annotations

import json

from mcp_agent_chatterbox.errors import ToolFault, tool_error
from mcp_agent_chatterbox.speak import speak_once


def register(server, ctx) -> None:
    cfg = ctx.cfg
    engine = ctx.engine

    @server.tool()
    def speak(
        text: str,
        model: str | None = None,
        voice: str | None = None,
        reference_clip: str | None = None,
        language: str | None = None,
        t3_model: str | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        play_audio: bool | None = None,
        wait: bool = False,
        filename: str | None = None,
    ) -> str:
        """
        Speak text aloud with Chatterbox on the local GPU, save it as a WAV, and
        by default play it through the system audio device.

        model:   "turbo" (default, 350M, English, no reference clip needed,
                 supports inline [laugh]/[chuckle] tags), "multilingual" (500M,
                 23 languages, requires a reference clip) or "original" (500M,
                 English, requires a reference clip).
        voice:   name of a reference clip in the voices directory (its filename
                 without extension). Required for multilingual/original.
        reference_clip: path to a wav/mp3/flac clip, as an alternative to voice=.
                 For best cloning quality use 5-15 s of clean speech.
        language: ISO 639-1 code, multilingual only (e.g. "de", "en", "fr").
        t3_model: "v2" (default) or "v3" — multilingual checkpoint.
        exaggeration/cfg_weight/temperature: style controls. Left unset, each
                 model keeps its own tuned defaults (turbo runs cfg_weight 0.0,
                 the 500M models 0.5).
        play_audio: set false to only write the file. Defaults to the server
                 setting (on).
        wait: block until playback finishes instead of returning immediately.
        filename: output basename; a safe name is generated when omitted.

        The first call downloads the model from Hugging Face (hundreds of MB to
        a few GB) and can take a while; later calls reuse the cached weights.
        Returns JSON with the output filename, duration, sample rate, model,
        device and whether it played.
        """
        try:
            payload = speak_once(
                text,
                cfg=cfg,
                engine=engine,
                model=model,
                voice=voice,
                reference_clip=reference_clip,
                language=language,
                t3_model=t3_model,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                play_audio=play_audio,
                wait=wait,
                filename=filename,
            )
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except ToolFault as fault:
            return fault.to_json()
        except Exception as e:
            return tool_error("speak", e)


__all__ = ["register"]
