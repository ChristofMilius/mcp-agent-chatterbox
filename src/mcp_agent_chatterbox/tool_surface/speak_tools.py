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
  * model="nano" is turbo's 110M sibling — same architecture, tags and
    built-in voice, tightest latency/memory budget (also CPU-capable).
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
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        norm_loudness: bool | None = None,
        seed: int | None = None,
        play_audio: bool | None = None,
        wait: bool = False,
        progressive: bool | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Speak text aloud with Chatterbox on the local GPU, save it as a WAV, and
        by default play it through the system audio device.

        model:   "turbo" (default, 350M, English, no reference clip needed,
                 supports inline [laugh]/[chuckle] tags), "nano" (110M,
                 turbo's smaller sibling, built-in voice, tightest budget),
                 "multilingual" (500M, 23 languages, requires a reference clip)
                 or "original" (500M, English, requires a reference clip).
        voice:   name of a reference clip in the voices directory (its filename
                 without extension). Required for multilingual/original.
        reference_clip: path to a wav/mp3/flac clip, as an alternative to voice=.
                 Supply as much clean continuous speech as you have. Short
                 references measurably increase fine clicks and crackle: a 40 s
                 reference beat an 11.8 s excerpt of the same recording by a
                 wide margin, several times the model's own run-to-run variance.
                 Under ~5 s is genuinely too little. Do not normalise or limit
                 the clip -- that shifts output level and adds artefacts. A
                 mastered or compressed recording is fine. Sample rate and
                 channel count need not match anything.
        language: ISO 639-1 code, multilingual only (e.g. "de", "en", "fr").
        t3_model: "v2" (default) or "v3" — multilingual checkpoint.
        temperature: sampling temperature (0.05-5.0). Honored by all models.
        top_p: nucleus-sampling cutoff (0.0-1.0). Honored by all models.
        top_k: top-k sampling size (0-1000). Honored by turbo/nano only — the
                 500M models have no such parameter.
        repetition_penalty: penalise repeated tokens (1.0-2.0). Honored by all
                 models.
        norm_loudness: normalize output to -27 LUFS. Honored by turbo/nano only
                 — the 500M models have no such parameter.
        exaggeration/cfg_weight: style controls (0.0-2.0 / 0.0-1.0). Honored
                 ONLY by the 500M models (multilingual/original); turbo/nano
                 ignore both, so setting them with model="turbo" is dropped
                 with a warning, not an error. cfg_weight>0 doubles the text
                 tokens for CFG guidance.
        seed: reseed torch (CPU + CUDA) so re-renders are reproducible within
                 the resident session; 0 (the upstream convention) or unset
                 keeps random sampling. Byte-identical output across a server
                 restart is not guaranteed — Chatterbox is nondeterministic
                 across CUDA kernel choices.
                 Left unset, every knob keeps the model's own tuned default
                 (e.g. turbo runs cfg_weight 0.0 with top_k 1000; the 500M
                 models run cfg_weight 0.5).
        play_audio: set false to only write the file. Defaults to the server
                 setting (on).
        wait: block until playback finishes instead of returning immediately.
        progressive: for long multi-chunk text with playback on, start playing
                 each sentence-chunk the moment it is rendered instead of
                 waiting for the whole utterance to generate first. Defaults
                 to the server setting (off). Ignored otherwise — single
                 utterances and playback-off calls behave exactly as before.
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
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
                seed=seed,
                play_audio=play_audio,
                wait=wait,
                progressive=progressive,
                filename=filename,
            )
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except ToolFault as fault:
            return fault.to_json()
        except Exception as e:
            return tool_error("speak", e)


__all__ = ["register"]
