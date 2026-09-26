"""
speak.py — the one synthesis path
=================================
Validation, model resolution, synthesis, file writing and playback, in that
order, as a single function returning a plain dict. Both entry points call it:
the MCP tool in tool_surface/speak_tools.py and the `speak` CLI subcommand.
Keeping one implementation means a bug cannot be fixed in the tool and left
broken on the command line (or vice versa).

Recoverable conditions — empty text, over-long text, unknown model, bad
language, missing reference clip, not enough VRAM — are raised as ToolFault so
each caller can render them its own way. Anything unexpected propagates.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp_agent_chatterbox.audio import duration_s, write_wav
from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.engine import ChatterboxEngine
from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.names import slugify, speech_filename
from mcp_agent_chatterbox.playback import play
from mcp_agent_chatterbox.registry import SUPPORTED_LANGUAGES, T3_MODELS, get_spec
from mcp_agent_chatterbox.voices import resolve_voice

logger = logging.getLogger(__name__)

_CLIP_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _validate_language(language: str | None) -> str | None:
    """Normalize and check an ISO 639-1 code against the multilingual set."""
    if language is None:
        return None
    code = language.strip().lower()
    if not code:
        return None
    if code not in SUPPORTED_LANGUAGES:
        raise ToolFault(
            "unsupported_language",
            f"{language!r} is not one of the {len(SUPPORTED_LANGUAGES)} languages the "
            "multilingual model speaks. Use an ISO 639-1 code, e.g. de, en, fr, es "
            "(call tts_status for the full list).",
            supported=sorted(SUPPORTED_LANGUAGES),
        )
    return code


def _validate_t3_model(t3_model: str | None) -> str | None:
    if t3_model is None:
        return None
    choice = t3_model.strip().lower()
    if not choice:
        return None
    if choice not in T3_MODELS:
        raise ToolFault(
            "unsupported_t3_model",
            f"t3_model must be one of {sorted(T3_MODELS)}; got {t3_model!r}.",
        )
    return choice


def _resolve_clip(voices_dir: Path, voice: str | None, reference_clip: str | None) -> str | None:
    """Turn voice=/reference_clip= into a validated path, or None."""
    if voice and reference_clip:
        raise ToolFault(
            "ambiguous_voice",
            "Pass either voice= (a name in the voices directory) or reference_clip= "
            "(an explicit path), not both.",
        )
    if voice:
        return str(resolve_voice(voices_dir, voice))
    if reference_clip:
        candidate = Path(reference_clip).expanduser()
        if not candidate.is_file():
            raise ToolFault(
                "reference_clip_not_found",
                f"No file at reference_clip {reference_clip!r}.",
            )
        if candidate.suffix.lower() not in _CLIP_SUFFIXES:
            raise ToolFault(
                "unsupported_reference_format",
                "Reference clip must be wav, mp3, flac, ogg or m4a; got "
                f"{candidate.suffix or 'no extension'}.",
            )
        return str(candidate)
    return None


def speak_once(
    text: str,
    *,
    cfg: Config,
    engine: ChatterboxEngine,
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
) -> dict:
    """
    Render `text`, save the WAV under the configured output dir, play it if asked.

    Returns a JSON-safe dict. Raises ToolFault for recoverable conditions.
    """
    body = (text or "").strip()
    if not body:
        raise ToolFault("empty_text", "No text to speak.")
    if len(body) > cfg.max_chars:
        raise ToolFault(
            "text_too_long",
            f"{len(body)} characters exceeds the {cfg.max_chars} character budget for "
            "one call. Split the text and speak it in parts.",
            chars=len(body),
            max_chars=cfg.max_chars,
        )

    spec = get_spec(model or cfg.model)
    if spec is None:
        raise ToolFault(
            "unknown_model",
            f"Unknown model {model!r}. Use turbo, multilingual or original.",
        )

    lang = _validate_language(language)
    if lang and spec.key != "multilingual":
        logger.warning(
            "[speak] language=%r ignored: only the multilingual model takes a language_id", lang
        )
        lang = None
    t3 = _validate_t3_model(t3_model)
    if t3 and spec.key != "multilingual":
        t3 = None

    clip = _resolve_clip(cfg.voices_dir, voice, reference_clip)

    if clip is None and not spec.stock_voice:
        # Chatterbox raises a bare
        # "AssertionError: Please prepare_conditionals first or specify
        # audio_prompt_path" from deep inside generate() when a stock-voice-less
        # model gets no prompt. Catch it here so the caller is told the fix.
        raise ToolFault(
            "reference_clip_required",
            f"{spec.label} has no built-in voice, so it needs a reference clip to "
            f"clone from. Pass voice=<name from list_voices> or reference_clip=<path>, "
            f"or use model='turbo' which speaks without one.",
            model=spec.key,
            requires=["voice", "reference_clip"],
        )

    result = engine.synthesize(
        body,
        spec.key,
        reference_clip=clip,
        language=lang,
        t3_model=t3,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        out_path = cfg.output_dir / f"{slugify(Path(filename).stem)}.wav"
    else:
        out_path = cfg.output_dir / speech_filename(body, spec.key)
    write_wav(result["wav"], result["sample_rate"], out_path)

    should_play = cfg.autoplay if play_audio is None else play_audio
    played, play_reason = (False, "playback disabled")
    if should_play:
        played, play_reason = play(out_path, wait=wait)

    payload = {
        "status": "ok",
        "text_chars": result["chars"],
        "model": result["model"],
        "device": result["device"],
        "sample_rate": result["sample_rate"],
        "duration_s": duration_s(result["wav"], result["sample_rate"]),
        "synthesis_s": result["elapsed_s"],
        "language": result["language"],
        "voice": voice,
        "reference_clip_used": result["reference_clip_used"],
        "filename": out_path.name,
        "output_dir": str(cfg.output_dir),
        "path": str(out_path),
        "played": played,
        "free_vram_mb": result["free_vram_mb"],
    }
    if play_reason and not played:
        payload["playback_note"] = play_reason
    return payload


__all__ = ["speak_once"]
