"""
speak.py — the one synthesis path
=================================
Validation, model resolution, synthesis, file writing and playback, in that
order, as a single function returning a plain dict. Both entry points call it:
the MCP tool in tool_surface/speak_tools.py and the `speak` CLI subcommand.
Keeping one implementation means a bug cannot be fixed in the tool and left
broken on the command line (or vice versa).

Long text is transparently chunked. Chatterbox's own generate() truncates
large inputs (turbo degrades past roughly 600 chars and comes back shorter
than a shorter prompt; the 500M models cap their output at 1000 speech
tokens), so a single oversized render would be garbled. speak_once splits the
text into sentence-aligned chunks of at most cfg.max_chunk_chars, renders each
with the same voice and knobs, concatenates the waveforms with a short silence
between them, and writes and plays one WAV. Short text takes the original
single-render path unchanged.

Recoverable conditions — empty text, over-long text, unknown model, bad
language, missing reference clip, not enough VRAM — are raised as ToolFault so
each caller can render them its own way. Anything unexpected propagates.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from mcp_agent_chatterbox.audio import duration_s, write_wav
from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.engine import ChatterboxEngine
from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.names import slugify, speech_filename
from mcp_agent_chatterbox.playback import play, player_available
from mcp_agent_chatterbox.registry import SUPPORTED_LANGUAGES, T3_MODELS, get_spec
from mcp_agent_chatterbox.voices import resolve_voice

logger = logging.getLogger(__name__)

_CLIP_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

_SENTENCE_ENDERS = {".", "!", "?", ";", "\n"}
_CLOSING_QUOTES = {'"', "'", ")", "]", "}", "\u201d", "\u2019"}


def _sentence_ends(text: str) -> list[int]:
    """Exclusive end indices of the sentence units in `text`."""
    ends: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] in _SENTENCE_ENDERS:
            j = i + 1
            while j < n and text[j] in _CLOSING_QUOTES:
                j += 1
            ends.append(j)
            i = j
        else:
            i += 1
    if not ends or ends[-1] < n:
        ends.append(n)
    return ends


def _split_segments(text: str, max_chunk: int) -> list[str]:
    """
    Greedily pack whole sentences into segments of at most `max_chunk` chars.

    A sentence breaks at '.', '!', '?' or ';' (plus any immediately following
    closing quotes) or at a newline. A single sentence longer than the budget
    is hard-split, preferring the last space that fits so words stay intact.
    """
    if len(text) <= max_chunk:
        return [text]

    segments: list[str] = []
    start = 0
    fit = 0  # last sentence boundary that still fits the current segment
    for end in _sentence_ends(text):
        if end - start <= max_chunk:
            fit = end
            continue
        if fit > start:
            segments.append(text[start:fit])
            start = fit
        while end - start > max_chunk:
            cut = text.rfind(" ", start, start + max_chunk)
            if cut < start:
                cut = start + max_chunk
            else:
                cut += 1  # take the space with it so words stay whole
            segments.append(text[start:cut])
            start = cut
        fit = end
    if fit > start:
        segments.append(text[start:fit])
    return segments


def _merge_chunks(results: list[dict], pause_s: float) -> dict:
    """Concatenate per-chunk waveforms with a short silence between them."""
    import torch

    first = results[0]
    combined = first["wav"]
    for r in results[1:]:
        gap = torch.zeros((1, int(first["sample_rate"] * pause_s)), dtype=combined.dtype)
        combined = torch.cat([combined, gap, r["wav"]], dim=1)
    return {
        "wav": combined,
        "sample_rate": first["sample_rate"],
        "model": first["model"],
        "device": first["device"],
        "elapsed_s": round(sum(r["elapsed_s"] for r in results), 2),
        "chars": sum(r["chars"] for r in results),
        "language": first["language"],
        "reference_clip_used": first["reference_clip_used"],
        "free_vram_mb": results[-1]["free_vram_mb"],
        "chunks": len(results),
    }


def _write_chunk_wav(result: dict, out_dir: Path, final_path: Path, index: int) -> Path:
    """Stage one chunk to its own file so playback can start before merging."""
    path = out_dir / f"{final_path.stem}.progressive-{index + 1}.wav"
    write_wav(result["wav"], result["sample_rate"], path)
    return path


def _stream_chunks(chunk_paths: queue.Queue[Path | None], pause_s: float) -> None:
    """
    Play chunk paths in arrival order; each lands on the speakers as it renders.

    Runs on a dedicated thread so the synchronous per-chunk PlaySound does not
    stall synthesis of the following chunk. The staged files are removed once
    played (best effort — a failure leaves them in the output dir for review).
    """
    seen: list[Path] = []
    try:
        while True:
            path = chunk_paths.get()
            if path is None:
                break
            seen.append(path)
            played, reason = play(path, wait=True)
            if not played:
                logger.warning("[speak] progressive chunk not played: %s (%s)", path.name, reason)
                continue
            if pause_s > 0:
                time.sleep(pause_s)
    finally:
        for staged in seen:
            try:
                staged.unlink()
            except OSError:
                pass


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


#: Bounds for the sampling/style knobs — the union of the ranges the upstream
#: Gradio apps use per model (turbo caps temperature at 2.0, the 500M apps at
#: 5.0, so the wider bound wins). seed=0 matches the apps' "0 for random".
_KNOB_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (0.05, 5.0),
    "top_p": (0.0, 1.0),
    "top_k": (0.0, 1000.0),
    "repetition_penalty": (1.0, 2.0),
    "exaggeration": (0.0, 2.0),
    "cfg_weight": (0.0, 1.0),
    "seed": (0.0, float("inf")),
}


def _validate_knobs(knobs: dict[str, float | int | bool | None]) -> None:
    """Fail fast on an out-of-range knob, before any weights load."""
    for name, value in knobs.items():
        if name == "norm_loudness" or value is None:
            continue
        low, high = _KNOB_RANGES.get(name, (float("-inf"), float("inf")))
        if not (low <= value <= high):
            raise ToolFault(
                "invalid_knob",
                f"{name}={value} is outside the allowed range [{low}, {high}]. "
                "Unset it to keep the model's own tuned default.",
                knob=name,
                value=value,
                min=low,
                max=high,
            )


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
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    norm_loudness: bool | None = None,
    seed: int | None = None,
    play_audio: bool | None = None,
    wait: bool = False,
    progressive: bool | None = None,
    filename: str | None = None,
) -> dict:
    """
    Render `text`, save the WAV under the configured output dir, play it if asked.

    Returns a JSON-safe dict. Raises ToolFault for recoverable conditions.

    With `progressive` (or CHATTERBOX_PROGRESSIVE) and playback enabled, long
    text does not wait for the full synthesis: the first rendered chunk starts
    playing while later chunks are still being generated. This only applies to
    multi-chunk text; a single short utterance plays unchanged.

    Knobs left as None keep the model's own tuned default. Knobs a model does
    not support (top_k/norm_loudness are turbo-only; min_p, exaggeration and
    cfg_weight do nothing on turbo) are dropped by the engine with a warning —
    see registry.ModelSpec.honored_knobs.
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

    _validate_knobs(
        {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        }
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

    segments = _split_segments(body, cfg.max_chunk_chars)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        out_path = cfg.output_dir / f"{slugify(Path(filename).stem)}.wav"
    else:
        out_path = cfg.output_dir / speech_filename(body, spec.key)

    should_play = cfg.autoplay if play_audio is None else play_audio

    playback_q: queue.Queue[Path | None] | None = None
    player: threading.Thread | None = None
    progressive_used = False

    if len(segments) == 1:
        result = engine.synthesize(
            body,
            spec.key,
            reference_clip=clip,
            language=lang,
            t3_model=t3,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=seed,
        )
    else:
        logger.info(
            "[speak] %d chars split into %d chunks (max %d)",
            len(body),
            len(segments),
            cfg.max_chunk_chars,
        )
        use_progressive = (
            (progressive if progressive is not None else cfg.progressive)
            and should_play
            and player_available()
        )
        pause_s = cfg.chunk_pause_ms / 1000.0
        if use_progressive:
            playback_q = queue.Queue()
            player = threading.Thread(
                target=_stream_chunks, args=(playback_q, pause_s), daemon=True
            )
            player.start()
        try:
            results: list[dict] = []
            for index, seg in enumerate(segments):
                r = engine.synthesize(
                    seg.strip(),
                    spec.key,
                    reference_clip=clip,
                    language=lang,
                    t3_model=t3,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    norm_loudness=norm_loudness,
                    seed=seed,
                )
                results.append(r)
                if playback_q is not None:
                    playback_q.put(_write_chunk_wav(r, cfg.output_dir, out_path, index))
        finally:
            if playback_q is not None:
                playback_q.put(None)
        result = _merge_chunks(results, pause_s)
        result["chars"] = len(body)
        progressive_used = player is not None

    write_wav(result["wav"], result["sample_rate"], out_path)

    played, play_reason = (False, "playback disabled")
    if should_play:
        if progressive_used:
            if wait and player is not None:
                player.join(timeout=180)
            played = True
        else:
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
        "progressive": progressive_used,
        "free_vram_mb": result["free_vram_mb"],
    }
    if "chunks" in result:
        payload["chunks"] = result["chunks"]
    if play_reason and not played:
        payload["playback_note"] = play_reason
    return payload


__all__ = ["speak_once"]
