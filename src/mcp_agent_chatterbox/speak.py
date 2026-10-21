"""
speak.py — queued synthesis and playback
========================================
The one synthesis path. Both entry points use it: the MCP tool in
tool_surface/speak_tools.py and the `speak` CLI subcommand. Keeping one
implementation means a bug cannot be fixed in the tool and left broken on the
command line (or vice versa).

How a call flows
----------------
submit() validates the request (text budget, model, language, reference clip,
knob ranges), splits the text into sentence chunks and puts the job on a
process-local FIFO. It returns immediately — the caller gets the message id and
the chunk filenames, not rendered audio.

A daemon worker then, for each chunk in order: renders it with the same voice
and knobs, writes it as ``<message_id>_<n>.wav`` and plays it the moment it is
written, sleeping ``CHATTERBOX_CHUNK_PAUSE_MS`` between chunks. The result is
speech that starts within one sentence and continues without gaps while the
rest of the reply is still being generated.

Why a queue and not an in-request loop
--------------------------------------
Chatterbox's generate() truncates large inputs (turbo degrades past roughly 600
chars and comes back shorter than a shorter prompt; the 500M models cap their
output at 1000 speech tokens), so long text must be split. Splitting per
sentence in an in-request loop still makes the tool call last as long as the
whole reply takes to render, and MCP clients time tool calls out on the order of
a minute — LM Studio's default is ~60 s. A timed-out call does not cancel the
render: the client reports failure while the server keeps synthesising, and a
restart kills the audio mid-sentence. A queue makes the call cost milliseconds
and moves the long work where a client timeout cannot touch it.

The FIFO also means two replies arriving back to back never overlap: the second
waits for the first to finish speaking.

Recoverable conditions — empty text, over-long text, unknown model, bad
language, missing reference clip — are raised as ToolFault by submit(), before
anything is queued, so the caller can render them its own way. Failures during
rendering (VRAM, a failed generate) cannot be reported to a caller that has
already been answered; they are logged and recorded in the last-result summary.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mcp_agent_chatterbox.audio import duration_s, write_wav
from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.names import slugify
from mcp_agent_chatterbox.playback import play, player_available
from mcp_agent_chatterbox.playback import stop as stop_playback
from mcp_agent_chatterbox.registry import SUPPORTED_LANGUAGES, T3_MODELS, get_spec
from mcp_agent_chatterbox.voices import resolve_voice

logger = logging.getLogger(__name__)

_CLIP_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class Synthesizer(Protocol):
    """
    The single engine method the worker needs.

    Declared as a protocol rather than a concrete ChatterboxEngine so the queue
    can be exercised with a fake engine — no weights, no GPU, no download. The
    signature mirrors ChatterboxEngine.synthesize(); if that method changes,
    pyright flags the mismatch at server.py instead of the worker silently
    passing a knob the engine no longer accepts.
    """

    def synthesize(
        self,
        text: str,
        model_key: str,
        *,
        reference_clip: str | None = None,
        language: str | None = None,
        t3_model: str | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_p: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        norm_loudness: bool | None = None,
        seed: int | None = None,
    ) -> dict: ...

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


def _hard_split(sentence: str, max_chunk: int) -> list[str]:
    """Break one over-long sentence into pieces of at most `max_chunk` chars.

    Prefers the last space that fits so words stay intact; a chunk of
    punctuation or an unbroken token is cut at the budget.
    """
    if len(sentence) <= max_chunk:
        return [sentence]
    parts: list[str] = []
    rest = sentence
    while len(rest) > max_chunk:
        cut = rest.rfind(" ", 0, max_chunk)
        if cut <= 0:
            cut = max_chunk
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return [p for p in parts if p]


def split_sentences(text: str, max_chunk: int) -> list[str]:
    """
    Split `text` into one chunk per sentence.

    A sentence breaks at '.', '!', '?' or ';' (plus any immediately following
    closing quotes) or at a newline. Only a sentence longer than `max_chunk` is
    split further, because a single generate() call truncates past roughly that
    size. The result is never empty for non-blank input.
    """
    body = (text or "").strip()
    if not body:
        return []
    chunks: list[str] = []
    start = 0
    for end in _sentence_ends(body):
        unit = body[start:end].strip()
        start = end
        if unit:
            chunks.extend(_hard_split(unit, max_chunk))
    return chunks


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


@dataclass(eq=False)
class _Job:
    """One accepted speak request, waiting on or inside the worker."""

    message_id: str
    chunks: list[str]
    model: str
    knobs: dict
    should_play: bool
    out_dir: Path
    pause_s: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    # Filled in by the worker as it goes.
    written: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    played_chunks: int = 0
    synthesis_s: float = 0.0
    duration_s: float = 0.0
    device: str | None = None

    def summary(self) -> dict:
        return {
            "message_id": self.message_id,
            "chunks": len(self.chunks),
            "written": list(self.written),
            "errors": list(self.errors),
            "played_chunks": self.played_chunks,
            "synthesis_s": round(self.synthesis_s, 2),
            "duration_s": round(self.duration_s, 2),
            "device": self.device,
            "cancelled": self.cancelled.is_set(),
        }


class Speaker:
    """
    FIFO of speak requests, rendered and played by one daemon thread.

    One instance per server (or per CLI invocation). The engine is shared with
    the rest of the process, so the worker takes no extra lock of its own —
    engine.synthesize() is already serialised and safe to call from any thread.
    """

    def __init__(self, cfg: Config, engine: Synthesizer) -> None:
        self.cfg = cfg
        self.engine = engine
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._guard = threading.Lock()
        self._idle = threading.Condition(self._guard)
        self._pending = 0  # queued + currently rendering jobs
        self._current: _Job | None = None
        self._live: set[_Job] = set()
        self._last: dict | None = None
        self._last_id = ""

    # -- public surface -------------------------------------------------------
    def submit(
        self,
        text: str,
        *,
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
        filename: str | None = None,
    ) -> dict:
        """
        Validate `text`, queue it, and return a JSON-safe acceptance payload.

        Raises ToolFault for recoverable conditions. Rendering happens on the
        worker; this returns in milliseconds.
        """
        cfg = self.cfg
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
                "[speak] language=%r ignored: only the multilingual model takes a "
                "language_id",
                lang,
            )
            lang = None
        t3 = _validate_t3_model(t3_model)
        if t3 and spec.key != "multilingual":
            t3 = None

        # A configured default voice (CHATTERBOX_VOICE) fills in when the call
        # names neither voice= nor reference_clip=. An explicit reference_clip=
        # must not trigger it: voice= and reference_clip= are mutually exclusive,
        # and the caller who brought their own clip means it.
        if not voice and not reference_clip and cfg.voice:
            voice = cfg.voice

        clip = _resolve_clip(cfg.voices_dir, voice, reference_clip)

        if clip is None and not spec.stock_voice:
            # Chatterbox raises a bare "AssertionError: Please prepare_conditionals
            # first or specify audio_prompt_path" from deep inside generate() when
            # a stock-voice-less model gets no prompt. Catch it here so the caller
            # is told the fix.
            raise ToolFault(
                "reference_clip_required",
                f"{spec.label} has no built-in voice, so it needs a reference clip to "
                f"clone from. Pass voice=<name from list_voices> or reference_clip=<path>, "
                f"or use model='turbo' which speaks without one.",
                model=spec.key,
                requires=["voice", "reference_clip"],
            )

        chunks = split_sentences(body, cfg.max_chunk_chars)
        message_id = slugify(Path(filename).stem) if filename else self._next_id()

        job = _Job(
            message_id=message_id,
            chunks=chunks,
            model=spec.key,
            knobs={
                "reference_clip": clip,
                "language": lang,
                "t3_model": t3,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "norm_loudness": norm_loudness,
                "seed": seed,
            },
            should_play=cfg.autoplay if play_audio is None else play_audio,
            out_dir=cfg.output_dir,
            pause_s=cfg.chunk_pause_ms / 1000.0,
        )
        with self._idle:
            queued_ahead = self._pending
            self._pending += 1
            self._live.add(job)

        self._ensure_worker()
        self._queue.put(job)

        logger.info(
            "[speak] %s accepted: %d chars, %d sentence chunks, %d ahead, play=%s",
            message_id,
            len(body),
            len(chunks),
            queued_ahead,
            job.should_play,
        )
        return {
            "status": "queued",
            "message_id": message_id,
            "text_chars": len(body),
            "chunks": len(chunks),
            "files": [self._chunk_name(job, i) for i in range(1, len(chunks) + 1)],
            "output_dir": str(cfg.output_dir),
            "play": job.should_play,
            "model": spec.key,
            "voice": voice,
            "queued_ahead": queued_ahead,
            "note": (
                "Synthesis runs in the background. The chunk files appear in order "
                "as each sentence is rendered and played; listen to them in filename "
                "order to replay the utterance."
            ),
        }

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every queued job has finished. True when drained."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout)

    def stop(self) -> dict:
        """
        Stop playback and drop everything still queued.

        Drains the queue and marks the active job cancelled; the worker checks
        the flag between sentences, so one unscheduled chunk is the most it can
        still start. The speaking chunk is cut immediately by the player.
        """
        drained = 0
        drained_ids: list[str] = []
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            with self._idle:
                job.cancelled.set()
                self._live.discard(job)
                self._pending -= 1
                self._last = job.summary()
                drained += 1
                drained_ids.append(job.message_id)
                self._idle.notify_all()
        with self._idle:
            current = self._current
            # Mark anything live cancelled — including a job the worker may have
            # just dequeued but not yet reported as active.
            for job in self._live:
                job.cancelled.set()
            self._idle.notify_all()
        stopped, reason = stop_playback()
        logger.info(
            "[speak] stop requested: %d queued job(s) cancelled, active=%s",
            drained,
            current.message_id if current is not None else None,
        )
        return {
            "stopped": stopped,
            "reason": reason,
            "cancelled_jobs": drained,
            "cancelled_ids": drained_ids,
            "cancelled_active": current is not None,
        }

    def status(self) -> dict:
        """Queue depth and the most recent completed job, for tts_status."""
        with self._guard:
            pending = self._pending
            active = self._current.message_id if self._current is not None else None
            last = dict(self._last) if self._last is not None else None
        return {"pending": pending, "active": active, "last": last}

    # -- worker ---------------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._guard:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run, name="chatterbox-speaker", daemon=True
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            with self._guard:
                self._current = job
            try:
                self._render(job)
            except Exception:  # noqa: BLE001 — one job must not kill the worker
                logger.exception("[speak] %s crashed the worker", job.message_id)
                job.errors.append("worker error (see log)")
            finally:
                with self._idle:
                    self._current = None
                    self._live.discard(job)
                    self._pending -= 1
                    self._last = job.summary()
                    self._idle.notify_all()

    def _render(self, job: _Job) -> None:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        total = len(job.chunks)
        for index, sentence in enumerate(job.chunks, start=1):
            if job.cancelled.is_set():
                logger.info("[speak] %s cancelled before chunk %d/%d", job.message_id, index, total)
                return
            try:
                result = self.engine.synthesize(sentence, job.model, **job.knobs)
            except ToolFault as fault:
                logger.warning(
                    "[speak] %s chunk %d/%d failed: %s - %s",
                    job.message_id,
                    index,
                    total,
                    fault.reason,
                    fault.message,
                )
                job.errors.append(f"chunk {index}: {fault.reason}")
                continue
            except Exception as exc:  # noqa: BLE001 — keep the rest of the reply
                logger.error(
                    "[speak] %s chunk %d/%d raised %s",
                    job.message_id,
                    index,
                    total,
                    type(exc).__name__,
                    exc_info=True,
                )
                job.errors.append(f"chunk {index}: {type(exc).__name__}")
                continue

            name = self._chunk_name(job, index)
            path = job.out_dir / name
            try:
                write_wav(result["wav"], result["sample_rate"], path)
            except OSError as exc:
                logger.error("[speak] %s could not write %s: %s", job.message_id, name, exc)
                job.errors.append(f"chunk {index}: write failed")
                continue

            job.written.append(name)
            job.synthesis_s += result.get("elapsed_s", 0.0)
            job.duration_s += duration_s(result["wav"], result["sample_rate"])
            job.device = job.device or result.get("device")

            if not job.should_play:
                continue
            if not player_available():
                job.errors.append(f"chunk {index}: no audio player")
                continue
            if job.cancelled.is_set():
                return
            played, reason = play(path, wait=True)
            if not played:
                logger.warning("[speak] %s chunk %d/%d not played: %s", job.message_id, index, total, reason)
                job.errors.append(f"chunk {index}: {reason or 'playback failed'}")
                continue
            job.played_chunks += 1
            if job.pause_s > 0 and index < total:
                time.sleep(job.pause_s)

        logger.info(
            "[speak] %s done: %d/%d chunks, %.2fs synthesis, %.2fs audio, played %d",
            job.message_id,
            len(job.written),
            total,
            job.synthesis_s,
            job.duration_s,
            job.played_chunks,
        )

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _chunk_name(job: _Job, index: int) -> str:
        return f"{job.message_id}_{index}.wav"

    def _next_id(self) -> str:
        """Millisecond timestamp, made unique within this process."""
        with self._guard:
            candidate = int(time.time() * 1000)
            previous = int(self._last_id) if self._last_id else 0
            value = max(candidate, previous + 1)
            self._last_id = str(value)
            return self._last_id


__all__ = ["Speaker", "Synthesizer", "split_sentences"]
