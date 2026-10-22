"""
speak.py — queued synthesis and playback
========================================
The one synthesis path. Both entry points use it: the resident worker
(worker.py) and the `speak` CLI subcommand. Keeping one implementation means a
bug cannot be fixed in the worker and left broken on the command line (or vice
versa).

How a call flows
----------------
submit() validates the request (text budget, model, language, reference clip,
knob ranges), splits the text into sentence chunks and puts the job on a
process-local FIFO. It returns as soon as the text is accepted — the caller gets
the message id and the chunk filenames, not rendered audio.

A worker thread then, for each chunk in order: renders it with the same voice
and knobs, writes it as ``<message_id>_<n>.wav`` and plays it in order. A second
thread keeps a small render-ahead buffer filled, so the next chunk is already
on disk when the current one ends; ``CHATTERBOX_CHUNK_PAUSE_MS`` is the pause
between chunks. The result is speech that starts within one sentence and
continues without gaps while the rest of the reply is still being generated.

Who waits, and who does not
---------------------------
submit() returns immediately and the FIFO does the long work, because the module
that drives it may not be able to wait. A caller that holds the FIFO without
waiting must arrange to outlive the work, which is the whole reason the worker
is a separate long-lived process rather than a thread inside the MCP server
(see worker.py). The two callers make opposite choices: the CLI's `speak`
subcommand calls wait() afterwards and blocks until the audio has been spoken —
no one aborts a terminal command, and the operator wants the summary — while
the worker answers with the acceptance payload and lets the utterance continue
in the background, because a tool call that waited would be cut off by a
harness with a client-side tool-call timeout (LM Studio). drain() is the exit
path that makes not-waiting safe for the worker: serve() runs it on the way
out, so accepted speech is finished rather than discarded.

Why a queue and not an in-request loop
--------------------------------------
Chatterbox's generate() truncates large inputs (turbo degrades past roughly 600
chars and comes back shorter than a shorter prompt; the 500M models cap their
output at 1000 speech tokens), so long text must be split. Splitting per
sentence in an in-request loop would make one call last as long as the whole
reply takes to render. The FIFO lets the render happen where it cannot be
touched by a client timeout, and it means two utterances arriving back to back
never overlap: the second waits for the first to finish speaking.

Recoverable conditions — empty text, over-long text, unknown model, bad
language, missing reference clip — are raised as ToolFault by submit(), before
anything is queued, so the caller can render them its own way. Failures during
rendering (VRAM, a failed generate) are recorded in the last-result summary and
reported back through the worker's status and speak responses.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Protocol

from mcp_agent_chatterbox.audio import duration_s, write_wav
from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.names import slugify
from mcp_agent_chatterbox.playback import play, player_available
from mcp_agent_chatterbox.playback import stop as stop_playback
from mcp_agent_chatterbox.registry import (
    SUPPORTED_LANGUAGES,
    T3_MODELS,
    get_spec,
    tags_for,
)
from mcp_agent_chatterbox.voices import resolve_voice

logger = logging.getLogger(__name__)

#: Speech rate used to estimate how long a chunk will take to SPEAK, measured on
#: this stack (nano/turbo, English, default voice): 1503 chars produced 85.8s of
#: audio, i.e. ~17.5 characters per second. Only ever used to compare one chunk
#: against another, so a rate that is off by 20% changes chunk sizes, not
#: correctness.
CHARS_PER_SECOND = 17.5

#: Rendering runs at ~0.42x realtime, i.e. a chunk renders in about 40% of the
#: time it takes to play. A chunk may therefore grow to roughly 2.4x its
#: predecessor (0.42 * 2.4 = 1.0) and still be ready before that predecessor's
#: playback ends. 2.2 keeps a little of that margin: the next render finishes at
#: ~0.92 of the previous playback, so the buffer creeps ahead instead of racing
#: the player. Going lower would break batching for evenly sized sentences, where
#: two sentences are exactly 2x one and would just miss the limit.
MAX_CHUNK_GROWTH = 2.2

#: Floor for that growth rule. Without it, an opening sentence like "Okay."
#: (0.2s) would cap the next chunk at 0.4s and force one chunk per sentence for
#: the rest of the reply.
MIN_CHUNK_SECONDS = 4.0

#: Audio that must be banked before playback starts. Covers an opening sentence
#: so short that the second chunk cannot finish rendering while it plays.
PREROLL_SECONDS = 2.0

#: Rendered chunks held ready ahead of the player. Only needs to absorb jitter;
#: the sizing rule above is what guarantees the player stays fed.
BUFFER_CHUNKS = 3


#: Every Speaker that has been constructed and not yet collected. The worker
#: thread is non-daemon (so real speech survives a per-turn harness), which
#: means a Speaker that is never stopped pins the interpreter. Tests and
#: embedders use stop_all_speakers() to release them; the long-lived server
#: registers drain() with atexit instead. Weak so this never keeps a Speaker
#: alive by itself.
_LIVE_SPEAKERS: weakref.WeakSet[Speaker] = weakref.WeakSet()

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

#: Characters that may form a punctuation run ('...', '?!', '!!!'). The run is
#: classified as a whole — see _sentence_ends.
_PUNCT_RUN = {".", "!", "?", "\u2026"}

#: What a bracketed token may contain to count as a tag attempt ([soft, contented
#: hum], [chuckle], [x]). Bracketed text with digits, code or nested brackets is
#: deliberately not matched — it is prose-like noise the tokenizer will speak.
_TAG_MATCH = re.compile(r"\[[A-Za-z][A-Za-z ,'\u2019-]*\]")


def _drop_unknown_tags(text: str, model_key: str) -> tuple[str, int]:
    """
    Canonicalize engine tags, drop every other bracketed token.

    Only tokens from the model's own vocabulary are real to the tokenizer
    (registry.tags_for); anything else in brackets is passed through as
    literal words, so caller-invented tags ([soft, contented hum]) — and tags
    from the WRONG model's vocabulary, like [whisper] on nano — used to be
    read out loud and mangle the audio. Known tags in the wrong case are
    lower-cased (or canonicalized); unknown ones are removed and the resulting
    double spaces collapsed.

    The tag sets differ per model: turbo/nano know [whispering] and
    '[clear throat]', the 500M models know [whisper] and [music] — see
    registry tags_for/TURBO_TAGS/ORIGINAL_TAGS.

    Returns (cleaned text, how many tokens were removed). The count is all
    that may ever be logged — the token text itself can carry conversation
    content, and conversations do not belong in the log.
    """
    known = tags_for(model_key)
    dropped = 0

    def _review(match: re.Match) -> str:
        nonlocal dropped
        token = match.group(0)
        canonical = token.lower()
        if canonical in known:
            return canonical
        dropped += 1
        return ""

    cleaned = _TAG_MATCH.sub(_review, text)
    if dropped:
        cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned, dropped


def _is_pause_run(run: str) -> bool:
    """True when a punctuation run is a written pause, not a sentence end."""
    return run.count(".") >= 2 or "\u2026" in run


def _sentence_ends(text: str) -> list[int]:
    """
    Exclusive end indices of the sentence units in `text`.

    A single '.', '!', '?' or ';' (plus any immediately following closing
    quotes) or a newline ends a sentence. A maximal run of '.', '!', '?' and
    '…' is classified as a whole: a run containing dots or the ellipsis glyph
    is a WRITTEN PAUSE and ends nothing, while a dotless run ('?!', '!!!')
    ends the sentence once.

    The pause rule is what keeps the speech smooth: the model writes its
    pauses as literal '...' mid-sentence ('Your report is... complete.'),
    and treating every dot in that run as its own terminator split the text
    into fragment pairs like 'Your report is. . .' plus a lonely
    'complete.' — dot-initial chunks the engine renders as near-silence, so
    the words after the pause dropped out of the audio and every boundary
    added a stutter. Inside one sentence the pause is simply part of the
    prosody the engine renders; measured on this stack (nano, 2026-10-22),
    a word-initial chunk containing '...' delivers its full length.
    """
    ends: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c in _PUNCT_RUN:
            start = i
            while i < n and text[i] in _PUNCT_RUN:
                i += 1
            run = text[start:i]
            if _is_pause_run(run):
                # Dots/ellipsis inside the sentence: keep scanning. Nothing
                # terminates here, so the run stays with its sentence.
                continue
            # A dotless run ('Wow!!!', 'What?!') ends the sentence once, at
            # the whole run — never as lonely '!!' fragments between ends.
            while i < n and text[i] in _CLOSING_QUOTES:
                i += 1
            ends.append(i)
        elif c in _SENTENCE_ENDERS:  # ';' or '\n': plain, single-character
            ends.append(i + 1)
            i += 1
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


def _speech_seconds(text: str) -> float:
    """Estimated spoken length of `text`, from the measured speech rate."""
    return len(text) / CHARS_PER_SECOND


def split_sentences(text: str, max_chunk: int, sentences_per_chunk: int = 1) -> list[str]:
    """
    Split `text` into chunks of whole sentences.

    A sentence breaks at '.', '!', '?' or ';' (plus any immediately following
    closing quotes) or at a newline. A written pause — '...', '…', or any dot
    run inside a sentence — does NOT break the sentence: it stays inside as
    prosody (see `_sentence_ends`). Consecutive '!!!' or '?!' end the sentence
    once rather than producing empty fragments between them. `sentences_per_chunk` consecutive sentences
    are batched into one chunk when they fit, because every chunk is a separate
    generate() call and each boundary costs a pause plus a fresh prosody start —
    batching keeps short sentences from sounding staccato. The batch is capped by
    `max_chunk` characters as well as by the sentence count, since a single
    generate() call truncates past roughly that size.

    It is also capped by *estimated speaking time* against the chunk before it
    (see `_group_sentences`): the worker renders chunk N+1 while chunk N is
    playing, so a chunk that needs longer to render than its predecessor takes
    to play leaves the player silent. Sizing by duration instead of sentence
    count is what keeps the ring buffer fed.

    The FIRST sentence always goes out alone, so audio starts as soon as one
    sentence is rendered rather than after a whole batch. Only a chunk that is
    still longer than `max_chunk` is split further, on word boundaries. The
    result is never empty for non-blank input.
    """
    body = (text or "").strip()
    if not body:
        return []
    units: list[str] = []
    start = 0
    for end in _sentence_ends(body):
        unit = body[start:end].strip()
        start = end
        if unit:
            units.append(unit)
    if not units:
        return []
    groups = [
        [units[0]],
        *_group_sentences(
            units[1:],
            max_chunk,
            sentences_per_chunk,
            previous_seconds=_speech_seconds(units[0]),
        ),
    ]
    chunks: list[str] = []
    for group in groups:
        chunks.extend(_hard_split(" ".join(group), max_chunk))
    return chunks


def _group_sentences(
    units: list[str], max_chunk: int, per_chunk: int, previous_seconds: float = 0.0
) -> list[list[str]]:
    """
    Batch consecutive sentences into chunks of comparable speaking time.

    Three limits apply at once: the sentence count, the character cap, and a
    duration limit derived from the PREVIOUS chunk. The last one is what makes
    the render pipeline self-sustaining: rendering costs roughly 0.42 of the
    audio it produces, so a chunk at most ~2.2x its predecessor always finishes
    inside that predecessor's playback. A single sentence over the limit still
    stands alone — it cannot be split without cutting mid-phrase.
    """
    if per_chunk <= 1:
        return [[unit] for unit in units]
    groups: list[list[str]] = []
    current: list[str] = []
    length = 0
    seconds = 0.0
    limit = _duration_limit(previous_seconds)
    for unit in units:
        unit_seconds = _speech_seconds(unit)
        added = len(unit) + (1 if current else 0)
        too_big = (
            len(current) >= per_chunk
            or length + added > max_chunk
            or seconds + unit_seconds > limit
        )
        if current and too_big:
            groups.append(current)
            limit = _duration_limit(seconds)
            current, length, seconds = [], 0, 0.0
            added = len(unit)
        current.append(unit)
        length += added
        seconds += unit_seconds
    if current:
        groups.append(current)
    return groups


def _duration_limit(previous_seconds: float) -> float:
    """Longest a chunk may be, given how long the previous one plays."""
    if previous_seconds <= 0:
        return float("inf")
    return max(previous_seconds * MAX_CHUNK_GROWTH, MIN_CHUNK_SECONDS)


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


class _Rendered(NamedTuple):
    """One chunk written to disk and ready to play."""

    #: 1-based chunk number, matching the filename suffix.
    number: int
    path: Path
    #: Measured spoken length of the written WAV, in seconds.
    seconds: float
    #: When the render finished (time.monotonic), for measuring dry time.
    ready_at: float


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
    #: Seconds from job start to the first audible chunk.
    first_audio_s: float | None = None
    #: Silence at each boundary the player could not avoid, in seconds.
    gaps: list[float] = field(default_factory=list)

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
            "first_audio_s": self.first_audio_s,
            "max_gap_s": max(self.gaps, default=None),
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
        # None is the shutdown sentinel put by drain(); it is why this is not
        # Queue[_Job]. The worker treats it as "queue empty, stop looping".
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._guard = threading.Lock()
        self._idle = threading.Condition(self._guard)
        self._pending = 0  # queued + currently rendering jobs
        self._current: _Job | None = None
        self._live: set[_Job] = set()
        self._last: dict | None = None
        self._last_id = ""
        _LIVE_SPEAKERS.add(self)

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
        queue's own worker thread; this call itself returns in milliseconds.
        The caller decides whether to wait for the audio (the worker and the CLI
        both do, via wait()).
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

        body, unknown_tags = _drop_unknown_tags(body, spec.key)
        if unknown_tags:
            logger.info(
                "[speak] %d invented bracket tag(s) dropped before splitting",
                unknown_tags,
            )
        chunks = split_sentences(body, cfg.max_chunk_chars, cfg.sentences_per_chunk)
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

    def teardown_report(self) -> None:
        """
        Log what the worker was doing when the process went away.

        Diagnostic-only, run from atexit *after* drain(). It is idle now in the
        healthy case; anything still outstanding here means the drain timed out
        or was never registered, which is the failure that produced a "queued"
        result with no files on disk.
        """
        with self._idle:
            pending = self._pending
            active = self._current
            worker = self._worker
            active_id = active.message_id if active is not None else None
            written = list(active.written) if active is not None else []
            total = len(active.chunks) if active is not None else 0
        if pending == 0 and active_id is None:
            logger.info("[speak] process exiting with an idle queue")
            return
        logger.error(
            "[speak] process exiting with work outstanding: pending=%d, "
            "active=%s, chunks_written=%d/%d, worker_alive=%s -- drain() did not "
            "clear the queue, so these chunks were never written",
            pending,
            active_id,
            len(written),
            total,
            worker.is_alive() if worker is not None else None,
        )

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
            if job is None:
                continue  # shutdown sentinel, not a job
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
            # NOT a daemon thread. A daemon worker is destroyed the instant the
            # process exits, and a stdio harness ends the process when the turn
            # is over -- measured directly: speak() returns "queued", the client
            # closes stdin, the server exits with code 0, and the worker is
            # killed mid-chunk with 0 of 13 files written. A non-daemon worker
            # keeps the process alive until the queue drains (see drain()).
            self._worker = threading.Thread(
                target=self._run, name="chatterbox-speaker", daemon=False
            )
            self._worker.start()

    def stop_worker(self) -> None:
        """
        Stop the worker thread, if one is running.

        The worker is non-daemon so that real speech survives a per-turn harness
        (see _ensure_worker). The cost of that choice is that the thread must be
        stopped explicitly or the interpreter will not exit, so anything that
        creates a Speaker and is not the long-lived server -- tests, embedders,
        a CLI one-shot -- should call this when it is done.

        Safe to call repeatedly and safe when no worker ever started. Any job
        still queued is left queued; call wait()/drain() first if it matters.
        """
        with self._guard:
            worker = self._worker
            self._worker = None
        if worker is None or not worker.is_alive():
            return
        self._queue.put(None)
        worker.join(timeout=5.0)

    def drain(self, timeout: float) -> None:
        """
        Wait for the queue to empty and stop the worker, bounded by `timeout`.

        Called by server.run() once the transport loop has returned -- i.e. when
        the client has disconnected and the turn is over -- so the worker can
        finish the speech it accepted instead of being killed mid-chunk. It has
        to run there rather than in atexit because atexit fires during
        interpreter shutdown, and a lazily imported dependency (torch) then
        raises "can't register atexit after shutdown"; the working chunk was
        then unreachable and the process hung.

        Bounded on purpose: a wedged engine must not keep the process alive
        forever, so on timeout the loss is logged and the worker is still
        stopped, because a non-daemon thread left running would pin the
        interpreter.
        """
        finished = self.wait(timeout)
        if not finished:
            logger.error(
                "[speak] drain timed out after %.1fs with work still queued; "
                "some chunks were not written",
                timeout,
            )
        # Stop the worker either way. It is non-daemon, so leaving it running
        # would pin the interpreter and turn this into a hang instead of a
        # logged, bounded loss.
        self.stop_worker()
        if finished:
            logger.info("[speak] queue drained cleanly before exit")

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                # Sentinel from drain(): the queue is empty and the process is
                # exiting, so leave the loop and let the interpreter finish.
                return
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
        if job.should_play and player_available():
            self._render_and_play(job, total)
        else:
            self._render_only(job, total)
        if not job.cancelled.is_set():
            first = f"{job.first_audio_s:.2f}s" if job.first_audio_s is not None else "n/a"
            gap = f"{max(job.gaps):.2f}s" if job.gaps else "none"
            logger.info(
                "[speak] %s done: %d/%d chunks, %.2fs synthesis, %.2fs audio, played %d, "
                "first audio %s, worst gap %s",
                job.message_id,
                len(job.written),
                total,
                job.synthesis_s,
                job.duration_s,
                job.played_chunks,
                first,
                gap,
            )

    def _render_only(self, job: _Job, total: int) -> None:
        """Render every chunk when nothing is competing for the time."""
        for index, sentence in enumerate(job.chunks, start=1):
            if job.cancelled.is_set():
                logger.info("[speak] %s cancelled before chunk %d/%d", job.message_id, index, total)
                return
            if self._render_chunk(job, index, sentence) is None:
                continue
            if job.should_play:
                # Asked to play, but this host has no player: one error per
                # chunk, because that is one missing file per chunk.
                job.errors.append(f"chunk {index}: no audio player")

    def _render_and_play(self, job: _Job, total: int) -> None:
        """
        Render ahead into a bounded buffer and play from it.

        Rendering costs about 0.4x the time the audio takes to play, so the work
        only has to start early enough. Three things keep the player fed:

        - chunk sizes are smoothed against each other when the text is split
          (see `split_sentences`), so a chunk's render fits inside the playback
          of the one before it by construction;
        - renders run on their own thread and feed a small queue, so the surplus
          from chunks that render faster than they play is banked instead of
          being thrown away at each boundary;
        - playback waits for `PREROLL_SECONDS` of audio, which covers an opening
          sentence too short to hide the second render.

        Playback stays strictly in order. The queue is bounded, so a long reply
        cannot make the GPU run a minute ahead of the speakers, and a cancel
        discards at most `BUFFER_CHUNKS` chunks.
        """
        stop = threading.Event()
        ready: queue.Queue[_Rendered | None] = queue.Queue(maxsize=BUFFER_CHUNKS)
        producer = threading.Thread(
            target=self._fill_buffer,
            args=(job, ready, stop),
            name=f"speak-render-{job.message_id}",
            daemon=True,
        )
        producer.start()
        started = time.monotonic()
        pause_end = started
        try:
            banked, finished = self._preroll(job, ready)
            while True:
                if not banked:
                    if finished:
                        return
                    item = self._take(job, ready)
                    if item is None:
                        return
                    banked.append(item)
                rendered = banked.pop(0)
                if job.cancelled.is_set():
                    logger.info(
                        "[speak] %s cancelled before chunk %d/%d",
                        job.message_id,
                        rendered.number,
                        total,
                    )
                    return
                if job.first_audio_s is None:
                    job.first_audio_s = round(time.monotonic() - started, 2)
                else:
                    # Silence nobody asked for: the next chunk was not ready
                    # when the pause after the previous one ended.
                    job.gaps.append(round(max(0.0, rendered.ready_at - pause_end), 2))
                played, reason = play(rendered.path, wait=True)
                if not played:
                    logger.warning(
                        "[speak] %s chunk %d/%d not played: %s",
                        job.message_id,
                        rendered.number,
                        total,
                        reason,
                    )
                    job.errors.append(f"chunk {rendered.number}: {reason or 'playback failed'}")
                    pause_end = time.monotonic()
                    continue
                job.played_chunks += 1
                if job.pause_s > 0 and rendered.number < total:
                    time.sleep(job.pause_s)
                pause_end = time.monotonic()
        finally:
            stop.set()
            producer.join()

    def _fill_buffer(
        self, job: _Job, ready: queue.Queue[_Rendered | None], stop: threading.Event
    ) -> None:
        """Thread body: render every chunk in order into the buffer."""
        total = len(job.chunks)
        try:
            for index, sentence in enumerate(job.chunks, start=1):
                if stop.is_set() or job.cancelled.is_set():
                    logger.info(
                        "[speak] %s cancelled before chunk %d/%d", job.message_id, index, total
                    )
                    return
                rendered = self._render_chunk(job, index, sentence)
                if rendered is None:
                    continue
                path, seconds = rendered
                item = _Rendered(index, path, seconds, time.monotonic())
                if not self._put(ready, item, stop, job):
                    return
        finally:
            # Never leave the consumer waiting on a producer that is gone.
            self._put(ready, None, stop, job)

    def _preroll(
        self, job: _Job, ready: queue.Queue[_Rendered | None]
    ) -> tuple[list[_Rendered], bool]:
        """
        Collect rendered chunks until `PREROLL_SECONDS` of audio is banked.

        Returns the banked chunks and whether the producer is already finished,
        because the end-of-stream marker is consumed here and the caller must not
        wait for it again. What exists is returned when the reply ends first, so a
        short reply still plays; the list is empty only if nothing was rendered.
        """
        banked: list[_Rendered] = []
        seconds = 0.0
        while seconds < PREROLL_SECONDS:
            item = self._take(job, ready)
            if item is None:
                return banked, True
            banked.append(item)
            seconds += item.seconds
        return banked, False

    @staticmethod
    def _put(
        ready: queue.Queue[_Rendered | None],
        item: _Rendered | None,
        stop: threading.Event,
        job: _Job,
    ) -> bool:
        """Blocking put that gives up when the job goes away. False = gave up."""
        while True:
            try:
                ready.put(item, timeout=0.2)
                return True
            except queue.Full:
                if stop.is_set() or job.cancelled.is_set():
                    return False

    @staticmethod
    def _take(job: _Job, ready: queue.Queue[_Rendered | None]) -> _Rendered | None:
        """Next rendered chunk, or None once the producer is done or cancelled."""
        while True:
            try:
                return ready.get(timeout=0.2)
            except queue.Empty:
                if job.cancelled.is_set():
                    return None

    def _render_chunk(self, job: _Job, index: int, sentence: str) -> tuple[Path, float] | None:
        """Synthesize and write one chunk. Returns its path and length, or None."""

        total = len(job.chunks)
        logger.info(
            "[speak] %s rendering chunk %d/%d (%d chars)",
            job.message_id,
            index,
            total,
            len(sentence),
        )
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
            return None
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
            return None

        name = self._chunk_name(job, index)
        path = job.out_dir / name
        try:
            write_wav(result["wav"], result["sample_rate"], path)
        except OSError as exc:
            logger.error("[speak] %s could not write %s: %s", job.message_id, name, exc)
            job.errors.append(f"chunk {index}: write failed")
            return None

        job.written.append(name)
        job.synthesis_s += result.get("elapsed_s", 0.0)
        seconds = duration_s(result["wav"], result["sample_rate"])
        job.duration_s += seconds
        job.device = job.device or result.get("device")
        return path, seconds

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


def stop_all_speakers() -> None:
    """
    Stop every live Speaker's worker thread.

    The worker is non-daemon so that real speech survives a per-turn harness.
    The price is that an unstopped worker pins the interpreter, so test suites
    and embedders call this on the way out. The server does not: it registers
    Speaker.drain with atexit, which waits for the queue first.
    """
    for speaker in list(_LIVE_SPEAKERS):
        speaker.stop_worker()


__all__ = ["Speaker", "Synthesizer", "split_sentences", "stop_all_speakers"]
