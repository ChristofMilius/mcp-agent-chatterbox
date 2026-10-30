"""
playback.py — play a rendered WAV through the local audio device
================================================================
The MCP server runs as a subprocess with no window of its own, so playback
goes through the Win32 wave API (winsound, stdlib) rather than a shell-out to a
player binary: no PATH dependency, no quoting hazards from a model-supplied
path, and no extra wheel to install.

Why a blocking play is started asynchronously
---------------------------------------------
PlaySound keeps its playback state *per thread*: a stop issued from a different
thread is a no-op on this host (measured — see the note on `stop()`), so only
the thread that started a sound can cut it short. A plain `SND_SYNC` call hands
that thread to Windows for the length of the utterance, which is exactly the
situation that made a long chunk uncancellable.

So `wait=True` starts the sound with `SND_ASYNC` and then blocks *in this
thread*, polling the stop flag in short slices. The wait the caller asked for is
unchanged; the thread now stays able to stop its own sound.

Non-Windows hosts degrade honestly: play() reports that no player is
available instead of pretending it succeeded.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

#: How long a blocking play sleeps between checks of the stop flag. This is the
#: worst-case delay between pressing Stop and the audio ending.
POLL_INTERVAL_S = 0.02

#: Set by stop(), read by the thread inside a blocking play().
_stop_requested = threading.Event()


def player_available() -> bool:
    """True when this host has a way to play audio synchronously or not."""
    return sys.platform == "win32"


def _wav_duration_s(path: str | Path) -> float:
    """
    Length of a PCM WAV in seconds, or 0.0 when the header cannot be read.

    A blocking play needs to know when to stop waiting; the header is the only
    cheap source for that and costs one file read.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        logger.warning("[playback] cannot read WAV header of %s: %s", Path(path).name, exc)
        return 0.0
    return round(frames / rate, 3) if rate > 0 else 0.0


def _purge(winsound) -> None:
    """Ask the wave device to drop this thread's current sound."""
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as exc:  # noqa: BLE001 — a failed purge must not raise
        logger.warning("[playback] purge failed: %s", exc)


def play(path: str | Path, wait: bool = False) -> tuple[bool, str | None]:
    """
    Play a WAV file.

    Returns (played, reason). `reason` is None on success and otherwise a
    short, path-free explanation ("no audio player on this platform",
    "file not found", "playback failed: ...").

    With `wait=True` the call returns once the audio has finished *or* stop()
    has been called; a stopped chunk counts as played, because cutting an
    utterance short is a normal request, not a playback failure.
    """
    target = Path(path)
    if not target.is_file():
        return False, "file not found"

    if not player_available():
        return False, f"no audio player on platform {sys.platform!r}"

    try:
        import winsound  # noqa: PLC0415
    except ImportError:
        return False, "winsound module unavailable"

    flags = winsound.SND_FILENAME | winsound.SND_NODEFAULT
    if not wait:
        try:
            winsound.PlaySound(str(target), flags | winsound.SND_ASYNC)
        except Exception as exc:  # noqa: BLE001 — any wave failure is a soft failure
            logger.warning("[playback] PlaySound failed: %s", exc)
            return False, f"playback failed: {type(exc).__name__}"
        return True, None

    duration = _wav_duration_s(target)
    if duration <= 0.0:
        # No header to time against: fall back to the blocking call and accept
        # that this one chunk cannot be cut short.
        try:
            winsound.PlaySound(str(target), flags)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[playback] PlaySound failed: %s", exc)
            return False, f"playback failed: {type(exc).__name__}"
        return True, None

    _stop_requested.clear()
    started = time.monotonic()
    try:
        winsound.PlaySound(str(target), flags | winsound.SND_ASYNC)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[playback] PlaySound failed: %s", exc)
        return False, f"playback failed: {type(exc).__name__}"

    deadline = started + duration
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return True, None
        if _stop_requested.wait(min(POLL_INTERVAL_S, remaining)):
            _purge(winsound)
            logger.info(
                "[playback] %s stopped after %.2fs of %.2fs",
                target.name,
                time.monotonic() - started,
                duration,
            )
            return True, None


def stop() -> tuple[bool, str | None]:
    """
    Stop whatever is currently playing. Used to cut off a long utterance.

    PlaySound's sound belongs to the thread that started it: `SND_PURGE` and
    `PlaySound(None, 0)` from another thread leave the sound running (measured
    on this host: a blocking 20 s play still ran its full length). So this sets
    the flag and the playing thread purges its own sound within
    `POLL_INTERVAL_S`.
    """
    if not player_available():
        return False, f"no audio player on platform {sys.platform!r}"
    try:
        import winsound  # noqa: PLC0415
    except ImportError:
        return False, "winsound module unavailable"

    _stop_requested.set()
    _purge(winsound)
    return True, None


__all__ = ["play", "player_available", "stop"]
