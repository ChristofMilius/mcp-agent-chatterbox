"""
playback.py — play a rendered WAV through the local audio device
================================================================
The MCP server runs as a subprocess with no window of its own, so playback
goes through the Win32 wave API (winsound, stdlib) rather than a shell-out to a
player binary: no PATH dependency, no quoting hazards from a model-supplied
path, and no extra wheel to install.

Winsound is synchronous by default and would stall the MCP request for the
length of the utterance, so the default is SND_ASYNC. Pass wait=True when the
caller needs the audio to have finished before it returns.

Non-Windows hosts degrade honestly: play() reports that no player is
available instead of pretending it succeeded.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def player_available() -> bool:
    """True when this host has a way to play audio synchronously or not."""
    return sys.platform == "win32"


def play(path: str | Path, wait: bool = False) -> tuple[bool, str | None]:
    """
    Play a WAV file.

    Returns (played, reason). `reason` is None on success and otherwise a
    short, path-free explanation ("no audio player on this platform",
    "file not found", "playback failed: ...").
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

    flags = winsound.SND_FILENAME
    if not wait:
        flags |= winsound.SND_ASYNC
        flags |= winsound.SND_NODEFAULT

    try:
        winsound.PlaySound(str(target), flags)
    except Exception as exc:  # noqa: BLE001 — any wave failure is a soft failure
        logger.warning("[playback] PlaySound failed: %s", exc)
        return False, f"playback failed: {type(exc).__name__}"

    return True, None


def stop() -> tuple[bool, str | None]:
    """Stop whatever is currently playing. Used to cut off a long utterance."""
    if not player_available():
        return False, f"no audio player on platform {sys.platform!r}"
    try:
        import winsound  # noqa: PLC0415
    except ImportError:
        return False, "winsound module unavailable"
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[playback] stop failed: %s", exc)
        return False, f"stop failed: {type(exc).__name__}"
    return True, None


__all__ = ["play", "player_available", "stop"]
