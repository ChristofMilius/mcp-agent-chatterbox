"""
voices.py — the reference-clip registry
=======================================
Voice cloning needs a short clean recording of the target speaker (roughly
5-15 s of speech, no music, no background noise). Rather than have the model
hand-wave absolute paths on every call, clips are dropped into one configured
directory and referred to by their filename stem.

That keeps two things true at once: the model only ever passes a short name
around, and nothing outside the voices directory can be reached by naming it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.registry import VOICE_SUFFIXES

logger = logging.getLogger(__name__)


def list_voices(voices_dir: Path) -> list[dict]:
    """
    Enumerate usable reference clips in `voices_dir`, sorted by name.

    Returns [] when the directory does not exist yet — a missing voices dir is
    a normal state for a fresh install, not an error.
    """
    if not voices_dir.is_dir():
        return []
    entries: list[dict] = []
    for path in sorted(voices_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VOICE_SUFFIXES:
            continue
        try:
            size_kb = round(path.stat().st_size / 1024, 1)
        except OSError:
            size_kb = None
        entries.append(
            {
                "name": path.stem,
                "filename": path.name,
                "path": str(path),
                "size_kb": size_kb,
            }
        )
    return entries


def resolve_voice(voices_dir: Path, name: str) -> Path:
    """
    Map a voice name to a clip inside `voices_dir`.

    Matching is case-insensitive against the filename stem, and a name given
    with its extension is accepted too. Raises ToolFault for an unknown name or
    an ambiguous prefix, so the model gets a choice it can act on rather than an
    opaque failure.
    """
    if not voices_dir.is_dir():
        raise ToolFault(
            "voices_dir_missing",
            f"No voices directory at {voices_dir.name}/ yet. Drop a 5-15 s clean "
            "wav/mp3 of the target speaker in there, or use model='turbo' which "
            "needs no reference clip.",
        )

    wanted = name.strip()
    if not wanted:
        raise ToolFault("voice_missing", "No voice name given.")

    wanted_stem = Path(wanted).stem.lower()
    exact: list[Path] = []
    partial: list[Path] = []
    for path in sorted(voices_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VOICE_SUFFIXES:
            continue
        if path.stem.lower() == wanted_stem:
            exact.append(path)
        elif path.stem.lower().startswith(wanted_stem):
            partial.append(path)

    if len(exact) == 1:
        return exact[0]
    if not exact and len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ToolFault(
            "voice_ambiguous",
            f"Voice {name!r} matches several clips: "
            + ", ".join(p.stem for p in partial)
            + ". Use the full name.",
        )

    available = [
        p.stem
        for p in sorted(voices_dir.iterdir(), key=lambda p: p.name.lower())
        if p.is_file() and p.suffix.lower() in VOICE_SUFFIXES
    ]
    raise ToolFault(
        "voice_not_found",
        f"No reference clip named {name!r} in {voices_dir.name}/."
        + (f" Available: {', '.join(available)}." if available else " The directory is empty."),
    )


__all__ = ["list_voices", "resolve_voice"]
