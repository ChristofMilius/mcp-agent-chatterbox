"""
voices.py — the reference-clip registry
=======================================
Voice cloning needs a clean recording of the target speaker. Rather than have
the model hand-wave absolute paths on every call, clips are dropped into one
configured directory and referred to by their filename stem.

That keeps two things true at once: the model only ever passes a short name
around, and nothing outside the voices directory can be reached by naming it.

On how much audio to supply — measured, not assumed
---------------------------------------------------
The earlier guidance here said "5-15 s". That is wrong, and it was wrong in the
direction that costs quality. A controlled A/B on one speaker's 40.1 s
mastered recording versus an 11.8 s excerpt of that same recording, five German
texts per arm, with a same-reference re-render arm as the run-to-run noise
floor:

    impulsive discontinuities per second (adaptive Laplacian impulse detector)
      threshold    40.1s ref    11.8s ref     noise floor
      k=6  (fine)      283.6        459.0        5.1
      k=10 (moderate)  108.8        173.6        7.7
      k=20 (large)      72.5         80.1        5.2

The short reference produced ~36% more moderate and ~44% more fine impulses,
several times the model's own sampling variance, and the effect persisted when
the gain change was controlled for. Worst-spike magnitude, HF energy and
zero-crossing irregularity were indistinguishable — so the added artefact is
fine crackle, not louder pops.

So: the failure mode is too LITTLE audio, not too much. A 40 s reference beat
an 11.8 s one. We did not test 20 s, 60 s or longer, so this does not
establish an optimum — only that truncating a good recording to the "5-15 s"
guidance measurably degrades it.

Two further measured facts, both counter to the folk advice:

* A processed or mastered source is FINE. The 40 s reference used above was an
  Audacity-mastered file and was the better of the two. Compression is not the
  problem; shortness is.
* Do not normalise, gain-match or limit the reference. Applying +23 dB of peak
  normalisation to the excerpt moved rendered output level by 12.1 dB (RMS
  -23.1 vs -35.2 dBFS, against a 0.35 dB level noise floor) and the hotter
  output was measurably grainier. Leave the source level alone.

Sample rate and channel count do not need to match anything: the 40 s
reference was 48 kHz stereo and worked without preparation. Chatterbox
resamples and downmixes internally.

Caveat worth keeping in mind: this is n=5 per arm on a single speaker, and the
detector measures discontinuities, not timbre, prosody or speaker similarity.
It quantifies one artefact class, not overall quality.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.registry import VOICE_SUFFIXES

logger = logging.getLogger(__name__)

#: Below this a clip is short enough that cloning quality measurably suffers.
#: Not a rejection threshold -- a warning, and deliberately conservative: the
#: A/B test compared 11.8s against 40.1s and found the short one worse, but did
#: not locate the boundary. Anything comfortably above this is known good.
SHORT_CLIP_S = 15.0


def _probe(path: Path) -> dict[str, Any]:
    """
    Read a clip's duration, sample rate and channel count without decoding it.

    Duration is the property that actually predicts clone quality (see the
    module docstring), so it belongs in the listing rather than being something
    the caller has to go and measure. Header reads only: `wave` for PCM WAV
    (stdlib, exact) and soundfile.info for everything else (mp3/flac/ogg/m4a
    container metadata, no decode). Anything unreadable -- an exotic codec, a
    truncated file -- degrades to nulls rather than failing the listing,
    because the clip may still be usable.

    (torchaudio.info() used to serve here, but torchaudio 2.11+ moved all I/O
    to the separate torchcodec package, and a container header does not
    justify a new dependency.)
    """
    probed: dict[str, Any] = {
        "duration_s": None,
        "sample_rate": None,
        "channels": None,
    }
    try:
        import soundfile as sf  # noqa: PLC0415 -- off the hot path

        meta = sf.info(str(path))
        probed["sample_rate"] = int(meta.samplerate)
        probed["channels"] = int(meta.channels)
        if meta.samplerate:
            probed["duration_s"] = round(meta.frames / meta.samplerate, 2)
    except Exception as exc:  # noqa: BLE001 -- header probing is best effort
        try:
            with wave.open(str(path), "rb") as w:
                probed["channels"] = w.getnchannels()
                probed["sample_rate"] = w.getframerate()
                if w.getframerate():
                    probed["duration_s"] = round(w.getnframes() / w.getframerate(), 2)
        except Exception:
            logger.debug("[voices] could not probe %s: %s", path.name, exc)
    return probed


def list_voices(voices_dir: Path) -> list[dict]:
    """
    Enumerate usable reference clips in `voices_dir`, sorted by name.

    Each entry carries duration, sample rate and channel count, plus a
    `quality_note` when the clip is short enough to be worth warning about.

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
        entry: dict[str, Any] = {
            "name": path.stem,
            "filename": path.name,
            "path": str(path),
            "size_kb": size_kb,
            **_probe(path),
        }
        duration = entry.get("duration_s")
        if duration is not None and duration < SHORT_CLIP_S:
            entry["quality_note"] = (
                f"Only {duration}s long. Short references measurably increase "
                "fine clicks and crackle in the output; if a longer clean "
                "recording of this speaker exists, prefer it."
            )
        entries.append(entry)
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
            f"No voices directory at {voices_dir.name}/ yet. Drop a clean "
            "wav/mp3 of the target speaker in there -- as much continuous clean "
            "speech as you have, since short references measurably crackle, "
            "though under ~5 s is genuinely too little. Do not normalise it. "
            "Or use model='turbo', which needs no reference clip.",
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
