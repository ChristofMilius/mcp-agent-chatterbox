"""
config.py — All configuration in one place
==========================================
Loads from environment variables. No secrets involved.

Path resolution:
  - All path fields are resolved to absolute paths at construction time.
  - Relative values (CHATTERBOX_OUTPUT_DIR, ...) are resolved relative to
    the project root (the directory containing the package), NOT the process
    CWD. Agent harnesses set an unpredictable CWD when spawning MCP server
    subprocesses.

Device selection:
  CHATTERBOX_DEVICE and CHATTERBOX_GPU_INDEX together decide which accelerator
  the models land on. "auto" is the default and picks the CUDA device with the
  most free memory, which matters on a multi-GPU box where one card is often
  occupied by something else.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root — the directory containing the package (src/mcp_agent_chatterbox).
# All relative path env vars are resolved against this, never against CWD.
_BASE = Path(__file__).resolve().parent.parent.parent

#: Free VRAM (MiB) a CUDA device must have before a model load is attempted.
#: Chatterbox weights are loaded straight onto the device (see the module
#: docstring in __init__.py for why there is no CPU-staging path), so a load
#: that starts with less headroom than this fails with a bare CUDA OOM part
#: way through reading the checkpoint. We check first and say something useful.
DEFAULT_VRAM_FLOOR_MB = 4096


def _resolve_path(raw: str) -> Path:
    """
    Resolve a path string to an absolute path.

    If the value from the environment is already absolute, return it
    normalized. If relative, resolve relative to the project root (_BASE),
    not the process CWD.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return (_BASE / p).resolve()


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """
    Read a boolean from the environment. Accepts 1/0/true/false/yes/no/on/off.

    An unrecognised value logs a warning and falls back to `default` rather than
    silently resolving to False. Treating garbage as False looks harmless but
    is not: a typo in CHATTERBOX_AUTOPLAY would switch audio off with no
    indication that the setting had been read at all.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "[config] %s=%r is not a recognised boolean (use 1/0, true/false, yes/no, on/off); "
        "using the default %s",
        name,
        raw.strip(),
        default,
    )
    return default


class Config:
    def __init__(self) -> None:
        self.output_dir: Path = _resolve_path(os.getenv("CHATTERBOX_OUTPUT_DIR", "tts_output"))
        self.voices_dir: Path = _resolve_path(os.getenv("CHATTERBOX_VOICES_DIR", "voices"))
        self.logs_dir: Path = _resolve_path(os.getenv("CHATTERBOX_LOGS_DIR", "logs"))

        #: Set the Hugging Face cache root (HF_HOME). Empty or unset leaves
        #: HF's own default untouched. Relative values resolve against the
        #: project root.
        raw_hf_home = os.getenv("CHATTERBOX_HF_HOME", "").strip()
        self.hf_home: Path | None = _resolve_path(raw_hf_home) if raw_hf_home else None

        #: Default model key when a speak() call does not name one.
        self.model: str = os.getenv("CHATTERBOX_MODEL", "turbo").strip().lower()

        #: Default reference clip (name in the voices dir) when a speak()
        #: call names neither voice= nor reference_clip=. Empty means no
        #: default: each model then uses its own voice, or reports
        #: reference_clip_required if it has none. Resolved like any other
        #: voice name, so a default that does not exist fails fast with
        #: voice_not_found instead of silently rendering in another voice.
        self.voice: str = os.getenv("CHATTERBOX_VOICE", "").strip()

        #: "auto" | "cpu" | "cuda" | "cuda:N". "auto" resolves at load time.
        self.device: str = os.getenv("CHATTERBOX_DEVICE", "auto").strip().lower()

        #: Pin a specific CUDA ordinal. Overrides the free-memory heuristic
        #: used by "auto", and is the setting to reach for when you know which
        #: card LM Studio is not sitting on.
        self.gpu_index: int | None = (
            _env_int("CHATTERBOX_GPU_INDEX", -1) if os.getenv("CHATTERBOX_GPU_INDEX") else None
        )
        if self.gpu_index is not None and self.gpu_index < 0:
            self.gpu_index = None

        self.vram_floor_mb: int = _env_int("CHATTERBOX_VRAM_MB", DEFAULT_VRAM_FLOOR_MB)
        self.strict_vram: bool = _env_bool("CHATTERBOX_STRICT_VRAM", False)
        self.autoplay: bool = _env_bool("CHATTERBOX_AUTOPLAY", True)

        #: When text is long enough to need chunking and playback is on, start
        #: playing each chunk the moment it is rendered instead of waiting for
        #: the whole utterance to synthesise first.
        self.progressive: bool = _env_bool("CHATTERBOX_PROGRESSIVE", False)

        #: Reject text longer than this instead of attempting a synthesis that
        #: will take minutes and blow the context window of whoever is waiting.
        self.max_chars: int = _env_int("CHATTERBOX_MAX_CHARS", 4000)

        #: Longest text a single generate() call is asked to render. Chatterbox
        #: truncates large inputs (turbo degrades past roughly 600 chars and
        #: comes back shorter than a shorter prompt), so text longer than this
        #: is sentence-split into per-chunk renders and concatenated into one
        #: WAV.
        self.max_chunk_chars: int = _env_int("CHATTERBOX_MAX_CHUNK_CHARS", 500)

        #: Silence in ms inserted between concatenated chunks, so sentence
        #: boundaries do not run directly into one another.
        self.chunk_pause_ms: int = _env_int("CHATTERBOX_CHUNK_PAUSE_MS", 250)

    def __repr__(self) -> str:
        return (
            f"Config("
            f"output_dir={self.output_dir!r}, "
            f"voices_dir={self.voices_dir!r}, "
            f"logs_dir={self.logs_dir!r}, "
            f"hf_home={self.hf_home!r}, "
            f"model={self.model!r}, "
            f"voice={self.voice!r}, "
            f"device={self.device!r}, "
            f"gpu_index={self.gpu_index!r}, "
            f"vram_floor_mb={self.vram_floor_mb!r}, "
            f"strict_vram={self.strict_vram!r}, "
            f"autoplay={self.autoplay!r}, "
            f"progressive={self.progressive!r}, "
            f"max_chars={self.max_chars!r}, "
            f"max_chunk_chars={self.max_chunk_chars!r}, "
            f"chunk_pause_ms={self.chunk_pause_ms!r}, "
            f")"
        )


__all__ = ["DEFAULT_VRAM_FLOOR_MB", "Config"]
