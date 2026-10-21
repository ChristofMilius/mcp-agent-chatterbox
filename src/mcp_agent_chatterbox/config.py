"""
config.py — All configuration in one place
==========================================
Loads from a project `.env` file, then the environment. No secrets involved.

Why the file exists: agent harnesses are not reliable carriers of environment
variables. LM Studio's `mcp.json` and Hermes both accept an `env` block and
neither was observed forwarding it to the server subprocess, so every
CHATTERBOX_* setting silently fell back to its compiled default — the server
came up on turbo with no voice while the config asked for nano and a clip.
Reading the settings from a file the server opens itself removes that
dependency entirely: the same config now applies under LM Studio, Hermes and
opencode, and there is exactly one place to edit.

Precedence is environment-over-file, deliberately. Values already present in
os.environ are never overwritten by the file, so an explicit export (a
one-off CHATTERBOX_MODEL=turbo, a CI job, a test's monkeypatch) still wins,
and the file is what you edit for the durable setting.

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

#: The project config file, read once at import. See the module docstring for
#: why settings live here instead of only in the harness's env block.
ENV_FILE = _BASE / ".env"

#: Names that load_env_file() filled in from the file (as opposed to names the
#: caller had already exported). Used by the test suite to undo import-time
#: loading so a developer's real .env cannot make a test machine-dependent.
ENV_FILE_KEYS: set[str] = set()

#: Port the resident synthesis worker listens on. The MCP server never
#: synthesizes: it hands the text to this worker, which outlives the server's
#: per-turn process, so a long reply is not killed when the harness ends the
#: turn. Both sides read the port from here, so a change of default cannot
#: drift between the server that connects and the worker that binds.
DEFAULT_WORKER_PORT = 8765

#: How long the MCP server waits for a connection to the worker before it gives
#: up and tells the caller to start one. Deliberately short: a refused connect
#: on loopback is immediate, and a server that has no worker must not make the
#: model wait for the tool call to time out.
DEFAULT_WORKER_CONNECT_TIMEOUT_S = 5

#: Free VRAM (MiB) a CUDA device must have before a model load is attempted.
#: Chatterbox weights are loaded straight onto the device (see the module
#: docstring in __init__.py for why there is no CPU-staging path), so a load
#: that starts with less headroom than this fails with a bare CUDA OOM part
#: way through reading the checkpoint. We check first and say something useful.
DEFAULT_VRAM_FLOOR_MB = 4096


def load_env_file(path: Path | None = None) -> Path | None:
    """
    Seed os.environ from the project `.env`, without overriding real exports.

    python-dotenv is used when importable (it is a declared dependency); if it
    is somehow missing we fall back to a minimal parser rather than silently
    starting on defaults, because that silent-default path is the exact failure
    this file exists to prevent.

    Every name this fills in is recorded in ENV_FILE_KEYS, so a test (or an
    embedder) can tell a value that came from the file apart from one the
    caller exported. Without that, a developer's real .env would leak into the
    suite via import-time loading and the tests would depend on the machine.

    Returns the path that was read, or None when there is no file.
    """
    target = path or ENV_FILE
    if not target.is_file():
        return None
    before = set(os.environ)
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover — declared dependency
        _load_env_fallback(target)
    else:
        load_dotenv(target, override=False)
    ENV_FILE_KEYS.clear()
    ENV_FILE_KEYS.update(set(os.environ) - before)
    return target


def _load_env_fallback(path: Path) -> None:
    """
    Parse `KEY=value` lines and fill in only what os.environ lacks.

    Deliberately minimal: blank lines and `#` comments are skipped, values may
    be quoted, and `export ` prefixes are tolerated. Late-bound values
    (${VAR}) are not expanded — install python-dotenv for anything exotic.
    """
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


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

        #: Reject text longer than this instead of attempting a synthesis that
        #: will take minutes and blow the context window of whoever is waiting.
        self.max_chars: int = _env_int("CHATTERBOX_MAX_CHARS", 4000)

        #: Hard cap on any single generate() call. Each chunk is one sentence, or
        #: a batch of `sentences_per_chunk` sentences; this bounds the batch, and
        #: bounds the exception where one sentence is so long it must be broken.
        #: Chatterbox truncates large inputs (turbo degrades past roughly 600
        #: chars and comes back shorter than a shorter prompt), so a chunk never
        #: exceeds this.
        self.max_chunk_chars: int = _env_int("CHATTERBOX_MAX_CHUNK_CHARS", 500)

        #: How many sentences go into one rendered chunk. Every chunk is a
        #: separate generate() call, so each boundary costs a pause plus a fresh
        #: prosody start -- with one sentence per chunk a reply made of short
        #: sentences sounds staccato. Batching two or three makes it flow. Set it
        #: to 1 for the old one-sentence-per-file behaviour. The first sentence
        #: always stays alone so audio starts as soon as it is rendered, and a
        #: batch never exceeds CHATTERBOX_MAX_CHUNK_CHARS regardless of this value.
        self.sentences_per_chunk: int = _env_int("CHATTERBOX_SENTENCES_PER_CHUNK", 3)

        #: Silence in ms inserted between consecutive chunks. With batching this
        #: is the pause between groups of sentences, not between every sentence.
        self.chunk_pause_ms: int = _env_int("CHATTERBOX_CHUNK_PAUSE_MS", 250)

        #: Seconds the server will wait, after the client has disconnected, for
        #: the speech queue to finish before letting the process exit. The
        #: worker is non-daemon and synthesis is background work, so without
        #: this wait a per-turn harness would kill the queue mid-chunk. Sized
        #: generously because it is only ever paid when work is outstanding:
        #: measured warm throughput is ~0.03-0.05 s/char, so the default covers
        #: roughly a 4000-char reply (CHATTERBOX_MAX_CHARS) plus the one-time
        #: model load. Set it to 0 to skip the wait entirely.
        self.drain_timeout_s: float = _env_int("CHATTERBOX_DRAIN_TIMEOUT", 300)

        #: Where the resident synthesis worker lives. The MCP server reads this
        #: to know where to send text; the `worker` subcommand reads it to know
        #: where to listen. Point them at different ports and the server will
        #: report that no worker is running.
        self.worker_port: int = _env_int("CHATTERBOX_WORKER_PORT", DEFAULT_WORKER_PORT)

        #: Bound on connecting to that worker. Only the connect, not the reply:
        #: rendering a reply takes as long as it takes, and cutting the read off
        #: would abandon audio that is still being spoken.
        self.worker_connect_timeout_s: float = _env_int(
            "CHATTERBOX_WORKER_CONNECT_TIMEOUT", DEFAULT_WORKER_CONNECT_TIMEOUT_S
        )

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
            f"max_chars={self.max_chars!r}, "
            f"max_chunk_chars={self.max_chunk_chars!r}, "
            f"sentences_per_chunk={self.sentences_per_chunk!r}, "
            f"chunk_pause_ms={self.chunk_pause_ms!r}, "
            f"drain_timeout_s={self.drain_timeout_s!r}, "
            f"worker_port={self.worker_port!r}, "
            f"worker_connect_timeout_s={self.worker_connect_timeout_s!r}, "
            f")"
        )


__all__ = [
    "DEFAULT_VRAM_FLOOR_MB",
    "DEFAULT_WORKER_CONNECT_TIMEOUT_S",
    "DEFAULT_WORKER_PORT",
    "Config",
]
