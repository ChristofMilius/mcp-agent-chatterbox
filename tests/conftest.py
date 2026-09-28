"""Shared fixtures. Every test runs on the CPU with no model download."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from mcp_agent_chatterbox.config import Config

# Env vars that Config reads. Cleared so a developer's own shell cannot leak
# into a test run and change the expected result.
_CONFIG_ENV = [
    "CHATTERBOX_OUTPUT_DIR",
    "CHATTERBOX_VOICES_DIR",
    "CHATTERBOX_LOGS_DIR",
    "CHATTERBOX_MODEL",
    "CHATTERBOX_DEVICE",
    "CHATTERBOX_GPU_INDEX",
    "CHATTERBOX_VRAM_MB",
    "CHATTERBOX_STRICT_VRAM",
    "CHATTERBOX_AUTOPLAY",
    "CHATTERBOX_MAX_CHARS",
    "CHATTERBOX_MAX_CHUNK_CHARS",
    "CHATTERBOX_CHUNK_PAUSE_MS",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _CONFIG_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cfg(tmp_path: Path) -> Iterator[Config]:
    """A Config rooted entirely inside tmp_path, with autoplay off."""
    monkey = pytest.MonkeyPatch()
    monkey.setenv("CHATTERBOX_OUTPUT_DIR", str(tmp_path / "out"))
    monkey.setenv("CHATTERBOX_VOICES_DIR", str(tmp_path / "voices"))
    monkey.setenv("CHATTERBOX_LOGS_DIR", str(tmp_path / "logs"))
    monkey.setenv("CHATTERBOX_AUTOPLAY", "0")
    try:
        yield Config()
    finally:
        monkey.undo()


class FakeTorch:
    """Just enough torch.cuda to drive device selection in the engine."""

    __version__ = "2.6.0+cu124"

    class version:  # noqa: N801 — mirrors torch.version.cuda
        cuda = "12.4"

    def __init__(self, *, available=True, count=1, free_mb=None, names=None):
        self.cuda = _FakeCuda(available, count, free_mb, names or {})
        self.manual_seed_calls: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def __repr__(self):
        return f"<FakeTorch cuda={self.cuda}>"


class _FakeCuda:
    def __init__(self, available, count, free_mb, names):
        self._available = available
        self._count = count
        self._free = free_mb or [0] * count
        self._names = names
        self.manual_seed_all_calls: list[int] = []

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def mem_get_info(self, index):
        free = self._free[index] * 1024 * 1024
        return free, 12 * 1024 * 1024 * 1024

    def get_device_name(self, index):
        if self._names:
            return self._names.get(index, f"fake-gpu-{index}")
        return f"fake-gpu-{index}"

    def empty_cache(self):
        pass

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)

    def __repr__(self):
        return f"cuda(available={self._available}, count={self._count})"


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a FakeTorch in place of the real torch, and hand back a setter."""

    def _install(**kwargs):
        fake = FakeTorch(**kwargs)
        monkeypatch.setattr("mcp_agent_chatterbox.engine._load_torch", lambda: fake)
        return fake

    return _install


class FakeEngine:
    """Stands in for ChatterboxEngine. Records calls, returns a tiny waveform."""

    def __init__(self, sample_rate=24000, samples=2400):
        import torch

        self.calls = []
        self.wav = torch.zeros(1, samples)
        self.sample_rate = sample_rate

    def synthesize(self, text, model, **kwargs):
        self.calls.append({"text": text, "model": model, **kwargs})
        return {
            "wav": self.wav,
            "sample_rate": self.sample_rate,
            "model": model,
            "device": "cpu",
            "chars": len(text),
            "elapsed_s": 0.01,
            "language": kwargs.get("language"),
            "reference_clip_used": bool(kwargs.get("reference_clip")),
            "free_vram_mb": None,
        }


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def silent_playback(monkeypatch):
    """Never touch the speakers during a test run."""
    monkeypatch.setattr("mcp_agent_chatterbox.speak.play", lambda *a, **k: (False, "stubbed"))
    return True


def pytest_report_header(config):
    return f"python: {sys.version.split()[0]}"
