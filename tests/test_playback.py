"""
Playback: the stop flag, and the per-thread rule that made a long chunk uncuttable.

Real audio is never touched — a fake `winsound` records the calls and the WAV is
silence, so these tests assert control flow and flags, not sound.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from mcp_agent_chatterbox import playback

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="winsound playback is Windows-only")

# The constants winsound exposes; the fake mirrors them so flag assertions read
# the same in a test as in the real call.
SND_FILENAME = 0x00020000
SND_NODEFAULT = 0x00000002
SND_ASYNC = 0x0001
SND_PURGE = 0x0040


class FakeWinsound:
    """Records every PlaySound call as (sound, flags)."""

    SND_FILENAME = SND_FILENAME
    SND_NODEFAULT = SND_NODEFAULT
    SND_ASYNC = SND_ASYNC
    SND_PURGE = SND_PURGE

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int]] = []

    def PlaySound(self, sound, flags):  # noqa: N802 — mirrors the stdlib name
        self.calls.append((sound, flags))

    def starts(self) -> list[tuple[str | None, int]]:
        return [c for c in self.calls if c[0] is not None]


@pytest.fixture
def fake_winsound(monkeypatch):
    module = FakeWinsound()
    monkeypatch.setitem(sys.modules, "winsound", module)
    playback._stop_requested.clear()
    yield module
    playback._stop_requested.clear()


def write_silence(path: Path, seconds: float, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return path


class TestDuration:
    def test_reads_the_wav_header(self, tmp_path):
        path = write_silence(tmp_path / "a.wav", 0.5)
        assert playback._wav_duration_s(path) == pytest.approx(0.5, abs=0.01)

    def test_unreadable_file_reports_zero(self, tmp_path):
        path = tmp_path / "notawav.wav"
        path.write_bytes(b"nope")
        assert playback._wav_duration_s(path) == 0.0


class TestPlay:
    def test_blocking_play_is_started_async(self, tmp_path, fake_winsound):
        # The whole point of the change: the waiting thread must keep control of
        # its own sound, so the start cannot be SND_SYNC.
        path = write_silence(tmp_path / "short.wav", 0.05)
        played, reason = playback.play(path, wait=True)

        assert (played, reason) == (True, None)
        ((sound, flags),) = fake_winsound.starts()
        assert sound == str(path)
        assert flags & SND_ASYNC
        assert not fake_winsound.calls[1:], "nothing should ask for a purge on a clean play"

    def test_fire_and_forget_still_returns_immediately(self, tmp_path, fake_winsound):
        path = write_silence(tmp_path / "long.wav", 5.0)
        started = time.monotonic()
        played, _reason = playback.play(path)

        assert played is True
        assert time.monotonic() - started < 1.0
        assert fake_winsound.starts()[0][1] & SND_ASYNC

    def test_missing_file_is_reported_not_played(self, tmp_path, fake_winsound):
        played, reason = playback.play(tmp_path / "absent.wav", wait=True)
        assert played is False
        assert reason == "file not found"
        assert fake_winsound.calls == []


class TestStop:
    def test_stop_cuts_a_blocking_play(self, tmp_path, fake_winsound):
        # 10 s of silence, stopped after a beat: the call must return long before
        # the file would have finished, and the purge must come from the playing
        # thread (that is the only thread whose sound PlaySound will drop).
        path = write_silence(tmp_path / "long.wav", 10.0)
        playing_thread: list[int] = []
        finished: list[float] = []

        def run() -> None:
            playing_thread.append(threading.get_ident())
            started = time.monotonic()
            playback.play(path, wait=True)
            finished.append(time.monotonic() - started)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.2)
        assert playback.stop() == (True, None)
        thread.join(timeout=5)

        assert not thread.is_alive(), "stop did not release the playing thread"
        assert finished[0] < 2.0
        purge = fake_winsound.calls[-1]
        assert purge[0] is None
        assert purge[1] & SND_PURGE

    def test_flag_is_cleared_by_the_next_play(self, tmp_path, fake_winsound):
        # A stop with nothing playing must not silence the next utterance.
        playback.stop()
        path = write_silence(tmp_path / "short.wav", 0.02)
        played, reason = playback.play(path, wait=True)

        assert (played, reason) == (True, None)
        assert not playback._stop_requested.is_set()
