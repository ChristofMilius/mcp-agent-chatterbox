"""
Replay and history: repeating an utterance that is already on disk.
===================================================================
A replay must never synthesize. The audio of a message that was spoken earlier
is still in the output directory, so repeating it is a file read; the engine in
these tests raises the moment it is called, which proves that by construction
rather than by inspecting a log.

The queue is shared with live speech on purpose: that is what stops a replay
from talking over an utterance that is still rendering, and what lets `stop`
cut a replay off. The cancellation test exercises exactly that promise.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.speak import Speaker
from mcp_agent_chatterbox.worker import _dispatch


class ExplodingEngine:
    """Stands in for the synthesizer: any render is a test failure."""

    def synthesize(self, *args, **kwargs):
        raise AssertionError("a replay must not synthesize")


@pytest.fixture(autouse=True)
def no_pause(cfg):
    """Chunk pauses exist to sound right, not to slow the suite down."""
    cfg.chunk_pause_ms = 0
    return cfg


def write_chunks(output_dir, message_id: str, count: int = 3):
    """Create fake chunk files. Their content is never decoded -- play is stubbed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number in range(1, count + 1):
        path = output_dir / f"{message_id}_{number}.wav"
        path.write_bytes(b"RIFF----WAVE")
        paths.append(path)
    return paths


class TestReplay:
    def test_plays_the_existing_files_in_order(self, cfg, monkeypatch):
        write_chunks(cfg.output_dir, "2026-10-30", 3)
        played: list[str] = []

        def record_play(path, wait=False):
            played.append(str(path))
            return True, None

        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", record_play)

        speaker = Speaker(cfg, ExplodingEngine())
        payload = speaker.replay("2026-10-30")

        assert payload["status"] == "queued"
        assert payload["chunks"] == 3
        assert speaker.wait(timeout=10)
        assert [Path(p).name for p in played] == [
            "2026-10-30_1.wav",
            "2026-10-30_2.wav",
            "2026-10-30_3.wav",
        ]
        assert speaker.status()["last"]["replay"] is True
        assert speaker.status()["last"]["played_chunks"] == 3

    def test_writes_no_new_files(self, cfg, monkeypatch):
        before = {path.name for path in write_chunks(cfg.output_dir, "abc", 2)}
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", lambda *a, **k: (True, None))

        speaker = Speaker(cfg, ExplodingEngine())
        speaker.replay("abc")
        assert speaker.wait(timeout=10)

        after = {path.name for path in cfg.output_dir.iterdir()}
        assert after == before

    def test_unknown_message_is_a_fault(self, cfg):
        speaker = Speaker(cfg, ExplodingEngine())
        with pytest.raises(ToolFault) as caught:
            speaker.replay("never-spoken")
        assert caught.value.reason == "message_not_found"

    def test_stop_cuts_a_replay_off(self, cfg, monkeypatch):
        write_chunks(cfg.output_dir, "long", 4)
        started = threading.Event()
        played: list[str] = []

        def slow_play(path, wait=False):
            played.append(path)
            started.set()
            # Long enough that stop() lands while this chunk is "playing".
            threading.Event().wait(0.3)
            return True, None

        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", slow_play)

        speaker = Speaker(cfg, ExplodingEngine())
        speaker.replay("long")
        assert started.wait(5)

        speaker.stop()
        speaker.wait(timeout=10)

        # The chunk that was playing when stop() fired cannot be un-played, but
        # nothing after it may start.
        assert len(played) < 4, played
        assert speaker.status()["pending"] == 0


class TestDispatch:
    def test_history_returns_the_messages_on_disk(self, cfg):
        write_chunks(cfg.output_dir, "older", 1)
        write_chunks(cfg.output_dir, "newer", 2)
        speaker = Speaker(cfg, ExplodingEngine())

        response, finished = _dispatch(speaker, {"op": "history"})

        assert finished is False
        assert response["ok"] is True
        assert [m["message_id"] for m in response["messages"]] == ["newer", "older"]
        assert response["output_dir"] == str(cfg.output_dir)

    def test_history_limit_is_capped_not_rejected(self, cfg):
        for index in range(3):
            write_chunks(cfg.output_dir, f"m{index}", 1)
        speaker = Speaker(cfg, ExplodingEngine())

        response, _ = _dispatch(speaker, {"op": "history", "limit": 2})
        assert response["ok"] is True
        assert len(response["messages"]) == 2

        # A junk limit falls back to the default instead of erroring: the caller
        # asked for a listing, and it still gets one.
        response, _ = _dispatch(speaker, {"op": "history", "limit": "all"})
        assert response["ok"] is True
        assert len(response["messages"]) == 3

    def test_replay_needs_a_message_id(self, cfg):
        speaker = Speaker(cfg, ExplodingEngine())
        response, _ = _dispatch(speaker, {"op": "replay"})
        assert response["ok"] is False
        assert response["reason"] == "bad_request"

    def test_replay_of_an_unknown_message_reports_the_reason(self, cfg):
        speaker = Speaker(cfg, ExplodingEngine())
        response, _ = _dispatch(speaker, {"op": "replay", "message_id": "nope"})
        assert response["ok"] is False
        assert response["reason"] == "message_not_found"
        assert response["message_id"] == "nope"
