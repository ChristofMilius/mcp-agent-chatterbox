"""
The speak path: validation, the reference-clip requirement, the queue, and the
chunk files.

Speaker is shared by the MCP tool and the CLI, so these tests cover both entry
points at once. The engine is faked, playback is stubbed, and every test drives
the real background worker — nothing here downloads weights or touches the
speakers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.registry import SUPPORTED_LANGUAGES, T3_MODELS
from mcp_agent_chatterbox.speak import (
    CHARS_PER_SECOND,
    MAX_CHUNK_GROWTH,
    MIN_CHUNK_SECONDS,
    Speaker,
    split_sentences,
)


@pytest.fixture(autouse=True)
def _no_sound(silent_playback):
    pass


@pytest.fixture(autouse=True)
def _no_chunk_pause(cfg):
    """Keep the worker from sleeping between chunks in every test."""
    cfg.chunk_pause_ms = 0


@pytest.fixture
def narrator(cfg):
    """Put one synthetic reference clip in the voices dir and return its name.

    Generic fixture name on purpose -- tests never reference a real local
    voice, and multilingual/original (which have no built-in voice) need a
    clip to exist at all.
    """
    cfg.voices_dir.mkdir(parents=True, exist_ok=True)
    (cfg.voices_dir / "narrator.wav").write_bytes(b"RIFF" + b"\0" * 32)
    return "narrator"


def run(cfg, engine, text="Build finished.", **kwargs):
    """Submit, block until the worker is idle, return (payload, last summary)."""
    speaker = Speaker(cfg, engine)
    payload = speaker.submit(text, **kwargs)
    assert speaker.wait(timeout=10), "worker did not drain the queue"
    last = speaker.status()["last"]
    assert last is not None
    return payload, last


class BlockingEngine:
    """A fake engine that parks inside synthesize() until the test releases it."""

    def __init__(self):
        import torch

        self.calls: list[dict] = []
        self.wav = torch.zeros(1, 2400)
        self.sample_rate = 24000
        self.entered = threading.Event()
        self.gate = threading.Event()

    def synthesize(self, text, model_key, **kwargs):
        self.entered.set()
        assert self.gate.wait(timeout=10), "test never released the engine"
        self.calls.append({"text": text, "model": model_key, **kwargs})
        return {
            "wav": self.wav,
            "sample_rate": self.sample_rate,
            "model": model_key,
            "device": "cpu",
            "chars": len(text),
            "elapsed_s": 0.01,
            "language": kwargs.get("language"),
            "reference_clip_used": bool(kwargs.get("reference_clip")),
            "free_vram_mb": None,
        }


class TestTextValidation:
    def test_rejects_empty(self, cfg, fake_engine):
        for bad in ("", "   ", "\n\t"):
            with pytest.raises(ToolFault) as exc:
                Speaker(cfg, fake_engine).submit(bad)
            assert exc.value.reason == "empty_text"
        assert fake_engine.calls == []

    def test_rejects_over_budget(self, cfg, fake_engine):
        cfg.max_chars = 10
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("x" * 11)
        assert exc.value.reason == "text_too_long"
        assert exc.value.extra["max_chars"] == 10
        assert fake_engine.calls == []

    def test_strips_surrounding_whitespace(self, cfg, fake_engine):
        run(cfg, fake_engine, "  Build finished.  ")
        assert fake_engine.calls[0]["text"] == "Build finished."


class TestModelValidation:
    def test_rejects_unknown_model(self, cfg, fake_engine):
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.", model="whisper")
        assert exc.value.reason == "unknown_model"
        assert "turbo" in exc.value.message

    def test_defaults_to_configured_model(self, cfg, fake_engine):
        # nano is a stock-voice model, so the configured default is exercised
        # on the built-in voice -- no reference clip involved.
        cfg.model = "nano"
        run(cfg, fake_engine)
        assert fake_engine.calls[0]["model"] == "nano"

    def test_full_hf_repo_id_is_rejected_clearly(self, cfg, fake_engine):
        # Models are addressed by short key. A repo id must be refused by name,
        # not silently swapped for some other model.
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit(
                "Build finished.", model="ResembleAI/chatterbox-turbo"
            )
        assert exc.value.reason == "unknown_model"
        assert fake_engine.calls == []

    def test_alias_is_accepted(self, cfg, fake_engine, narrator):
        run(cfg, fake_engine, model="multi", voice=narrator)
        assert fake_engine.calls[0]["model"] == "multilingual"


class TestLanguageValidation:
    def test_rejects_unsupported(self, cfg, fake_engine):
        # Language validation runs before the clip check, so no voice is needed.
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.", model="multilingual", language="xx")
        assert exc.value.reason == "unsupported_language"
        supported = cast("list[str]", exc.value.extra["supported"])
        assert "de" in supported

    def test_normalizes_case_and_whitespace(self, cfg, fake_engine, narrator):
        run(cfg, fake_engine, model="multilingual", language="  DE  ", voice=narrator)
        assert fake_engine.calls[0]["language"] == "de"

    def test_ignored_for_non_multilingual(self, cfg, fake_engine):
        # turbo takes no language; passing one must not reach generate().
        run(cfg, fake_engine, model="turbo", language="de")
        assert fake_engine.calls[0]["language"] is None

    def test_blank_language_is_ignored(self, cfg, fake_engine):
        run(cfg, fake_engine, model="turbo", language="   ")
        assert fake_engine.calls[0]["language"] is None

    def test_every_advertised_language_passes(self, cfg, fake_engine, narrator):
        for code in SUPPORTED_LANGUAGES:
            run(cfg, fake_engine, model="multilingual", language=code, voice=narrator)
        assert len(fake_engine.calls) == len(SUPPORTED_LANGUAGES)


class TestT3ModelValidation:
    def test_rejects_unknown(self, cfg, fake_engine):
        # t3 validation runs before the clip check, so no voice is needed.
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.", model="multilingual", t3_model="v9")
        assert exc.value.reason == "unsupported_t3_model"

    def test_accepts_each_advertised(self, cfg, fake_engine, narrator):
        for choice in T3_MODELS:
            run(cfg, fake_engine, model="multilingual", voice=narrator, t3_model=choice)
        assert {c["t3_model"] for c in fake_engine.calls} == set(T3_MODELS)

    def test_ignored_for_turbo(self, cfg, fake_engine):
        run(cfg, fake_engine, model="turbo", t3_model="v3")
        assert fake_engine.calls[0]["t3_model"] is None


class TestReferenceClipRequirement:
    def test_multilingual_without_clip_is_refused(self, cfg, fake_engine):
        # Chatterbox would raise a bare AssertionError here; the point of this
        # check is to return something the caller can act on.
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.", model="multilingual", language="de")
        assert exc.value.reason == "reference_clip_required"
        assert fake_engine.calls == []

    def test_original_without_clip_is_refused(self, cfg, fake_engine):
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.", model="original")
        assert exc.value.reason == "reference_clip_required"

    def test_turbo_needs_no_clip(self, cfg, fake_engine):
        payload, _ = run(cfg, fake_engine, model="turbo")
        assert payload["status"] == "queued"

    def test_voice_and_reference_clip_are_exclusive(self, cfg, fake_engine, narrator, tmp_path):
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"RIFF")
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit(
                "Build finished.", model="turbo", voice=narrator, reference_clip=str(clip)
            )
        assert exc.value.reason == "ambiguous_voice"


class TestReferenceClipResolution:
    @pytest.fixture
    def voices_dir(self, cfg):
        cfg.voices_dir.mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "narrator.wav").write_bytes(b"RIFF" + b"\0" * 32)
        return cfg.voices_dir

    def test_voice_name_resolves_to_path(self, cfg, fake_engine, voices_dir):
        run(cfg, fake_engine, model="multilingual", voice="narrator", language="en")
        assert fake_engine.calls[0]["reference_clip"].endswith("narrator.wav")

    def test_explicit_path_accepted(self, cfg, fake_engine, voices_dir, tmp_path):
        clip = tmp_path / "elsewhere.wav"
        clip.write_bytes(b"RIFF")
        run(cfg, fake_engine, model="multilingual", reference_clip=str(clip), language="en")
        assert fake_engine.calls[0]["reference_clip"] == str(clip)

    def test_missing_path_is_reported(self, cfg, fake_engine, voices_dir, tmp_path):
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit(
                "Build finished.", model="multilingual", reference_clip=str(tmp_path / "gone.wav")
            )
        assert exc.value.reason == "reference_clip_not_found"

    def test_wrong_extension_is_reported(self, cfg, fake_engine, voices_dir, tmp_path):
        bad = tmp_path / "clip.txt"
        bad.write_bytes(b"nope")
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit(
                "Build finished.", model="multilingual", reference_clip=str(bad)
            )
        assert exc.value.reason == "unsupported_reference_format"


class TestDefaultVoice:
    def test_used_when_the_call_names_no_voice(self, cfg, fake_engine, narrator):
        # CHATTERBOX_VOICE behaves exactly as if the caller had passed voice=.
        cfg.voice = narrator
        payload, _ = run(cfg, fake_engine)
        assert fake_engine.calls[0]["reference_clip"].endswith("narrator.wav")
        assert payload["voice"] == "narrator"

    def test_explicit_voice_wins(self, cfg, fake_engine, narrator):
        cfg.voice = "other"
        (cfg.voices_dir / "other.wav").write_bytes(b"RIFF")
        run(cfg, fake_engine, voice=narrator)
        assert fake_engine.calls[0]["reference_clip"].endswith("narrator.wav")

    def test_explicit_clip_wins_without_ambiguity(self, cfg, fake_engine, narrator, tmp_path):
        # An explicit reference_clip must not collide with the default voice:
        # voice= and reference_clip= are exclusive, and a caller who brought
        # their own clip means it.
        cfg.voice = narrator
        clip = tmp_path / "mine.wav"
        clip.write_bytes(b"RIFF")
        run(cfg, fake_engine, reference_clip=str(clip))
        assert fake_engine.calls[0]["reference_clip"] == str(clip)

    def test_missing_default_fails_fast(self, cfg, fake_engine, narrator):
        # A default voice that is not in the voices dir must not silently
        # render in another voice; the call fails with voice_not_found.
        cfg.voice = "ghost"
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.")
        assert exc.value.reason == "voice_not_found"
        assert fake_engine.calls == []


class TestOutput:
    def test_accepts_without_rendering(self, cfg):
        # The whole point of the queue: submit returns before any audio exists.
        engine = BlockingEngine()
        speaker = Speaker(cfg, engine)
        payload = speaker.submit("One. Two.")

        assert payload["status"] == "queued"
        assert engine.entered.wait(timeout=10)  # the worker picked it up
        assert engine.calls == []  # ...and is still rendering

        engine.gate.set()
        assert speaker.wait(timeout=10)

    def test_writes_one_file_per_sentence(self, cfg, fake_engine):
        # One file per SENTENCE is the sentences_per_chunk=1 mode; batching is
        # covered in TestChunking.
        cfg.sentences_per_chunk = 1
        payload, last = run(cfg, fake_engine, "Alpha. Beta. Gamma.")
        assert payload["chunks"] == 3
        assert payload["files"] == [
            f"{payload['message_id']}_1.wav",
            f"{payload['message_id']}_2.wav",
            f"{payload['message_id']}_3.wav",
        ]
        for name in payload["files"]:
            path = cfg.output_dir / name
            assert path.is_file()
            assert path.stat().st_size > 0
        assert last["written"] == payload["files"]

    def test_reports_duration_and_rate(self, cfg, fake_engine):
        _, last = run(cfg, fake_engine)
        # 2400 samples at 24000 Hz is 0.1 s.
        assert last["duration_s"] == 0.1
        assert last["device"] == "cpu"
        assert last["played_chunks"] == 0

    def test_explicit_filename_is_used(self, cfg, fake_engine):
        payload, _ = run(cfg, fake_engine, filename="take-one")
        assert payload["files"] == ["take-one_1.wav"]
        assert (cfg.output_dir / "take-one_1.wav").is_file()

    def test_explicit_filename_is_sanitized(self, cfg, fake_engine):
        payload, _ = run(cfg, fake_engine, filename="../../escape")
        name = payload["files"][0]
        assert "/" not in name
        assert "\\" not in name
        assert ".." not in name
        assert (cfg.output_dir / name).is_file()

    def test_creates_missing_output_dir(self, cfg, fake_engine):
        assert not cfg.output_dir.exists()
        run(cfg, fake_engine)
        assert cfg.output_dir.is_dir()

    def test_message_ids_are_unique(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        ids = {speaker.submit(f"Take {i}.")["message_id"] for i in range(5)}
        assert len(ids) == 5
        assert speaker.wait(timeout=10)

    def test_result_is_json_serialisable(self, cfg, fake_engine):
        payload, last = run(cfg, fake_engine)
        assert json.loads(json.dumps(payload))["status"] == "queued"
        assert json.loads(json.dumps(last))["message_id"] == payload["message_id"]


class TestPlayback:
    def test_disabled_by_config(self, cfg, fake_engine, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play", lambda *a, **k: calls.append(a) or (True, None)
        )
        cfg.autoplay = False
        payload, last = run(cfg, fake_engine, play_audio=False)
        assert payload["play"] is False
        assert calls == []
        assert last["played_chunks"] == 0

    def test_explicit_override_wins(self, cfg, fake_engine, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.player_available", lambda: True
        )
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play", lambda *a, **k: calls.append(a) or (True, None)
        )
        cfg.autoplay = False
        payload, last = run(cfg, fake_engine, play_audio=True)
        assert payload["play"] is True
        assert len(calls) == 1
        assert last["played_chunks"] == 1

    def test_one_play_call_per_chunk_in_order(self, cfg, fake_engine, monkeypatch):
        cfg.sentences_per_chunk = 1
        calls = []
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play", lambda *a, **k: calls.append(a[0]) or (True, None)
        )
        payload, last = run(cfg, fake_engine, "Alpha. Beta. Gamma.", play_audio=True)
        assert [path.name for path in calls] == payload["files"]
        assert last["played_chunks"] == 3

    def test_next_chunk_renders_while_the_current_one_plays(
        self, cfg, fake_engine, monkeypatch
    ):
        # play() blocks the worker thread, so rendering has to run ahead of it.
        # Without that overlap the GPU idles for the whole utterance and every
        # chunk boundary costs a full render's worth of silence.
        # Preroll off: 0.1s fake chunks would otherwise render the whole reply
        # before playback starts, which would hide the overlap under test.
        monkeypatch.setattr("mcp_agent_chatterbox.speak.PREROLL_SECONDS", 0.0)
        events: list[tuple[str, str]] = []
        lock = threading.Lock()

        def record(kind: str, what: str) -> None:
            with lock:
                events.append((kind, what))

        real_synthesize = fake_engine.synthesize

        def slow_synthesize(text, model_key, **kwargs):
            record("synth-start", text)
            time.sleep(0.1)
            result = real_synthesize(text, model_key, **kwargs)
            record("synth-end", text)
            return result

        def slow_play(path, wait=True):
            record("play-start", Path(path).name)
            time.sleep(0.4)
            record("play-end", Path(path).name)
            return True, None

        fake_engine.synthesize = slow_synthesize
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", slow_play)

        cfg.sentences_per_chunk = 1
        payload, last = run(cfg, fake_engine, "Alpha. Beta. Gamma.", play_audio=True)
        assert last["played_chunks"] == 3

        # The second chunk began rendering before the first had finished playing.
        first_play_end = events.index(("play-end", f"{payload['message_id']}_1.wav"))
        second_synth_start = events.index(("synth-start", "Beta."))
        assert second_synth_start < first_play_end, events

        # ...but playback itself stayed strictly in order.
        played = [what for kind, what in events if kind == "play-start"]
        assert played == [f"{payload['message_id']}_{i}.wav" for i in (1, 2, 3)]

    def test_playback_waits_for_a_preroll(self, cfg, fake_engine, monkeypatch):
        # A short opening sentence cannot hide the next render, so playback does
        # not start until enough audio is already on disk to cover it.
        cfg.sentences_per_chunk = 1
        events: list[tuple[str, str]] = []
        real_synthesize = fake_engine.synthesize

        def record_synthesize(text, model_key, **kwargs):
            result = real_synthesize(text, model_key, **kwargs)
            events.append(("synth", text))
            return result

        def record_play(path, wait=True):
            events.append(("play", Path(path).name))
            return True, None

        # Fake chunks are 0.1s, so 0.25s of preroll means three of them.
        monkeypatch.setattr("mcp_agent_chatterbox.speak.PREROLL_SECONDS", 0.25)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", record_play)
        fake_engine.synthesize = record_synthesize

        payload, last = run(cfg, fake_engine, "Alpha. Beta. Gamma. Delta.", play_audio=True)
        first_play = events.index(("play", f"{payload['message_id']}_1.wav"))
        rendered_first = sum(1 for kind, _ in events[:first_play] if kind == "synth")
        assert rendered_first >= 3, events
        assert last["played_chunks"] == 4

    def test_first_audio_and_gap_are_reported(self, cfg, fake_engine, monkeypatch):
        # Dry time is what the pipeline is supposed to avoid, so a run has to
        # say whether it happened rather than leave it to the listener's ear.
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", lambda *a, **k: (True, None))
        cfg.sentences_per_chunk = 1
        _, last = run(cfg, fake_engine, "Alpha. Beta. Gamma.", play_audio=True)
        assert last["first_audio_s"] is not None
        assert last["first_audio_s"] >= 0
        assert last["max_gap_s"] == 0.0

    def test_no_playback_means_no_gap_figures(self, cfg, fake_engine):
        _, last = run(cfg, fake_engine, play_audio=False)
        assert last["first_audio_s"] is None
        assert last["max_gap_s"] is None

    def test_failure_is_reported_not_raised(self, cfg, fake_engine, monkeypatch):
        # A machine with no sound card must still get its chunk files written.
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play",
            lambda *a, **k: (False, "no audio device"),
        )
        payload, last = run(cfg, fake_engine, play_audio=True)
        assert payload["status"] == "queued"
        assert last["written"] == payload["files"]
        assert last["played_chunks"] == 0
        assert last["errors"] == ["chunk 1: no audio device"]


class TestWorkerLifecycle:
    """
    The worker is non-daemon so real speech survives a per-turn harness, which
    means it must be stoppable or the interpreter never exits. These cover the
    stop/drain contract that keeps that true.
    """

    def test_worker_is_not_a_daemon(self, cfg, fake_engine):
        # The whole point: a daemon thread is destroyed with the process, which
        # is what silently discarded queued speech when a client hung up.
        speaker = Speaker(cfg, fake_engine)
        speaker.submit("One.")
        assert speaker.wait(timeout=10)
        worker = speaker._worker
        assert worker is not None and not worker.daemon

    def test_stop_worker_ends_the_thread(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        speaker.submit("One.")
        assert speaker.wait(timeout=10)
        speaker.stop_worker()
        worker = speaker._worker
        assert worker is None or not worker.is_alive()

    def test_stop_worker_is_safe_when_nothing_started(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        speaker.stop_worker()  # no worker was ever created
        speaker.stop_worker()  # and calling twice is fine

    def test_stop_worker_leaves_no_live_thread(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        speaker.submit("One.")
        assert speaker.wait(timeout=10)
        speaker.stop_worker()
        # Only the main thread may remain; a leaked worker would hang exit.
        names = {t.name for t in threading.enumerate()}
        assert "chatterbox-speaker" not in names

    def test_drain_waits_for_queued_work(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        payload = speaker.submit("One. Two. Three.")
        assert speaker.drain(timeout=10) is None  # returns None, blocks
        written = speaker.status()["last"]["written"]
        assert len(written) == payload["chunks"]
        speaker.stop_worker()

    def test_drain_times_out_without_hanging(self, cfg):
        # A wedged engine must not pin the process: drain has to give up at the
        # bound AND stop the worker, otherwise this is a hang instead of a loss.
        engine = BlockingEngine()
        speaker = Speaker(cfg, engine)
        speaker.submit("One.")
        assert engine.entered.wait(timeout=10)
        speaker.drain(timeout=0.2)
        engine.gate.set()
        worker = speaker._worker
        assert worker is None or not worker.is_alive()

    def test_stop_survives_a_shutdown_sentinel_in_the_queue(self, cfg, fake_engine):
        # stop() walks the queue with get_nowait(); the None sentinel drain()
        # puts there is not a job and must not be counted or cancelled.
        speaker = Speaker(cfg, fake_engine)
        speaker.submit("One.")
        assert speaker.wait(timeout=10)
        speaker._queue.put(None)
        state = speaker.stop()
        assert state["cancelled_jobs"] == 0
        assert state["cancelled_ids"] == []
        speaker.stop_worker()


class TestQueue:
    LONG = "Alpha red green blue. Beta yellow orange purple. Gamma pink black white."

    def test_jobs_render_in_submission_order(self, cfg):
        engine = BlockingEngine()
        speaker = Speaker(cfg, engine)
        first = speaker.submit("One. Two.")
        assert engine.entered.wait(timeout=10)
        second = speaker.submit("Three.")
        assert first["queued_ahead"] == 0
        assert second["queued_ahead"] == 1
        engine.gate.set()
        assert speaker.wait(timeout=10)
        assert [c["text"] for c in engine.calls] == ["One.", "Two.", "Three."]
        assert speaker.status()["last"]["message_id"] == second["message_id"]

    def test_second_job_waits_for_the_first(self, cfg):
        engine = BlockingEngine()
        speaker = Speaker(cfg, engine)
        speaker.submit("One.")
        assert engine.entered.wait(timeout=10)
        speaker.submit("Two.")
        assert engine.calls == []  # the first chunk is still rendering
        engine.gate.set()
        assert speaker.wait(timeout=10)
        assert [c["text"] for c in engine.calls] == ["One.", "Two."]

    def test_stop_cancels_queued_jobs(self, cfg, monkeypatch):
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.stop_playback", lambda: (True, None)
        )
        engine = BlockingEngine()
        speaker = Speaker(cfg, engine)
        first = speaker.submit("One. Two.")
        assert engine.entered.wait(timeout=10)
        second = speaker.submit("Three.")

        state = speaker.stop()
        assert state["cancelled_jobs"] == 1
        assert state["cancelled_ids"] == [second["message_id"]]
        assert state["cancelled_active"] is True

        engine.gate.set()
        assert speaker.wait(timeout=10)
        assert [c["text"] for c in engine.calls] == ["One."]
        last = speaker.status()["last"]
        assert last["message_id"] == first["message_id"]
        assert last["written"] == [f"{first['message_id']}_1.wav"]

    def test_chunk_pause_between_chunks(self, cfg, fake_engine, monkeypatch):
        import time as real_time

        class _Clock:
            def __init__(self):
                self.sleeps: list[float] = []

            def sleep(self, seconds):
                self.sleeps.append(seconds)

            def time(self):
                return real_time.time()

            def monotonic(self):
                return real_time.monotonic()

        clock = _Clock()
        monkeypatch.setattr("mcp_agent_chatterbox.speak.time", clock)
        monkeypatch.setattr("mcp_agent_chatterbox.speak.player_available", lambda: True)
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play", lambda *a, **k: (True, None)
        )
        cfg.chunk_pause_ms = 250
        cfg.sentences_per_chunk = 1
        run(cfg, fake_engine, "Alpha. Beta. Gamma.", play_audio=True)
        # A pause after every chunk except the last.
        assert clock.sleeps == [0.25, 0.25]


class TestStyleParameters:
    def test_explicit_values_forwarded(self, cfg, fake_engine):
        run(cfg, fake_engine, exaggeration=0.4, cfg_weight=0.7, temperature=0.9)
        call = fake_engine.calls[0]
        assert call["exaggeration"] == 0.4
        assert call["cfg_weight"] == 0.7
        assert call["temperature"] == 0.9

    def test_unset_values_stay_none(self, cfg, fake_engine):
        # None means "let the model use its own tuned default".
        run(cfg, fake_engine)
        call = fake_engine.calls[0]
        assert call["exaggeration"] is None
        assert call["cfg_weight"] is None
        assert call["temperature"] is None


class TestSamplingKnobs:
    def test_explicit_values_forwarded(self, cfg, fake_engine):
        run(
            cfg,
            fake_engine,
            top_p=0.9,
            top_k=500,
            repetition_penalty=1.5,
            norm_loudness=False,
            seed=42,
        )
        call = fake_engine.calls[0]
        assert call["top_p"] == 0.9
        assert call["top_k"] == 500
        assert call["repetition_penalty"] == 1.5
        assert call["norm_loudness"] is False
        assert call["seed"] == 42

    def test_unset_values_stay_none(self, cfg, fake_engine):
        run(cfg, fake_engine)
        call = fake_engine.calls[0]
        assert call["top_p"] is None
        assert call["top_k"] is None
        assert call["repetition_penalty"] is None
        assert call["norm_loudness"] is None
        assert call["seed"] is None

    def test_chunks_share_the_sampling_knobs(self, cfg, fake_engine):
        cfg.sentences_per_chunk = 1
        run(cfg, fake_engine, "Alpha. " * 6, top_p=0.9, top_k=500, seed=7)
        assert len(fake_engine.calls) == 6
        for call in fake_engine.calls:
            assert call["top_p"] == 0.9
            assert call["top_k"] == 500
            assert call["seed"] == 7


class TestKnobValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": 5.5},
            {"top_p": 1.5},
            {"top_p": -0.1},
            {"top_k": 1500},
            {"top_k": -1},
            {"repetition_penalty": 0.5},
            {"repetition_penalty": 2.5},
            {"exaggeration": 3.0},
            {"cfg_weight": 1.5},
            {"seed": -1},
        ],
    )
    def test_out_of_range_knob_raises(self, cfg, fake_engine, kwargs):
        with pytest.raises(ToolFault) as exc:
            Speaker(cfg, fake_engine).submit("Build finished.", **kwargs)
        assert exc.value.reason == "invalid_knob"
        assert "range" in exc.value.message
        assert fake_engine.calls == []

    def test_boundary_values_accepted(self, cfg, fake_engine):
        run(
            cfg,
            fake_engine,
            temperature=5.0,
            top_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            exaggeration=2.0,
            cfg_weight=1.0,
            seed=0,
        )
        assert len(fake_engine.calls) == 1

    def test_zero_seed_is_accepted_for_random(self, cfg, fake_engine):
        # Upstream Gradio apps treat 0 as "random"; it is a valid request.
        payload, _ = run(cfg, fake_engine, seed=0)
        assert payload["status"] == "queued"


class TestSentenceSplitting:
    def test_one_sentence_is_one_chunk(self):
        assert split_sentences("Hello.", 500) == ["Hello."]

    def test_every_sentence_is_its_own_chunk(self):
        assert split_sentences("One. Two! Three?", 500) == ["One.", "Two!", "Three?"]

    def test_closing_quote_stays_with_the_sentence(self):
        chunks = split_sentences('He said "go." Then left.', 500)
        assert chunks == ['He said "go."', "Then left."]

    def test_splits_at_newlines(self):
        chunks = split_sentences("Line one.\nLine two.\nLine three.", 500)
        assert chunks == ["Line one.", "Line two.", "Line three."]

    def test_empty_input_is_no_chunks(self):
        assert split_sentences("", 500) == []
        assert split_sentences("   \n\t ", 500) == []

    def test_hard_splits_an_oversize_sentence_on_word_boundaries(self):
        text = "The quick brown fox jumps over the lazy dog."  # one sentence, 46 chars
        chunks = split_sentences(text, 20)
        assert len(chunks) >= 2
        assert all(len(s) <= 20 for s in chunks)
        assert " ".join(chunks).split() == text.split()

    def test_no_punctuation_hard_splits_evenly(self):
        text = "word " * 30  # 150 chars, no enders at all
        chunks = split_sentences(text, 40)
        assert all(len(s) <= 40 for s in chunks)
        assert " ".join(chunks).split() == text.split()

    def test_batching_keeps_the_first_sentence_alone(self):
        # Speech has to start as soon as one sentence is rendered, so a batch
        # never swallows the opening sentence.
        chunks = split_sentences("One. Two. Three. Four.", 500, 3)
        assert chunks == ["One.", "Two. Three. Four."]

    def test_batching_respects_the_sentence_count(self):
        chunks = split_sentences("One. Two. Three. Four. Five.", 500, 2)
        assert chunks == ["One.", "Two. Three.", "Four. Five."]

    def test_batching_breaks_early_at_the_character_cap(self):
        # Two sentences fit, the third does not: the batch stops at two rather
        # than overflowing the cap.
        chunks = split_sentences("aaaaaa. bbbbbb. cccccc.", 14, 3)
        assert chunks == ["aaaaaa.", "bbbbbb.", "cccccc."]

    def test_batching_of_one_is_the_old_behaviour(self):
        chunks = split_sentences("One. Two. Three.", 500, 1)
        assert chunks == ["One.", "Two.", "Three."]

    def test_a_chunk_never_outgrows_the_one_before_it(self):
        # The renderer has to finish chunk N+1 inside chunk N's playback, or the
        # player runs dry. Rendering costs ~0.42 of the audio it produces, so a
        # chunk at most ~2.2x its predecessor still fits; this is the rule that
        # keeps that true instead of hoping the sentence count works out.
        sentences = ["Opening line."] + [
            f"This is sentence {i} with a few words in it." for i in range(1, 20)
        ]
        chunks = split_sentences(" ".join(sentences), 500, 3)
        assert len(chunks) > 2
        for previous, current in zip(chunks, chunks[1:]):
            allowed = max(
                len(previous) / CHARS_PER_SECOND * MAX_CHUNK_GROWTH,
                MIN_CHUNK_SECONDS,
            )
            assert len(current) / CHARS_PER_SECOND <= allowed, (previous, current)

    def test_evenly_sized_sentences_still_batch(self):
        # Two equal sentences are exactly 2x one of them, so the growth factor
        # must leave room for that. Otherwise every sentence would end up alone
        # and the staccato delivery would come back through the timing rule.
        sentence = "This sentence is about seventy characters long, give or take a word."
        chunks = split_sentences(" ".join([sentence] * 13), 500, 3)
        assert any(chunk.count(".") > 1 for chunk in chunks), chunks

    def test_a_long_sentence_is_not_rejoined_by_batching(self):
        # hard_split pieces must each stand alone; merging them back would
        # exceed the cap that made them split in the first place.
        text = "First. " + "word " * 20 + ". Last."
        chunks = split_sentences(text, 30, 3)
        assert chunks[0] == "First."
        assert all(len(s) <= 30 for s in chunks)
        assert " ".join(chunks).split() == text.split()


class TestChunking:
    def test_short_text_is_still_one_chunk_per_sentence_in_single_mode(self, cfg, fake_engine):
        cfg.sentences_per_chunk = 1
        payload, _ = run(cfg, fake_engine, "Alpha. Beta. Gamma.")
        assert payload["chunks"] == 3
        assert [c["text"] for c in fake_engine.calls] == ["Alpha.", "Beta.", "Gamma."]

    def test_short_sentences_are_batched_by_default(self, cfg, fake_engine):
        # Each chunk is one generate() call, so batching short sentences removes
        # the boundary pauses that made the delivery sound staccato.
        payload, _ = run(cfg, fake_engine, "Alpha. Beta. Gamma. Delta.")
        # First sentence alone (fast start), then the rest in one batch of 3.
        assert payload["chunks"] == 2
        assert [c["text"] for c in fake_engine.calls] == ["Alpha.", "Beta. Gamma. Delta."]

    def test_batch_never_exceeds_the_character_cap(self, cfg, fake_engine):
        # The count is a wish, the char cap is a hard limit: Chatterbox truncates
        # a long input, so a batch must break early rather than overflow.
        cfg.sentences_per_chunk = 3
        cfg.max_chunk_chars = 30
        payload, _ = run(cfg, fake_engine, "One two three. Four five six. Seven eight nine.")
        assert payload["chunks"] >= 3
        assert all(len(call["text"]) <= 30 for call in fake_engine.calls)

    def test_over_long_sentence_is_capped(self, cfg, fake_engine):
        cfg.max_chunk_chars = 40
        payload, _ = run(cfg, fake_engine, "Alpha " * 20)
        assert payload["chunks"] > 1
        assert all(len(call["text"]) <= 40 for call in fake_engine.calls)

    def test_one_file_per_chunk(self, cfg, fake_engine):
        payload, last = run(cfg, fake_engine, "Alpha. " * 3)
        files = sorted(p.name for p in cfg.output_dir.glob("*.wav"))
        assert files == sorted(payload["files"])
        assert last["written"] == payload["files"]

    def test_chunks_share_the_style_knobs(self, cfg, fake_engine):
        run(cfg, fake_engine, "Alpha. " * 3, temperature=0.9, exaggeration=0.4)
        for call in fake_engine.calls:
            assert call["temperature"] == 0.9
            assert call["exaggeration"] == 0.4
