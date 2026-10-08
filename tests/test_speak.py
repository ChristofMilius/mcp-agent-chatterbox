"""
The speak path: validation, the reference-clip requirement, and the happy path.

speak_once is shared by the MCP tool and the CLI, so these tests cover both
entry points at once. The engine is faked and playback is stubbed, so nothing
here downloads weights or touches the speakers.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.registry import SUPPORTED_LANGUAGES, T3_MODELS
from mcp_agent_chatterbox.speak import _split_segments, speak_once


@pytest.fixture(autouse=True)
def _no_sound(silent_playback):
    pass


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
    return speak_once(text, cfg=cfg, engine=engine, **kwargs)


class TestTextValidation:
    def test_rejects_empty(self, cfg, fake_engine):
        for bad in ("", "   ", "\n\t"):
            with pytest.raises(ToolFault) as exc:
                run(cfg, fake_engine, bad)
            assert exc.value.reason == "empty_text"
        assert fake_engine.calls == []

    def test_rejects_over_budget(self, cfg, fake_engine):
        cfg.max_chars = 10
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, "x" * 11)
        assert exc.value.reason == "text_too_long"
        assert exc.value.extra["max_chars"] == 10
        assert fake_engine.calls == []

    def test_strips_surrounding_whitespace(self, cfg, fake_engine):
        run(cfg, fake_engine, "  Build finished.  ")
        assert fake_engine.calls[0]["text"] == "Build finished."


class TestModelValidation:
    def test_rejects_unknown_model(self, cfg, fake_engine):
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="whisper")
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
            run(cfg, fake_engine, model="ResembleAI/chatterbox-turbo")
        assert exc.value.reason == "unknown_model"
        assert fake_engine.calls == []

    def test_alias_is_accepted(self, cfg, fake_engine, narrator):
        run(cfg, fake_engine, model="multi", voice=narrator)
        assert fake_engine.calls[0]["model"] == "multilingual"


class TestLanguageValidation:
    def test_rejects_unsupported(self, cfg, fake_engine):
        # Language validation runs before the clip check, so no voice is needed.
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="multilingual", language="xx")
        assert exc.value.reason == "unsupported_language"
        supported = cast("list[str]", exc.value.extra["supported"])
        assert "de" in supported

    def test_normalizes_case_and_whitespace(self, cfg, fake_engine, tmp_path):
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "narrator.wav").write_bytes(b"RIFF")
        run(cfg, fake_engine, model="multilingual", language="  DE  ", voice="narrator")
        assert fake_engine.calls[0]["language"] == "de"

    def test_ignored_for_non_multilingual(self, cfg, fake_engine):
        # turbo takes no language; passing one must not reach generate().
        run(cfg, fake_engine, model="turbo", language="de")
        assert fake_engine.calls[0]["language"] is None

    def test_blank_language_is_ignored(self, cfg, fake_engine):
        run(cfg, fake_engine, model="turbo", language="   ")
        assert fake_engine.calls[0]["language"] is None

    def test_every_advertised_language_passes(self, cfg, fake_engine, tmp_path):
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "narrator.wav").write_bytes(b"RIFF")
        for code in SUPPORTED_LANGUAGES:
            run(cfg, fake_engine, model="multilingual", language=code, voice="narrator")
        assert len(fake_engine.calls) == len(SUPPORTED_LANGUAGES)


class TestT3ModelValidation:
    def test_rejects_unknown(self, cfg, fake_engine):
        # t3 validation runs before the clip check, so no voice is needed.
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="multilingual", t3_model="v9")
        assert exc.value.reason == "unsupported_t3_model"

    def test_accepts_each_advertised(self, cfg, fake_engine, tmp_path):
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "narrator.wav").write_bytes(b"RIFF")
        for choice in T3_MODELS:
            run(cfg, fake_engine, model="multilingual", voice="narrator", t3_model=choice)
        assert {c["t3_model"] for c in fake_engine.calls} == set(T3_MODELS)

    def test_ignored_for_turbo(self, cfg, fake_engine):
        run(cfg, fake_engine, model="turbo", t3_model="v3")
        assert fake_engine.calls[0]["t3_model"] is None


class TestReferenceClipRequirement:
    def test_multilingual_without_clip_is_refused(self, cfg, fake_engine):
        # Chatterbox would raise a bare AssertionError here; the point of this
        # check is to return something the caller can act on.
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="multilingual", language="de")
        assert exc.value.reason == "reference_clip_required"
        assert fake_engine.calls == []

    def test_original_without_clip_is_refused(self, cfg, fake_engine):
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="original")
        assert exc.value.reason == "reference_clip_required"

    def test_turbo_needs_no_clip(self, cfg, fake_engine):
        assert run(cfg, fake_engine, model="turbo")["status"] == "ok"

    def test_voice_and_reference_clip_are_exclusive(self, cfg, fake_engine, tmp_path):
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"RIFF")
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "narrator.wav").write_bytes(b"RIFF")
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="turbo", voice="narrator", reference_clip=str(clip))
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
            run(cfg, fake_engine, model="multilingual", reference_clip=str(tmp_path / "gone.wav"))
        assert exc.value.reason == "reference_clip_not_found"

    def test_wrong_extension_is_reported(self, cfg, fake_engine, voices_dir, tmp_path):
        bad = tmp_path / "clip.txt"
        bad.write_bytes(b"nope")
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="multilingual", reference_clip=str(bad))
        assert exc.value.reason == "unsupported_reference_format"


class TestOutput:
    def test_writes_a_wav(self, cfg, fake_engine):
        result = run(cfg, fake_engine)
        path = cfg.output_dir / result["filename"]
        assert path.is_file()
        assert path.stat().st_size > 0
        assert result["path"] == str(path)

    def test_reports_duration_and_rate(self, cfg, fake_engine):
        result = run(cfg, fake_engine)
        # 2400 samples at 24000 Hz is 0.1 s.
        assert result["sample_rate"] == 24000
        assert result["duration_s"] == 0.1

    def test_explicit_filename_is_used(self, cfg, fake_engine):
        result = run(cfg, fake_engine, filename="take-one")
        assert result["filename"] == "take-one.wav"
        assert (cfg.output_dir / "take-one.wav").is_file()

    def test_explicit_filename_is_sanitized(self, cfg, fake_engine):
        result = run(cfg, fake_engine, filename="../../escape")
        assert "/" not in result["filename"]
        assert ".." not in result["filename"]
        assert (cfg.output_dir / result["filename"]).is_file()

    def test_creates_missing_output_dir(self, cfg, fake_engine):
        assert not cfg.output_dir.exists()
        run(cfg, fake_engine)
        assert cfg.output_dir.is_dir()

    def test_result_is_json_serialisable(self, cfg, fake_engine):
        result = run(cfg, fake_engine)
        assert json.loads(json.dumps(result))["status"] == "ok"


class TestPlayback:
    def test_disabled_by_config(self, cfg, fake_engine):
        cfg.autoplay = False
        result = run(cfg, fake_engine)
        assert result["played"] is False

    def test_explicit_override_wins(self, cfg, fake_engine, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play",
            lambda *a, **k: calls.append(a) or (True, ""),
        )
        cfg.autoplay = False
        result = run(cfg, fake_engine, play_audio=True)
        assert result["played"] is True
        assert len(calls) == 1

    def test_failure_is_reported_not_raised(self, cfg, fake_engine, monkeypatch):
        # A machine with no sound card must still get its WAV written.
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play",
            lambda *a, **k: (False, "no audio device"),
        )
        result = run(cfg, fake_engine, play_audio=True)
        assert result["status"] == "ok"
        assert result["played"] is False
        assert result["playback_note"] == "no audio device"


class TestProgressive:
    LONG = "Alpha red green blue. Beta yellow orange purple. Gamma pink black white."

    def _record_play(self, monkeypatch):
        calls = []

        def fake_play(*a, **k):
            calls.append(a[0])
            return True, None

        monkeypatch.setattr("mcp_agent_chatterbox.speak.play", fake_play)
        return calls

    def test_single_chunk_ignores_progressive(self, cfg, fake_engine, monkeypatch):
        calls = self._record_play(monkeypatch)
        result = run(cfg, fake_engine, progressive=True, play_audio=True)
        assert result["progressive"] is False
        assert len(calls) == 1  # the final file, played as always

    def test_streams_chunks_in_order_and_cleans_up(self, cfg, fake_engine, monkeypatch):
        calls = self._record_play(monkeypatch)
        cfg.max_chunk_chars = 30
        result = run(cfg, fake_engine, text=self.LONG, progressive=True, play_audio=True, wait=True)
        segments = _split_segments(self.LONG, 30)
        assert len(segments) >= 2
        assert result["progressive"] is True
        assert result["played"] is True
        assert result["chunks"] == len(segments)
        assert len(calls) == len(segments)
        # Chunks played in rendering order, back-to-back, no final-file play.
        assert all(str(c).endswith(f".progressive-{i + 1}.wav") for i, c in enumerate(calls))
        staged = list(cfg.output_dir.glob("*.progressive-*.wav"))
        assert staged == []  # played chunks are removed again
        finals = list(cfg.output_dir.glob("*.wav"))
        assert len(finals) == 1  # only the stitched artifact remains

    def test_requires_playback(self, cfg, fake_engine, monkeypatch):
        calls = self._record_play(monkeypatch)
        cfg.max_chunk_chars = 30
        result = run(
            cfg, fake_engine, text=self.LONG, progressive=True, play_audio=False, wait=True
        )
        assert result["progressive"] is False
        assert calls == []
        assert list(cfg.output_dir.glob("*.progressive-*.wav")) == []

    def test_enabled_by_config_and_wait_joins(self, cfg, fake_engine, monkeypatch):
        calls = self._record_play(monkeypatch)
        cfg.max_chunk_chars = 30
        cfg.progressive = True
        result = run(cfg, fake_engine, text=self.LONG, play_audio=True, wait=True)
        assert result["progressive"] is True
        assert len(calls) == len(_split_segments(self.LONG, 30))

    def test_failed_chunks_do_not_stop_cleanup(self, cfg, fake_engine, monkeypatch):
        monkeypatch.setattr(
            "mcp_agent_chatterbox.speak.play",
            lambda *a, **k: (False, "no audio device"),
        )
        cfg.max_chunk_chars = 30
        result = run(cfg, fake_engine, text=self.LONG, progressive=True, play_audio=True, wait=True)
        assert result["progressive"] is True
        assert list(cfg.output_dir.glob("*.progressive-*.wav")) == []


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
        cfg.max_chunk_chars = 40
        run(cfg, fake_engine, "Alpha. " * 6, top_p=0.9, top_k=500, seed=7)
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
            run(cfg, fake_engine, **kwargs)
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
        result = run(cfg, fake_engine, seed=0)
        assert result["status"] == "ok"


class TestSegmentSplitting:
    def test_short_text_is_one_segment(self):
        assert _split_segments("Hello.", 500) == ["Hello."]

    def test_packs_sentences_up_to_budget(self):
        text = "Sentence one. " * 5
        segs = _split_segments(text, 40)
        assert all(len(s) <= 40 for s in segs)
        assert len(segs) == 3
        assert "".join("".join(s.split()) for s in segs) == "".join(text.split())

    def test_hard_splits_an_oversize_sentence_on_word_boundaries(self):
        text = "The quick brown fox jumps over the lazy dog."  # 46 chars, one sentence
        segs = _split_segments(text, 20)
        assert len(segs) >= 2
        assert all(len(s) <= 20 for s in segs)
        words = text.split()
        joined_words = " ".join(segs).split()
        assert "".join(joined_words) == "".join(words)

    def test_splits_at_newlines(self):
        # The newline sticks to the preceding sentence; speak_one strips each
        # segment before synthesis, so the rendered text is clean either way.
        segs = _split_segments("Line one.\nLine two.\nLine three.", 12)
        assert [s.strip() for s in segs] == ["Line one.", "Line two.", "Line three."]

    def test_no_punctuation_hard_splits_evenly(self):
        text = "word " * 30  # 150 chars, no enders at all
        segs = _split_segments(text, 40)
        assert all(len(s) <= 40 for s in segs)
        assert "".join("".join(s.split()) for s in segs) == "".join(text.split())


class TestChunking:
    def test_short_text_is_a_single_call(self, cfg, fake_engine):
        run(cfg, fake_engine, "Alpha. Beta. Gamma.")
        assert len(fake_engine.calls) == 1
        assert "chunks" not in run(cfg, fake_engine, "Alpha. Beta. Gamma.")

    def test_long_text_is_split_into_chunks(self, cfg, fake_engine):
        cfg.max_chunk_chars = 40
        body = "Alpha. " * 6
        result = run(cfg, fake_engine, body)
        assert len(fake_engine.calls) == 2
        assert all(len(call["text"]) <= 40 for call in fake_engine.calls)
        assert result["chunks"] == 2

    def test_chunked_result_is_one_wav_with_silence(self, cfg, fake_engine):
        cfg.max_chunk_chars = 40
        result = run(cfg, fake_engine, "Alpha. " * 6)
        # 2 chunks of 2400 samples + 1 inter-chunk gap (250 ms @ 24 kHz).
        assert result["sample_rate"] == 24000
        assert result["duration_s"] == pytest.approx(0.45)
        assert result["text_chars"] == len(("Alpha. " * 6).strip())

    def test_chunked_output_still_writes_a_single_file(self, cfg, fake_engine):
        cfg.max_chunk_chars = 40
        result = run(cfg, fake_engine, "Alpha. " * 6)
        path = cfg.output_dir / result["filename"]
        assert path.is_file()
        assert result["filename"] == path.name

    def test_chunks_share_the_style_knobs(self, cfg, fake_engine):
        cfg.max_chunk_chars = 40
        run(cfg, fake_engine, "Alpha. " * 6, temperature=0.9, exaggeration=0.4)
        for call in fake_engine.calls:
            assert call["temperature"] == 0.9
            assert call["exaggeration"] == 0.4
