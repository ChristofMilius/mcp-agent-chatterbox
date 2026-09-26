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
from mcp_agent_chatterbox.speak import speak_once


@pytest.fixture(autouse=True)
def _no_sound(silent_playback):
    pass


@pytest.fixture
def seven(cfg):
    """Put one reference clip in the voices dir and return its name."""
    cfg.voices_dir.mkdir(parents=True, exist_ok=True)
    (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF" + b"\0" * 32)
    return "seven"


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

    def test_defaults_to_configured_model(self, cfg, fake_engine, seven):
        cfg.model = "original"
        run(cfg, fake_engine, voice=seven)
        assert fake_engine.calls[0]["model"] == "original"

    def test_full_hf_repo_id_is_rejected_clearly(self, cfg, fake_engine):
        # Models are addressed by short key. A repo id must be refused by name,
        # not silently swapped for some other model.
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="ResembleAI/chatterbox-turbo")
        assert exc.value.reason == "unknown_model"
        assert fake_engine.calls == []

    def test_alias_is_accepted(self, cfg, fake_engine, seven):
        run(cfg, fake_engine, model="multi", voice=seven)
        assert fake_engine.calls[0]["model"] == "multilingual"


class TestLanguageValidation:
    def test_rejects_unsupported(self, cfg, fake_engine):
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="multilingual", language="xx", voice="seven")
        assert exc.value.reason == "unsupported_language"
        supported = cast("list[str]", exc.value.extra["supported"])
        assert "de" in supported

    def test_normalizes_case_and_whitespace(self, cfg, fake_engine, tmp_path):
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF")
        run(cfg, fake_engine, model="multilingual", language="  DE  ", voice="seven")
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
        (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF")
        for code in SUPPORTED_LANGUAGES:
            run(cfg, fake_engine, model="multilingual", language=code, voice="seven")
        assert len(fake_engine.calls) == len(SUPPORTED_LANGUAGES)


class TestT3ModelValidation:
    def test_rejects_unknown(self, cfg, fake_engine, tmp_path):
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF")
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="multilingual", voice="seven", t3_model="v9")
        assert exc.value.reason == "unsupported_t3_model"

    def test_accepts_each_advertised(self, cfg, fake_engine, tmp_path):
        (cfg.voices_dir).mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF")
        for choice in T3_MODELS:
            run(cfg, fake_engine, model="multilingual", voice="seven", t3_model=choice)
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
        (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF")
        with pytest.raises(ToolFault) as exc:
            run(cfg, fake_engine, model="turbo", voice="seven", reference_clip=str(clip))
        assert exc.value.reason == "ambiguous_voice"


class TestReferenceClipResolution:
    @pytest.fixture
    def voices_dir(self, cfg):
        cfg.voices_dir.mkdir(parents=True, exist_ok=True)
        (cfg.voices_dir / "seven.wav").write_bytes(b"RIFF" + b"\0" * 32)
        return cfg.voices_dir

    def test_voice_name_resolves_to_path(self, cfg, fake_engine, voices_dir):
        run(cfg, fake_engine, model="multilingual", voice="seven", language="en")
        assert fake_engine.calls[0]["reference_clip"].endswith("seven.wav")

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
