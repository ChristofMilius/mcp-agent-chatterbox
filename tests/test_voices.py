"""Voice lookup: exact names, unique prefixes, and refusing to guess."""

from __future__ import annotations

import wave

import pytest

from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.voices import SHORT_CLIP_S, list_voices, resolve_voice


def _write_wav(path, seconds, rate=24000, channels=1):
    """A real, decodable WAV header -- _probe reads container metadata."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * channels * seconds))
    return path


@pytest.fixture
def voices(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    for name in ("seven.wav", "narrator.mp3", "Ann.WAV"):
        (d / name).write_bytes(b"RIFF" + b"\0" * 64)
    (d / "notes.txt").write_text("not a clip")
    (d / "subdir").mkdir()
    return d


class TestListVoices:
    def test_lists_only_audio(self, voices):
        names = {v["name"] for v in list_voices(voices)}
        assert names == {"seven", "narrator", "Ann"}

    def test_excludes_non_audio(self, voices):
        assert "notes" not in {v["name"] for v in list_voices(voices)}

    def test_missing_dir_is_empty_not_an_error(self, tmp_path):
        # A fresh install has no voices/ yet; that is a normal state.
        assert list_voices(tmp_path / "nope") == []

    def test_reports_size(self, voices):
        entry = next(v for v in list_voices(voices) if v["name"] == "seven")
        assert entry["size_kb"] == pytest.approx(0.1, abs=0.01)


class TestClipMetadata:
    """Duration is the property that predicted clone quality, so it is listed."""

    def test_reports_duration_rate_channels(self, tmp_path):
        d = tmp_path / "voices"
        d.mkdir()
        _write_wav(d / "long.wav", 40.1, rate=48000, channels=2)
        entry = list_voices(d)[0]
        assert entry["duration_s"] == pytest.approx(40.1, abs=0.01)
        assert entry["sample_rate"] == 48000
        assert entry["channels"] == 2

    def test_unreadable_clip_degrades_to_nulls(self, voices):
        # The fixture writes RIFF + noise, not a real header. Listing must still
        # succeed -- a clip we cannot probe may still be usable.
        entry = next(v for v in list_voices(voices) if v["name"] == "seven")
        assert entry["duration_s"] is None
        assert entry["sample_rate"] is None
        assert "quality_note" not in entry

    def test_short_clip_is_flagged(self, tmp_path):
        d = tmp_path / "voices"
        d.mkdir()
        _write_wav(d / "brief.wav", SHORT_CLIP_S - 5)
        entry = list_voices(d)[0]
        assert "quality_note" in entry
        assert "crackle" in entry["quality_note"]

    def test_long_clip_is_not_flagged(self, tmp_path):
        # The 40.1s reference that measurably beat an 11.8s excerpt must not
        # be warned about.
        d = tmp_path / "voices"
        d.mkdir()
        _write_wav(d / "seven.wav", 40.1, rate=48000, channels=2)
        assert "quality_note" not in list_voices(d)[0]

    def test_unknown_duration_is_never_flagged(self, tmp_path):
        d = tmp_path / "voices"
        d.mkdir()
        (d / "opaque.wav").write_bytes(b"RIFF" + b"\0" * 64)
        assert "quality_note" not in list_voices(d)[0]


class TestResolveVoice:
    def test_exact_match(self, voices):
        assert resolve_voice(voices, "seven").name == "seven.wav"

    def test_case_insensitive(self, voices):
        assert resolve_voice(voices, "SEVEN").name == "seven.wav"
        assert resolve_voice(voices, "ann").name == "Ann.WAV"

    def test_extension_accepted(self, voices):
        assert resolve_voice(voices, "seven.wav").name == "seven.wav"

    def test_unique_prefix(self, voices):
        assert resolve_voice(voices, "narr").name == "narrator.mp3"

    def test_ambiguous_prefix_refuses_to_guess(self, tmp_path):
        d = tmp_path / "voices"
        d.mkdir()
        (d / "clone_a.wav").write_bytes(b"RIFF")
        (d / "clone_b.wav").write_bytes(b"RIFF")
        with pytest.raises(ToolFault) as exc:
            resolve_voice(d, "clone")
        # The message must list the candidates so the caller can retry.
        assert exc.value.reason == "voice_ambiguous"
        assert "clone_a" in exc.value.message

    def test_exact_beats_prefix(self, tmp_path):
        # "sam" is a prefix of "samantha", but "sam" itself must win.
        d = tmp_path / "voices"
        d.mkdir()
        (d / "sam.wav").write_bytes(b"RIFF")
        (d / "samantha.wav").write_bytes(b"RIFF")
        assert resolve_voice(d, "sam").name == "sam.wav"

    def test_unknown_name_lists_available(self, voices):
        with pytest.raises(ToolFault) as exc:
            resolve_voice(voices, "nobody")
        assert exc.value.reason == "voice_not_found"
        assert "seven" in exc.value.message

    def test_empty_name(self, voices):
        with pytest.raises(ToolFault) as exc:
            resolve_voice(voices, "  ")
        assert exc.value.reason == "voice_missing"

    def test_missing_dir_suggests_turbo(self, tmp_path):
        with pytest.raises(ToolFault) as exc:
            resolve_voice(tmp_path / "nope", "seven")
        assert exc.value.reason == "voices_dir_missing"
        assert "turbo" in exc.value.message

    def test_path_traversal_is_not_resolved(self, voices, tmp_path):
        # resolve_voice only ever looks at stems inside the voices dir, so a
        # traversal string must not escape it.
        with pytest.raises(ToolFault) as exc:
            resolve_voice(voices, "../../secret")
        assert exc.value.reason in {"voice_not_found", "voice_ambiguous"}
