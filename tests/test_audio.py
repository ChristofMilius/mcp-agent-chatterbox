"""
The WAV container, and the metadata inside it.
===============================================
A chunk file is the only record of what was spoken, so `write_wav` can carry a
RIFF ``LIST/INFO`` chunk alongside the samples. Two properties matter: the tags
must come back exactly as written (UTF-8 included, since replies are not always
English), and adding them must not disturb the audio — the file has to stay a
canonical WAV that `wave`, winsound and ffmpeg all read.

The parser below is deliberately independent of the writer: it walks the RIFF
structure the way a consumer would, so a mistake in the writer cannot pass by
agreeing with itself.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest

from mcp_agent_chatterbox.audio import _with_info

_RATE = 24000
_FRAMES = 240


def make_wav(frames: int = _FRAMES, rate: int = _RATE) -> bytes:
    """A canonical mono PCM16 WAV, the shape `wave` produces for a render."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def read_info(wav_bytes: bytes) -> dict[str, str]:
    """Every LIST/INFO tag in the file, decoded the way a reader would."""
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    tags: dict[str, str] = {}
    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        body = wav_bytes[offset + 8 : offset + 8 + size]
        if chunk_id == b"LIST" and body[:4] == b"INFO":
            position = 4
            while position + 8 <= len(body):
                code = body[position : position + 4].decode("ascii")
                length = struct.unpack_from("<I", body, position + 4)[0]
                value = body[position + 8 : position + 8 + length]
                tags[code] = value.rstrip(b"\x00").decode("utf-8")
                position += 8 + length + (length & 1)
        offset += 8 + size + (size & 1)
    return tags


def read_frames(wav_bytes: bytes) -> tuple[int, int]:
    """(frame count, sample rate) read back through the stdlib reader."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes(), w.getframerate()


class TestInfoChunk:
    def test_tags_come_back_unchanged(self):
        tagged = _with_info(make_wav(), {"INAM": "1712345678901 #1/3", "ICMT": "Build finished."})
        assert read_info(tagged) == {"INAM": "1712345678901 #1/3", "ICMT": "Build finished."}

    def test_the_audio_is_untouched(self):
        # The chunk goes in ahead of `data`; the samples must still be readable
        # and the length unchanged, or the added tags would be a regression.
        tagged = _with_info(make_wav(), {"ICMT": "Hello."})
        assert read_frames(tagged) == (_FRAMES, _RATE)
        original = make_wav()
        assert tagged.endswith(original[original.index(b"data") :])

    def test_utf8_values_round_trip(self):
        # Replies are often German; INFO is byte-oriented, so the writer has to
        # pick an encoding and the reader has to get the same text back.
        text = "Grüße aus München — schön, dass du da bist."
        assert read_info(_with_info(make_wav(), {"ICMT": text}))["ICMT"] == text

    def test_an_odd_length_value_leaves_the_file_even(self):
        # RIFF chunks are padded to even sizes. Get this wrong and the next
        # chunk starts one byte early, which corrupts the samples silently.
        tagged = _with_info(make_wav(), {"ICMT": "odd"})
        assert len(tagged) % 2 == 0
        assert read_frames(tagged) == (_FRAMES, _RATE)

    def test_no_info_means_the_file_is_unchanged(self):
        original = make_wav()
        assert _with_info(original, {}) == original

    def test_an_embedded_nul_cannot_truncate_the_tag(self):
        # A NUL inside the value would end the string early for every reader.
        assert read_info(_with_info(make_wav(), {"ICMT": "a\x00b"}))["ICMT"] == "ab"

    def test_a_malformed_code_is_refused(self):
        with pytest.raises(ValueError, match="4 ASCII"):
            _with_info(make_wav(), {"CMT": "too short"})

    def test_a_non_wav_is_refused(self):
        with pytest.raises(ValueError, match="RIFF/WAVE"):
            _with_info(b"not a wav", {"ICMT": "x"})
