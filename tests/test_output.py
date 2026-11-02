"""
The output directory as an inventory.
=====================================
The chunk files are the only record that a message was ever spoken, so the
parsing rules here decide what the panel offers to replay. Three of them are
easy to get wrong and silent when wrong: chunk numbers must sort numerically
(``_10`` after ``_9``), a message id may itself contain underscores, and a
caller-supplied id must be compared literally rather than used as a glob.

The transcript is reconstructed from the files, and the panel asks for it every
two seconds, so the extraction is cached: these tests cover both the joining
order and the cache invalidation.
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import pytest

from mcp_agent_chatterbox import output
from mcp_agent_chatterbox.audio import _with_info
from mcp_agent_chatterbox.output import chunk_files, group_chunks, scan_output


def touch(path: Path, size: int = 4, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def tagged(path: Path, text: str, code: str = "ICMT") -> Path:
    """A real WAV carrying `text` in a RIFF INFO tag, as a render would write."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 240)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_with_info(buffer.getvalue(), {code: text}))
    return path


class TestGrouping:
    def test_chunk_numbers_sort_numerically(self, tmp_path):
        for number in (10, 1, 2):
            touch(tmp_path / f"msg_{number}.wav")
        grouped = group_chunks(tmp_path)
        assert [number for number, _path in grouped["msg"]] == [1, 2, 10]

    def test_an_id_may_contain_underscores(self, tmp_path):
        touch(tmp_path / "my_long_name_3.wav")
        touch(tmp_path / "my_long_name_1.wav")
        grouped = group_chunks(tmp_path)
        assert set(grouped) == {"my_long_name"}
        assert [p.name for _n, p in grouped["my_long_name"]] == [
            "my_long_name_1.wav",
            "my_long_name_3.wav",
        ]

    def test_non_chunk_files_are_ignored(self, tmp_path):
        touch(tmp_path / "notes.txt")
        touch(tmp_path / "msg.wav")
        touch(tmp_path / "msg_x.wav")
        (tmp_path / "subdir_1.wav").mkdir()
        assert group_chunks(tmp_path) == {}

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert group_chunks(tmp_path / "nope") == {}


class TestChunkFiles:
    def test_only_the_requested_message(self, tmp_path):
        touch(tmp_path / "one_1.wav")
        touch(tmp_path / "two_1.wav")
        touch(tmp_path / "one_2.wav")
        assert [p.name for p in chunk_files(tmp_path, "one")] == ["one_1.wav", "one_2.wav"]

    def test_a_glob_character_in_the_id_matches_nothing(self, tmp_path):
        # If this were passed to glob(), '*' would list every message there is.
        touch(tmp_path / "one_1.wav")
        assert chunk_files(tmp_path, "*") == []


class TestScan:
    def test_newest_first_with_totals(self, tmp_path):
        touch(tmp_path / "old_1.wav", size=10, mtime=1_000)
        touch(tmp_path / "new_1.wav", size=5, mtime=3_000)
        touch(tmp_path / "new_2.wav", size=7, mtime=3_100)

        messages = scan_output(tmp_path)

        assert [m["message_id"] for m in messages] == ["new", "old"]
        assert messages[0]["chunks"] == 2
        assert messages[0]["bytes"] == 12
        assert messages[0]["modified"] == 3_100
        assert messages[0]["files"] == ["new_1.wav", "new_2.wav"]

    def test_limit_truncates_to_the_newest(self, tmp_path):
        for index in range(5):
            touch(tmp_path / f"m{index}_1.wav", mtime=100 + index)
        messages = scan_output(tmp_path, limit=2)
        assert [m["message_id"] for m in messages] == ["m4", "m3"]

    def test_empty_directory(self, tmp_path):
        assert scan_output(tmp_path) == []


class TestTranscript:
    """
    What each file says, read back out of the file itself.

    The panel shows this next to the play button, so it has to be the message in
    playback order — and it has to survive the two properties of a poll: it runs
    constantly, and it must not be fooled by a file that was replaced.
    """

    def test_chunks_are_joined_in_playback_order(self, tmp_path):
        # Written out of order on purpose: the joined text must follow chunk
        # numbers, not directory order.
        tagged(tmp_path / "msg_10.wav", "Eleventh.")
        tagged(tmp_path / "msg_2.wav", "Second.")
        tagged(tmp_path / "msg_1.wav", "First.")

        messages = scan_output(tmp_path)

        assert messages[0]["transcript"] == "First. Second. Eleventh."

    def test_a_file_without_tags_contributes_nothing(self, tmp_path):
        # Files rendered before the tags existed still replay; their text is
        # simply unknown, and the panel has to survive that.
        tagged(tmp_path / "msg_1.wav", "Known.")
        touch(tmp_path / "msg_2.wav")

        assert scan_output(tmp_path)[0]["transcript"] == "Known."

    def test_a_truncated_file_does_not_blank_the_listing(self, tmp_path):
        # One bad old file must not cost the whole history, which is the only
        # route back to the audio.
        tagged(tmp_path / "msg_1.wav", "Good.")
        broken = tmp_path / "msg_2.wav"
        broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")

        messages = scan_output(tmp_path)

        assert [m["message_id"] for m in messages] == ["msg"]
        assert messages[0]["transcript"] == "Good."

    def test_the_extraction_is_cached_between_polls(self, tmp_path, monkeypatch):
        tagged(tmp_path / "msg_1.wav", "Read once.")
        calls: list[Path] = []
        real = output.read_info
        monkeypatch.setattr(output, "read_info", lambda path: calls.append(path) or real(path))

        scan_output(tmp_path)
        scan_output(tmp_path)

        assert len(calls) == 1

    def test_a_replaced_file_is_read_again(self, tmp_path, monkeypatch):
        # A re-render writes a different chunk at the same path. Serving the
        # cached text would show a transcript the file no longer says. The
        # replacement is the same length on purpose, so only a changed mtime can
        # reveal it.
        path = tagged(tmp_path / "msg_1.wav", "First take.")
        calls: list[Path] = []
        real = output.read_info
        monkeypatch.setattr(output, "read_info", lambda p: calls.append(p) or real(p))
        assert scan_output(tmp_path)[0]["transcript"] == "First take."

        tagged(path, "Other take.")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))

        assert scan_output(tmp_path)[0]["transcript"] == "Other take."
        assert len(calls) == 2

    def test_a_deleted_file_drops_out_of_the_listing(self, tmp_path):
        tagged(tmp_path / "msg_1.wav", "Here.")
        tagged(tmp_path / "msg_2.wav", "Gone.").unlink()

        message = scan_output(tmp_path)[0]

        assert message["files"] == ["msg_1.wav"]
        assert message["transcript"] == "Here."


@pytest.mark.parametrize("limit", [-1, 0])
def test_a_non_positive_limit_returns_nothing(tmp_path, limit):
    tagged(tmp_path / "msg_1.wav", "Text.")
    assert scan_output(tmp_path, limit=limit) == []
