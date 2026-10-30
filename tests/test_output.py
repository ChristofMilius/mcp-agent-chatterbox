"""
The output directory as an inventory.
=====================================
The chunk files are the only record that a message was ever spoken, so the
parsing rules here decide what the panel offers to replay. Three of them are
easy to get wrong and silent when wrong: chunk numbers must sort numerically
(``_10`` after ``_9``), a message id may itself contain underscores, and a
caller-supplied id must be compared literally rather than used as a glob.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp_agent_chatterbox.output import chunk_files, group_chunks, scan_output


def touch(path: Path, size: int = 4, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
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
