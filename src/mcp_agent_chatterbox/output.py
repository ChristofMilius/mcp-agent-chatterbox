"""
output.py — what has already been rendered
==========================================
The chunk files in the output directory are the record of every utterance that
was ever spoken: ``<message_id>_<n>.wav``, one file per rendered chunk. Nothing
in the worker remembers a message after its summary is replaced, so anything
that wants to *replay* an utterance, or to list what can be replayed, has to
read that directory.

Two callers use this module and they must agree on what a chunk file is:

  * the worker, which answers ``history`` (what is on disk) and ``replay``
    (play one message's files again, without re-rendering);
  * the panel, which shows the history and offers a play button per message.

Ordering is by the numeric suffix, never by plain string sort: ``_10.wav`` must
come after ``_9.wav``, not between ``_1.wav`` and ``_2.wav``.

Message ids may contain underscores — a caller-supplied ``filename=`` is
slugified and keeps its separators — so the id is everything before the LAST
``_<digits>.wav``, which is why the pattern is greedy on the id and anchored on
the suffix.

Transcript
----------
Each chunk file records its own text in a RIFF ``ICMT`` tag (see ``audio.py``),
so the spoken text of a message is recovered by reading the files rather than
remembered anywhere. Reading a tag means opening the file, and ``scan_output``
runs on every panel poll (every 2 s), so the extraction is cached by
``(path, mtime_ns, size)``: a directory of a hundred old chunks costs a hundred
``stat`` calls, not a hundred file reads.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from mcp_agent_chatterbox.audio import read_info

logger = logging.getLogger(__name__)

#: Read transcripts, keyed by path: ``(mtime_ns, size, text)``. Bounded by
#: `_TEXT_CACHE_LIMIT` and pruned oldest-first, since a long-lived worker sees
#: every message ever rendered while a poll only needs the recent ones.
_text_cache: dict[Path, tuple[int, int, str]] = {}
_TEXT_CACHE_LIMIT = 2000


def _chunk_text(path: Path) -> str:
    """
    The text a chunk file says, from its ``ICMT`` tag. Empty when it has none.

    Cached on the file's identity, so a re-render at the same path is picked up
    (a new mtime or size) while a repeated poll over old files is not re-read.
    """
    try:
        stat = path.stat()
    except OSError:
        return ""
    cached = _text_cache.get(path)
    if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
        return cached[2]
    text = read_info(path).get("ICMT", "").strip()
    if len(_text_cache) >= _TEXT_CACHE_LIMIT:
        # Plain dicts keep insertion order, so this drops the oldest entries.
        for stale in list(_text_cache)[: _TEXT_CACHE_LIMIT // 4]:
            del _text_cache[stale]
    _text_cache[path] = (stat.st_mtime_ns, stat.st_size, text)
    return text


#: One rendered chunk: ``<message_id>_<n>.wav``. The id group is greedy so an
#: id containing underscores stays whole ('my_long_name_3.wav' -> 'my_long_name').
_CHUNK_NAME = re.compile(r"^(?P<id>.+)_(?P<number>\d+)\.wav$")


def group_chunks(output_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """
    Every chunk file in `output_dir`, grouped by message id.

    Each group is sorted by chunk number. Unreadable directories and files that
    disappear mid-scan are skipped rather than raised: this is an inventory for
    a UI, and a listing that fails because one old file was deleted is worse
    than a listing that omits it.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    try:
        entries = list(output_dir.iterdir())
    except OSError:
        return groups
    for path in entries:
        match = _CHUNK_NAME.match(path.name)
        if match is None:
            continue
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        groups.setdefault(match.group("id"), []).append((int(match.group("number")), path))
    for items in groups.values():
        items.sort(key=lambda item: item[0])
    return groups


def chunk_files(output_dir: Path, message_id: str) -> list[Path]:
    """
    The chunk files of one message, in playback order. Empty when none exist.

    `message_id` is compared literally against the parsed id, never used as a
    glob: a caller-supplied ``*`` or ``?`` must not list the whole directory.
    """
    if not message_id:
        return []
    return [path for _number, path in group_chunks(output_dir).get(message_id, [])]


def scan_output(output_dir: Path, limit: int = 20) -> list[dict]:
    """
    Recent messages, newest first, as JSON-safe dicts.

    `modified` is the newest chunk's mtime as seconds since the epoch, and
    `bytes` the total size of the message's files, so a caller can show an age
    and a size without touching the filesystem itself.

    `transcript` is the chunk texts joined in playback order: the whole message,
    reconstructed from the files. It is empty for files written before the tags
    existed, which the panel shows as unknown rather than blank.
    """
    messages: list[dict] = []
    for message_id, items in group_chunks(output_dir).items():
        newest = 0.0
        total = 0
        parts: list[str] = []
        for _number, path in items:
            try:
                stat = path.stat()
            except OSError:
                continue
            newest = max(newest, stat.st_mtime)
            total += stat.st_size
            text = _chunk_text(path)
            if text:
                parts.append(text)
        messages.append(
            {
                "message_id": message_id,
                "chunks": len(items),
                "bytes": total,
                "modified": round(newest, 3),
                "files": [path.name for _number, path in items],
                "transcript": " ".join(parts),
            }
        )
    messages.sort(key=lambda message: message["modified"], reverse=True)
    return messages[: max(0, limit)]


__all__ = ["chunk_files", "group_chunks", "scan_output"]
