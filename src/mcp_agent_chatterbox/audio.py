"""
audio.py — waveform → file, and duration arithmetic
====================================================
Chatterbox's generate() hands back a torch tensor on CPU, shaped (1, N) for
every one of the four models. Saving and measuring are wrapped here so the
tool layer never has to care about rank or device.

WAV writing is done with the stdlib `wave` module: the waveform is mono
PCM16 at a fixed sample rate, which is precisely writable without any media
backend. torchaudio's save() used to serve, but torchaudio 2.11+ moved all
I/O to the separate torchcodec package, and a flat PCM16 write does not
justify a new dependency.

Metadata
--------
`write_wav` takes an optional `info` mapping and writes it as a RIFF
``LIST/INFO`` chunk ahead of the samples: 4-character codes such as ``ICMT``
(comment), ``INAM`` (title) and ``ISFT`` (software). This is the standard WAV
tag container — ffprobe, foobar2000, VLC and Windows read it, and a reader that
does not know the chunk skips it, so the audio path is unaffected. The stdlib
`wave` module cannot emit extra chunks, so the file is built in memory and the
chunk spliced in before ``data``.
"""

from __future__ import annotations

import io
import logging
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _as_batched_tensor(wav):
    """Coerce a Chatterbox return value to a (channels, samples) CPU tensor."""
    import torch  # noqa: PLC0415

    if isinstance(wav, torch.Tensor):
        tensor = wav.detach().to("cpu")
    else:
        tensor = torch.as_tensor(wav)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor


def num_samples(wav) -> int:
    """Number of audio samples in the waveform."""
    return int(_as_batched_tensor(wav).shape[-1])


def duration_s(wav, sample_rate: int) -> float:
    """Length of the waveform in seconds, rounded to two decimals."""
    if sample_rate <= 0:
        return 0.0
    return round(num_samples(wav) / sample_rate, 2)


def read_info(path: str | Path, limit: int = 256 * 1024) -> dict[str, str]:
    """
    Every RIFF INFO tag in a WAV file, or {} when it has none.

    Only the front of the file is read, since the tag chunk is written ahead of
    the samples. Unreadable and malformed files return {} rather than raising:
    this feeds a history listing, and one old truncated file must not blank the
    whole page.
    """
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(limit)
    except OSError:
        return {}
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return {}
    tags: dict[str, str] = {}
    offset = 12
    while offset + 8 <= len(head):
        chunk_id = head[offset : offset + 4]
        size = struct.unpack_from("<I", head, offset + 4)[0]
        if chunk_id == b"data":
            break
        body = head[offset + 8 : offset + 8 + size]
        if chunk_id == b"LIST" and body[:4] == b"INFO":
            position = 4
            while position + 8 <= len(body):
                code = body[position : position + 4].decode("ascii", "replace")
                length = struct.unpack_from("<I", body, position + 4)[0]
                value = body[position + 8 : position + 8 + length]
                tags[code] = value.rstrip(b"\x00").decode("utf-8", "replace")
                position += 8 + length + (length & 1)
        offset += 8 + size + (size & 1)
    return tags


def _info_chunk(info: Mapping[str, str]) -> bytes:
    """
    One RIFF ``LIST/INFO`` chunk carrying `info`.

    Values are NUL-terminated UTF-8, which is what ffmpeg writes and what
    ffprobe, foobar2000 and VLC read back; the length includes the terminator,
    and both subchunk bodies and the whole chunk are padded to even sizes as
    RIFF requires. Codes are validated because a three-character one would be
    written happily and then be invisible to every reader.
    """
    body = bytearray(b"INFO")
    for code, value in info.items():
        if len(code) != 4 or not code.isascii():
            raise ValueError(f"INFO tag code must be 4 ASCII characters, got {code!r}")
        raw = value.replace("\x00", "").encode("utf-8") + b"\x00"
        if len(raw) % 2:
            raw += b"\x00"
        body += code.encode("ascii") + struct.pack("<I", len(raw)) + raw
    if len(body) % 2:
        body += b"\x00"
    return b"LIST" + struct.pack("<I", len(body)) + bytes(body)


def _with_info(wav_bytes: bytes, info: Mapping[str, str]) -> bytes:
    """`wav_bytes` with an INFO chunk spliced in before its samples."""
    if not info:
        return wav_bytes
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        if chunk_id == b"data":
            break
        offset += 8 + size + (size & 1)
    else:
        raise ValueError("no data chunk")
    merged = wav_bytes[:offset] + _info_chunk(info) + wav_bytes[offset:]
    # The RIFF size counts everything after its own id and size fields.
    return merged[:4] + struct.pack("<I", len(merged) - 8) + merged[8:]


def write_wav(
    wav,
    sample_rate: int,
    path: str | Path,
    info: Mapping[str, str] | None = None,
) -> Path:
    """
    Write a waveform to `path` as a WAV file, creating parent directories.

    `info` is an optional mapping of 4-character RIFF INFO codes to text; see
    the module docstring. Returns the resolved path actually written.
    """
    import wave

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(_as_batched_tensor(wav).squeeze(0).cpu().numpy(), dtype=np.float32)
    pcm = np.clip(np.round(frames * 32767), -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    target.write_bytes(_with_info(buffer.getvalue(), info or {}))
    logger.info("[audio] wrote %s (%d samples @ %d Hz)", target.name, num_samples(wav), sample_rate)
    return target


__all__ = ["duration_s", "num_samples", "write_wav"]
