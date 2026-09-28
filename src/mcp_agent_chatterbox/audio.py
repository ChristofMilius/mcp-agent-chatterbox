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
"""

from __future__ import annotations

import logging
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


def write_wav(wav, sample_rate: int, path: str | Path) -> Path:
    """
    Write a waveform to `path` as a WAV file, creating parent directories.

    Returns the resolved path actually written.
    """
    import wave

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(_as_batched_tensor(wav).squeeze(0).cpu().numpy(), dtype=np.float32)
    pcm = np.clip(np.round(frames * 32767), -32768, 32767).astype(np.int16)
    with target.open("wb") as fh, wave.open(fh, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    logger.info("[audio] wrote %s (%d samples @ %d Hz)", target.name, num_samples(wav), sample_rate)
    return target


__all__ = ["duration_s", "num_samples", "write_wav"]
