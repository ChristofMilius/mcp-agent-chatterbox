"""
engine.py — lazy Chatterbox engine with bounded VRAM residency
==============================================================
The model is downloaded and loaded on the first synthesize() call, never at
server startup: a status or voice-listing call must not pay the multi-second
torch import plus a multi-gigabyte checkpoint fetch.

Two constraints shape the implementation.

1. One resident model. MAX_RESIDENT is 1. A 350M turbo model and a 500M
   multilingual model do not both fit on a 12 GB card that already has an LLM
   resident, so asking for a different model evicts the current one and empties
   the allocator cache. The cache is class-level so that repeated
   create_server() calls in tests and reloads share one set of weights instead
   of pulling gigabytes per instantiation.

2. Check VRAM before loading, not during. Chatterbox's from_pretrained() reads
   the checkpoint straight onto the target device (its .to() is incomplete —
   see the module docstring in __init__.py), so a load started with too little
   headroom dies with a bare torch.cuda.OutOfMemoryError part way through,
   after the download. resolve_device() picks the emptiest card by default and
   load() refuses early under CHATTERBOX_STRICT_VRAM, so the operator gets a
   message naming the actual problem.
"""

from __future__ import annotations

import functools
import gc
import importlib.util
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar, cast

from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.registry import ModelSpec, get_spec, load_model_class

logger = logging.getLogger(__name__)

#: How many models may be resident at once. See the class docstring.
MAX_RESIDENT = 1

#: Key under which a loaded model is stored: (model_key, t3_model or "").
_CACHE_KEY = tuple[str, str]

_MODEL_CACHE: dict[_CACHE_KEY, Any] = {}


def _has_wheel(spec_name: str) -> bool:
    """True when a module is importable (find_spec). Indirection for tests."""
    return importlib.util.find_spec(spec_name) is not None


def _load_torch():
    """Import torch. Separate function so tests can stub it."""
    import torch  # noqa: PLC0415

    return torch


F = TypeVar("F", bound=Callable[..., Any])


def _serialised[F: Callable[..., Any]](fn: F) -> F:
    """Run `fn` under the engine lock. Re-entrant, so synthesize() may load()."""

    @functools.wraps(fn)
    def wrapper(self: ChatterboxEngine, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)

    return cast(F, wrapper)


class ChatterboxEngine:
    """Owns model lifecycle, device selection and synthesis."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._device: str | None = None
        self._loaded_key: _CACHE_KEY | None = None
        # Guards the resident-model critical section: load / evict / generate.
        #
        # This is invisible over stdio, where the MCP transport serialises one
        # client's tool calls. It is not invisible over streamable-http or sse,
        # where several clients share this one process. Without the lock:
        #   * two threads can both miss the _MODEL_CACHE check and both call
        #     from_pretrained(), putting two copies of the weights on the card
        #     and OOMing a 12 GB GPU that the preflight check already cleared;
        #   * a request for model B can _evict_others() -- gc.collect() and
        #     torch.cuda.empty_cache() -- while request A is still generating
        #     with the model being evicted.
        # An RLock, not a Lock, because synthesize() calls load() while holding
        # it. One model, one GPU: serialising is the correct behaviour, not a
        # limitation. Reads (status, free_vram_mb) deliberately stay lock-free
        # so a 10s model load never blocks a tts_status call.
        self._lock = threading.RLock()

    # -- device selection -----------------------------------------------------
    def resolve_device(self) -> str:
        """
        Decide which device the next load should use.

        Precedence: explicit CHATTERBOX_DEVICE, then CHATTERBOX_GPU_INDEX,
        then "auto" — which uses CUDA if any card is visible and picks the one
        with the most free memory, falling back to CPU.

        Cached after the first successful resolution so that every call in a
        session lands on the same card; a model already resident is never
        silently moved.
        """
        if self._device is not None:
            return self._device

        torch = _load_torch()
        requested = (self.cfg.device or "auto").lower()

        if requested not in {"auto", ""}:
            if requested.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(
                    "[engine] %s requested but CUDA is unavailable — falling back to cpu",
                    requested,
                )
                requested = "cpu"
            self._device = requested
            return self._device

        if self.cfg.gpu_index is not None:
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if self.cfg.gpu_index >= count:
                raise ToolFault(
                    "gpu_index_out_of_range",
                    f"CHATTERBOX_GPU_INDEX={self.cfg.gpu_index} but only {count} "
                    "CUDA device(s) are visible.",
                )
            self._device = f"cuda:{self.cfg.gpu_index}"
            return self._device

        if torch.cuda.is_available():
            best = self._emptiest_cuda(torch)
            if best is not None:
                self._device = f"cuda:{best}"
                logger.info("[engine] auto-selected %s", self._device)
                return self._device

        self._device = "cpu"
        logger.info("[engine] no usable CUDA device — running on cpu")
        return self._device

    @staticmethod
    def _emptiest_cuda(torch) -> int | None:
        """Index of the CUDA device with the most free memory, or None."""
        best_idx: int | None = None
        best_free = -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _total = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001 — a card that cannot be queried is skipped
                continue
            if free > best_free:
                best_free = free
                best_idx = i
        return best_idx

    def cuda_index(self) -> int | None:
        """Ordinal of the resolved CUDA device, or None when on CPU."""
        device = self.resolve_device()
        if not device.startswith("cuda"):
            return None
        tail = device.split(":", 1)[1] if ":" in device else "0"
        try:
            return int(tail)
        except ValueError:
            return 0

    def free_vram_mb(self) -> int | None:
        """Free VRAM on the resolved device, in MiB. None when on CPU."""
        if self.cuda_index() is None:
            return None
        torch = _load_torch()
        try:
            free, _total = torch.cuda.mem_get_info(self.cuda_index())
        except Exception:  # noqa: BLE001
            return None
        return int(free // (1024 * 1024))

    # -- status ---------------------------------------------------------------
    def is_loaded(self) -> bool:
        return self._loaded_key is not None and self._loaded_key in _MODEL_CACHE

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_key[0] if self._loaded_key else None

    def status(self) -> dict:
        """Report runtime readiness without loading any weights."""
        chatterbox_ok = _has_wheel("chatterbox")
        torch_ok = _has_wheel("torch")
        torchaudio_ok = _has_wheel("torchaudio")

        cuda_available: bool | None = None
        device_name: str | None = None
        device_count = 0
        torch_version: str | None = None
        cuda_build: str | None = None

        if torch_ok:
            torch = _load_torch()
            # Version strings are cosmetic; the CUDA probe decides the device.
            # Reading them in the same try block meant a hiccup on either would
            # silently report a CUDA box as CPU-only.
            try:
                torch_version = torch.__version__
                cuda_build = torch.version.cuda
            except Exception:  # noqa: BLE001
                pass
            try:
                cuda_available = bool(torch.cuda.is_available())
                device_count = torch.cuda.device_count() if cuda_available else 0
            except Exception:  # noqa: BLE001
                cuda_available = None

        # Resolve the device only if nothing has pinned it yet. CUDA first:
        # the earlier version of this block tested the fallback before the
        # CUDA branch, which set the device to "cpu" and made the CUDA branch
        # unreachable.
        if self._device is None:
            if cuda_available and device_count:
                best = self._emptiest_cuda(_load_torch())
                self._device = f"cuda:{best}" if best is not None else "cpu"
            else:
                self._device = "cpu"

        if device_count and self._device and self._device.startswith("cuda"):
            idx = self.cuda_index()
            if idx is not None and idx < device_count:
                try:
                    device_name = _load_torch().cuda.get_device_name(idx)
                except Exception:  # noqa: BLE001
                    device_name = None

        from mcp_agent_chatterbox.playback import player_available  # noqa: PLC0415

        return {
            "chatterbox_installed": chatterbox_ok,
            "torch_installed": torch_ok,
            "torchaudio_installed": torchaudio_ok,
            "torch_version": torch_version,
            "cuda_build": cuda_build,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "device": self._device,
            "device_name": device_name,
            "free_vram_mb": self.free_vram_mb() if self._device else None,
            "vram_floor_mb": self.cfg.vram_floor_mb,
            "strict_vram": self.cfg.strict_vram,
            "model_loaded": self.is_loaded(),
            "loaded_model": self.loaded_model,
            "max_resident_models": MAX_RESIDENT,
            "autoplay": self.cfg.autoplay,
            "player_available": player_available(),
            "output_dir": str(self.cfg.output_dir),
            "voices_dir": str(self.cfg.voices_dir),
            "default_model": self.cfg.model,
            "max_chars": self.cfg.max_chars,
        }

    def gpu_report(self) -> list[dict]:
        """Per-GPU name and free/total VRAM, for tts_status and the CLI doctor."""
        if not _has_wheel("torch"):
            return []
        torch = _load_torch()
        if not torch.cuda.is_available():
            return []
        report: list[dict] = []
        selected = self._device
        for i in range(torch.cuda.device_count()):
            entry: dict[str, Any] = {"index": i, "selected": selected == f"cuda:{i}"}
            try:
                entry["name"] = torch.cuda.get_device_name(i)
            except Exception:  # noqa: BLE001
                entry["name"] = None
            try:
                free, total = torch.cuda.mem_get_info(i)
                entry["free_mb"] = int(free // (1024 * 1024))
                entry["total_mb"] = int(total // (1024 * 1024))
            except Exception:  # noqa: BLE001
                entry["free_mb"] = None
                entry["total_mb"] = None
            report.append(entry)
        return report

    # -- model lifecycle ------------------------------------------------------
    def _preflight(self, device: str) -> None:
        """Warn or refuse when the target device has too little free VRAM."""
        if not device.startswith("cuda"):
            return
        free = self.free_vram_mb()
        if free is None or free >= self.cfg.vram_floor_mb:
            return
        msg = (
            f"{device} has {free} MiB free but at least {self.cfg.vram_floor_mb} MiB is "
            "needed to load a Chatterbox model. Free the GPU (unload the model in LM "
            "Studio, or stop another process) or set CHATTERBOX_GPU_INDEX to a card "
            "with more headroom."
        )
        if self.cfg.strict_vram:
            raise ToolFault(
                "insufficient_vram", msg, free_mb=free, required_mb=self.cfg.vram_floor_mb
            )
        logger.warning("[engine] %s", msg)

    @_serialised
    def load(self, model_key: str, t3_model: str | None = None) -> tuple[Any, str, ModelSpec]:
        """
        Return (model, device, spec) for `model_key`, loading it if needed.

        The cache is process-wide (see _MODEL_CACHE) so weights are fetched and
        deserialised once per process no matter how many engines are built.
        """
        spec = get_spec(model_key)
        if spec is None:
            raise ToolFault(
                "unknown_model",
                f"Unknown model {model_key!r}. Known models: turbo, multilingual, original.",
            )

        key: _CACHE_KEY = (spec.key, t3_model or "")
        if key in _MODEL_CACHE:
            if self._loaded_key != key:
                self._evict_others(key)
            self._loaded_key = key
            self._device = self._device or self.resolve_device()
            return _MODEL_CACHE[key], self._device, spec

        if not _has_wheel("chatterbox") or not _has_wheel("torch"):
            raise ToolFault(
                "dependencies_missing",
                "chatterbox-tts / torch are not importable — run `uv sync` in the "
                "mcp-agent-chatterbox project.",
            )

        device = self.resolve_device()
        self._preflight(device)

        self._evict_others(key)

        model_cls = load_model_class(spec)
        logger.info(
            "[engine] loading %s on %s (first use; weights download once, then cached on disk)",
            spec.key,
            device,
        )
        started = time.perf_counter()
        try:
            if t3_model:
                model = model_cls.from_pretrained(device=device, t3_model=t3_model)
            else:
                model = model_cls.from_pretrained(device=device)
        except Exception as exc:  # noqa: BLE001
            logger.error("[engine] %s load failed: %s", spec.key, exc, exc_info=True)
            self._raise_load_failure(spec, exc, device)
        elapsed = time.perf_counter() - started

        _MODEL_CACHE[key] = model
        self._loaded_key = key
        logger.info(
            "[engine] %s ready in %.1fs (device=%s, free VRAM now %s MiB)",
            spec.key,
            elapsed,
            device,
            self.free_vram_mb(),
        )
        return model, device, spec

    def _raise_load_failure(self, spec: ModelSpec, exc: Exception, device: str) -> NoReturn:
        """Turn a load failure into an actionable ToolFault where possible."""
        name = type(exc).__name__
        if "OutOfMemory" in name:
            free = self.free_vram_mb()
            raise ToolFault(
                "cuda_out_of_memory",
                f"Loading {spec.key} on {device} ran out of VRAM"
                + (f" ({free} MiB still free afterwards)." if free is not None else ".")
                + " Free the GPU or switch cards with CHATTERBOX_GPU_INDEX.",
                free_mb=free,
            ) from exc
        if spec.key == "turbo" and device.startswith("cuda"):
            logger.debug("[engine] turbo CUDA load error (may be benign): %s", exc)
        raise ToolFault(
            "model_load_failed",
            f"Could not load {spec.label} on {device} ({name}). "
            "Weights download from Hugging Face on first use — check the server log "
            "for the underlying error.",
        ) from exc

    def _evict_others(self, keep: _CACHE_KEY) -> None:
        """Drop every cached model except `keep`, then reclaim VRAM."""
        stale = [k for k in _MODEL_CACHE if k != keep]
        if not stale:
            return
        for k in stale:
            logger.info("[engine] evicting %s to bound VRAM residency", k[0])
            _MODEL_CACHE.pop(k, None)
        if self._loaded_key is not None and self._loaded_key != keep:
            self._loaded_key = None
        gc.collect()
        if _has_wheel("torch"):
            try:
                torch = _load_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    @_serialised
    def unload(self) -> dict:
        """Release the resident model and return the freed VRAM, if any."""
        was = self.loaded_model
        before = self.free_vram_mb()
        self._evict_others(("__none__", ""))
        after = self.free_vram_mb()
        return {
            "unloaded": was,
            "was_loaded": was is not None,
            "free_vram_mb_before": before,
            "free_vram_mb_after": after,
            "freed_mb": (after - before) if (before is not None and after is not None) else None,
        }

    # -- synthesis ------------------------------------------------------------
    @_serialised
    def synthesize(
        self,
        text: str,
        model_key: str,
        *,
        reference_clip: str | None = None,
        language: str | None = None,
        t3_model: str | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_p: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        norm_loudness: bool | None = None,
        seed: int | None = None,
    ) -> dict:
        """
        Render `text` to a waveform and return it with metadata.

        kwargs left as None fall back to the per-model defaults in the registry
        rather than to this function's signature, because the three Chatterbox
        models are tuned differently and a single set of numbers would degrade
        two of them.

        Any caller-set knob outside `spec.honored_knobs` is dropped with a
        warning instead of reaching generate(): turbo ignores min_p, exaggeration
        and cfg_weight, and the 500M models have no top_k/norm_loudness parameter
        at all — forwarding them would TypeError or do nothing. `seed` reseeds
        torch (CPU + CUDA) so a render is reproducible within a resident
        session (the same text + seed then renders byte-identical audio);
        0 (the upstream Gradio apps' convention) or None keeps random sampling.
        Signed reproducibility across processes is not guaranteed: Chatterbox
        does not enable torch's deterministic algorithms, so a fresh process
        may pick different CUDA fast-path kernels and drift in low bits — the
        upstream apps' seed has the same limitation.
        """
        model, device, spec = self.load(model_key, t3_model=t3_model)

        if not spec.stock_voice and not reference_clip:
            raise ToolFault(
                "reference_clip_required",
                f"{spec.label} has no built-in voice and needs a reference clip. "
                "Pass voice='<name>' (a clip in the voices dir) or "
                "reference_clip='<path to wav/mp3>'. Use model='turbo' if you want "
                "to speak without one.",
                model=spec.key,
            )

        # Only forward knobs the caller actually set. Anything left as None
        # keeps Chatterbox's own tuned value for that model, which differs
        # across the three (turbo runs cfg_weight=0.0, the 500M models 0.5).
        overrides = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "top_p": top_p,
            "top_k": top_k,
            "norm_loudness": norm_loudness,
        }
        set_but_unhonored = sorted(
            k for k, v in overrides.items() if v is not None and k not in spec.honored_knobs
        )
        if set_but_unhonored:
            logger.warning(
                "[engine] %s does not support %s — ignoring",
                spec.key,
                ", ".join(set_but_unhonored),
            )
        kwargs: dict[str, Any] = {
            k: v
            for k, v in overrides.items()
            if v is not None and k in spec.honored_knobs
        }
        if reference_clip:
            kwargs["audio_prompt_path"] = reference_clip
        if language:
            kwargs["language_id"] = language

        logger.info(
            "[engine] synthesize model=%s chars=%d device=%s kwargs=%s",
            spec.key,
            len(text),
            device,
            sorted(kwargs),
        )
        if seed is not None and seed != 0:
            torch = _load_torch()
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info("[engine] seed=%d set for reproducible render", seed)
        started = time.perf_counter()
        try:
            wav = model.generate(text, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("[engine] generate failed: %s", exc, exc_info=True)
            raise ToolFault(
                "synthesis_failed",
                f"{spec.label} failed to synthesize this text ({type(exc).__name__}). "
                "Check the server log for details.",
            ) from exc
        elapsed = time.perf_counter() - started

        return {
            "wav": wav,
            "sample_rate": int(getattr(model, "sr", 24000)),
            "model": spec.key,
            "device": device,
            "elapsed_s": round(elapsed, 2),
            "chars": len(text),
            "language": language,
            "reference_clip_used": bool(reference_clip),
            "free_vram_mb": self.free_vram_mb(),
        }


__all__ = ["MAX_RESIDENT", "ChatterboxEngine"]
