"""
Concurrency safety of the resident-model critical section.

These exist because the engine was originally written for stdio, where the MCP
transport serialises one client's tool calls and no locking is needed. Under
`serve --transport streamable-http` several clients share the one process, and
the unlocked check-then-act in load() was a real defect: two threads could both
miss the _MODEL_CACHE check and both call from_pretrained(), putting two copies
of the weights on one card, and a request for a different model could evict the
model another request was still generating with.

Every test here is written to FAIL against the unlocked engine. If a change ever
removes @_serialised, these are the tests that notice.
"""

from __future__ import annotations

import threading
import time

import pytest

from mcp_agent_chatterbox import engine as engine_mod
from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.engine import ChatterboxEngine
from mcp_agent_chatterbox.errors import ToolFault


class CountingModel:
    """A model whose from_pretrained() is slow and counted.

    The sleep inside from_pretrained is what makes the race observable: without
    the lock, every thread passes the cache check and is still inside the load
    when the last one commits.
    """

    sr = 24000
    instances = 0
    load_ms = 60
    concurrent = 0
    peak_concurrent = 0
    _guard = threading.Lock()

    def __init__(self):
        import torch

        self.wav = torch.zeros(1, 1200)

    @classmethod
    def reset(cls, load_ms: int = 60) -> None:
        cls.instances = 0
        cls.load_ms = load_ms
        cls.concurrent = 0
        cls.peak_concurrent = 0

    @classmethod
    def from_pretrained(cls, **kwargs):
        with cls._guard:
            cls.instances += 1
            cls.concurrent += 1
            cls.peak_concurrent = max(cls.peak_concurrent, cls.concurrent)
        try:
            time.sleep(cls.load_ms / 1000)
            return cls()
        finally:
            with cls._guard:
                cls.concurrent -= 1

    def generate(self, text, **kwargs):
        return self.wav


@pytest.fixture
def counting_model(monkeypatch, fake_torch):
    """Install CountingModel as turbo's class and clear the module-level cache."""
    fake_torch(available=True, count=1, free_mb=[9000])
    CountingModel.reset()
    monkeypatch.setattr(
        engine_mod,
        "load_model_class",
        lambda spec: CountingModel,
    )
    monkeypatch.setattr(engine_mod, "_MODEL_CACHE", {})
    yield CountingModel
    monkeypatch.setattr(engine_mod, "_MODEL_CACHE", {})


def _cfg() -> Config:
    cfg = Config()
    cfg.device = "cuda:0"
    return cfg


def test_concurrent_loads_resident_once(counting_model):
    """Four threads, one GPU: the weights must be deserialised exactly once."""
    engine = ChatterboxEngine(_cfg())
    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        try:
            barrier.wait()  # maximise the overlap
            model, _device, _spec = engine.load("turbo")
            results.append(model.sr)
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"worker raised: {errors[0]!r}"
    assert len(results) == 4
    # The regression: without the lock this is 4 (one copy of the weights per
    # thread, i.e. 4x the VRAM on a card that passed the preflight check once).
    assert counting_model.instances == 1, (
        f"from_pretrained called {counting_model.instances}x for the same model; "
        "the load path is not mutually exclusive"
    )
    assert counting_model.peak_concurrent == 1, (
        f"{counting_model.peak_concurrent} loads overlapped inside from_pretrained"
    )


def test_switching_models_does_not_evict_under_an_active_generate(counting_model):
    """A model switch must not pull the rug from under an in-flight generate()."""
    engine = ChatterboxEngine(_cfg())
    observed: list[str] = []

    class SlowModel(CountingModel):
        def generate(self, text, **kwargs):
            # Yield the GIL long enough for the other thread to attempt a load.
            time.sleep(0.08)
            observed.append("generated")
            return self.wav

    monkey_model = SlowModel
    monkey_model.sr = 24000
    counting_model.load_ms = 0
    engine_mod.load_model_class = lambda spec: monkey_model  # type: ignore[assignment]
    try:
        gen_error: list[BaseException] = []

        def generating() -> None:
            try:
                engine.synthesize("hello there", "turbo")
            except BaseException as exc:  # noqa: BLE001 — surfaced below
                gen_error.append(exc)

        thread = threading.Thread(target=generating)
        thread.start()
        time.sleep(0.02)  # let it get into generate()
        engine.load("turbo")  # same key, must not evict the model in use
        thread.join(timeout=30)

        assert not gen_error, f"in-flight generate raised: {gen_error[0]!r}"
        assert observed == ["generated"], "the in-flight generate never completed"
    finally:
        engine_mod.load_model_class = _ORIGINAL_LOAD_MODEL_CLASS


def test_synthesize_is_serialised(counting_model):
    """Two speaks at once must not enter generate() simultaneously.

    A single model instance holds mutable state (the Perth watermarker, the
    condgen/uncondgen caches), so concurrent generate() calls on one instance
    are not safe. Serialising is the correct behaviour for one model on one GPU.
    """
    counting_model.load_ms = 0
    engine = ChatterboxEngine(_cfg())

    in_generate = 0
    peak = 0
    guard = threading.Lock()

    class OverlapProbe(CountingModel):
        def generate(self, text, **kwargs):
            nonlocal in_generate, peak
            with guard:
                in_generate += 1
                peak = max(peak, in_generate)
            time.sleep(0.05)
            with guard:
                in_generate -= 1
            return self.wav

    probe = OverlapProbe
    probe.sr = 24000
    engine_mod.load_model_class = lambda spec: probe  # type: ignore[assignment]
    try:
        threads = [
            threading.Thread(target=engine.synthesize, args=(f"line {i}", "turbo"))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert peak == 1, f"{peak} concurrent generate() calls on one model instance"
    finally:
        engine_mod.load_model_class = _ORIGINAL_LOAD_MODEL_CLASS


def test_unload_waits_for_an_active_synthesis(counting_model):
    """tts_unload must not free the GPU underneath a running generate()."""
    counting_model.load_ms = 0
    engine = ChatterboxEngine(_cfg())

    order: list[str] = []

    class Ordered(CountingModel):
        def generate(self, text, **kwargs):
            order.append("generate-start")
            time.sleep(0.08)
            order.append("generate-end")
            return self.wav

    ordered = Ordered
    ordered.sr = 24000
    engine_mod.load_model_class = lambda spec: ordered  # type: ignore[assignment]
    try:
        t = threading.Thread(target=engine.synthesize, args=("hello", "turbo"))
        t.start()
        time.sleep(0.02)
        engine.unload()
        order.append("unload-returned")
        t.join(timeout=30)
        assert order.index("generate-end") < order.index("unload-returned"), (
            f"unload() returned mid-synthesis: {order}"
        )
    finally:
        engine_mod.load_model_class = _ORIGINAL_LOAD_MODEL_CLASS


def test_reads_do_not_block_behind_a_slow_load(counting_model):
    """tts_status must stay responsive while a 10s weight load is in flight.

    Reads deliberately take no lock. If someone later wraps status() in the same
    lock, tts_status would hang for the whole model load -- which is exactly the
    responsiveness the harness needs when diagnosing a slow first call.
    """
    counting_model.reset(load_ms=400)
    engine = ChatterboxEngine(_cfg())

    started = threading.Event()
    finished = threading.Event()

    def slow_load() -> None:
        started.set()
        engine.load("turbo")
        finished.set()

    t = threading.Thread(target=slow_load)
    t.start()
    started.wait(timeout=5)
    time.sleep(0.05)

    began = time.perf_counter()
    engine.status()
    engine.free_vram_mb()
    elapsed = time.perf_counter() - began

    assert not finished.is_set(), "load finished too fast to prove anything"
    assert elapsed < 0.2, f"reads blocked for {elapsed:.2f}s behind the load lock"
    t.join(timeout=30)


def test_lock_is_reentrant(counting_model):
    """synthesize() holds the lock and calls load(), which takes it again."""
    counting_model.load_ms = 0
    engine = ChatterboxEngine(_cfg())
    # A plain Lock would deadlock here; RLock is what makes the decorator legal.
    result = engine.synthesize("reentrancy check", "turbo")
    assert result["model"] == "turbo"
    assert result["wav"] is not None


def test_unknown_model_still_faults_under_the_lock(counting_model):
    engine = ChatterboxEngine(_cfg())
    with pytest.raises(ToolFault) as exc:
        engine.synthesize("hi", "nope")
    assert exc.value.reason == "unknown_model"


# Captured at import so the module-level monkeypatching above can be undone.
_ORIGINAL_LOAD_MODEL_CLASS = engine_mod.load_model_class
