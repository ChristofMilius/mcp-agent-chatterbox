"""
Device selection and status, against a fake torch.

No weights are loaded here. These cover the logic that decides *where* a model
would land and what status reports before anything is resident — the part that
was wrong once already: status() checked its CPU fallback before its CUDA
branch, so a CUDA-capable box reported device "cpu" and free_vram_mb null.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import torch

from mcp_agent_chatterbox import engine as engine_module
from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.engine import ChatterboxEngine, _apply_hf_home, _harden_reference_dtype
from mcp_agent_chatterbox.errors import ToolFault
from mcp_agent_chatterbox.registry import (
    MODEL_SPECS,
    SUPPORTED_LANGUAGES,
    describe_models,
    get_spec,
)


def engine_with(monkeypatch, device="auto", gpu_index=None) -> ChatterboxEngine:
    cfg = Config()
    cfg.device = device
    cfg.gpu_index = gpu_index
    return ChatterboxEngine(cfg)


def spec_key(name: str | None) -> str:
    """get_spec(...).key, asserting the lookup actually resolved."""
    spec = get_spec(name)
    assert spec is not None, f"expected {name!r} to resolve to a model"
    return spec.key


class TestAutoDevice:
    def test_picks_emptiest_card(self, monkeypatch, fake_torch):
        # Card 1 has more free memory, so it must win.
        fake_torch(available=True, count=2, free_mb=[2000, 9000])
        assert engine_with(monkeypatch).resolve_device() == "cuda:1"

    def test_picks_emptiest_when_card0_wins(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 2000])
        assert engine_with(monkeypatch).resolve_device() == "cuda:0"

    def test_single_card(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=1, free_mb=[8000])
        assert engine_with(monkeypatch).resolve_device() == "cuda:0"

    def test_falls_back_to_cpu_without_cuda(self, monkeypatch, fake_torch):
        fake_torch(available=False, count=0)
        assert engine_with(monkeypatch).resolve_device() == "cpu"

    def test_falls_back_to_cpu_when_no_card_can_be_queried(self, monkeypatch, fake_torch):
        # device_count() lies, but mem_get_info() fails on every ordinal.
        class Broken:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def device_count():
                    return 2

                @staticmethod
                def mem_get_info(_i):
                    raise RuntimeError("driver not ready")

        monkeypatch.setattr("mcp_agent_chatterbox.engine._load_torch", lambda: Broken())
        assert engine_with(monkeypatch).resolve_device() == "cpu"

    def test_cached_after_first_resolution(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[2000, 9000])
        e = engine_with(monkeypatch)
        assert e.resolve_device() == "cuda:1"
        # A resident model must never be silently moved to another card.
        fake_torch(available=True, count=2, free_mb=[9000, 2000])
        assert e.resolve_device() == "cuda:1"


class TestExplicitDevice:
    def test_explicit_cpu(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 9000])
        assert engine_with(monkeypatch, device="cpu").resolve_device() == "cpu"

    def test_explicit_cuda_n(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 9000])
        assert engine_with(monkeypatch, device="cuda:1").resolve_device() == "cuda:1"

    def test_cuda_requested_but_unavailable_degrades_to_cpu(self, monkeypatch, fake_torch):
        fake_torch(available=False, count=0)
        e = engine_with(monkeypatch, device="cuda")
        assert e.resolve_device() == "cpu"

    def test_explicit_device_beats_gpu_index(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 9000])
        e = engine_with(monkeypatch, device="cuda:0", gpu_index=1)
        assert e.resolve_device() == "cuda:0"


class TestGpuIndexPin:
    def test_pins_requested_ordinal(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 2000])
        # The pin must not be overridden by the emptiest-card heuristic.
        assert engine_with(monkeypatch, gpu_index=0).resolve_device() == "cuda:0"

    def test_out_of_range_raises(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 9000])
        e = engine_with(monkeypatch, gpu_index=5)
        with pytest.raises(ToolFault) as exc:
            e.resolve_device()
        assert exc.value.reason == "gpu_index_out_of_range"
        assert "2" in exc.value.message


class TestCudaIndex:
    def test_none_on_cpu(self, monkeypatch, fake_torch):
        fake_torch(available=False, count=0)
        assert engine_with(monkeypatch).cuda_index() is None

    def test_ordinal_from_cuda_n(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=2, free_mb=[9000, 9000])
        assert engine_with(monkeypatch, device="cuda:1").cuda_index() == 1

    def test_bare_cuda_defaults_to_zero(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=1, free_mb=[9000])
        assert engine_with(monkeypatch, device="cuda").cuda_index() == 0


class TestFreeVram:
    def test_reports_mib_on_cuda(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=1, free_mb=[8192])
        e = engine_with(monkeypatch)
        assert e.free_vram_mb() == 8192

    def test_none_on_cpu(self, monkeypatch, fake_torch):
        fake_torch(available=False, count=0)
        e = engine_with(monkeypatch)
        assert e.free_vram_mb() is None


class TestStatus:
    def test_status_picks_a_cuda_device_not_cpu(self, monkeypatch, fake_torch):
        # Regression: the fallback used to be tested first, so a CUDA box
        # reported device "cpu" and free_vram_mb null.
        fake_torch(available=True, count=2, free_mb=[2000, 9000], names={1: "RTX 3060"})
        st = engine_with(monkeypatch).status()
        assert st["device"] == "cuda:1"
        assert st["device_name"] == "RTX 3060"
        assert st["free_vram_mb"] == 9000

    def test_status_reports_cpu_when_no_cuda(self, monkeypatch, fake_torch):
        fake_torch(available=False, count=0)
        st = engine_with(monkeypatch).status()
        assert st["device"] == "cpu"
        assert st["free_vram_mb"] is None

    def test_status_loads_nothing(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=1, free_mb=[8000])
        e = engine_with(monkeypatch)
        st = e.status()
        assert st["model_loaded"] is False
        assert st["loaded_model"] is None
        assert e.is_loaded() is False

    def test_status_reports_one_resident_max(self, monkeypatch, fake_torch):
        fake_torch(available=True, count=1, free_mb=[8000])
        assert engine_with(monkeypatch).status()["max_resident_models"] == 1

    def test_status_survives_a_broken_version_string(self, monkeypatch):
        # A cosmetic __version__ read must not be able to demote a CUDA box to
        # CPU: device selection and version reporting are independent.
        class NoVersion:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def device_count():
                    return 1

                @staticmethod
                def mem_get_info(_i):
                    return 8 * 1024 * 1024 * 1024, 12 * 1024 * 1024 * 1024

                @staticmethod
                def get_device_name(_i):
                    return "fake-gpu-0"

            @property
            def __version__(self):
                raise RuntimeError("broken build string")

        monkeypatch.setattr("mcp_agent_chatterbox.engine._load_torch", lambda: NoVersion())
        monkeypatch.setattr("mcp_agent_chatterbox.engine._has_wheel", lambda _n: True)
        st = engine_with(monkeypatch).status()
        assert st["device"] == "cuda:0"


class TestApplyHfHome:
    def test_empty_home_leaves_environment_untouched(self, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        _apply_hf_home(None)
        assert "HF_HOME" not in os.environ

    def test_sets_hf_home_from_configured_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_HOME", raising=False)
        target = tmp_path / "hf"
        _apply_hf_home(target)
        assert os.environ["HF_HOME"] == str(target)

    def test_already_pointing_there_is_a_noop(self, monkeypatch, tmp_path):
        target = tmp_path / "hf"
        monkeypatch.setenv("HF_HOME", str(target))
        _apply_hf_home(target)
        assert os.environ["HF_HOME"] == str(target)


class TestRegistry:  # noqa: N801 — the registry is imported into this test module
    def test_describe_lists_all_four_models(self):
        keys = {m["key"] for m in describe_models()}
        assert keys == {"turbo", "nano", "multilingual", "original"}

    def test_turbo_and_nano_have_a_stock_voice(self):
        stock = {m["key"] for m in describe_models() if m["stock_voice"]}
        assert stock == {"turbo", "nano"}

    def test_models_without_stock_voice_are_flagged(self):
        for m in describe_models():
            assert m["requires_reference_clip"] is not m["stock_voice"]

    def test_advertised_languages(self):
        assert "de" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) == 23

    def test_language_values_are_names_not_codes(self):
        assert SUPPORTED_LANGUAGES["de"] == "German"

    def test_aliases_resolve(self):
        assert spec_key("multi") == "multilingual"
        assert spec_key("mtl") == "multilingual"
        assert spec_key("chatterbox-turbo") == "turbo"
        assert spec_key("chatterbox-nano") == "nano"
        assert spec_key("english") == "original"

    def test_aliases_are_case_insensitive(self):
        assert spec_key("MULTI") == "multilingual"
        assert spec_key("  Multi  ") == "multilingual"

    def test_full_hf_repo_id_is_rejected(self):
        # Models are addressed by short key only. A repo id must fail loudly
        # rather than silently falling back to some default model.
        assert get_spec("ResembleAI/chatterbox-turbo") is None

    def test_unknown_model_is_none(self):
        assert get_spec("whisper") is None
        assert get_spec("") is None
        assert get_spec(None) is None

    def test_per_model_defaults_differ(self):
        # The reason defaults live in the registry: turbo and the 500M models
        # are tuned differently, and one global default would degrade two.
        assert MODEL_SPECS["turbo"].defaults["cfg_weight"] == 0.0
        assert MODEL_SPECS["multilingual"].defaults["cfg_weight"] == 0.5
        assert MODEL_SPECS["original"].defaults["cfg_weight"] == 0.5

    def test_tunable_lists_only_honored_knobs(self):
        # turbo accepts min_p/exaggeration/cfg_weight in its signature but
        # ignores them, so tuning must not advertise them.
        turbo = next(m for m in describe_models() if m["key"] == "turbo")
        assert "top_k" in turbo["tunable"]
        assert "norm_loudness" in turbo["tunable"]
        assert "temperature" in turbo["tunable"]
        assert "min_p" not in turbo["tunable"]
        assert "exaggeration" not in turbo["tunable"]
        assert "cfg_weight" not in turbo["tunable"]

    def test_500m_tunable_excludes_turbo_only_knobs(self):
        multi = next(m for m in describe_models() if m["key"] == "multilingual")
        assert "min_p" in multi["tunable"]
        assert "exaggeration" in multi["tunable"]
        assert "cfg_weight" in multi["tunable"]
        assert "top_k" not in multi["tunable"]
        assert "norm_loudness" not in multi["tunable"]

    def test_multilingual_repetition_penalty_default_is_1_2(self):
        # Mirrors the upstream-snapshot default (mtl_tts.generate). The PyPI
        # 0.1.7 release used 2.0; master retuned it to 1.2.
        assert MODEL_SPECS["multilingual"].defaults["repetition_penalty"] == 1.2

    def test_nano_uses_turbo_class_with_nano_load_flag(self):
        nano = MODEL_SPECS["nano"]
        assert nano.module == "chatterbox.tts_turbo"
        assert nano.attr == "ChatterboxTurboTTS"
        assert nano.load_kwargs == {"nano": True}
        assert nano.params == "110M"
        assert nano.stock_voice is True

    def test_turbo_has_no_extra_load_kwargs(self):
        # turbo is the class default; only nano diverges within the family.
        assert MODEL_SPECS["turbo"].load_kwargs == {}


class TestLoadKwargs:
    """load() must forward spec.load_kwargs into from_pretrained()."""

    @pytest.fixture(autouse=True)
    def clear_model_cache(self):
        engine_module._MODEL_CACHE.clear()
        yield
        engine_module._MODEL_CACHE.clear()

    def _recording_cls(self, monkeypatch):
        class Recording:
            calls: list[dict] = []

            @classmethod
            def from_pretrained(cls, **kwargs):
                Recording.calls.append(kwargs)
                return object.__new__(cls)

        monkeypatch.setattr(engine_module, "load_model_class", lambda spec: Recording)
        return Recording

    def _engine(self, device="cpu") -> ChatterboxEngine:
        cfg = Config()
        cfg.device = device
        eng = ChatterboxEngine(cfg)
        eng.resolve_device = lambda: device
        eng._preflight = lambda device: None
        return eng

    def test_nano_forwards_nano_flag(self, monkeypatch):
        recording = self._recording_cls(monkeypatch)
        self._engine().load("nano")
        assert recording.calls == [{"device": "cpu", "nano": True}]

    def test_turbo_forwards_no_extra_kwargs(self, monkeypatch):
        recording = self._recording_cls(monkeypatch)
        self._engine().load("turbo")
        assert recording.calls == [{"device": "cpu"}]

    def test_t3_model_is_added_for_multilingual(self, monkeypatch):
        recording = self._recording_cls(monkeypatch)
        self._engine().load("multilingual", t3_model="v3")
        assert recording.calls == [{"device": "cpu", "t3_model": "v3"}]


class FakeModel:
    """Stands in for a chatterbox model: records generate() kwargs."""

    sr = 24000

    def __init__(self):
        self.renders: list[tuple[str, dict]] = []

    def norm_loudness(self, wav, sr):
        return wav

    def generate(self, text: str, **kwargs):
        self.renders.append((text, kwargs))
        return torch.zeros(1, 2400)


class TestSynthesize:
    """synthesize()'s knob handling, with load() stubbed so no weights load."""

    @pytest.fixture
    def fake_model(self):
        return FakeModel()

    def _engine(self, monkeypatch, spec_name, fake_model):
        engine = ChatterboxEngine(Config())
        spec = get_spec(spec_name)
        monkeypatch.setattr(
            engine, "load", lambda model_key, t3_model=None: (fake_model, "cpu", spec)
        )
        return engine

    def test_forwards_turbo_honored_knobs(self, monkeypatch, fake_model):
        e = self._engine(monkeypatch, "turbo", fake_model)
        e.synthesize(
            "hi",
            "turbo",
            top_k=500,
            norm_loudness=False,
            top_p=0.9,
            repetition_penalty=1.5,
            temperature=0.7,
        )
        _text, kwargs = fake_model.renders[0]
        assert kwargs["top_k"] == 500
        assert kwargs["norm_loudness"] is False
        assert kwargs["top_p"] == 0.9
        assert kwargs["repetition_penalty"] == 1.5
        assert kwargs["temperature"] == 0.7

    def test_drops_unhonored_turbo_knobs_with_warning(self, monkeypatch, fake_model, caplog):
        e = self._engine(monkeypatch, "turbo", fake_model)
        with caplog.at_level(logging.WARNING, logger="mcp_agent_chatterbox.engine"):
            e.synthesize("hi", "turbo", exaggeration=0.4, cfg_weight=0.7, min_p=0.1)
        _text, kwargs = fake_model.renders[0]
        assert "exaggeration" not in kwargs
        assert "cfg_weight" not in kwargs
        assert "min_p" not in kwargs
        assert "exaggeration" in caplog.text

    def test_drops_top_k_and_norm_loudness_for_500m(self, monkeypatch, fake_model, caplog):
        e = self._engine(monkeypatch, "multilingual", fake_model)
        with caplog.at_level(logging.WARNING, logger="mcp_agent_chatterbox.engine"):
            e.synthesize(
                "hi",
                "multilingual",
                reference_clip="clip.wav",
                top_k=500,
                norm_loudness=False,
                top_p=0.9,
                repetition_penalty=1.5,
            )
        _text, kwargs = fake_model.renders[0]
        assert "top_k" not in kwargs
        assert "norm_loudness" not in kwargs
        assert kwargs["top_p"] == 0.9
        assert kwargs["repetition_penalty"] == 1.5

    def test_min_p_forwarded_for_500m(self, monkeypatch, fake_model):
        e = self._engine(monkeypatch, "multilingual", fake_model)
        e.synthesize("hi", "multilingual", reference_clip="clip.wav", min_p=0.1)
        _text, kwargs = fake_model.renders[0]
        assert kwargs["min_p"] == 0.1

    def test_unset_knobs_reach_generate_as_nothing(self, monkeypatch, fake_model):
        e = self._engine(monkeypatch, "turbo", fake_model)
        e.synthesize("hi", "turbo")
        _text, kwargs = fake_model.renders[0]
        assert kwargs == {}

    def test_seed_seeds_torch_on_cpu_and_cuda(self, monkeypatch, fake_model, fake_torch):
        torch_fake = fake_torch(available=True, count=1, free_mb=[8000])
        e = self._engine(monkeypatch, "turbo", fake_model)
        e.synthesize("hi", "turbo", seed=42)
        assert torch_fake.manual_seed_calls == [42]
        assert torch_fake.cuda.manual_seed_all_calls == [42]

    def test_zero_seed_means_random(self, monkeypatch, fake_model, fake_torch):
        # Upstream Gradio apps use 0 for "random"; it must not seed.
        torch_fake = fake_torch(available=True, count=1, free_mb=[8000])
        e = self._engine(monkeypatch, "turbo", fake_model)
        e.synthesize("hi", "turbo", seed=0)
        assert torch_fake.manual_seed_calls == []
        assert torch_fake.cuda.manual_seed_all_calls == []

    def test_no_seed_no_seeding(self, monkeypatch, fake_model, fake_torch):
        torch_fake = fake_torch(available=True, count=1, free_mb=[8000])
        e = self._engine(monkeypatch, "turbo", fake_model)
        e.synthesize("hi", "turbo")
        assert torch_fake.manual_seed_calls == []
        assert torch_fake.cuda.manual_seed_all_calls == []

    def test_returns_wav_and_metadata(self, monkeypatch, fake_model):
        e = self._engine(monkeypatch, "turbo", fake_model)
        result = e.synthesize("hi", "turbo")
        assert result["model"] == "turbo"
        assert result["chars"] == 2
        assert result["wav"].shape == (1, 2400)


class TestReferenceDtypeShim:
    """The float64 upcast in upstream norm_loudness() must never reach torch.

    norm_loudness() scales by a pyloudnorm np.float64 loudness, promoting the
    float32 wav to float64; s3tokenizer's float32 mel filters and the voice
    encoder's LSTM then both refuse the double array. The shim pins the output
    back to float32. The real load() path applies it; here we test the wrapper.
    """

    def test_coerces_float64_numpy_to_float32(self):
        model = FakeModel()
        model.norm_loudness = lambda wav, sr: np.asarray(wav, dtype=np.float64)

        model.norm_loudness = _harden_reference_dtype(model.norm_loudness)
        out = model.norm_loudness(np.zeros(8, dtype=np.float32), 24000)

        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32

    def test_leaves_float32_numpy_unchanged(self):
        model = FakeModel()
        model.norm_loudness = lambda wav, sr: np.asarray(wav, dtype=np.float32)

        model.norm_loudness = _harden_reference_dtype(model.norm_loudness)
        out = model.norm_loudness(np.zeros(8, dtype=np.float32), 24000)

        assert out.dtype == np.float32

    def test_coerces_float64_tensor_to_float32(self):
        model = FakeModel()
        model.norm_loudness = lambda wav, sr: torch.zeros(8, dtype=torch.float64)

        model.norm_loudness = _harden_reference_dtype(model.norm_loudness)
        out = model.norm_loudness(np.zeros(8, dtype=np.float32), 24000)

        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
