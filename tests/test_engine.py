"""
Device selection and status, against a fake torch.

No weights are loaded here. These cover the logic that decides *where* a model
would land and what status reports before anything is resident — the part that
was wrong once already: status() checked its CPU fallback before its CUDA
branch, so a CUDA-capable box reported device "cpu" and free_vram_mb null.
"""

from __future__ import annotations

import pytest

from mcp_agent_chatterbox.config import Config
from mcp_agent_chatterbox.engine import ChatterboxEngine
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


class TestRegistry:  # noqa: N801 — the registry is imported into this test module
    def test_describe_lists_all_three_models(self):
        keys = {m["key"] for m in describe_models()}
        assert keys == {"turbo", "multilingual", "original"}

    def test_only_turbo_has_a_stock_voice(self):
        stock = {m["key"] for m in describe_models() if m["stock_voice"]}
        assert stock == {"turbo"}

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
