"""Config parsing: env vars, booleans, and the resolve-against-root-not-CWD rule."""

from __future__ import annotations

import os
from pathlib import Path

from mcp_agent_chatterbox.config import DEFAULT_VRAM_FLOOR_MB, Config

# The project root config.py anchors relative paths to.
_BASE = Path(__file__).resolve().parent.parent


class TestDefaults:
    def test_sensible_defaults(self):
        c = Config()
        assert c.model == "turbo"
        assert c.voice == ""
        assert c.device == "auto"
        assert c.gpu_index is None
        assert c.vram_floor_mb == DEFAULT_VRAM_FLOOR_MB
        assert c.strict_vram is False
        assert c.autoplay is True
        assert c.progressive is False
        assert c.max_chars == 4000
        assert c.max_chunk_chars == 500
        assert c.chunk_pause_ms == 250
        assert c.hf_home is None


class TestHfHome:
    def test_defaults_to_none(self):
        c = Config()
        assert c.hf_home is None

    def test_absolute_path_kept(self, monkeypatch, tmp_path):
        target = tmp_path / "hf"
        monkeypatch.setenv("CHATTERBOX_HF_HOME", str(target))
        assert Config().hf_home == target

    def test_relative_path_resolves_against_project_root(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CHATTERBOX_HF_HOME", "hf_cache")
        c = Config()
        assert c.hf_home == (_BASE / "hf_cache").resolve()

    def test_empty_value_is_none(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_HF_HOME", "   ")
        assert Config().hf_home is None


class TestPathResolution:
    def test_relative_paths_resolve_against_project_root_not_cwd(self, monkeypatch, tmp_path):
        # The whole point: an MCP harness may spawn us from anywhere.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CHATTERBOX_OUTPUT_DIR", "tts_output")
        c = Config()
        assert c.output_dir == (_BASE / "tts_output").resolve()
        assert c.output_dir.is_absolute()

    def test_absolute_paths_are_kept(self, monkeypatch, tmp_path):
        target = tmp_path / "elsewhere"
        monkeypatch.setenv("CHATTERBOX_OUTPUT_DIR", str(target))
        assert Config().output_dir == target

    def test_all_path_fields_are_absolute(self):
        c = Config()
        for value in (c.output_dir, c.voices_dir, c.logs_dir):
            assert isinstance(value, Path)
            assert value.is_absolute()


class TestBooleans:
    def test_truthy_spellings(self, monkeypatch):
        for raw in ("1", "true", "TRUE", "yes", "on", "On"):
            monkeypatch.setenv("CHATTERBOX_AUTOPLAY", raw)
            assert Config().autoplay is True, raw

    def test_falsy_spellings(self, monkeypatch):
        for raw in ("0", "false", "FALSE", "no", "off", "OFF"):
            monkeypatch.setenv("CHATTERBOX_AUTOPLAY", raw)
            assert Config().autoplay is False, raw

    def test_blank_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_AUTOPLAY", "   ")
        assert Config().autoplay is True

    def test_garbage_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_AUTOPLAY", "perhaps")
        assert Config().autoplay is True


class TestProgressive:
    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_PROGRESSIVE", "1")
        assert Config().progressive is True

    def test_defaults_off(self, monkeypatch):
        monkeypatch.delenv("CHATTERBOX_PROGRESSIVE", raising=False)
        assert Config().progressive is False


class TestIntegers:
    def test_parses(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_VRAM_MB", "8192")
        assert Config().vram_floor_mb == 8192

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_VRAM_MB", "lots")
        assert Config().vram_floor_mb == DEFAULT_VRAM_FLOOR_MB

    def test_blank_falls_back(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_MAX_CHARS", "")
        assert Config().max_chars == 4000

    def test_chunk_settings_parse(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_MAX_CHUNK_CHARS", "300")
        monkeypatch.setenv("CHATTERBOX_CHUNK_PAUSE_MS", "150")
        c = Config()
        assert c.max_chunk_chars == 300
        assert c.chunk_pause_ms == 150

    def test_chunk_settings_blank_fall_back(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_MAX_CHUNK_CHARS", "")
        monkeypatch.setenv("CHATTERBOX_CHUNK_PAUSE_MS", "")
        c = Config()
        assert c.max_chunk_chars == 500
        assert c.chunk_pause_ms == 250


class TestGpuIndex:
    def test_unset_is_none(self):
        assert Config().gpu_index is None

    def test_empty_string_is_none(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_GPU_INDEX", "")
        assert Config().gpu_index is None

    def test_parses_zero(self, monkeypatch):
        # 0 is a valid ordinal; a falsy check would lose it.
        monkeypatch.setenv("CHATTERBOX_GPU_INDEX", "0")
        assert Config().gpu_index == 0

    def test_negative_becomes_none(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_GPU_INDEX", "-1")
        assert Config().gpu_index is None

    def test_garbage_becomes_none(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_GPU_INDEX", "first")
        assert Config().gpu_index is None


class TestDefaultVoice:
    def test_unset_is_empty(self):
        assert Config().voice == ""

    def test_set_is_kept(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_VOICE", "  Narrator  ")
        # Voice names resolve case-insensitively, so only whitespace is
        # trimmed, never lowercased (compare CHATTERBOX_MODEL).
        assert Config().voice == "Narrator"

    def test_blank_is_empty(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_VOICE", "   ")
        assert Config().voice == ""


class TestModelAndDevice:
    def test_model_lowercased_and_stripped(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_MODEL", "  Turbo  ")
        assert Config().model == "turbo"

    def test_device_lowercased_and_stripped(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_DEVICE", "  CUDA:1 ")
        assert Config().device == "cuda:1"


class TestRepr:
    def test_repr_does_not_raise(self):
        text = repr(Config())
        assert "Config(" in text
        # A path is fine in a repr; a secret would not be, and there are none.
        assert "password" not in text.lower()


def test_no_fallback_values_for_credentials():
    """
    Sanitization guard: no credential-shaped env var may carry a hardcoded
    default. The project holds no secrets, so the real check is that the only
    env vars it reads are the documented CHATTERBOX_* settings — an
    os.getenv("API_KEY", "x") anywhere would show up here as an unexpected name.
    """
    import re

    import mcp_agent_chatterbox

    package = Path(mcp_agent_chatterbox.__file__).parent
    read = re.compile(r"""os\.getenv\(\s*["']([A-Z0-9_]+)["']""")
    found: set[str] = set()
    for path in package.rglob("*.py"):
        found.update(read.findall(path.read_text(encoding="utf-8")))

    unexpected = {n for n in found if not n.startswith("CHATTERBOX_")}
    assert not unexpected, f"unexpected env vars in the package: {sorted(unexpected)}"


def test_no_credential_defaults_anywhere():
    """A hardcoded default for a credential-shaped name is the banned pattern."""
    import re

    import mcp_agent_chatterbox

    package = Path(mcp_agent_chatterbox.__file__).parent
    # os.getenv(NAME, <literal>) — the second positional argument is the default.
    pattern = re.compile(
        r"""os\.getenv\(\s*["']([A-Z0-9_]*(?:PASSWORD|TOKEN|SECRET|API_KEY|CREDENTIAL)[A-Z0-9_]*)"""
        r"""["']\s*,""",
        re.IGNORECASE,
    )
    for path in package.rglob("*.py"):
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} gives {hits} a hardcoded fallback value"


def test_env_is_not_written_at_import():
    # Importing the package must not mutate the process environment.
    before = dict(os.environ)
    import importlib

    import mcp_agent_chatterbox.config as config_module

    importlib.reload(config_module)
    assert dict(os.environ) == before
