"""Config parsing: env vars, booleans, and the resolve-against-root-not-CWD rule."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcp_agent_chatterbox.config import (
    DEFAULT_VRAM_FLOOR_MB,
    DEFAULT_WORKER_CONNECT_TIMEOUT_S,
    DEFAULT_WORKER_PORT,
    Config,
)

# The project root config.py anchors relative paths to.
_BASE = Path(__file__).resolve().parent.parent


class TestDefaults:
    def test_sensible_defaults(self):
        c = Config()
        assert c.model == "nano"
        assert c.voice == ""
        assert c.device == "auto"
        assert c.gpu_index is None
        assert c.vram_floor_mb == DEFAULT_VRAM_FLOOR_MB
        assert c.strict_vram is False
        assert c.autoplay is True
        assert c.max_chars == 4000
        assert c.max_chunk_chars == 500
        assert c.sentences_per_chunk == 3
        assert c.chunk_pause_ms == 250
        assert c.drain_timeout_s == 300
        assert c.worker_port == DEFAULT_WORKER_PORT
        assert c.worker_connect_timeout_s == DEFAULT_WORKER_CONNECT_TIMEOUT_S
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
        monkeypatch.setenv("CHATTERBOX_SENTENCES_PER_CHUNK", "")
        monkeypatch.setenv("CHATTERBOX_CHUNK_PAUSE_MS", "")
        c = Config()
        assert c.max_chunk_chars == 500
        assert c.sentences_per_chunk == 3
        assert c.chunk_pause_ms == 250

    def test_sentences_per_chunk_parses(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_SENTENCES_PER_CHUNK", "1")
        assert Config().sentences_per_chunk == 1

    def test_drain_timeout_parses(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_DRAIN_TIMEOUT", "45")
        assert Config().drain_timeout_s == 45

    def test_drain_timeout_zero_is_allowed(self, monkeypatch):
        # 0 means "do not wait at all", a legitimate choice, not a fallback.
        monkeypatch.setenv("CHATTERBOX_DRAIN_TIMEOUT", "0")
        assert Config().drain_timeout_s == 0

    def test_drain_timeout_blank_falls_back(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_DRAIN_TIMEOUT", "")
        assert Config().drain_timeout_s == 300

    def test_worker_port_parses(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_WORKER_PORT", "9123")
        assert Config().worker_port == 9123

    def test_worker_port_blank_falls_back(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_WORKER_PORT", "")
        assert Config().worker_port == DEFAULT_WORKER_PORT

    def test_worker_connect_timeout_parses(self, monkeypatch):
        monkeypatch.setenv("CHATTERBOX_WORKER_CONNECT_TIMEOUT", "30")
        assert Config().worker_connect_timeout_s == 30


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


class TestEnvFile:
    """
    The project config file: the mechanism that replaces the harness env block.

    These run against a tmp .env, never the repo's own, and each one restores
    both os.environ and ENV_FILE_KEYS so no later test sees leaked state.
    """

    @pytest.fixture
    def env_file(self, tmp_path: Path, monkeypatch):
        """Write KEY=value lines to a tmp .env and yield its path."""
        import mcp_agent_chatterbox.config as config_module

        before = dict(os.environ)
        before_keys = set(config_module.ENV_FILE_KEYS)

        def _write(text: str) -> Path:
            path = tmp_path / ".env"
            path.write_text(text, encoding="utf-8")
            return path

        yield _write

        for name in set(os.environ) - set(before):
            monkeypatch.delenv(name, raising=False)
        for name, value in before.items():
            if os.environ.get(name) != value:
                monkeypatch.setenv(name, value)
        config_module.ENV_FILE_KEYS.clear()
        config_module.ENV_FILE_KEYS.update(before_keys)

    def test_missing_file_returns_none(self, tmp_path: Path):
        from mcp_agent_chatterbox.config import load_env_file

        assert load_env_file(tmp_path / "nope.env") is None

    def test_file_values_reach_config(self, env_file):
        from mcp_agent_chatterbox.config import load_env_file

        path = env_file("CHATTERBOX_MODEL=nano\nCHATTERBOX_VOICE=seven\n")
        assert load_env_file(path) == path
        cfg = Config()
        assert cfg.model == "nano"
        assert cfg.voice == "seven"

    def test_exported_value_beats_file(self, env_file, monkeypatch):
        # The documented precedence: an explicit export wins over the file, so
        # a one-off override and a test monkeypatch both behave as expected.
        from mcp_agent_chatterbox.config import load_env_file

        monkeypatch.setenv("CHATTERBOX_MODEL", "turbo")
        load_env_file(env_file("CHATTERBOX_MODEL=nano\n"))
        assert Config().model == "turbo"

    def test_env_file_keys_records_only_file_sourced_names(self, env_file, monkeypatch):
        import mcp_agent_chatterbox.config as config_module

        monkeypatch.setenv("CHATTERBOX_MODEL", "turbo")  # already present
        config_module.load_env_file(
            env_file("CHATTERBOX_MODEL=nano\nCHATTERBOX_VOICE=seven\n")
        )
        # VOICE came from the file; MODEL did not, so it is not recorded and a
        # test suite can free the former without disturbing the latter.
        assert "CHATTERBOX_VOICE" in config_module.ENV_FILE_KEYS
        assert "CHATTERBOX_MODEL" not in config_module.ENV_FILE_KEYS

    def test_quotes_comments_and_export_are_handled(self, env_file):
        from mcp_agent_chatterbox.config import load_env_file

        load_env_file(
            env_file(
                "# a comment\n"
                "\n"
                'export CHATTERBOX_VOICE="Narrator"\n'
                "CHATTERBOX_MODEL='nano'\n"
                "not-a-pair\n"
            )
        )
        cfg = Config()
        assert cfg.voice == "Narrator"
        assert cfg.model == "nano"

    def test_fallback_parser_matches_the_dotenv_path(self, env_file):
        # _load_env_fallback exists for the degraded case where python-dotenv is
        # absent. It must parse the same file the same way, or the fallback
        # silently changes behaviour instead of degrading.
        from mcp_agent_chatterbox.config import _load_env_fallback

        _load_env_fallback(
            env_file("# c\nexport CHATTERBOX_VOICE=\"Narrator\"\nCHATTERBOX_MODEL=nano\n")
        )
        cfg = Config()
        assert cfg.voice == "Narrator"
        assert cfg.model == "nano"

    def test_fallback_parser_does_not_override_existing(self, env_file, monkeypatch):
        from mcp_agent_chatterbox.config import _load_env_fallback

        monkeypatch.setenv("CHATTERBOX_MODEL", "turbo")
        _load_env_fallback(env_file("CHATTERBOX_MODEL=nano\n"))
        assert os.environ["CHATTERBOX_MODEL"] == "turbo"


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


def test_import_does_not_override_exported_values(monkeypatch):
    """
    Importing the package seeds os.environ from .env but must never clobber a
    value the caller exported. This replaces an older test that asserted import
    mutates nothing at all: that was true while settings came only from the
    harness, but the point of .env is precisely to mutate the environment at
    import. The invariant worth keeping is environment-over-file precedence —
    an explicit export still wins, so a one-off override and a monkeypatch both
    behave as the caller expects.
    """
    import importlib

    monkeypatch.setenv("CHATTERBOX_MODEL", "turbo")

    import mcp_agent_chatterbox.config as config_module

    importlib.reload(config_module)

    assert os.environ["CHATTERBOX_MODEL"] == "turbo"
