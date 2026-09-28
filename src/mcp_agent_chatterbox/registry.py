"""
registry.py — what the three Chatterbox models are and how to drive them
=======================================================================
Chatterbox exposes three `generate()` signatures that do not agree with each
other, and the per-model defaults are not the same either. Turbo is tuned for
latency with cfg_weight=0.0 / exaggeration=0.0, while the 500M models expect
0.5 / 0.5. Passing a single set of defaults to all three audibly degrades two
of them, so the defaults live here per model and the tool layer resolves
"unspecified" against this table rather than against its own signature.

The import targets are stored as strings and imported on first use. Importing
chatterbox eagerly costs several seconds and pulls in torch, so a server that
is only being asked for status must not pay that.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

#: ISO 639-1 codes ChatterboxMultilingualTTS accepts, with English names.
#: Mirrors chatterbox.mtl_tts.SUPPORTED_LANGUAGES; duplicated here so that
#: language validation works before the (slow) chatterbox import.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

#: Paralinguistic tags the turbo model understands natively. Passed through
#: verbatim to the tokenizer, so this list is documentation for the model, not
#: validation — Chatterbox will raise on anything it cannot tokenize.
TURBO_TAGS: tuple[str, ...] = (
    "[laugh]",
    "[chuckle]",
    "[cough]",
    "[sigh]",
    "[gasp]",
    "[hiccup]",
    "[sniff]",
    "[clearthroat]",
    "[groan]",
    "[whisper]",
)

#: Reference clip suffixes accepted for voice cloning.
VOICE_SUFFIXES: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the tool layer needs to know about one Chatterbox model."""

    key: str
    label: str
    module: str
    attr: str
    params: str
    languages: str
    #: True when the checkpoint carries a built-in voice, i.e. the model can
    #: speak without a reference clip. Only turbo does.
    stock_voice: bool
    #: generate() kwargs, with the values Chatterbox itself defaults to. The
    #: tool layer omits any kwarg the caller left unspecified so the library
    #: default applies.
    defaults: dict[str, Any] = field(default_factory=dict)
    #: The subset of generate() kwargs that actually have an effect for this
    #: model. The three signatures overlap but do not agree: turbo honors
    #: top_k/norm_loudness but ignores min_p/exaggeration/cfg_weight, while the
    #: 500M models honor min_p/exaggeration/cfg_weight and have no
    #: top_k/norm_loudness parameter at all. The engine drops any caller-set
    #: knob outside this set (with a warning) instead of letting it crash or
    #: silently do nothing.
    honored_knobs: frozenset[str] = frozenset()
    #: Extra note surfaced in tts_status / tool descriptions.
    note: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "turbo": ModelSpec(
        key="turbo",
        label="Chatterbox-Turbo",
        module="chatterbox.tts_turbo",
        attr="ChatterboxTurboTTS",
        params="350M",
        languages="English",
        stock_voice=True,
        honored_knobs=frozenset(
            {"repetition_penalty", "top_p", "top_k", "norm_loudness", "temperature"}
        ),
        defaults={
            "repetition_penalty": 1.2,
            "min_p": 0.00,
            "top_p": 0.95,
            "top_k": 1000,
            "exaggeration": 0.0,
            "cfg_weight": 0.0,
            "temperature": 0.8,
            "norm_loudness": True,
        },
        note=(
            "Lowest latency and VRAM. Has a built-in voice, so no reference clip "
            "is needed. Understands paralinguistic tags such as [laugh] and "
            "[chuckle] inline in the text."
        ),
    ),
    "multilingual": ModelSpec(
        key="multilingual",
        label="Chatterbox-Multilingual",
        module="chatterbox.mtl_tts",
        attr="ChatterboxMultilingualTTS",
        params="500M",
        languages="23 languages (incl. de, fr, es, zh)",
        stock_voice=False,
        honored_knobs=frozenset(
            {"repetition_penalty", "min_p", "top_p", "exaggeration", "cfg_weight", "temperature"}
        ),
        defaults={
            "repetition_penalty": 2.0,
            "min_p": 0.05,
            "top_p": 1.0,
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        },
        note=(
            "Requires a reference clip (audio_prompt) — it has no stock voice. "
            "Pass language=... as an ISO 639-1 code. t3_model selects the "
            "checkpoint: 'v2' (default) or 'v3'."
        ),
    ),
    "original": ModelSpec(
        key="original",
        label="Chatterbox",
        module="chatterbox.tts",
        attr="ChatterboxTTS",
        params="500M",
        languages="English",
        stock_voice=False,
        honored_knobs=frozenset(
            {"repetition_penalty", "min_p", "top_p", "exaggeration", "cfg_weight", "temperature"}
        ),
        defaults={
            "repetition_penalty": 1.2,
            "min_p": 0.05,
            "top_p": 1.0,
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        },
        note=(
            "The original English model. Requires a reference clip (audio_prompt) "
            "— it has no stock voice. The one to reach for when tuning "
            "exaggeration/cfg_weight for delivery style."
        ),
    ),
}

#: Accepted spellings for the multilingual checkpoint.
T3_MODELS: dict[str, str] = {"v2": "v2", "v3": "v3"}


def get_spec(model_key: str | None) -> ModelSpec | None:
    """Look up a model spec, tolerating a few obvious aliases."""
    if not model_key:
        return None
    key = model_key.strip().lower()
    aliases = {
        "en": "original",
        "english": "original",
        "mtl": "multilingual",
        "multi": "multilingual",
        "chatterbox": "original",
        "tts": "original",
        "chatterbox-turbo": "turbo",
        "chatterbox-tts": "original",
        "chatterbox-multilingual": "multilingual",
    }
    key = aliases.get(key, key)
    return MODEL_SPECS.get(key)


def model_keys() -> list[str]:
    """Model keys in registry order."""
    return list(MODEL_SPECS)


def describe_models() -> list[dict]:
    """Model specs as plain JSON-safe dicts, for tts_status."""
    return [
        {
            "key": s.key,
            "label": s.label,
            "parameters": s.params,
            "languages": s.languages,
            "stock_voice": s.stock_voice,
            "requires_reference_clip": not s.stock_voice,
            "tunable": sorted(k for k in s.defaults if k in s.honored_knobs),
            "note": s.note,
        }
        for s in MODEL_SPECS.values()
    ]


def load_model_class(spec: ModelSpec):
    """Import the Chatterbox class for `spec`. Deferred: torch is slow to import."""
    module = importlib.import_module(spec.module)
    return getattr(module, spec.attr)


__all__ = [
    "MODEL_SPECS",
    "SUPPORTED_LANGUAGES",
    "T3_MODELS",
    "TURBO_TAGS",
    "VOICE_SUFFIXES",
    "ModelSpec",
    "describe_models",
    "get_spec",
    "load_model_class",
    "model_keys",
]
