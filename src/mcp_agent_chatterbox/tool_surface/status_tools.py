"""
status_tools — readiness and VRAM reclamation
=============================================
tts_status: which models exist, which device would be used, how much VRAM is
            free on each card, and whether weights are already resident.
            Loads nothing, so it is safe to call as a preflight.
tts_unload:  drop the resident model and give the VRAM back. The escape hatch
            that matters on a shared GPU.
"""

from __future__ import annotations

import json

from mcp_agent_chatterbox.errors import tool_error
from mcp_agent_chatterbox.registry import (
    SUPPORTED_LANGUAGES,
    T3_MODELS,
    TURBO_TAGS,
    describe_models,
)


def register(server, ctx) -> None:
    engine = ctx.engine
    cfg = ctx.cfg

    @server.tool()
    def tts_status() -> str:
        """
        Report Chatterbox TTS readiness: installed packages, resolved device,
        per-GPU free/total VRAM, the resident model, the models available, the
        languages the multilingual model speaks, the configured default voice
        (CHATTERBOX_VOICE), and the paralinguistic tags turbo understands.
        Loads no weights — safe as a preflight before a speak call. On a
        machine where a large model already occupies a GPU, check free_mb here
        first: a card with less than ~4096 MB free will fail to load.
        """
        try:
            payload = {
                "status": "ok",
                "runtime": engine.status(),
                "gpus": engine.gpu_report(),
                "models": describe_models(),
                "default_voice": cfg.voice or None,
                "multilingual_languages": SUPPORTED_LANGUAGES,
                "multilingual_t3_models": sorted(T3_MODELS),
                "turbo_paralinguistic_tags": list(TURBO_TAGS),
            }
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("tts_status", e)

    @server.tool()
    def tts_unload() -> str:
        """
        Release the currently loaded Chatterbox model and return its VRAM to
        the GPU. Use this when another process needs the card, or to drop a
        500M model before a lighter turbo load. Safe to call when nothing is
        loaded.
        """
        try:
            result = engine.unload()
            return json.dumps(
                {
                    "status": "ok",
                    **result,
                    "note": (
                        "Model released. The next speak call reloads it, which "
                        "costs seconds and may re-download nothing (weights stay "
                        "in the Hugging Face cache)."
                        if result["was_loaded"]
                        else "No model was loaded."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:
            return tool_error("tts_unload", e)


__all__ = ["register"]
