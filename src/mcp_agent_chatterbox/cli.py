"""
cli.py — command-line interface
==============================
Subcommands:
  serve    Run the MCP server (default). --http switches to streamable-http.
  doctor   Offline diagnostics (paths, torch/CUDA, per-GPU VRAM, voices).
  speak    Synthesize one utterance from the shell — the same code path the
           MCP tool uses, so it is the fastest way to check a fresh install.

The CLI is the operator console, not the model-facing surface; output only
reports paths/state, no secrets.
"""

from __future__ import annotations

import argparse
import sys

from mcp_agent_chatterbox import __version__


def _cmd_serve(args) -> int:
    from mcp_agent_chatterbox.server import run

    transport = "streamable-http" if args.http else "stdio"
    try:
        run(transport=transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_doctor(args) -> int:
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.engine import ChatterboxEngine
    from mcp_agent_chatterbox.playback import player_available
    from mcp_agent_chatterbox.registry import describe_models
    from mcp_agent_chatterbox.voices import list_voices

    print(f"mcp-agent-chatterbox {__version__} - doctor\n")

    try:
        cfg = Config()
    except Exception as e:
        print(f"[config] {type(e).__name__}: {e}")
        return 1

    print("  Configuration:")
    print(f"    output dir   : {cfg.output_dir}")
    print(f"    voices dir   : {cfg.voices_dir}")
    print(f"    logs dir     : {cfg.logs_dir}")
    print(f"    default model: {cfg.model}")
    print(f"    device       : {cfg.device}")
    print(
        f"    gpu index    : {cfg.gpu_index if cfg.gpu_index is not None else '(auto: emptiest card)'}"
    )
    print(f"    vram floor   : {cfg.vram_floor_mb} MiB (strict={cfg.strict_vram})")
    print(f"    autoplay     : {cfg.autoplay} (player available: {player_available()})")
    print(f"    max chars    : {cfg.max_chars}")

    engine = ChatterboxEngine(cfg)
    status = engine.status()
    print("\n  Runtime:")
    for k, v in status.items():
        if k in {"output_dir", "voices_dir", "default_model", "autoplay"}:
            continue
        print(f"    {k:24s} {v}")

    gpus = engine.gpu_report()
    if gpus:
        print("\n  GPUs:")
        for g in gpus:
            mark = " <- selected" if g.get("selected") else ""
            free = g.get("free_mb")
            total = g.get("total_mb")
            mem = f"{free} / {total} MiB free" if free is not None else "unknown"
            print(f"    [{g['index']}] {g.get('name') or 'unknown'}: {mem}{mark}")
            if free is not None and free < cfg.vram_floor_mb:
                print(
                    f"        WARNING: only {free} MiB free — below the {cfg.vram_floor_mb} MiB "
                    "floor. Unload the model in LM Studio or pick another card with "
                    "CHATTERBOX_GPU_INDEX."
                )
    else:
        print("\n  GPUs: none visible (running on CPU)")

    print("\n  Models:")
    for m in describe_models():
        voice = "built-in voice" if m["stock_voice"] else "needs a reference clip"
        print(f"    {m['key']:14s} {m['parameters']:5s} {m['languages']:34s} {voice}")

    voices = list_voices(cfg.voices_dir)
    print(f"\n  Reference clips ({len(voices)} in {cfg.voices_dir}):")
    if voices:
        for v in voices:
            size = f"{v['size_kb']} KB" if v.get("size_kb") is not None else "?"
            print(f"    {v['name']:24s} {v['filename']:28s} {size}")
    else:
        print("    (none - turbo speaks with its built-in voice)")

    return 0


def _cmd_speak(args) -> int:
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.engine import ChatterboxEngine
    from mcp_agent_chatterbox.errors import ToolFault
    from mcp_agent_chatterbox.logging_setup import setup_logging
    from mcp_agent_chatterbox.speak import speak_once

    cfg = Config()
    setup_logging(str(cfg.logs_dir))

    try:
        result = speak_once(
            args.text,
            cfg=cfg,
            engine=ChatterboxEngine(cfg),
            model=args.model,
            voice=args.voice,
            reference_clip=args.reference_clip,
            language=args.language,
            t3_model=args.t3_model,
            play_audio=not args.no_play,
            wait=args.wait,
        )
    except ToolFault as fault:
        print(f"Failed  : {fault.reason} - {fault.message}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed  : {type(e).__name__} (see {cfg.logs_dir})", file=sys.stderr)
        return 1

    print(f"Wrote   : {result['path']}")
    print(f"Model   : {result['model']} on {result['device']}")
    print(
        f"Duration: {result['duration_s']}s @ {result['sample_rate']} Hz "
        f"(synthesis {result['synthesis_s']}s)"
    )
    if result.get("played"):
        print("Played  : yes")
    elif result.get("playback_note"):
        print(f"Played  : no ({result['playback_note']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-agent-chatterbox",
        description=(
            "MCP server for local GPU-accelerated text-to-speech with Chatterbox "
            "(turbo, multilingual, zero-shot voice cloning)."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Run the MCP server (default).")
    p_serve.add_argument("--http", action="store_true", help="Use streamable-http transport.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_doc = sub.add_parser("doctor", help="Diagnose configuration, CUDA and voices.")
    p_doc.set_defaults(func=_cmd_doctor)

    p_speak = sub.add_parser("speak", help="Synthesize one utterance and save/play it.")
    p_speak.add_argument("text", help="Text to speak.")
    p_speak.add_argument("--model", default=None, help="turbo | multilingual | original")
    p_speak.add_argument("--voice", default=None, help="Reference clip name in the voices dir.")
    p_speak.add_argument("--reference-clip", default=None, help="Path to a reference clip.")
    p_speak.add_argument("--language", default=None, help="ISO 639-1 code (multilingual only).")
    p_speak.add_argument("--t3-model", default=None, help="v2 | v3 (multilingual only).")
    p_speak.add_argument("--no-play", action="store_true", help="Only write the WAV.")
    p_speak.add_argument("--wait", action="store_true", help="Block until playback finishes.")
    p_speak.set_defaults(func=_cmd_speak)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default action: serve over stdio (what MCP harnesses expect).
        args.http = False
        args.host = "127.0.0.1"
        args.port = 8000
        return _cmd_serve(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
