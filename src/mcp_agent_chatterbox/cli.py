"""
cli.py — command-line interface
==============================
Subcommands:
  serve    Run the MCP server (default). --http switches to streamable-http.
           The server does not synthesize; it forwards to the resident worker.
  worker   Run the resident synthesis worker. Start this once, before the
           first spoken reply, and leave it running.
  doctor   Offline diagnostics (paths, torch/CUDA, per-GPU VRAM, voices).
  panel    Local control panel in a browser: stop the audio, replay an earlier
           message, speak something by hand. On demand; it owns no audio.
  speak    Synthesize one utterance from the shell — the same code path the
           resident worker uses, so it is the fastest way to check a fresh
           install.

The CLI is the operator console, not the model-facing surface; output only
reports paths/state, no secrets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp_agent_chatterbox import __version__


def _cmd_serve(args) -> int:
    from mcp_agent_chatterbox.server import run

    transport = "streamable-http" if args.http else "stdio"
    try:
        run(transport=transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_worker(args) -> int:
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.logging_setup import setup_logging
    from mcp_agent_chatterbox.worker import serve

    cfg = Config()
    setup_logging(str(cfg.logs_dir))
    port = args.port if args.port is not None else cfg.worker_port
    print(f"chatterbox worker listening on 127.0.0.1:{port} (Ctrl+C to stop)")
    print(f"  model   : {cfg.model}")
    print(f"  voice   : {cfg.voice or '(model default)'}")
    print(f"  output  : {cfg.output_dir}")
    sys.stdout.flush()
    try:
        return serve(cfg, port=port, host="127.0.0.1")
    except KeyboardInterrupt:
        return 0


def _cmd_worker_status(args) -> int:
    """Report whether a worker is up, and what it is doing."""
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.worker import WorkerUnavailable, send_request

    cfg = Config()
    port = args.port if args.port is not None else cfg.worker_port
    try:
        response = send_request({"op": "status"}, port=port, timeout=cfg.worker_connect_timeout_s)
    except WorkerUnavailable as exc:
        print(f"no worker on 127.0.0.1:{port} ({exc})")
        print("Start one with: mcp-agent-chatterbox worker")
        return 1
    print(f"worker on 127.0.0.1:{port}: {response.get('status', 'unknown')}")
    last = response.get("last")
    if last:
        print(f"  last    : {last.get('message_id')} ({len(last.get('written', []))} file(s))")
    return 0 if response.get("ok") else 1


def _cmd_worker_stop(args) -> int:
    """Ask the resident worker to drain its queue and exit."""
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.worker import WorkerUnavailable, send_request

    cfg = Config()
    port = args.port if args.port is not None else cfg.worker_port
    try:
        response = send_request({"op": "shutdown"}, port=port, timeout=cfg.worker_connect_timeout_s)
    except WorkerUnavailable as exc:
        print(f"no worker on 127.0.0.1:{port} ({exc})")
        return 1
    print(f"worker on 127.0.0.1:{port}: {response.get('status')} (finishing the queue first)")
    return 0


def _cmd_panel(args) -> int:
    """Run the local control panel (stop, replay, speak by hand)."""
    from mcp_agent_chatterbox.config import Config
    from mcp_agent_chatterbox.logging_setup import setup_logging
    from mcp_agent_chatterbox.panel import DEFAULT_PANEL_PORT, serve

    cfg = Config()
    setup_logging(str(cfg.logs_dir))
    port = args.port if args.port is not None else DEFAULT_PANEL_PORT
    try:
        return serve(port, worker_port=cfg.worker_port, open_browser=not args.no_open)
    except KeyboardInterrupt:
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
    print(f"    chunk size   : {cfg.max_chunk_chars} chars")
    print(f"    chunk pause  : {cfg.chunk_pause_ms} ms")
    print(f"    worker port  : {cfg.worker_port} (server forwards speak() here)")

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
    from mcp_agent_chatterbox.speak import Speaker

    cfg = Config()
    setup_logging(str(cfg.logs_dir))

    speaker = Speaker(cfg, ChatterboxEngine(cfg))
    try:
        payload = speaker.submit(
            args.text,
            model=args.model,
            voice=args.voice,
            reference_clip=args.reference_clip,
            language=args.language,
            t3_model=args.t3_model,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            norm_loudness=False if args.no_norm_loudness else None,
            seed=args.seed,
            play_audio=not args.no_play,
        )
    except ToolFault as fault:
        print(f"Failed  : {fault.reason} - {fault.message}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed  : {type(e).__name__} (see {cfg.logs_dir})", file=sys.stderr)
        return 1

    print(f"Message : {payload['message_id']}")
    print(f"Chunks  : {payload['chunks']} sentence(s), {payload['text_chars']} chars")
    knobs = {
        k: v
        for k, v in {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "norm_loudness": False if args.no_norm_loudness else None,
            "seed": args.seed,
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
        }.items()
        if v is not None
    }
    if knobs:
        print("Knobs   : " + ", ".join(f"{k}={v}" for k, v in knobs.items()))

    # The CLI is the operator console, so unlike the MCP tool it blocks: the
    # shell should not return while the worker is still rendering.
    speaker.wait()
    last = speaker.status()["last"] or {}
    for name in last.get("written", []):
        print(f"Wrote   : {cfg.output_dir / name}")
    if last:
        print(f"Model   : {payload['model']} on {last.get('device') or 'unknown'}")
        print(
            f"Duration: {last.get('duration_s')}s "
            f"(synthesis {last.get('synthesis_s')}s)"
        )
    print(f"Played  : {last.get('played_chunks', 0)} chunk(s)")
    if last.get("first_audio_s") is not None:
        gap = last.get("max_gap_s")
        print(f"Timing  : first audio {last['first_audio_s']}s, worst gap {gap}s")
    for error in last.get("errors", []):
        print(f"Warning : {error}", file=sys.stderr)
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

    p_worker = sub.add_parser(
        "worker", help="Run the resident synthesis worker (start this first)."
    )
    p_worker.add_argument(
        "--port",
        type=int,
        default=None,
        help="Loopback port to listen on (default: CHATTERBOX_WORKER_PORT).",
    )
    p_worker.set_defaults(func=_cmd_worker)

    p_wstatus = sub.add_parser(
        "worker-status", help="Report whether the resident worker is up."
    )
    p_wstatus.add_argument("--port", type=int, default=None)
    p_wstatus.set_defaults(func=_cmd_worker_status)

    p_wstop = sub.add_parser(
        "worker-stop", help="Ask the resident worker to drain its queue and exit."
    )
    p_wstop.add_argument("--port", type=int, default=None)
    p_wstop.set_defaults(func=_cmd_worker_stop)

    p_panel = sub.add_parser(
        "panel",
        help="Run the local control panel: stop, replay an old message, speak by hand.",
    )
    p_panel.add_argument(
        "--port",
        type=int,
        default=None,
        help="Loopback port to serve on (default: 8766).",
    )
    p_panel.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the page in the default browser.",
    )
    p_panel.set_defaults(func=_cmd_panel)

    p_speak = sub.add_parser("speak", help="Synthesize one utterance and save/play it.")
    p_speak.add_argument("text", help="Text to speak.")
    p_speak.add_argument("--model", default=None, help="turbo | multilingual | original")
    p_speak.add_argument("--voice", default=None, help="Reference clip name in the voices dir.")
    p_speak.add_argument("--reference-clip", default=None, help="Path to a reference clip.")
    p_speak.add_argument("--language", default=None, help="ISO 639-1 code (multilingual only).")
    p_speak.add_argument("--t3-model", default=None, help="v2 | v3 (multilingual only).")
    p_speak.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature (0.05-2.0)."
    )
    p_speak.add_argument(
        "--exaggeration", type=float, default=None, help="Style control (500M models only)."
    )
    p_speak.add_argument(
        "--cfg-weight", type=float, default=None, help="CFG guidance weight (500M models only)."
    )
    p_speak.add_argument(
        "--top-p", type=float, default=None, help="Nucleus-sampling cutoff (0.0-1.0)."
    )
    p_speak.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling size (0-1000, turbo only)."
    )
    p_speak.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalise repeated tokens (1.0-2.0).",
    )
    p_speak.add_argument(
        "--no-norm-loudness",
        action="store_true",
        help="Do not normalize output to -27 LUFS (turbo only).",
    )
    p_speak.add_argument(
        "--seed", type=int, default=None, help="Seed torch (reproducible re-render within a session)."
    )
    p_speak.add_argument("--no-play", action="store_true", help="Only write the WAV files.")
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
    # The module path is prepended so this file is runnable on its own
    # (`python <...>/cli.py worker`), which is how worker.spawn() starts a
    # worker without depending on the console script being on PATH.
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    # Imported before anything else can: the console-script entry point has
    # already imported the package by now, but a direct run of this file has
    # not, and the relative imports below would fail without it.
    from mcp_agent_chatterbox.cli import main as _main

    sys.exit(_main())


__all__ = ["build_parser", "main"]
