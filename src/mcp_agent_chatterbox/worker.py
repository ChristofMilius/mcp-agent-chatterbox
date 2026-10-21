"""
worker.py — the long-lived synthesis worker
===========================================
Wait on a socket for speak requests, render them with the resident model,
answer every request with a JSON line. This is the process that actually
synthesizes; nothing else does.

Why it is separate
------------------
Chatterbox synthesis was first served from the MCP server process itself. Inside
that process it reliably stopped after two sentences and never resumed, while
the identical Speaker worker driven from a plain script finished every sentence
of the same text. It reproduced over stdio and over streamable-http, so the
transport was not the cause, and it reproduced with the pipe held open and with
it closed early, so it was not the shutdown path either. The MCP SDK's process
is simply a bad host for multi-second background CUDA work.

So the server was dropped from the synthesis path. The MCP tool now hands the
text to this worker and returns; the worker is a plain Python process with no
transport, no event loop and no SDK, which is the configuration that works.

The same stall, from the user's side, was worse than a dropped sentence: the
MCP server died with its client's turn, so a reply that took longer than the
turn to render was killed mid-word regardless. A resident process outlives any
turn, and the model stays loaded between utterances.

Protocol
--------
One request per connection, newline-delimited JSON in both directions, over a
loopback TCP socket. The worker is strictly single-threaded: it accepts one
connection, handles it to completion, then accepts the next. That serialisation
is deliberate -- one GPU holds one resident model, and Chatterbox's model
instance carries mutable state (the Perth watermarker, the conditionals cache),
so two concurrent generate() calls on it are not safe.

  request   {"op": "speak", "text": "...", ...knobs}
            {"op": "status"}
            {"op": "unload"}    drop the resident model, return its VRAM
            {"op": "stop"}      cancel queued work and stop playback
            {"op": "shutdown"}  drain, then exit

  response  {"ok": true,  "status": "done"|"idle"|"busy"|"stopping"|..., ...}
            {"ok": false, "reason": "<ToolFault reason>", "message": "..."}

A response is always written before the connection closes, including for a
request that could not be parsed, so a caller never has to guess why a connect
went nowhere.

TCP rather than a Windows named pipe
------------------------------------
A pipe would need pywin32 (`win32pipe`, `win32file`), which this project does
not depend on, and the hand-rolled `\\\\.\\pipe\\` path adds real complexity for
no benefit here: the listening socket is bound to 127.0.0.1, so it is not
reachable from another machine, and the requests are small and local. The one
honest cost is that any local process can connect to the port, so a local
process can ask this worker to speak. It cannot read anything back beyond the
queue summary.

Startup, and who starts it
--------------------------
`mcp-agent-chatterbox worker` runs one in the foreground. Because that process
loads torch (~4s) it is not started lazily from the MCP tool: the tool would pay
the import on every turn and lose the responsiveness the resident worker exists
to provide. It is started ahead of time by the harness, or by hand; see
README.md.

Discovery is by port, so the client needs nothing beyond the port number:
`CHATTERBOX_WORKER_PORT` (8765 by default). `spawn()` exists for a caller that
wants to own one directly.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from mcp_agent_chatterbox.errors import ToolFault

logger = logging.getLogger(__name__)

#: Default loopback port. It lives in config.py next to every other setting;
#: imported here so a change of default cannot drift between the two modules.
from mcp_agent_chatterbox.config import DEFAULT_WORKER_PORT as DEFAULT_PORT  # noqa: E402

#: Backlog for the listening socket. The worker accepts one caller at a time and
#: each request is a single short line, so a handful of pending connections is
#: already more than the design produces.
_LISTEN_BACKLOG = 16

#: Longest request line accepted, so a runaway caller cannot make the worker
#: buffer without bound. The text budget is 4000 chars (CHATTERBOX_MAX_CHARS);
#: this leaves room for the JSON envelope and the knob arguments.
_MAX_REQUEST_BYTES = 64 * 1024


def encode(payload: dict) -> bytes:
    """Serialise a request or response as one newline-terminated JSON line."""
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _read_line(sock: socket.socket) -> bytes:
    """
    Read one request line, or whatever arrives before the socket closes.

    Stops at the first newline *in the bytes received so far*, not at the first
    newline in the newest chunk: a single recv() can carry the end of one line
    and the start of the next, and taking the whole chunk would then hand the
    parser two lines jammed together.
    """
    buffer = bytearray()
    while len(buffer) < _MAX_REQUEST_BYTES:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer.extend(chunk)
        if b"\n" in buffer:
            break
    return bytes(buffer).split(b"\n", 1)[0]


def _receive_line(sock: socket.socket, timeout: float) -> dict | None:
    """Read one request line. None when the caller sends nothing."""
    sock.settimeout(timeout)
    try:
        raw = _read_line(sock)
    except OSError:
        return None
    if not raw.strip():
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Not JSON: the caller gets a written response rather than a silent
        # close, so "the worker is not running" and "the worker did not
        # understand me" stay distinguishable.
        return {"op": "__unparseable__"}
    if not isinstance(decoded, dict):
        return {"op": "__unparseable__"}
    return decoded


def _safe_send(sock: socket.socket, payload: dict) -> None:
    try:
        sock.sendall(encode(payload))
    except OSError:
        logger.debug("[worker] caller disconnected before the response was sent")


def send_request(
    request: dict,
    port: int = DEFAULT_PORT,
    *,
    host: str = "127.0.0.1",
    timeout: float = 5.0,
) -> dict:
    """
    Send one request to the resident worker and return its response.

    Raises WorkerUnavailable when nothing is listening -- the caller turns that
    into a message that says how to start one.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(encode(request))
            # The worker answers only once the work is done, so the read
            # timeout is not `timeout`: a five-sentence reply takes tens of
            # seconds to render and must not be mistaken for a dead worker.
            sock.settimeout(None)
            raw = _read_line(sock)
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        raise WorkerUnavailable(f"no worker listening on {host}:{port} ({type(exc).__name__})") from exc
    if not raw.strip():
        raise WorkerUnavailable(f"the worker on {host}:{port} closed without answering")
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerUnavailable(
            f"the worker on {host}:{port} sent an unreadable response ({type(exc).__name__})"
        ) from exc
    if not isinstance(response, dict):
        raise WorkerUnavailable(f"the worker on {host}:{port} sent a non-object response")
    return response


class WorkerUnavailable(RuntimeError):
    """Nothing is listening on the worker port, or it stopped answering."""


def worker_script() -> str:
    """Absolute path of this package's cli.py, runnable without an install."""
    return str(Path(__file__).resolve().parent / "cli.py")


def spawn(
    port: int = DEFAULT_PORT,
    *,
    config_file: Path | None = None,
    python: str | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """
    Start a detached worker process. Returns the handle; the caller owns it.

    Detached on purpose: the worker outlives whatever launched it, which is the
    whole point of a resident worker. stdout/stderr go to `log_path` (the
    project log directory by default) rather than a pipe, because a pipe nobody
    drains fills up and blocks the child mid-write.
    """
    executable = python or sys.executable
    command = [executable, worker_script(), "worker", "--port", str(port)]
    if config_file is not None:
        command += ["--config", str(config_file)]

    kwargs: dict[str, Any] = {
        "cwd": str(Path(__file__).resolve().parent.parent.parent),
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        # Without this the child dies with its console. Not a "process group":
        # this project does not manage worker lifetime by pid.
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["stdout"] = kwargs["stderr"] = log_path.open("a", encoding="utf-8")
    else:
        kwargs["stdout"] = kwargs["stderr"] = subprocess.DEVNULL

    logger.info("[worker] spawning %s", " ".join(command))
    return subprocess.Popen(command, **kwargs)  # noqa: S603 — fixed args, own script


def wait_until_ready(port: int = DEFAULT_PORT, *, timeout: float = 60.0) -> bool:
    """
    Block until the worker answers a status request, or `timeout` elapses.

    Readiness is reported by the worker itself, so a listening socket that
    cannot yet speak (torch still importing) does not read as ready.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = send_request({"op": "status"}, port, timeout=1.0)
        except WorkerUnavailable:
            time.sleep(0.25)
            continue
        if response.get("ok"):
            return True
        time.sleep(0.25)
    return False


class Server:
    """
    The accept loop, around a callable that answers one request.

    Split from `serve()` so the protocol can be exercised against a fake
    handler without loading a model or binding a fixed port.
    """

    def __init__(self, handle, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.handle = handle
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._stopped = threading.Event()

    def bind(self) -> int:
        """Bind and listen. Returns the port actually bound (0 -> ephemeral)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(_LISTEN_BACKLOG)
        self._sock = sock
        self.port = sock.getsockname()[1]
        return self.port

    def serve_forever(self) -> None:
        """
        Accept and answer requests until a handler ends the loop.

        Nothing a caller sends and nothing a handler raises may end the loop: a
        resident worker that a single bad request can kill is worse than no
        worker, because it fails silently on the next utterance too. The failure
        is logged and reported to the caller, then the loop continues.
        """
        if self._sock is None:
            self.bind()
        assert self._sock is not None
        while not self._stopped.is_set():
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                if self._stopped.is_set():
                    return
                raise
            with conn:
                request = _receive_line(conn, timeout=5.0)
                if request is None:
                    continue
                try:
                    response, finished = self.handle(request)
                except Exception:  # noqa: BLE001 — one request must not kill the worker
                    logger.exception("[worker] handler raised on op=%r", request.get("op"))
                    response, finished = (
                        {
                            "ok": False,
                            "reason": "worker_error",
                            "message": "The worker failed to handle this request (see the log).",
                        },
                        False,
                    )
                _safe_send(conn, response)
            if finished:
                self._stopped.set()

    def stop(self) -> None:
        self._stopped.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()


def serve(
    cfg,
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    *,
    ready: threading.Event | None = None,
) -> int:
    """
    Run the worker until it is asked to shut down.

    Returns a process exit code. `ready` is set once the socket is listening,
    which is how a supervisor knows the port is up without polling it.
    """
    from mcp_agent_chatterbox.engine import ChatterboxEngine
    from mcp_agent_chatterbox.speak import Speaker

    speaker = Speaker(cfg, ChatterboxEngine(cfg))
    server = Server(lambda req: _dispatch(speaker, req), port=port, host=host)
    bound = server.bind()
    logger.info(
        "[worker] listening on %s:%d (model=%s, voice=%s, output=%s)",
        host,
        bound,
        cfg.model,
        cfg.voice or "unset",
        cfg.output_dir,
    )
    if ready is not None:
        ready.set()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[worker] interrupted; draining the queue")
    finally:
        speaker.drain(cfg.drain_timeout_s)
        server.stop()
    return 0


def _dispatch(speaker, request: dict) -> tuple[dict, bool]:
    """
    Answer one request. Returns (response, should_exit).

    Every failure a caller can act on comes back as ok=false with a ToolFault
    reason; nothing here raises out of the loop.
    """
    op = request.get("op")

    if op == "__unparseable__":
        return {
            "ok": False,
            "reason": "bad_request",
            "message": "Every request must be one JSON object per line.",
        }, False

    if op == "speak":
        text = request.get("text")
        if not isinstance(text, str):
            return {
                "ok": False,
                "reason": "bad_request",
                "message": "speak needs a string text field.",
            }, False
        try:
            # The worker blocks until the reply has been spoken. That is the
            # contract: a client with a bounded read timeout would otherwise
            # abandon a render that is still running and report a failure for
            # audio that arrives afterwards. Nothing is lost by waiting -- the
            # server-side tool call that talks to this worker has no timeout.
            payload = speaker.submit(
                text,
                model=request.get("model"),
                voice=request.get("voice"),
                reference_clip=request.get("reference_clip"),
                language=request.get("language"),
                t3_model=request.get("t3_model"),
                exaggeration=request.get("exaggeration"),
                cfg_weight=request.get("cfg_weight"),
                temperature=request.get("temperature"),
                top_p=request.get("top_p"),
                top_k=request.get("top_k"),
                repetition_penalty=request.get("repetition_penalty"),
                norm_loudness=request.get("norm_loudness"),
                seed=request.get("seed"),
                play_audio=request.get("play_audio"),
                filename=request.get("filename"),
            )
        except ToolFault as fault:
            return {"ok": False, "reason": fault.reason, "message": fault.message, **fault.extra}, False
        except Exception as exc:  # noqa: BLE001 — one bad request must not kill the worker
            logger.exception("[worker] speak failed")
            return {
                "ok": False,
                "reason": "worker_error",
                "message": f"{type(exc).__name__} while accepting the text (see the log)",
            }, False

        speaker.wait()
        last = speaker.status()["last"] or {}
        return (
            {
                "ok": True,
                "status": "done",
                "message_id": payload["message_id"],
                "chunks": payload["chunks"],
                "written": last.get("written", []),
                "errors": last.get("errors", []),
                "played_chunks": last.get("played_chunks", 0),
                "synthesis_s": last.get("synthesis_s", 0.0),
                "duration_s": last.get("duration_s", 0.0),
                "first_audio_s": last.get("first_audio_s"),
                "max_gap_s": last.get("max_gap_s"),
                "device": last.get("device"),
                "output_dir": payload["output_dir"],
                "model": payload["model"],
                "voice": payload["voice"],
            },
            False,
        )

    if op == "status":
        state = speaker.status()
        return {"ok": True, "status": "idle" if state["pending"] == 0 else "busy", **state}, False

    if op == "unload":
        # The speaker holds the resident engine; it is the same object the
        # render loop calls, so unloading evicts the model everything uses.
        engine = getattr(speaker, "engine", None)
        unload = getattr(engine, "unload", None)
        if unload is None:
            return {
                "ok": False,
                "reason": "unload_unsupported",
                "message": "This worker holds no engine with an unload method.",
            }, False
        return {"ok": True, **unload()}, False

    if op == "stop":
        return {"ok": True, "status": "stopping", **speaker.stop()}, False

    if op == "shutdown":
        logger.info("[worker] shutdown requested")
        return {"ok": True, "status": "shutting_down"}, True

    return {
        "ok": False,
        "reason": "unknown_op",
        "message": f"Unknown op {op!r}; expected speak, status, unload, stop or shutdown.",
    }, False


__all__ = [
    "DEFAULT_PORT",
    "Server",
    "WorkerUnavailable",
    "encode",
    "send_request",
    "serve",
    "spawn",
    "wait_until_ready",
    "worker_script",
]
