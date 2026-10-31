"""
panel.py — a local control panel for the resident worker
========================================================
A small web page with the controls that have no home in a chat window: stop the
audio that is playing, play an earlier message again, speak something by hand.

Why a separate process
----------------------
The panel does not play audio and does not synthesize. Every button is one op
sent to the resident worker over the same loopback protocol the MCP server uses
(worker.send_request). That is not laziness, it is the only arrangement that
keeps `stop` honest: playback is `winsound`, whose sound belongs to the thread
that started it, so only the worker's own playback thread can cut it short (see
playback.py). A panel that played the WAVs itself would be uncancellable from
the worker, so a replay would keep talking after Stop.

Why HTTP and not a GUI toolkit
------------------------------
Nothing here needs a window library: the page is one HTML file, the server is
http.server from the standard library, and there is no dependency to add, no
packaging step, and no toolkit version to track. It also means the page opens in
whatever browser is at hand — including a browser pane embedded in an editor or
agent app later — because it is just a URL.

Where the buttons come from
---------------------------
The output directory is the history (see output.py): every message ever spoken
left its chunks on disk, so replaying one costs a file read. Replay jobs go
through the worker's queue, so a replay never overlaps speech that is still
rendering, and Stop cuts it off like any other utterance.

Starting and stopping the worker
--------------------------------
The panel can also start the resident worker and ask it to exit, because both
are needed to hand the GPU to something else (ComfyUI, a game) and get it back
without a terminal. Starting uses `worker.spawn`, which is detached, so the
worker still outlives the panel that launched it. Stopping sends the `shutdown`
op — not `stop`, which means "stop the audio that is playing". Neither action
loads a model: the worker is an idle listener until something speaks, so the
panel still owns no GPU state. Both sit behind the same action guard, which
matters more here than anywhere else: these two are process-level.

Binding and the guard
---------------------
The socket is bound to 127.0.0.1, so it is not reachable from another machine.
That is not the whole story: any web page the user visits can POST to a
localhost port. The actions therefore require a custom header, which a
cross-origin request cannot send without a CORS preflight this server never
approves, and an `Origin` header, when present, must be loopback. The page
itself is same-origin, so it sends both without effort.
"""

from __future__ import annotations

import json
import logging
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from mcp_agent_chatterbox.config import DEFAULT_WORKER_PORT
from mcp_agent_chatterbox.worker import (
    WorkerUnavailable,
    send_request,
    spawn,
    wait_until_ready,
)

logger = logging.getLogger(__name__)

#: Default loopback port for the panel. Separate from the worker's port: they
#: are two different protocols (HTTP here, newline JSON there) and must never be
#: confused by a hand-typed URL.
DEFAULT_PANEL_PORT = 8766

#: Marker header every action must carry. A cross-origin POST cannot set it
#: without a preflight, which this server does not answer.
_ACTION_HEADER = "X-Chatterbox"

#: How many messages the history shows.
_HISTORY_LIMIT = 50

#: Largest request body accepted. The speak box holds a full text budget of
#: 4000 characters, so this is generous and still bounded.
_MAX_BODY = 64 * 1024

#: Hostnames that count as "this machine" for the Origin check.
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}

#: How long a start request waits for the worker to answer. The worker imports
#: torch before it can answer `status`, so this is generous but bounded; the
#: request blocks on it rather than answering "started" before it is true.
_START_TIMEOUT_S = 30.0


def page_html() -> str:
    """The panel page, read from the package data file next to this module."""
    return (resources.files(__package__) / "panel.html").read_text(encoding="utf-8")


class PanelServer(ThreadingHTTPServer):
    """Threaded HTTP server that remembers which worker port to talk to."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], worker_port: int) -> None:
        super().__init__(address, PanelHandler)
        self.worker_port = worker_port


class PanelHandler(BaseHTTPRequestHandler):
    """
    Routes for the page. Voice actions are forwarded to the worker unchanged;
    the two worker-lifecycle routes act on the process itself.

    The handler holds no state of its own: the worker owns the queue, the model
    and the output directory, so a panel restart loses nothing but the page.
    """

    server_version = "chatterbox-panel"

    @property
    def worker_port(self) -> int:
        server = self.server
        assert isinstance(server, PanelServer)
        return server.worker_port

    # -- routing --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, "text/html; charset=utf-8", page_html().encode("utf-8"))
            return
        if path == "/api/state":
            self._state()
            return
        self._json(404, {"ok": False, "reason": "not_found", "message": f"No route for {path}."})

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        path = urlsplit(self.path).path
        if not self._guard():
            return
        body = self._body()
        if body is None:
            return
        if path == "/api/speak":
            self._forward({"op": "speak", "text": body.get("text"), "play_audio": True})
        elif path == "/api/replay":
            self._forward({"op": "replay", "message_id": body.get("message_id")})
        elif path == "/api/stop":
            self._forward({"op": "stop"})
        elif path == "/api/unload":
            self._forward({"op": "unload"})
        elif path == "/api/worker/start":
            self._start_worker()
        elif path == "/api/worker/stop":
            self._stop_worker()
        else:
            self._json(
                404, {"ok": False, "reason": "not_found", "message": f"No route for {path}."}
            )

    # -- worker calls ---------------------------------------------------------
    def _state(self) -> None:
        """One poll's worth of state: the queue, plus what can be replayed."""
        try:
            status = send_request({"op": "status"}, port=self.worker_port)
            history = send_request(
                {"op": "history", "limit": _HISTORY_LIMIT}, port=self.worker_port
            )
        except WorkerUnavailable as exc:
            self._json(
                503,
                {
                    "ok": False,
                    "reason": "worker_not_running",
                    "message": (
                        f"The resident worker is not reachable on port {self.worker_port} "
                        f"({exc}). Start it with `mcp-agent-chatterbox worker`."
                    ),
                },
            )
            return
        self._json(
            200,
            {
                "ok": True,
                "worker": status,
                "messages": history.get("messages", []),
                "output_dir": history.get("output_dir"),
            },
        )

    def _forward(self, request: dict[str, Any]) -> None:
        """Send one op to the worker and relay its answer verbatim."""
        try:
            response = send_request(request, port=self.worker_port)
        except WorkerUnavailable as exc:
            self._json(
                503,
                {
                    "ok": False,
                    "reason": "worker_not_running",
                    "message": (
                        f"The resident worker is not reachable on port {self.worker_port} "
                        f"({exc}). Start it with `mcp-agent-chatterbox worker`."
                    ),
                },
            )
            return
        status = 200 if response.get("ok") else 409
        self._json(status, response)

    # -- worker lifecycle -----------------------------------------------------
    def _start_worker(self) -> None:
        """
        Start a detached worker, unless one is already answering.

        Idempotent on purpose: the button is a toggle, so a double click — or a
        click on a label that is one poll behind — must not start a second
        worker that cannot bind the port and dies with a confusing error.
        """
        if self._worker_answers():
            self._json(200, {"ok": True, "status": "already_running"})
            return
        try:
            spawn(port=self.worker_port)
        except OSError as exc:
            self._json(
                500,
                {
                    "ok": False,
                    "reason": "spawn_failed",
                    "message": f"Could not start the worker: {exc}",
                },
            )
            return
        if wait_until_ready(self.worker_port, timeout=_START_TIMEOUT_S):
            self._json(200, {"ok": True, "status": "started"})
            return
        self._json(
            503,
            {
                "ok": False,
                "reason": "worker_start_timeout",
                "message": (
                    f"A worker was started but did not answer on port {self.worker_port} "
                    f"within {_START_TIMEOUT_S:.0f}s."
                ),
            },
        )

    def _stop_worker(self) -> None:
        """
        Ask the worker to drain its queue and exit.

        `shutdown`, not `stop`: the latter cuts off playback and leaves the
        process (and its port) alive, which is not what "Stop worker" means.
        The worker exits by itself, so there is no pid for the panel to kill.
        """
        self._forward({"op": "shutdown"})

    def _worker_answers(self) -> bool:
        """True when a worker is up and answering, not merely that a port is bound."""
        try:
            return bool(send_request({"op": "status"}, port=self.worker_port).get("ok"))
        except WorkerUnavailable:
            return False

    # -- plumbing -------------------------------------------------------------
    def _guard(self) -> bool:
        """Reject actions a random web page could otherwise trigger."""
        if self.headers.get(_ACTION_HEADER) != "1":
            self._json(
                403,
                {
                    "ok": False,
                    "reason": "missing_action_header",
                    "message": f"Actions require the {_ACTION_HEADER}: 1 header.",
                },
            )
            return False
        origin = self.headers.get("Origin")
        if origin is not None and urlsplit(origin).hostname not in _LOOPBACK:
            self._json(
                403,
                {
                    "ok": False,
                    "reason": "foreign_origin",
                    "message": f"Actions from {origin!r} are not allowed.",
                },
            )
            return False
        return True

    def _body(self) -> dict[str, Any] | None:
        """Parse the JSON request body, or answer 400 and return None."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > _MAX_BODY:
            self._json(
                413, {"ok": False, "reason": "body_too_large", "message": "Request body too large."}
            )
            return None
        raw = self.rfile.read(length)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(
                400,
                {
                    "ok": False,
                    "reason": "bad_request",
                    "message": "The request body must be one JSON object.",
                },
            )
            return None
        if not isinstance(decoded, dict):
            self._json(
                400,
                {
                    "ok": False,
                    "reason": "bad_request",
                    "message": "The request body must be one JSON object.",
                },
            )
            return None
        return decoded

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            # The tab was closed or reloaded mid-response; nothing to do and
            # nothing worth logging.
            pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server's name
        logger.info("[panel] %s %s", self.address_string(), format % args)


def serve(
    port: int = DEFAULT_PANEL_PORT,
    *,
    worker_port: int = DEFAULT_WORKER_PORT,
    open_browser: bool = True,
) -> int:
    """
    Run the panel until interrupted. Returns a process exit code.

    `open_browser` is on by default because the panel is started on demand and
    the point of starting it is to look at it. Pass False in tests or on a
    machine with no default browser.
    """
    server = PanelServer(("127.0.0.1", port), worker_port)
    bound = server.server_address[1]
    url = f"http://127.0.0.1:{bound}/"
    logger.info("[panel] listening on %s (worker port %d)", url, worker_port)
    print(f"chatterbox panel on {url} (worker port {worker_port}, Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — a missing browser is not a failure
            logger.warning("[panel] could not open a browser; visit %s yourself", url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[panel] interrupted")
    finally:
        server.server_close()
    return 0


__all__ = ["DEFAULT_PANEL_PORT", "PanelHandler", "PanelServer", "page_html", "serve"]
