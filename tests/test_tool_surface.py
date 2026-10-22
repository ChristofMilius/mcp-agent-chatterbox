"""
The MCP tool surface over the worker socket.

The tool surface and the worker are two halves of one contract, so these tests
run a real worker process loop (the same `Server` the `worker` subcommand
uses) with patched registrars, then call the registered handlers through the
real `register()` closures. A mock of `send_request` would have made "the tool
sends what the worker expects" untestable, which is the half that broke when the
tool stopped synthesizing in-process.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from mcp_agent_chatterbox.context import AppContext
from mcp_agent_chatterbox.tool_surface import speak_tools, status_tools, voice_tools
from mcp_agent_chatterbox.worker import Server

_WAIT = 10.0


class FakeServer:
    """Collects the tools a registrar decorates, so they can be called direct."""

    def __init__(self):
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class ScriptedWorker:
    """
    A worker that answers speak/status/unload/stop from a script.

    Records every request it received, so a test can assert on what the tool
    actually put on the wire rather than on what it returned.
    """

    def __init__(self, speak=None):
        self.requests: list[dict] = []
        self.speak_response = speak or {
            "ok": True,
            "status": "queued",
            "message_id": "1700000000000",
            "text_chars": 5,
            "chunks": 2,
            "files": ["1700000000000_1.wav", "1700000000000_2.wav"],
            "play": True,
            "model": "nano",
            "voice": "narrator",
            "queued_ahead": 0,
            "output_dir": "/tmp/out",
        }

    def __call__(self, request: dict) -> tuple[dict, bool]:
        self.requests.append(request)
        op = request.get("op")
        if op == "speak":
            return self.speak_response, False
        if op == "status":
            return {"ok": True, "status": "idle", "pending": 0, "active": None, "last": None}, False
        if op == "unload":
            return {"ok": True, "was_loaded": True, "model": "nano"}, False
        if op == "stop":
            return {
                "ok": True,
                "stopped": True,
                "reason": None,
                "cancelled_jobs": 0,
                "cancelled_ids": [],
                "cancelled_active": False,
            }, False
        return {"ok": False, "reason": "unknown_op", "message": op}, False


@pytest.fixture(autouse=True)
def _isolated_port(cfg):
    """
    Point every test at its own free loopback port.

    The default port is where a developer's *resident* worker lives, and the
    whole point of this design is that one stays running -- so a test that bound
    or probed the default would talk to that worker instead of its own fake, and
    "no worker running" would flake depending on the developer's machine.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg.worker_port = sock.getsockname()[1]


@pytest.fixture
def worker(cfg):
    """Run a real accept loop on cfg's port, with a scripted handler."""
    scripted = ScriptedWorker()
    server = Server(scripted, port=cfg.worker_port)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield scripted
    finally:
        server.stop()
        thread.join(timeout=5.0)
        assert not thread.is_alive()


@pytest.fixture
def tools(cfg):
    """Register every domain on a FakeServer and return the handlers by name."""
    ctx = AppContext(cfg=cfg)
    server = FakeServer()
    status_tools.register(server, ctx)
    speak_tools.register(server, ctx)
    voice_tools.register(server, ctx)
    return server.tools


def _body(raw: str) -> dict:
    return json.loads(raw)


class TestSpeakForwardsToTheWorker:
    def test_the_reply_is_reported_as_accepted_not_spoken(self, tools, worker):
        # Fire and forget: the tool result must NOT promise the record of a
        # played utterance, because it answers before playback finishes. A
        # result shaped like "spoken" would teach the model to rely on fields
        # (played_chunks, errors) that cannot exist yet.
        body = _body(tools["speak"](text="Line one. Line two."))

        assert body["status"] == "accepted"
        assert body["status"] != "spoken"
        assert body["message_id"] == "1700000000000"
        assert body["files"] == ["1700000000000_1.wav", "1700000000000_2.wav"]
        assert body["play"] is True
        assert "played_chunks" not in body
        assert "errors" not in body

    def test_the_text_and_set_knobs_reach_the_worker(self, tools, worker):
        tools["speak"](text="Grüße.", voice="narrator", play_audio=False, top_k=1000)

        request = worker.requests[-1]
        assert request["op"] == "speak"
        assert request["text"] == "Grüße."
        assert request["voice"] == "narrator"
        assert request["play_audio"] is False
        assert request["top_k"] == 1000

    def test_unset_knobs_are_omitted_not_nulled(self, tools, worker):
        # An omitted knob must keep the model's own tuned default; sending an
        # explicit null would make "unset" and "set to nothing" the same thing.
        tools["speak"](text="Hello.")

        request = worker.requests[-1]
        for name in ("model", "voice", "temperature", "seed", "filename", "exaggeration"):
            assert name not in request, f"{name} should not have been sent"

    def test_no_worker_is_a_reason_the_model_can_act_on(self, tools, cfg):
        # Nothing is listening: this is the state after a reboot, before anyone
        # started the worker, and it must be a clear instruction, not a crash.
        body = _body(tools["speak"](text="Hello."))

        assert body["status"] == "failed"
        assert body["reason"] == "worker_not_running"
        assert "worker" in body["message"]
        assert str(cfg.worker_port) in body["message"]

    def test_a_worker_side_fault_is_passed_through_unchanged(self, tools, cfg):
        scripted = ScriptedWorker(
            speak={
                "ok": False,
                "reason": "reference_clip_required",
                "message": "nano needs a reference clip.",
                "model": "original",
                "requires": ["voice", "reference_clip"],
            }
        )
        server = Server(scripted, port=cfg.worker_port)
        server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = _body(tools["speak"](text="Hello.", model="original"))
        finally:
            server.stop()
            thread.join(timeout=5.0)

        assert body["status"] == "failed"
        assert body["reason"] == "reference_clip_required"
        assert body["requires"] == ["voice", "reference_clip"]

    def test_the_error_string_never_carries_a_path(self, tools, cfg):
        # A handler that raises must not kill the worker, and must not leak a
        # traceback to the model: tool_error() is the boundary, and the worker
        # reports the failure as a reason instead of dying.
        def explode(_request):
            raise ValueError("boom")

        server = Server(explode, port=cfg.worker_port)
        server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            raw = tools["speak"](text="Hello.")
        finally:
            server.stop()
            thread.join(timeout=5.0)

        body = _body(raw)
        assert body["status"] == "failed"
        assert body["reason"] == "worker_error"
        assert "Traceback" not in raw
        assert "boom" not in raw

    def test_a_handler_failure_does_not_take_the_worker_down(self, cfg):
        # A resident worker that one bad request can kill fails again on the
        # next utterance, silently. The loop must survive and keep serving.
        calls: list[dict] = []

        def flaky(request):
            calls.append(request)
            if len(calls) == 1:
                raise RuntimeError("first one blows up")
            return {"ok": True, "status": "idle"}, False

        server = Server(flaky, port=cfg.worker_port)
        server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from mcp_agent_chatterbox.worker import send_request

            first = send_request({"op": "status"}, cfg.worker_port, timeout=_WAIT)
            second = send_request({"op": "status"}, cfg.worker_port, timeout=_WAIT)
        finally:
            server.stop()
            thread.join(timeout=5.0)

        assert first["ok"] is False and first["reason"] == "worker_error"
        assert second["ok"] is True, "the worker stopped serving after one bad handler"
        assert len(calls) == 2


class TestStatusAndControl:
    def test_status_reports_the_worker_when_it_is_up(self, tools, worker):
        body = _body(tools["tts_status"]())

        assert body["worker_running"] is True
        assert body["worker"]["status"] == "idle"
        assert body["default_model"]
        assert body["turbo_paralinguistic_tags"]

    def test_status_still_answers_without_a_worker(self, tools, cfg):
        # The point of this tool is to be a preflight, including when the thing
        # that is missing is the worker itself.
        body = _body(tools["tts_status"]())

        assert body["status"] == "ok"
        assert body["worker_running"] is False
        assert "worker_not_running" in body["worker_note"]
        assert body["models"]

    def test_unload_asks_the_worker(self, tools, worker):
        body = _body(tools["tts_unload"]())

        assert body["status"] == "ok"
        assert body["was_loaded"] is True

    def test_unload_without_a_worker_explains_the_state(self, tools, cfg):
        body = _body(tools["tts_unload"]())

        assert body["status"] == "failed"
        assert body["reason"] == "worker_not_running"

    def test_stop_speech_asks_the_worker(self, tools, worker):
        body = _body(tools["stop_speech"]())

        assert body["status"] == "ok"
        assert worker.requests[-1]["op"] == "stop"

    def test_stop_speech_without_a_worker_is_not_an_error(self, tools, cfg):
        body = _body(tools["stop_speech"]())

        assert body["status"] == "failed"
        assert body["reason"] == "worker_not_running"


class TestWorkerConfig:
    def test_the_tool_and_the_worker_read_the_same_port(self, cfg, tools, worker):
        # If these ever diverge, speak() reports worker_not_running while a
        # perfectly healthy worker sits on another port.
        body = _body(tools["tts_status"]())
        assert body["worker_port"] == cfg.worker_port
        assert worker.requests, "the tool never reached the worker on its configured port"
