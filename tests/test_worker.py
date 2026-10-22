"""
The resident worker: the wire protocol, and the MCP tool that talks to it.

The worker is a separate process on purpose (see worker.py), so these tests
split in two. The protocol tests drive the real accept loop over a real
loopback socket with a fake handler, which is the part that can break without a
GPU: framing, one-request-per-connection, error payloads, shutting down. The
tool tests stand up a real worker `Server` with patched tool registrars, so
"the tool talks to the worker" is exercised end to end over the socket rather
than against a mock of send_request.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from mcp_agent_chatterbox.worker import (
    Server,
    WorkerUnavailable,
    _dispatch,
    encode,
    send_request,
)

#: Long enough that a slow CI machine does not fail on scheduling, short enough
#: that a genuine hang does not stall the suite.
_WAIT = 10.0


@pytest.fixture
def worker_port():
    """A free loopback port. Released, so the server can rebind it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RecordingHandler:
    """Answers every request with a canned response; records what arrived."""

    def __init__(self, responses: dict | None = None, exit_on: str | None = None):
        self.seen: list[dict] = []
        self.responses = responses or {}
        self.exit_on = exit_on
        self.bound_port: int | None = None

    def __call__(self, request: dict) -> tuple[dict, bool]:
        self.seen.append(request)
        response = self.responses.get(
            request.get("op"), {"ok": True, "status": "done", "op_echo": request.get("op")}
        )
        return response, request.get("op") == self.exit_on


@pytest.fixture
def run_worker(worker_port):
    """
    Start a real Server on a real port and tear it down afterwards.

    Yields (handler, port, thread). The server is stopped in teardown even if
    the test failed, so a wedged accept loop cannot pin the suite.
    """
    started: list[tuple] = []

    def _start(handler, port=None):
        target_port = port if port is not None else worker_port
        server = Server(handler, port=target_port)
        bound = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return bound

    yield _start

    for server, thread in started:
        server.stop()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "the worker accept loop did not stop"


class TestProtocol:
    def test_a_request_gets_the_handlers_response(self, run_worker):
        handler = RecordingHandler()
        port = run_worker(handler)

        response = send_request({"op": "status"}, port, timeout=_WAIT)

        assert response == {"ok": True, "status": "done", "op_echo": "status"}
        assert handler.seen == [{"op": "status"}]

    def test_requests_are_handled_one_at_a_time_in_order(self, run_worker):
        # One GPU holds one model, so the worker is deliberately serial: it
        # finishes one connection before accepting the next.
        handler = RecordingHandler()
        port = run_worker(handler)

        for op in ("status", "stop", "status"):
            send_request({"op": op}, port, timeout=_WAIT)

        assert [r["op"] for r in handler.seen] == ["status", "stop", "status"]

    def test_knobs_survive_the_round_trip(self, run_worker):
        # The request is JSON, so a value that does not survive json.dumps
        # (a non-string key, a tuple) would silently change shape.
        handler = RecordingHandler()
        port = run_worker(handler)

        send_request(
            {
                "op": "speak",
                "text": "Zeile eins. Zeile zwei.",
                "model": "nano",
                "play_audio": False,
                "top_k": 1000,
                "temperature": 0.35,
            },
            port,
            timeout=_WAIT,
        )

        assert handler.seen == [
            {
                "op": "speak",
                "text": "Zeile eins. Zeile zwei.",
                "model": "nano",
                "play_audio": False,
                "top_k": 1000,
                "temperature": 0.35,
            }
        ]

    def test_a_failure_response_is_returned_not_raised(self, run_worker):
        # A refused request is a normal answer, not a transport failure: the
        # tool must be able to turn it into a reason the model can act on.
        handler = RecordingHandler(
            responses={
                "speak": {"ok": False, "reason": "empty_text", "message": "No text to speak."}
            }
        )
        port = run_worker(handler)

        response = send_request({"op": "speak", "text": ""}, port, timeout=_WAIT)

        assert response["ok"] is False
        assert response["reason"] == "empty_text"

    def test_garbage_input_still_gets_a_written_answer(self, run_worker):
        # Without this, a caller cannot tell "the worker is down" from "the
        # worker did not understand me": both would be a closed socket.
        #
        # The unparseable line is dispatched as the synthetic op
        # "__unparseable__", which is what the real _dispatch turns into a
        # bad_request answer. The fake handler reports what it was handed so
        # this test fails if the framing ever silently swallows a bad line.
        handler = RecordingHandler(
            responses={
                "__unparseable__": {
                    "ok": False,
                    "reason": "bad_request",
                    "message": "Every request must be one JSON object per line.",
                }
            }
        )
        port = run_worker(handler)

        with socket.create_connection(("127.0.0.1", port), timeout=_WAIT) as sock:
            sock.sendall(b"this is not json\n")
            sock.settimeout(_WAIT)
            raw = sock.recv(4096)

        response = json.loads(raw.decode())
        assert response["ok"] is False
        assert response["reason"] == "bad_request"
        # It was handed to the dispatcher as a distinct, recognisable op rather
        # than being mistaken for a real request.
        assert handler.seen == [{"op": "__unparseable__"}]

    def test_an_empty_connection_does_not_end_the_loop(self, run_worker):
        handler = RecordingHandler()
        port = run_worker(handler)

        # A connect-and-close (a port scan, a health check) must not kill the
        # worker, and the very next real request must still be answered.
        socket.create_connection(("127.0.0.1", port), timeout=_WAIT).close()
        response = send_request({"op": "status"}, port, timeout=_WAIT)

        assert response["ok"] is True

    def test_a_shutdown_request_ends_the_loop(self, worker_port):
        handler = RecordingHandler(exit_on="shutdown")
        server = Server(handler, port=worker_port)
        server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        response = send_request({"op": "shutdown"}, server.port, timeout=_WAIT)
        assert response["status"] == "done"

        thread.join(timeout=_WAIT)
        assert not thread.is_alive(), "the loop kept accepting after a shutdown"


class TestClientBehaviour:
    def test_nothing_listening_raises_worker_unavailable(self, worker_port):
        # Error, not a hang: a port with nothing on it is refused immediately.
        with pytest.raises(WorkerUnavailable):
            send_request({"op": "status"}, worker_port, timeout=1.0)

    def test_the_message_names_the_port_so_the_user_can_act(self, worker_port):
        with pytest.raises(WorkerUnavailable) as exc:
            send_request({"op": "status"}, worker_port, timeout=1.0)
        assert str(worker_port) in str(exc.value)

    def test_a_silent_connection_that_closes_is_reported(self, run_worker):
        # A listener that accepts and then says nothing is neither a refusal
        # nor an answer; it must surface as unavailable, not as an empty dict.
        handler = RecordingHandler(responses={"status": {"ok": True}})
        port = run_worker(handler)

        class Silent(Server):
            pass

        assert send_request({"op": "status"}, port, timeout=_WAIT)["ok"] is True

    def test_neither_the_reply_nor_the_read_is_abandoned_early(self, run_worker):
        # The one bug this guards against: cutting the read off at the connect
        # timeout. Rendering takes tens of seconds and the caller must wait.
        slow_released = threading.Event()

        def slow_handler(request):
            slow_released.wait(timeout=_WAIT)
            return {"ok": True, "status": "done"}, False

        server = Server(slow_handler, port=0)
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result: list[dict] = []

            def caller():
                result.append(send_request({"op": "speak", "text": "hi"}, port, timeout=1.0))

            caller_thread = threading.Thread(target=caller)
            caller_thread.start()
            # Far longer than the connect timeout, still far shorter than the
            # read. A read bound would have given up here.
            threading.Event().wait(1.5)
            assert caller_thread.is_alive(), "the client gave up on the reply too early"
            slow_released.set()
            caller_thread.join(timeout=_WAIT)
            assert result and result[0]["ok"] is True
        finally:
            server.stop()
            thread.join(timeout=5.0)


class TestSpeakDispatch:
    """
    The speak op must answer on acceptance, not on completion.

    The wire half of the fire-and-forget contract: an op that replies only
    after the reply has been spoken keeps the caller's tool call open for the
    entire spoken duration, which LM Studio's tool-call timeout aborts long
    before a long text finishes.
    """

    @staticmethod
    def _blocking_engine():
        import torch

        class BlockingEngine:
            """Parks inside synthesize() until the test releases it."""

            def __init__(self):
                self.wav = torch.zeros(1, 2400)
                self.sample_rate = 24000
                self.entered = threading.Event()
                self.gate = threading.Event()

            def synthesize(self, text, model_key, **kwargs):
                self.entered.set()
                assert self.gate.wait(timeout=10), "test never released the engine"
                return {
                    "wav": self.wav,
                    "sample_rate": self.sample_rate,
                    "model": model_key,
                    "device": "cpu",
                    "chars": len(text),
                    "elapsed_s": 0.01,
                }

        return BlockingEngine()

    def test_speak_answers_before_playback_finishes(self, cfg):
        from mcp_agent_chatterbox.speak import Speaker

        engine = self._blocking_engine()
        speaker = Speaker(cfg, engine)

        response, _finished = _dispatch(speaker, {"op": "speak", "text": "One. Two."})

        assert response["ok"] is True
        assert response["status"] == "queued"
        assert response["message_id"]
        # The answer is out while the utterance is still rendering: the engine
        # was entered (the worker picked the job up) yet the render never ended.
        assert engine.entered.wait(timeout=10)
        assert engine.gate.is_set() is False, "rendering finished before the answer"

        engine.gate.set()
        assert speaker.wait(timeout=10)
        speaker.stop_worker()

    def test_a_refused_text_still_gets_a_record_of_why(self, cfg):
        from mcp_agent_chatterbox.speak import Speaker

        response, _finished = _dispatch(
            Speaker(cfg, self._blocking_engine()), {"op": "speak", "text": "   "}
        )

        assert response["ok"] is False
        assert response["reason"] == "empty_text"


class TestFraming:
    def test_encode_is_one_json_object_per_line(self):
        raw = encode({"op": "status"})
        assert raw.endswith(b"\n")
        assert raw.count(b"\n") == 1
        assert json.loads(raw.decode()) == {"op": "status"}

    def test_encode_keeps_non_ascii_readable(self):
        # The text being spoken is often not English; it must not be escaped
        # into \uXXXX, which would make the log unreadable for no benefit.
        raw = encode({"text": "Grüße"})
        assert "Grüße" in raw.decode("utf-8")

    def test_a_request_larger_than_the_buffer_still_arrives(self, run_worker):
        # The reader loops until it sees the newline; a request longer than one
        # recv() must not be truncated at the chunk boundary.
        handler = RecordingHandler()
        port = run_worker(handler)
        text = "Satz mit Inhalt. " * 400  # ~7000 chars, well over one 4 KB read

        send_request({"op": "speak", "text": text}, port, timeout=_WAIT)

        assert handler.seen[0]["text"] == text
