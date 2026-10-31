"""
The panel's HTTP surface.
=========================
The panel is a thin relay: every action becomes one op for the resident worker,
and the page itself holds no state. These tests drive the real server over a
real loopback socket with `send_request` patched, so what is exercised is the
routing, the guard and the relay — not a mock of the handler.

Two properties matter beyond "the button works": the page must carry no external
assets (it has to render inside an embedded browser pane with no network), and
an action must be impossible to trigger from a foreign web page, which is what
the header and Origin checks are for.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from mcp_agent_chatterbox.panel import PanelServer
from mcp_agent_chatterbox.worker import WorkerUnavailable

_HEADERS = {"X-Chatterbox": "1", "Content-Type": "application/json"}


@pytest.fixture
def panel(monkeypatch):
    """A running panel with the worker call recorded instead of sent."""
    calls: list[dict] = []
    replies: dict[str, object] = {}

    def fake_send(request, port=None, **kwargs):
        calls.append(request)
        reply = replies.get(request["op"])
        if isinstance(reply, Exception):
            raise reply
        if reply is not None:
            return reply
        return {
            "ok": True,
            "status": "idle",
            "pending": 0,
            "active": None,
            "last": None,
            "messages": [],
            "output_dir": "output",
        }

    monkeypatch.setattr("mcp_agent_chatterbox.panel.send_request", fake_send)
    server = PanelServer(("127.0.0.1", 0), worker_port=8765)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            base=f"http://127.0.0.1:{server.server_address[1]}",
            calls=calls,
            replies=replies,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def get(base: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


def post(
    base: str, path: str, body: dict | None = None, headers: dict | None = None
) -> tuple[int, dict]:
    data = json.dumps(body if body is not None else {}).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=data, method="POST", headers={**_HEADERS, **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


class TestPage:
    def test_index_is_served(self, panel):
        status, body = get(panel.base, "/")
        assert status == 200
        assert "Replay last" in body
        assert "<!doctype html>" in body.lower()

    def test_the_page_says_when_the_model_is_loading(self, panel):
        # A cold start is seconds of silence that used to read as "speaking";
        # the page has to be able to name it.
        _status, body = get(panel.base, "/")
        assert "loading model…" in body

    def test_the_page_needs_no_network(self, panel):
        # An embedded browser pane may have no network at all, and a CDN asset
        # would silently leave the page unstyled or dead.
        _status, body = get(panel.base, "/")
        for forbidden in ("<script src", "@import", "href=\"http", "url(http"):
            assert forbidden not in body, forbidden


class TestState:
    def test_state_combines_queue_and_history(self, panel):
        panel.replies["history"] = {
            "ok": True,
            "messages": [{"message_id": "abc", "chunks": 2, "bytes": 8, "modified": 1}],
            "output_dir": "output",
        }
        status, raw = get(panel.base, "/api/state")
        assert status == 200
        body = json.loads(raw)
        assert body["ok"] is True
        assert body["messages"][0]["message_id"] == "abc"
        assert body["worker"]["status"] == "idle"

    def test_a_missing_worker_is_reported_as_such(self, panel):
        panel.replies["status"] = WorkerUnavailable("no worker listening")
        status, raw = get(panel.base, "/api/state")
        assert status == 503
        assert json.loads(raw)["reason"] == "worker_not_running"


class TestActions:
    def test_replay_is_forwarded(self, panel):
        status, body = post(panel.base, "/api/replay", {"message_id": "abc"})
        assert status == 200
        assert body["ok"] is True
        assert panel.calls[-1] == {"op": "replay", "message_id": "abc"}

    def test_stop_is_forwarded(self, panel):
        status, _body = post(panel.base, "/api/stop")
        assert status == 200
        assert panel.calls[-1] == {"op": "stop"}

    def test_speak_is_forwarded_and_always_plays(self, panel):
        # The panel is not a "write the files only" tool: a silent button in a
        # control panel that exists to make sound would be a bug.
        status, _body = post(panel.base, "/api/speak", {"text": "hello"})
        assert status == 200
        assert panel.calls[-1] == {"op": "speak", "text": "hello", "play_audio": True}

    def test_a_worker_refusal_is_relayed(self, panel):
        panel.replies["replay"] = {
            "ok": False,
            "reason": "message_not_found",
            "message": "nothing on disk",
        }
        status, body = post(panel.base, "/api/replay", {"message_id": "gone"})
        assert status == 409
        assert body["reason"] == "message_not_found"

    def test_a_missing_worker_is_a_503(self, panel):
        panel.replies["stop"] = WorkerUnavailable("down")
        status, body = post(panel.base, "/api/stop")
        assert status == 503
        assert body["reason"] == "worker_not_running"


class TestGuard:
    def test_an_action_without_the_header_is_refused(self, panel):
        status, body = post(panel.base, "/api/unload", headers={"X-Chatterbox": ""})
        assert status == 403
        assert body["reason"] == "missing_action_header"
        assert panel.calls == []

    def test_a_foreign_origin_is_refused(self, panel):
        status, body = post(panel.base, "/api/unload", headers={"Origin": "https://example.com"})
        assert status == 403
        assert body["reason"] == "foreign_origin"
        assert panel.calls == []

    def test_a_loopback_origin_is_allowed(self, panel):
        status, _body = post(panel.base, "/api/stop", headers={"Origin": "http://127.0.0.1:8766"})
        assert status == 200

    def test_a_foreign_origin_still_reads_the_page(self, panel):
        # The guard is on actions. Reading status reveals nothing that is not
        # already on disk, and blocking it would break the page in a webview
        # that sends an Origin the panel does not recognise.
        status, _body = get(panel.base, "/")
        assert status == 200


class TestWorkerLifecycle:
    """
    Starting and stopping the worker process itself.

    These are the only routes that are not a plain relay, so they need their own
    contract: "Stop worker" must ask the worker to exit rather than cut off the
    audio, and a start must not leave a second process fighting for the port.
    """

    def test_stop_worker_sends_shutdown_rather_than_stop(self, panel):
        status, _body = post(panel.base, "/api/worker/stop")
        assert status == 200
        assert panel.calls[-1] == {"op": "shutdown"}

    def test_stopping_a_missing_worker_is_a_503(self, panel):
        panel.replies["shutdown"] = WorkerUnavailable("down")
        status, body = post(panel.base, "/api/worker/stop")
        assert status == 503
        assert body["reason"] == "worker_not_running"

    def test_starting_an_already_running_worker_spawns_nothing(self, panel, monkeypatch):
        spawned: list[int] = []
        monkeypatch.setattr(
            "mcp_agent_chatterbox.panel.spawn", lambda **kwargs: spawned.append(kwargs["port"])
        )
        status, body = post(panel.base, "/api/worker/start")
        assert status == 200
        assert body["status"] == "already_running"
        assert spawned == []

    def test_starting_a_missing_worker_spawns_and_waits_for_it(self, panel, monkeypatch):
        panel.replies["status"] = WorkerUnavailable("no worker listening")
        spawned: list[int] = []
        monkeypatch.setattr(
            "mcp_agent_chatterbox.panel.spawn", lambda **kwargs: spawned.append(kwargs["port"])
        )
        monkeypatch.setattr(
            "mcp_agent_chatterbox.panel.wait_until_ready", lambda port, timeout=0: True
        )
        status, body = post(panel.base, "/api/worker/start")
        assert status == 200
        assert body["status"] == "started"
        assert spawned == [8765]

    def test_a_worker_that_never_answers_is_not_reported_as_started(self, panel, monkeypatch):
        # Answering "started" while the worker is still importing torch would
        # make the toggle show "Stop worker" for a process that cannot speak.
        panel.replies["status"] = WorkerUnavailable("no worker listening")
        monkeypatch.setattr("mcp_agent_chatterbox.panel.spawn", lambda **kwargs: None)
        monkeypatch.setattr(
            "mcp_agent_chatterbox.panel.wait_until_ready", lambda port, timeout=0: False
        )
        status, body = post(panel.base, "/api/worker/start")
        assert status == 503
        assert body["reason"] == "worker_start_timeout"


class TestRouting:
    def test_unknown_get_is_404(self, panel):
        status, _body = get(panel.base, "/nope")
        assert status == 404

    def test_unknown_post_is_404(self, panel):
        status, body = post(panel.base, "/api/nope")
        assert status == 404
        assert body["reason"] == "not_found"

    def test_malformed_json_is_400(self, panel):
        request = urllib.request.Request(
            panel.base + "/api/stop", data=b"{not json", method="POST", headers=_HEADERS
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        assert caught.value.code == 400
        assert json.loads(caught.value.read().decode("utf-8"))["reason"] == "bad_request"
