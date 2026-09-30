"""HTTP session, admission, event-stream, and shutdown contracts for the app."""

import argparse
import asyncio
import contextlib
import importlib.resources
import json
import sqlite3
import threading
from pathlib import Path

import httpx
import pytest
import uvicorn

import agent_qa.app
from agent_qa.__main__ import main
from agent_qa.app import create_app
from agent_qa.model import ModelRef
from agent_qa.providers.config import ProvidersConfig
from agent_qa.storage import Storage

CONFIG = ProvidersConfig.model_validate(
    {
        "providers": {
            "local": {
                "base_url": "http://unit.test/v1",
                "models": {"qwen": {"context_window": 8}},
            }
        }
    }
)
TARGET = ModelRef(provider="local", model="qwen")
PROMPT = "Answer briefly."
COOKIE = "agent_qa_session"
REJECTION = {"error": "request rejected"}


def choice(delta, finish=None):
    event_choice = {"index": 0, "delta": delta}
    if finish is not None:
        event_choice["finish_reason"] = finish
    return {"choices": [event_choice]}


def usage_event():
    return {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}


def sse(*events):
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return body + b"data: [DONE]\n\n"


def sse_open(*events):
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)


def text_stream(fragments):
    events = [choice({"role": "assistant", "content": piece}) for piece in fragments]
    events.append(choice({}, finish="stop"))
    events.append(usage_event())
    return sse(*events)


def stream_step(*pieces, gate=None):
    def step():
        async def source():
            head, tail = pieces[:-1], pieces[-1:]
            for piece in head:
                yield piece
            if gate is not None:
                await gate.wait()
            for piece in tail:
                yield piece

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=source()
        )

    return step


class Recorder:
    def __init__(self):
        self.requests = []


def make_client(recorder, *steps):
    remaining = list(steps)

    def handler(item):
        recorder.requests.append(item)
        return remaining.pop(0)()

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def build_app(path, http):
    return create_app(CONFIG, TARGET, path, PROMPT, http=http)


def asgi_client(app, raise_app_exceptions=True):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions),
        base_url="http://testserver",
    )


def session_id_from(response):
    cookie = response.headers.get("set-cookie", "")
    assert cookie.startswith(f"{COOKIE}="), cookie
    return cookie.split(";")[0].split("=", 1)[1]


def cookie_headers(session_id):
    return {"Cookie": f"{COOKIE}={session_id}"}


def dump(path):
    connection = sqlite3.connect(path)
    try:
        sessions = connection.execute(
            "SELECT id, current_turn_id, usage_total FROM sessions ORDER BY rowid"
        ).fetchall()
        entries = connection.execute(
            "SELECT session_id, turn_id, request_id, kind, payload FROM entries ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    return sessions, entries


async def next_event(queue):
    return await asyncio.wait_for(queue.get(), 5)


async def drain_to_end(queue):
    while True:
        name, data = await next_event(queue)
        if name == "turn_end":
            return data


def sse_scope(session_id):
    path = "/api/events"
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"cookie", f"{COOKIE}={session_id}".encode())],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 123),
        "root_path": "",
    }


def frames_of(raw):
    """Parse complete ``event:/data:`` frames from accumulated stream bytes."""
    data = bytes(raw)
    cut = data.rfind(b"\n\n")
    if cut == -1:
        return []
    if not data.endswith(b"\n\n"):
        data = data[: cut + 2]
    parsed = []
    for frame in data.split(b"\n\n"):
        if not frame:
            continue
        lines = frame.split(b"\n")
        assert lines[0].startswith(b"event: "), frame
        assert lines[1].startswith(b"data: "), frame
        parsed.append(
            (lines[0][7:].decode(), json.loads(lines[1][6:].decode("utf-8")))
        )
    return parsed


def capture_streaming(monkeypatch):
    """Retain every frames() generator the endpoint passes to its response."""
    captured = []
    original = agent_qa.app.StreamingResponse

    def capture(content, *args, **kwargs):
        captured.append(content)
        return original(content, *args, **kwargs)

    monkeypatch.setattr(agent_qa.app, "StreamingResponse", capture)
    return captured


class OwnedHttpx:
    """Replaces the app's httpx view to record and mock the owned client."""

    def __init__(self, handler):
        self.handler = handler
        self.created = []

    Timeout = staticmethod(httpx.Timeout)

    def AsyncClient(self, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler), **kwargs)
        self.created.append(client)
        return client


class Harness:
    """One real uvicorn server for the app, stopped with graceful timeout zero."""

    def __init__(self, app):
        self.server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=0, lifespan="on",
                timeout_graceful_shutdown=0,
            )
        )
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.server.serve())

        async def poll():
            while not self.server.started:
                if self.task.done():
                    self.task.result()
                await asyncio.sleep(0)

        await asyncio.wait_for(poll(), 5)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def stop(self):
        self.server.should_exit = True
        await asyncio.wait_for(self.task, 5)


def endless_body(pulled):
    async def source():
        pulled.set()
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    return source()


# --- P5: HTTP admission and session ----------------------------------------------


def test_get_session_resolves_creates_and_secures_cookies(tmp_path):
    async def run():
        app = build_app(tmp_path / "db.sqlite3", make_client(Recorder()))
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                empty = await client.get("/api/session")
                assert empty.status_code == 200
                assert empty.json() == {"messages": [], "turn": None, "outcome": None}
                cookie = empty.headers.get("set-cookie", "")
                assert "HttpOnly" in cookie and "Path=/" in cookie
                assert "SameSite=lax" in cookie
                assert "Secure" not in cookie
                assert "Expires" not in cookie and "Max-Age" not in cookie
                session_id = session_id_from(empty)

                known = await client.get("/api/session")
                assert known.status_code == 200
                assert "set-cookie" not in known.headers
                assert (await app.state.storage.snapshot(session_id)) is not None

                stale = await client.get("/api/session", headers=cookie_headers("gone"))
                assert stale.status_code == 200
                assert session_id_from(stale) != session_id

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                ) as secure_client:
                    secure = await secure_client.get("/api/session")
                    assert "Secure" in secure.headers.get("set-cookie", "")

    asyncio.run(run())


def test_two_cookie_jars_never_share_sessions(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder(), stream_step(text_stream(["A"]))))
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as first, asgi_client(app) as second:
                sid_a = session_id_from(await first.get("/api/session"))
                sid_b = session_id_from(await second.get("/api/session"))
                assert sid_a != sid_b

                watch = app.state.runtime.subscribe(sid_a)
                posted = await first.post(
                    "/api/message", content=json.dumps({"request_id": "r1", "text": "q"})
                )
                assert posted.status_code == 200
                await drain_to_end(watch)

                snapshot_a = await first.get("/api/session")
                assert snapshot_a.json()["messages"] == [
                    {"role": "user", "text": "q"},
                    {"role": "assistant", "text": "A"},
                ]
                snapshot_b = await second.get("/api/session")
                assert snapshot_b.json() == {"messages": [], "turn": None, "outcome": None}
                assert await app.state.storage.snapshot(sid_b) is not None
            sessions, entries = dump(path)
            assert len(sessions) == 2
            assert {entry[0] for entry in entries} == {sid_a}

    asyncio.run(run())


def test_post_and_events_never_create_sessions(tmp_path):
    async def run():
        app = build_app(tmp_path / "db.sqlite3", make_client(Recorder()))
        async with app.router.lifespan_context(app):
            sessions, _ = dump(tmp_path / "db.sqlite3")
            assert sessions == []
            async with asgi_client(app) as client:
                assert (await client.post("/api/message", content=b"{}")).status_code == 404
                assert (
                    await client.post(
                        "/api/message", headers=cookie_headers("gone"),
                        content=json.dumps({"request_id": "r9", "text": "q"}).encode(),
                    )
                ).status_code == 404

                events = await client.get("/api/events", headers=cookie_headers("gone"))
                assert events.status_code == 404
                assert events.json() == REJECTION
                assert app.state.runtime._subscribers == {}
            sessions, _ = dump(tmp_path / "db.sqlite3")
            assert sessions == []

    asyncio.run(run())


def test_missing_cookie_post_rejects_before_reading_body(tmp_path):
    async def run():
        app = build_app(tmp_path / "db.sqlite3", make_client(Recorder()))
        async with app.router.lifespan_context(app):
            pulled = asyncio.Event()
            async with asgi_client(app) as client:
                async with client.stream(
                    "POST", "/api/message", content=endless_body(pulled)
                ) as stream:
                    assert stream.status_code == 404
                    assert json.loads(await stream.aread()) == REJECTION
            assert not pulled.is_set(), "the body was read before the cookie check"

    asyncio.run(run())


def test_text_and_request_id_length_boundaries(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(
            path,
            make_client(
                Recorder(),
                stream_step(text_stream(["ok"])),
                stream_step(text_stream(["ok"])),
            ),
        )
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                astral = "\U0001F600" * 50_000
                watch = app.state.runtime.subscribe(sid)
                accepted = await client.post(
                    "/api/message",
                    headers=cookie_headers(sid),
                    content=json.dumps({"request_id": "r", "text": astral}).encode(),
                )
                assert accepted.status_code == 200
                await drain_to_end(watch)
                assert len(json.dumps(astral)) == 12 * 50_000 + 2
                before = dump(path)
                too_long = await client.post(
                    "/api/message",
                    headers=cookie_headers(sid),
                    content=json.dumps({"request_id": "r", "text": astral + "😀"}).encode(),
                )
                assert too_long.status_code == 400
                assert too_long.json() == REJECTION
                assert dump(path) == before

                long_id = await client.post(
                    "/api/message",
                    headers=cookie_headers(sid),
                    content=json.dumps({"request_id": "x" * 128, "text": "q"}).encode(),
                )
                assert long_id.status_code == 200
                await drain_to_end(watch)
                before = dump(path)
                over_id = await client.post(
                    "/api/message",
                    headers=cookie_headers(sid),
                    content=json.dumps({"request_id": "x" * 129, "text": "q"}).encode(),
                )
                assert over_id.status_code == 400
                assert over_id.json() == REJECTION
                assert dump(path) == before
                _, entries = dump(path)
                accepted_rows = [row for row in entries if row[2] in ("r", "x" * 128)]
                assert len(accepted_rows) == 2
                assert all(len(row[2]) in (1, 128) for row in accepted_rows)

    asyncio.run(run())


def test_body_size_cap_boundary_and_transport_shapes(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(
            path,
            make_client(
                Recorder(),
                stream_step(text_stream(["ok"])),
                stream_step(text_stream(["ok"])),
            ),
        )
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                base = json.dumps({"request_id": "declared-cap", "text": "a" * 100}).encode()
                at_cap = b" " * (1_048_576 - len(base)) + base
                assert len(at_cap) == 1_048_576
                chunked_base = json.dumps({"request_id": "chunked-cap", "text": "a" * 100}).encode()
                chunked_at_cap = b" " * (1_048_576 - len(chunked_base)) + chunked_base
                assert len(chunked_at_cap) == 1_048_576
                over_base = json.dumps({"request_id": "over-cap", "text": "a" * 100}).encode()
                over = b" " * (1_048_577 - len(over_base)) + over_base
                assert len(over) == 1_048_577

                async def chunks(body, size):
                    for start in range(0, len(body), size):
                        yield body[start : start + size]

                headers = cookie_headers(sid)
                watch = app.state.runtime.subscribe(sid)
                declared_request = client.build_request(
                    "POST", "/api/message", headers=headers, content=at_cap
                )
                assert declared_request.headers["content-length"] == "1048576"
                declared = await client.send(declared_request)
                assert declared.status_code == 200
                assert (await drain_to_end(watch))["status"] == "success"
                chunked_request = client.build_request(
                    "POST", "/api/message", headers=headers,
                    content=chunks(chunked_at_cap, 65_536),
                )
                assert "content-length" not in chunked_request.headers
                chunked = await client.send(chunked_request)
                assert chunked.status_code == 200
                assert chunked.json()["turn_id"] != declared.json()["turn_id"]
                assert (await drain_to_end(watch))["status"] == "success"
                cap_id = chunked.json()["turn_id"]
                before_duplicate = dump(path)
                duplicate_cap = await client.post("/api/message", headers=headers, content=chunked_at_cap)
                assert duplicate_cap.status_code == 200
                assert duplicate_cap.json() == {"turn_id": cap_id}
                assert dump(path) == before_duplicate, "a duplicate must not write"
                assert watch.empty(), "a duplicate must not publish events"

                before = dump(path)
                rejected = await client.post("/api/message", headers=headers, content=over)
                assert rejected.status_code == 400
                assert rejected.json() == REJECTION
                single = await client.post("/api/message", headers=headers, content=over)
                assert single.status_code == 400
                chunked_over = await client.post(
                    "/api/message", headers=headers, content=chunks(over, 65_536)
                )
                assert chunked_over.status_code == 400
                assert chunked_over.json() == REJECTION
                assert dump(path) == before, "an over-cap body must never be admitted"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("body",),
    [
        (b"{",),
        (b"",),
        (b"\xff\xfe{}",),
        (b"[]",),
        (b"5",),
        (b"null",),
        (json.dumps({"text": "q"}).encode(),),
        (json.dumps({"request_id": "r"}).encode(),),
        (json.dumps({"request_id": "", "text": "q"}).encode(),),
        (json.dumps({"request_id": "r", "text": ""}).encode(),),
        (json.dumps({"request_id": 7, "text": "q"}).encode(),),
        (json.dumps({"request_id": "r", "text": 7}).encode(),),
        (json.dumps({"request_id": "r", "text": None}).encode(),),
        (json.dumps({"request_id": "r", "text": "SENTINEL-rejected-input", "extra": 1}).encode(),),
    ],
)
def test_rejected_bodies_never_admit_or_echo(tmp_path, body):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                before = dump(path)
                response = await client.post(
                    "/api/message", headers=cookie_headers(sid), content=body
                )
                assert response.status_code == 400
                assert response.json() == REJECTION
                assert "SENTINEL" not in response.text
                assert dump(path) == before, "rejected input must not write"

    asyncio.run(run())


def test_admission_status_and_body_per_state(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        recorder = Recorder()
        app = build_app(
            path,
            make_client(recorder, stream_step(first, rest, gate=gate)),
        )
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                headers = cookie_headers(sid)
                watch = app.state.runtime.subscribe(sid)

                fresh = await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r1", "text": "q1"}).encode(),
                )
                assert fresh.status_code == 200
                turn_id = fresh.json()["turn_id"]

                duplicate = await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r1", "text": "replacement"}).encode(),
                )
                assert duplicate.status_code == 200
                assert duplicate.json() == {"turn_id": turn_id}

                busy = await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r2", "text": "q2"}).encode(),
                )
                assert busy.status_code == 409
                assert busy.json() == REJECTION

                unknown = await client.post(
                    "/api/message", headers=cookie_headers("gone"),
                    content=json.dumps({"request_id": "r9", "text": "q"}).encode(),
                )
                assert unknown.status_code == 404
                assert unknown.json() == REJECTION

                running = await client.get("/api/session", headers=headers)
                assert running.json()["turn"] == {"id": turn_id}
                assert running.json()["outcome"] is None
                assert running.json()["messages"] == [{"role": "user", "text": "q1"}]

                runtime = app.state.runtime
                runtime._stopping = True
                closed = await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r3", "text": "q3"}).encode(),
                )
                runtime._stopping = False
                assert closed.status_code == 503
                assert closed.json() == REJECTION

                gate.set()
                assert (await drain_to_end(watch))["status"] == "success"
                assert len(recorder.requests) == 1
                _, entries = dump(path)
                assert [row[3] for row in entries if row[0] == sid] == [
                    "user", "assistant", "turn_settlement",
                ]

    asyncio.run(run())


def test_snapshot_body_carries_latest_outcome(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        truncated = sse(choice({"role": "assistant", "content": "Hel"}))
        app = build_app(
            path,
            make_client(
                Recorder(), stream_step(text_stream(["A"])), stream_step(truncated)
            ),
        )
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                headers = cookie_headers(sid)
                watch = app.state.runtime.subscribe(sid)
                await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r1", "text": "q1"}).encode(),
                )
                assert (await drain_to_end(watch))["status"] == "success"
                await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r2", "text": "q2"}).encode(),
                )
                assert (await drain_to_end(watch))["status"] == "failure"

                snapshot = await client.get("/api/session", headers=headers)
                assert snapshot.json() == {
                    "messages": [
                        {"role": "user", "text": "q1"},
                        {"role": "assistant", "text": "A"},
                        {"role": "user", "text": "q2"},
                        {"role": "assistant", "text": "Hel"},
                    ],
                    "turn": None,
                    "outcome": "failure",
                }

    asyncio.run(run())


def test_corrupt_entry_produces_default_500(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "INSERT INTO entries (session_id, turn_id, request_id, kind, payload)"
                    " VALUES (?, ?, NULL, 'user', ?)",
                    (sid, "t", "not json"),
                )
                connection.commit()
            finally:
                connection.close()
            async with asgi_client(app, raise_app_exceptions=False) as client:
                response = await client.get("/api/session", headers=cookie_headers(sid))
            assert response.status_code == 500
            assert response.text == "Internal Server Error"

    asyncio.run(run())


# --- P6: SSE delivery and lifetime ------------------------------------------------


def test_sse_subscribes_before_headers_and_unsubscribes_on_exit(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        captured = capture_streaming(monkeypatch)
        async with app.router.lifespan_context(app):
            sid = await app.state.storage.create_session()
            runtime = app.state.runtime
            frames = []
            disconnected = asyncio.Event()

            async def receive():
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    assert runtime._subscribers.get(sid), "no subscription at headers"
                    headers = {
                        bytes(k).decode(): bytes(v).decode() for k, v in message["headers"]
                    }
                    assert message["status"] == 200
                    assert headers["content-type"].startswith("text/event-stream")
                    assert headers["cache-control"] == "no-cache"
                    runtime.publish(sid, "delta", {"turn_id": "t", "text": "x"})
                    runtime.publish(sid, "turn_end", {"turn_id": "t", "status": "success"})
                elif message["type"] == "http.response.body" and message.get("body"):
                    frames.append(message["body"])
                    if b"turn_end" in message["body"]:
                        disconnected.set()

            await asyncio.wait_for(
                app(sse_scope(sid), receive, send), 5
            )
            assert len(captured) == 1, "the response must be built through frames()"
            assert frames_of(b"".join(frames)) == [
                ("delta", {"turn_id": "t", "text": "x"}),
                ("turn_end", {"turn_id": "t", "status": "success"}),
            ]
            assert runtime._subscribers == {}, "the exit subscription leaked"

    asyncio.run(run())


def test_settlement_before_and_after_subscription(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(
            path,
            make_client(
                Recorder(), stream_step(text_stream(["A"])), stream_step(text_stream(["B"]))
            ),
        )
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                headers = cookie_headers(sid)
                runtime = app.state.runtime
                watch = runtime.subscribe(sid)

                committed = await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r1", "text": "q1"}).encode(),
                )
                first_turn = committed.json()["turn_id"]
                assert (await drain_to_end(watch)) == {
                    "turn_id": first_turn, "status": "success",
                }

                disconnected = asyncio.Event()
                started = asyncio.Event()
                frames = []

                async def receive():
                    await disconnected.wait()
                    return {"type": "http.disconnect"}

                async def send(message):
                    if message["type"] == "http.response.start":
                        assert runtime._subscribers.get(sid)
                        started.set()
                    elif message["type"] == "http.response.body" and message.get("body"):
                        frames.append(message["body"])
                        if b"turn_end" in message["body"]:
                            disconnected.set()

                stream = asyncio.create_task(app(sse_scope(sid), receive, send))
                await asyncio.wait_for(started.wait(), 5)

                onopen = await client.get("/api/session", headers=headers)
                assert onopen.json() == {
                    "messages": [
                        {"role": "user", "text": "q1"},
                        {"role": "assistant", "text": "A"},
                    ],
                    "turn": None,
                    "outcome": "success",
                }

                second = await client.post(
                    "/api/message", headers=headers,
                    content=json.dumps({"request_id": "r2", "text": "q2"}).encode(),
                )
                second_turn = second.json()["turn_id"]
                await asyncio.wait_for(stream, 5)
                live = frames_of(b"".join(frames))
                assert ("delta", {"turn_id": second_turn, "text": "B"}) in live
                assert (
                    "turn_end",
                    {"turn_id": second_turn, "status": "success"},
                ) in live
                after_end = await client.get("/api/session", headers=headers)
                assert after_end.json()["messages"][-1] == {"role": "assistant", "text": "B"}
                assert after_end.json()["turn"] is None
                assert after_end.json()["outcome"] == "success"
                assert runtime._subscribers == {sid: {watch}}

    asyncio.run(run())


def test_sse_cancelled_while_blocked_unsubscribes_on_response_exit(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        captured = capture_streaming(monkeypatch)
        async with app.router.lifespan_context(app):
            sid = await app.state.storage.create_session()
            runtime = app.state.runtime

            async def silent_receive():
                await asyncio.Future()

            started = asyncio.Event()
            entered = asyncio.Event()
            body_blocked = asyncio.Event()

            async def queue_sender(message):
                if message["type"] == "http.response.start":
                    assert runtime._subscribers.get(sid)
                    started.set()
                elif message["type"] == "http.response.body":
                    pytest.fail("an idle queue must not produce a frame")

            async def start_sender(message):
                if message["type"] == "http.response.start":
                    assert runtime._subscribers.get(sid)
                    runtime.publish(sid, "delta", {"turn_id": "t", "text": "x"})
                    started.set()
                elif message["type"] == "http.response.body" and message.get("body"):
                    entered.set()
                    await body_blocked.wait()

            task = asyncio.create_task(app(sse_scope(sid), silent_receive, queue_sender))
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert len(captured) == 1
            assert not entered.is_set()
            assert runtime._subscribers == {}, "the queue.get cancellation leaked"

            started = asyncio.Event()
            entered = asyncio.Event()
            body_blocked = asyncio.Event()
            task = asyncio.create_task(app(sse_scope(sid), silent_receive, start_sender))
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert len(captured) == 2
            assert runtime._subscribers == {}, "the send cancellation leaked"

    asyncio.run(run())


def test_sse_setup_rejections_create_no_subscription(tmp_path):
    async def run():
        app = build_app(tmp_path / "db.sqlite3", make_client(Recorder()))
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime

            async def receive():
                await asyncio.Future()

            async def send(message):
                if message["type"] == "http.response.body":
                    assert json.loads(message.get("body", b"{}")) == REJECTION

            scope = sse_scope("nobody")
            scope["headers"] = []
            await asyncio.wait_for(app(scope, receive, send), 5)
            assert runtime._subscribers == {}
            await asyncio.wait_for(app(sse_scope("gone"), receive, send), 5)
            assert runtime._subscribers == {}

            async def failed_snapshot(session_id):
                raise sqlite3.OperationalError("snapshot fault")

            app.state.storage.snapshot = failed_snapshot
            async with asgi_client(app, raise_app_exceptions=False) as client:
                response = await client.get("/api/events", headers=cookie_headers("gone"))
            assert response.status_code == 500
            assert runtime._subscribers == {}

    asyncio.run(run())


def test_real_server_disconnect_leaves_runner_completing(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        recorder = Recorder()
        app = build_app(
            path, make_client(recorder, stream_step(first, rest, gate=gate))
        )
        harness = Harness(app)
        url = await harness.start()
        try:
            async with httpx.AsyncClient(base_url=url) as client:
                sid = session_id_from(await asyncio.wait_for(client.get("/api/session"), 5))
                headers = cookie_headers(sid)
                watch = app.state.runtime.subscribe(sid)
                stream = await asyncio.wait_for(
                    client.send(
                        client.build_request("GET", "/api/events", headers=headers),
                        stream=True,
                    ),
                    5,
                )
                assert stream.headers["content-type"].startswith("text/event-stream")
                posted = await asyncio.wait_for(
                    client.post(
                        "/api/message", headers=headers,
                        content=json.dumps({"request_id": "r1", "text": "q1"}).encode(),
                    ),
                    5,
                )
                turn_id = posted.json()["turn_id"]
                raw = bytearray()

                async def until_delta():
                    async for piece in stream.aiter_bytes(1):
                        raw.extend(piece)
                        if raw.endswith(b"\n\n"):
                            return

                await asyncio.wait_for(until_delta(), 5)
                assert frames_of(raw)[0] == (
                    "delta", {"turn_id": turn_id, "text": "Hel"},
                )
                await stream.aclose()

                async def drained():
                    while app.state.runtime._subscribers.get(sid) != {watch}:
                        await asyncio.sleep(0)

                await asyncio.wait_for(drained(), 5)
                assert app.state.runtime._running[sid].turn_id == turn_id
                assert not app.state.runtime._running[sid].task.done()
                gate.set()
                assert (await drain_to_end(watch))["status"] == "success"
                snapshot = await asyncio.wait_for(client.get("/api/session", headers=headers), 5)
                assert snapshot.json() == {
                    "messages": [
                        {"role": "user", "text": "q1"},
                        {"role": "assistant", "text": "Hello"},
                    ],
                    "turn": None,
                    "outcome": "success",
                }
                assert len(recorder.requests) == 1
        finally:
            gate.set()
            await harness.stop()

    asyncio.run(run())


# --- P7: lifespan shutdown --------------------------------------------------------


def test_startup_interrupts_persisted_running_turns_without_replay(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            sid = await storage.create_session()
            turn_id, fresh = await storage.admit(sid, "r1", "q")
            assert fresh
            assert (await storage.snapshot(sid)).current_turn_id == turn_id
        finally:
            await storage.close()

        recorder = Recorder()
        http = make_client(recorder)
        app = build_app(path, http)
        try:
            async with app.router.lifespan_context(app):
                async with asgi_client(app) as client:
                    response = await client.get("/api/session", headers=cookie_headers(sid))
                assert response.json() == {
                    "messages": [{"role": "user", "text": "q"}],
                    "turn": None,
                    "outcome": "interrupted",
                }
                snapshot = await app.state.storage.snapshot(sid)
                assert [kind for kind, _ in snapshot.entries] == [
                    "user", "signal", "turn_settlement",
                ]
                assert recorder.requests == []
        finally:
            await http.aclose()

    asyncio.run(run())


def test_real_server_shutdown_before_first_model_body(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        entered = asyncio.Event()
        release = asyncio.Event()

        async def source():
            entered.set()
            await release.wait()
            yield text_stream(["late"])

        http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=source(),
            )
        ))
        app = build_app(path, http)
        harness = Harness(app)
        try:
            url = await harness.start()
            async with httpx.AsyncClient(base_url=url) as client:
                sid = session_id_from(await client.get("/api/session"))
                response = await client.post(
                    "/api/message", headers=cookie_headers(sid),
                    json={"request_id": "r1", "text": "q"},
                )
                assert response.status_code == 200
                await asyncio.wait_for(entered.wait(), 5)
                record = app.state.runtime._running[sid]
                assert record.text == ""
                await harness.stop()
                assert record.task.cancelled()
                assert not http.is_closed
            sessions, entries = dump(path)
            assert sessions[0][1] is None
            assert [row[3] for row in entries] == ["user", "signal", "turn_settlement"]
            assert json.loads(entries[-1][4]) == {"status": "interrupted"}
        finally:
            release.set()
            if harness.task is not None and not harness.task.done():
                await harness.stop()
            await http.aclose()

    asyncio.run(run())


def test_shutdown_interrupts_partial_and_closes_owned_http(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / "db.sqlite3"
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        recorder = Recorder()
        remaining = [stream_step(first, rest, gate=gate)]

        def handler(item):
            recorder.requests.append(item)
            return remaining.pop(0)()

        shim = OwnedHttpx(handler)
        monkeypatch.setattr(agent_qa.app, "httpx", shim)
        app = create_app(CONFIG, TARGET, path, PROMPT)
        harness = Harness(app)
        url = await harness.start()
        try:
            async with httpx.AsyncClient(base_url=url) as client:
                sid = session_id_from(await asyncio.wait_for(client.get("/api/session"), 5))
                headers = cookie_headers(sid)
                stream = await asyncio.wait_for(
                    client.send(
                        client.build_request("GET", "/api/events", headers=headers),
                        stream=True,
                    ),
                    5,
                )
                posted = await asyncio.wait_for(
                    client.post(
                        "/api/message", headers=headers,
                        content=json.dumps({"request_id": "r1", "text": "q1"}).encode(),
                    ),
                    5,
                )
                turn_id = posted.json()["turn_id"]
                chunks = []

                async def until_delta():
                    async for piece in stream.aiter_raw():
                        chunks.append(piece)
                        if b"Hel" in b"".join(chunks):
                            return

                await asyncio.wait_for(until_delta(), 5)
                assert b"Hel" in b"".join(chunks)
                await harness.stop()
                assert len(shim.created) == 1
                assert shim.created[0].is_closed, "the owned client must be closed"
            sessions, entries = dump(path)
            assert sessions[0][0] == sid and sessions[0][1] is None
            assert [row[3] for row in entries if row[0] == sid] == [
                "user", "assistant", "signal", "turn_settlement",
            ]
            assert json.loads(entries[1][4])["message"]["content"] == "Hel"
            assert json.loads(entries[3][4]) == {"status": "interrupted"}
            assert len(recorder.requests) == 1
        finally:
            gate.set()
            if harness.task is not None and not harness.task.done():
                harness.server.should_exit = True
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(harness.task, 5)

    asyncio.run(run())


def test_caller_injected_http_remains_open(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        client = make_client(Recorder(), stream_step(text_stream(["A"])))
        app = build_app(path, client)
        async with app.router.lifespan_context(app):
            sid = await app.state.storage.create_session()
            watch = app.state.runtime.subscribe(sid)
            async with asgi_client(app) as posted:
                response = await posted.post(
                    "/api/message", headers=cookie_headers(sid),
                    content=json.dumps({"request_id": "r1", "text": "q"}).encode(),
                )
            assert response.status_code == 200
            await drain_to_end(watch)
        assert client.is_closed is False, "shutdown closed the caller-owned client"

    asyncio.run(run())


def test_task_cancelled_before_first_body_writes_nothing(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        harness = Harness(app)
        url = await harness.start()
        try:
            async with httpx.AsyncClient(base_url=url) as client:
                sid = session_id_from(await asyncio.wait_for(client.get("/api/session"), 5))
                pulled = asyncio.Event()
                task = asyncio.create_task(
                    client.post(
                        "/api/message",
                        headers=cookie_headers(sid),
                        content=endless_body(pulled),
                    )
                )
                await asyncio.wait_for(pulled.wait(), 5)
                task.cancel()
                with contextlib.suppress(httpx.HTTPError, asyncio.CancelledError):
                    await task
                await harness.stop()
            sessions, entries = dump(path)
            assert sessions[0][0] == sid and sessions[0][1] is None
            assert entries == [], "a bodyless cancelled request must not admit"
        finally:
            await harness.stop()

    asyncio.run(run())


def test_cancelled_admission_commits_and_shutdown_scan_catches_it(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        entered = threading.Event()
        release = threading.Event()
        storage = None

        async with app.router.lifespan_context(app):
            storage = app.state.storage
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                original = storage._admit

                def blocked(session_id, request_id, text):
                    entered.set()
                    release.wait()
                    return original(session_id, request_id, text)

                storage._admit = blocked
                task = asyncio.create_task(
                    client.post(
                        "/api/message", headers=cookie_headers(sid),
                        content=json.dumps({"request_id": "r1", "text": "q"}).encode(),
                    )
                )
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    assert app.state.runtime._running == {}
                finally:
                    release.set()
        sessions, entries = dump(path)
        assert sessions[0][1] is None
        assert [row[3] for row in entries if row[0] == sid] == [
            "user", "signal", "turn_settlement",
        ]
        assert json.loads(entries[1][4]) == {"status": "interrupted"}

    asyncio.run(run())


def test_stopping_during_awaited_admission_returns_turn_without_spawn(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        app = build_app(path, make_client(Recorder()))
        entered = asyncio.Event()
        release = asyncio.Event()

        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                sid = session_id_from(await client.get("/api/session"))
                storage = app.state.storage
                original = storage.admit

                async def blocked(session_id, request_id, text):
                    entered.set()
                    await release.wait()
                    return await original(session_id, request_id, text)

                storage.admit = blocked
                task = asyncio.create_task(
                    client.post(
                        "/api/message", headers=cookie_headers(sid),
                        content=json.dumps({"request_id": "r1", "text": "q"}).encode(),
                    )
                )
                await asyncio.wait_for(entered.wait(), 5)
                app.state.runtime._stopping = True
                release.set()
                response = await asyncio.wait_for(task, 5)
                assert response.status_code == 200
                turn_id = response.json()["turn_id"]
                assert app.state.runtime._running == {}
        sessions, entries = dump(path)
        assert sessions[0][1] is None
        assert [row[3] for row in entries if row[0] == sid] == [
            "user", "signal", "turn_settlement",
        ]
        assert json.loads(entries[1][4]) == {"status": "interrupted"}

    asyncio.run(run())


def test_failed_scan_still_closes_resources(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / "db.sqlite3"
        shim = OwnedHttpx(lambda item: httpx.Response(500))
        monkeypatch.setattr(agent_qa.app, "httpx", shim)
        app = create_app(CONFIG, TARGET, path, PROMPT)
        cm = app.router.lifespan_context(app)
        await cm.__aenter__()
        storage = app.state.storage
        sid = await storage.create_session()

        async def failing():
            raise RuntimeError("scan fault")

        storage.interrupt_all = failing
        with pytest.raises(RuntimeError, match="scan fault"):
            await cm.__aexit__(None, None, None)
        assert shim.created[0].is_closed, "the owned client must close despite the scan fault"
        with pytest.raises(RuntimeError):
            await storage.snapshot(sid)

    asyncio.run(run())


def test_two_apps_on_separate_dbs_have_separate_runtime_state(tmp_path):
    async def run():
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        app_a = build_app(tmp_path / "a.sqlite3", make_client(Recorder(), stream_step(first, rest, gate=gate)))
        app_b = build_app(tmp_path / "b.sqlite3", make_client(Recorder(), stream_step(text_stream(["B"]))))
        async with app_a.router.lifespan_context(app_a):
            async with app_b.router.lifespan_context(app_b):
                runtime_a, runtime_b = app_a.state.runtime, app_b.state.runtime
                assert runtime_a is not runtime_b
                assert app_a.state.storage is not app_b.state.storage
                sid_a = await app_a.state.storage.create_session()
                assert await app_b.state.storage.snapshot(sid_a) is None

                watch_a = runtime_a.subscribe(sid_a)
                async with asgi_client(app_a) as posted:
                    response = await posted.post(
                        "/api/message", headers=cookie_headers(sid_a),
                        content=json.dumps({"request_id": "r1", "text": "q1"}).encode(),
                    )
                assert response.status_code == 200
                turn_a = response.json()["turn_id"]
                assert list(runtime_a._running) == [sid_a]
                assert runtime_b._running == {}

                sid_b = await app_b.state.storage.create_session()
                watch_b = runtime_b.subscribe(sid_b)
                async with asgi_client(app_b) as second:
                    response_b = await second.post(
                        "/api/message", headers=cookie_headers(sid_b),
                        content=json.dumps({"request_id": "r1", "text": "q2"}).encode(),
                    )
                assert response_b.status_code == 200
                assert (await drain_to_end(watch_b))["status"] == "success"
                assert runtime_a._running.get(sid_a).turn_id == turn_a
            # B is closed while A's lifespan is still active.
            assert runtime_b._stopping
            assert runtime_a._running[sid_a].turn_id == turn_a
            assert not runtime_a._running[sid_a].task.done()
            gate.set()
            assert (await drain_to_end(watch_a))["status"] == "success"

    asyncio.run(run())


# --- P8 launcher cells ------------------------------------------------------------


def launch(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "local": {
                        "base_url": "http://unit.test/v1",
                        "models": {"qwen": {"context_window": 8}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("sys", encoding="utf-8")
    return [
        "--config", str(config_path),
        "--provider", "local",
        "--model", "qwen",
        "--prompt", str(prompt_path),
    ]


def test_main_rejects_invalid_launch_inputs(tmp_path):
    def with_value(arguments, option, value):
        replaced = list(arguments)
        replaced[replaced.index(option) + 1] = value
        return replaced

    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2

    arguments = launch(tmp_path)
    with pytest.raises(SystemExit) as caught:
        main(with_value(arguments, "--config", str(tmp_path / "absent.json")))
    assert str(caught.value) == "configuration file cannot be read"
    assert caught.value.__cause__ is None

    with pytest.raises(SystemExit) as caught:
        main(with_value(arguments, "--provider", "other"))
    assert str(caught.value) == "unknown provider or model"
    assert caught.value.__cause__ is None

    with pytest.raises(SystemExit) as caught:
        main(with_value(arguments, "--model", "other"))
    assert str(caught.value) == "unknown provider or model"
    assert caught.value.__cause__ is None

    with pytest.raises(SystemExit) as caught:
        main(with_value(arguments, "--prompt", str(tmp_path / "absent.txt")))
    assert str(caught.value) == "prompt file cannot be read as UTF-8"
    assert caught.value.__cause__ is None

    unreadable = tmp_path / "binary.txt"
    unreadable.write_bytes(b"\xff\xfe{}")
    with pytest.raises(SystemExit) as caught:
        main(with_value(arguments, "--prompt", str(unreadable)))
    assert str(caught.value) == "prompt file cannot be read as UTF-8"
    assert caught.value.__cause__ is None


def test_unwritable_db_lifespan_fails_without_chaining(tmp_path):
    app = build_app(tmp_path / "missing-dir" / "db.sqlite3", make_client(Recorder()))

    async def run():
        with pytest.raises(RuntimeError, match="database startup failed") as caught:
            await app.router.lifespan_context(app).__aenter__()
        assert caught.value.__cause__ is None

    asyncio.run(run())


# --- P8 static mount ----------------------------------------------------------------


def test_static_pages_serve_from_a_changed_cwd(tmp_path, monkeypatch):
    async def run():
        app = build_app(tmp_path / "db.sqlite3", make_client(Recorder()))
        # A cwd-relative asset resolution would break under a foreign cwd.
        monkeypatch.chdir(tmp_path)
        async with app.router.lifespan_context(app):
            async with asgi_client(app) as client:
                page = await client.get("/")
                assert page.status_code == 200
                assert page.headers["content-type"].startswith("text/html")
                script = await client.get("/app.js")
                assert script.status_code == 200
                assert "javascript" in script.headers["content-type"]
                styles = await client.get("/style.css")
                assert styles.status_code == 200
                assert styles.headers["content-type"].startswith("text/css")
                static = importlib.resources.files("agent_qa") / "static"
                assert page.text == (static / "index.html").read_text(encoding="utf-8")
                assert script.text == (static / "app.js").read_text(encoding="utf-8")
                assert styles.text == (static / "style.css").read_text(encoding="utf-8")
                assert (await client.get("/api/nothing")).status_code == 404
                session = await client.get("/api/session")
                assert session.status_code == 200

    asyncio.run(run())


# --- Manual P9 fixture launcher (never started by pytest) ---------------------------


def manual_ui():
    """Serve the production app with a canned 20-fragment provider stream."""
    parser = argparse.ArgumentParser(prog="tests/test_app.py")
    parser.add_argument("--manual-ui", action="store_true", required=True)
    parser.add_argument("--db", required=True)
    arguments = parser.parse_args()

    calls = 0

    async def stream():
        for index in range(20):
            delta = {"content": f" fragment-{index}"}
            if index == 0:
                delta["role"] = "assistant"
            frame = json.dumps({"choices": [{"index": 0, "delta": delta}]}).encode()
            yield b"data: " + frame + b"\n\n"
            await asyncio.sleep(0.2)
        yield sse(choice({}, finish="stop"), usage_event())

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=stream()
        )

    config = ProvidersConfig.model_validate(
        {
            "providers": {
                "local": {
                    "base_url": "http://fixture.invalid/v1",
                    "models": {"fixture": {"context_window": 8}},
                }
            }
        }
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        config,
        ModelRef(provider="local", model="fixture"),
        Path(arguments.db),
        "You are answering questions for a manual browser check.",
        http=http,
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, timeout_graceful_shutdown=0)
    finally:
        asyncio.run(http.aclose())
    print(f"model calls: {calls}")


if __name__ == "__main__":
    manual_ui()
