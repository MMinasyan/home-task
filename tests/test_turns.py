"""Durable-history projections and the server-owned turn runtime."""

import asyncio
import json
import logging
import sqlite3

import httpx
import pytest

from agent_qa.model import Message, ModelRef, ModelRequest, ToolCall
from agent_qa.providers._cc import encode_request
from agent_qa.providers.chat_completions import ChatCompletionsClient
from agent_qa.providers.config import ProvidersConfig
from agent_qa.storage import Snapshot, Storage
from agent_qa.turns import (
    Runtime,
    client_messages,
    decode_message,
    encode_message,
    model_context,
)

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
KEYED_CONFIG = ProvidersConfig.model_validate(
    {
        "providers": {
            "local": {
                "base_url": "http://unit.test/v1",
                "api_key_env": "UNIT_TURNS_MISSING_KEY",
                "models": {"qwen": {"context_window": 8}},
            }
        }
    }
)
TARGET = ModelRef(provider="local", model="qwen")
FOREIGN = ModelRef(provider="local", model="other")
PROMPT = "Answer briefly."
INTERRUPTED_NOTE = "The previous response ended before completion."


def dump(path):
    """Complete durable rows for exact comparison."""
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
    """Event bytes without the [DONE] terminator, for splicing streams."""
    return b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )


def text_stream(fragments):
    events = [choice({"role": "assistant", "content": piece}) for piece in fragments]
    events.append(choice({}, finish="stop"))
    events.append(usage_event())
    return sse(*events)


class Recorder:
    """Captures the HTTP requests made through one mock transport."""

    def __init__(self):
        self.requests = []

    def bodies(self):
        return [json.loads(item.content) for item in self.requests]


def make_client(recorder, *steps):
    remaining = list(steps)

    def handler(item):
        recorder.requests.append(item)
        return remaining.pop(0)()

    return ChatCompletionsClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def stream_step(*pieces, gate=None):
    """One HTTP 200 SSE response delivered as byte chunks.

    With a *gate*, the source pauses before the final piece until the gate is
    set, holding the stream open.
    """
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


async def open_runtime(path, recorder, *steps):
    storage = await Storage.open(path)
    runtime = Runtime(storage, make_client(recorder, *steps), CONFIG, TARGET, PROMPT)
    return storage, runtime


async def next_event(queue):
    return await asyncio.wait_for(queue.get(), 5)


async def drain_to_end(queue):
    """Read queued events until the terminal one; return its data."""
    while True:
        name, data = await next_event(queue)
        if name == "turn_end":
            return data


# --- message codec -------------------------------------------------------------


def test_message_codec_round_trips_through_the_real_encoder():
    message = Message(
        role="assistant",
        content="Լուծում",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="look",
                arguments='{"q": 1}',
                extra={"priority": 1},
                function_extra={"signature": "s1"},
            ),
            ToolCall(id="call_2", name="echo", arguments="{}"),
        ),
        source=TARGET,
        extra={"reasoning_content": "մտածել", "note": None, "path": ["a", {"b": 2}]},
    )
    stored = json.loads(json.dumps(encode_message(message)))
    assert stored == {
        "role": "assistant",
        "content": "Լուծում",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "look",
                "arguments": '{"q": 1}',
                "extra": {"priority": 1},
                "function_extra": {"signature": "s1"},
            },
            {
                "id": "call_2",
                "name": "echo",
                "arguments": "{}",
                "extra": {},
                "function_extra": {},
            },
        ],
        "tool_call_id": None,
        "source": {"provider": "local", "model": "qwen"},
        "extra": {"reasoning_content": "մտածել", "note": None, "path": ["a", {"b": 2}]},
    }
    assert encode_message(decode_message(stored)) == stored

    same_source = encode_request(
        model_key="qwen", request=ModelRequest(messages=(decode_message(stored),)),
        target=TARGET,
    )["messages"][0]
    assert same_source == {
        "role": "assistant",
        "content": "Լուծում",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "look", "arguments": '{"q": 1}', "signature": "s1"},
                "priority": 1,
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            },
        ],
        "reasoning_content": "մտածել",
        "note": None,
        "path": ["a", {"b": 2}],
    }

    foreign = encode_request(
        model_key="qwen", request=ModelRequest(messages=(decode_message(stored),)),
        target=FOREIGN,
    )["messages"][0]
    assert foreign == {
        "role": "assistant",
        "content": "Լուծում",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "look", "arguments": '{"q": 1}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            },
        ],
    }

    textless = json.loads(
        json.dumps(encode_message(Message(role="assistant", content=None)))
    )
    assert textless == {
        "role": "assistant",
        "content": None,
        "tool_calls": [],
        "tool_call_id": None,
        "source": None,
        "extra": {},
    }
    assert encode_message(decode_message(textless)) == textless


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({}, KeyError),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "tool_call_id": None,
                "source": None,
            },
            KeyError,
        ),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "tool_call_id": None,
                "source": {"provider": "local"},
                "extra": {},
            },
            TypeError,
        ),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{}],
                "tool_call_id": None,
                "source": None,
                "extra": {},
            },
            KeyError,
        ),
    ],
)
def test_decode_propagates_unusable_payloads(value, error):
    with pytest.raises(error):
        decode_message(value)


def test_projections_project_real_history_exactly(tmp_path):
    async def run():
        storage = await Storage.open(tmp_path / "db.sqlite3")
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "What is A?")
            answered = Message(
                role="assistant",
                content="A",
                source=TARGET,
                extra={"reasoning_content": "r"},
            )
            assert await storage.settle(
                session_id, turn_id, "success",
                {"message": encode_message(answered), "usage": {"tokens": 2}},
            )
            turn_id, _ = await storage.admit(session_id, "r2", "And B?")
            silent = Message(
                role="assistant", content=None, extra={"reasoning_content": "silent"}
            )
            assert await storage.settle(
                session_id, turn_id, "success",
                {"message": encode_message(silent), "usage": None},
            )
            turn_id, _ = await storage.admit(session_id, "r3", "And C?")
            assert await storage.settle(session_id, turn_id, "interrupted", None)
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None

            assert model_context(snapshot, PROMPT) == (
                Message(role="system", content=PROMPT),
                Message(role="user", content="What is A?"),
                answered,
                Message(role="user", content="And B?"),
                silent,
                Message(role="user", content="And C?"),
                Message(role="system", content=INTERRUPTED_NOTE),
            )
            assert client_messages(snapshot) == [
                {"role": "user", "text": "What is A?"},
                {"role": "assistant", "text": "A"},
                {"role": "user", "text": "And B?"},
                {"role": "user", "text": "And C?"},
            ]
        finally:
            await storage.close()

    asyncio.run(run())


def test_projections_reject_unusable_history():
    unknown = Snapshot(None, (("widget", "{}"),))
    with pytest.raises(ValueError):
        model_context(unknown, PROMPT)
    with pytest.raises(ValueError):
        client_messages(unknown)

    broken = Snapshot(None, (("user", "not json"),))
    with pytest.raises(ValueError):
        model_context(broken, PROMPT)

    missing = Snapshot(None, (("assistant", '{"role": "assistant"}'),))
    with pytest.raises(KeyError):
        model_context(missing, PROMPT)

    empty = Snapshot(None, ())
    assert model_context(empty, PROMPT) == (Message(role="system", content=PROMPT),)
    assert client_messages(empty) == []


# --- transient event hub --------------------------------------------------------


def test_hub_fans_out_and_unsubscribes_idempotently():
    # The hub touches only its own dictionaries; the unused services stay None.
    runtime = Runtime(None, None, CONFIG, TARGET, PROMPT)  # type: ignore[arg-type]
    first = runtime.subscribe("s")
    second = runtime.subscribe("s")
    runtime.publish("s", "delta", {"turn_id": "t", "text": "x"})
    assert first.get_nowait() == ("delta", {"turn_id": "t", "text": "x"})
    assert second.get_nowait() == ("delta", {"turn_id": "t", "text": "x"})
    runtime.unsubscribe("s", first)
    runtime.unsubscribe("s", first)
    runtime.publish("s", "turn_end", {"turn_id": "t", "status": "success"})
    assert second.get_nowait() == ("turn_end", {"turn_id": "t", "status": "success"})
    runtime.unsubscribe("s", second)
    runtime.unsubscribe("s", second)
    assert runtime._subscribers == {}
    runtime.publish("missing", "delta", {})  # an unknown session is a no-op


# --- admission, runner, and shutdown -------------------------------------------


def test_submit_runs_one_model_request_in_a_server_task(tmp_path):
    async def run():
        recorder = Recorder()
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(first, rest, gate=gate),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            # Admission returns while the model stream is still held: the
            # runner is an independent server task.
            turn_id = await asyncio.wait_for(runtime.submit(session_id, "r1", "q"), 5)
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "Hel"},
            )
            record = runtime._running[session_id]
            assert record.turn_id == turn_id
            assert record.task is not None and not record.task.done()
            assert record.text == "Hel"

            gate.set()
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "lo"},
            )
            assert await next_event(queue) == (
                "turn_end", {"turn_id": turn_id, "status": "success"},
            )
            assert runtime._running == {}

            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "assistant", "turn_settlement",
            ]
            assert json.loads(snapshot.entries[1][1]) == {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "tool_calls": [],
                    "tool_call_id": None,
                    "source": {"provider": "local", "model": "qwen"},
                    "extra": {},
                },
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            }
            sessions, _ = dump(tmp_path / "db.sqlite3")
            assert json.loads(sessions[0][2]) == {
                "prompt_tokens": 3, "completion_tokens": 5,
            }

            # Exactly one model request, built from the history, with no tools.
            assert len(recorder.requests) == 1
            assert recorder.bodies()[0] == encode_request(
                model_key="qwen",
                request=ModelRequest(messages=(
                    Message(role="system", content=PROMPT),
                    Message(role="user", content="q"),
                )),
                target=TARGET,
            )
        finally:
            gate.set()
            await storage.close()

    asyncio.run(run())


def test_durable_history_composes_the_real_provider_request(tmp_path):
    async def run():
        recorder = Recorder()
        gate = asyncio.Event()
        reasoning = sse(
            choice({"role": "assistant", "reasoning_content": "think"}),
            choice({"extra_content": {"google": {"thought_signature": "s"}}}),
            choice({"content": "answer"}),
            choice({}, finish="stop"),
            usage_event(),
        )
        first_b = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest_b = sse(
            choice({"content": "lo"}), choice({}, finish="stop"), usage_event()
        )
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(reasoning),
            stream_step(first_b, rest_b, gate=gate),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            await runtime.submit(session_id, "r1", "What is A?")
            assert (await drain_to_end(queue))["status"] == "success"
            second = await runtime.submit(session_id, "r2", "And B?")
            # Held at the second stream: the history now ends with the new
            # user entry, before its own answer exists.
            assert await next_event(queue) == (
                "delta", {"turn_id": second, "text": "Hel"},
            )
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            context = model_context(snapshot, PROMPT)

            gate.set()
            assert (await drain_to_end(queue))["status"] == "success"
            # The captured wire body is exactly the real encoder's output for
            # the decoded durable history.
            assert recorder.bodies()[1] == encode_request(
                model_key="qwen",
                request=ModelRequest(messages=context),
                target=TARGET,
            )
            replayed = recorder.bodies()[1]["messages"][-2]
            assert replayed["reasoning_content"] == "think"
            assert replayed["extra_content"] == {"google": {"thought_signature": "s"}}
            assert "source" not in replayed and "usage" not in replayed
            assert recorder.bodies()[0]["messages"] == [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": "What is A?"},
            ]
        finally:
            gate.set()
            await storage.close()

    asyncio.run(run())


def test_duplicate_submit_returns_the_turn_and_spawns_nothing(tmp_path):
    async def run():
        recorder = Recorder()
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(first, rest, gate=gate),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            turn_id = await runtime.submit(session_id, "r1", "first")
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "Hel"},
            )
            # While running: same turn, no second runner, no second request.
            assert await runtime.submit(session_id, "r1", "replacement") == turn_id
            assert len(runtime._running) == 1
            assert len(recorder.requests) == 1

            gate.set()
            assert (await drain_to_end(queue))["status"] == "success"
            # After settlement the duplicate still resolves without a new call.
            assert await runtime.submit(session_id, "r1", "again") == turn_id
            assert len(recorder.requests) == 1
            assert runtime._running == {}
            _, entries = dump(tmp_path / "db.sqlite3")
            assert [row[3] for row in entries] == [
                "user", "assistant", "turn_settlement",
            ]
        finally:
            gate.set()
            await storage.close()

    asyncio.run(run())


def test_unknown_busy_and_closed_are_rejected_without_writes(tmp_path):
    async def run():
        recorder = Recorder()
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(first, rest, gate=gate),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            assert await runtime.submit("missing", "r1", "x") == "unknown"
            turn_id = await runtime.submit(session_id, "r1", "q")
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "Hel"},
            )
            assert await runtime.submit(session_id, "r2", "later") == "busy"
            # A stop already set before submitting admission rejects.
            runtime._stopping = True
            assert await runtime.submit(session_id, "r3", "late") == "closed"
            _, entries = dump(tmp_path / "db.sqlite3")
            assert [row[3] for row in entries] == ["user"]

            gate.set()
            assert (await drain_to_end(queue))["status"] == "success"
        finally:
            gate.set()
            await storage.close()

    asyncio.run(run())


def test_tool_response_settles_failure_and_leaves_no_dangling_call(tmp_path):
    async def run():
        recorder = Recorder()
        tooling = sse(
            choice({"role": "assistant", "content": "Hel"}),
            choice({"content": "lo"}),
            choice(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "look", "arguments": "{}"},
                        }
                    ]
                },
                finish="tool_calls",
            ),
        )
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(tooling), stream_step(text_stream(["after"])),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            turn_id = await runtime.submit(session_id, "r1", "q")
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "Hel"},
            )
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "lo"},
            )
            assert await next_event(queue) == (
                "turn_end", {"turn_id": turn_id, "status": "failure"},
            )
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "assistant", "signal", "turn_settlement",
            ]
            # The partial answer persists text-only, without the tool work.
            assert json.loads(snapshot.entries[1][1]) == {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "tool_calls": [],
                    "tool_call_id": None,
                    "source": None,
                    "extra": {},
                },
                "usage": None,
            }
            assert json.loads(snapshot.entries[2][1]) == {"status": "failure"}

            # The next captured request carries no dangling tool call.
            await runtime.submit(session_id, "r2", "again")
            assert (await drain_to_end(queue))["status"] == "success"
            body = recorder.bodies()[1]
            assert "tool_calls" not in json.dumps(body)
            assert "tools" not in body
        finally:
            await storage.close()

    asyncio.run(run())


def test_entry_validation_failure_settles_failure_without_http(tmp_path, monkeypatch):
    monkeypatch.delenv("UNIT_TURNS_MISSING_KEY", raising=False)

    async def run():
        recorder = Recorder()
        storage = await Storage.open(tmp_path / "db.sqlite3")
        runtime = Runtime(storage, make_client(recorder), KEYED_CONFIG, TARGET, PROMPT)
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            turn_id = await runtime.submit(session_id, "r1", "q")
            assert await next_event(queue) == (
                "turn_end", {"turn_id": turn_id, "status": "failure"},
            )
            assert recorder.requests == []
            assert runtime._running == {}
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "signal", "turn_settlement",
            ]
        finally:
            await storage.close()

    asyncio.run(run())


def test_truncated_stream_settles_failure_with_partial_text(tmp_path):
    async def run():
        recorder = Recorder()
        truncated = sse(choice({"role": "assistant", "content": "Hel"}))
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder, stream_step(truncated)
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            turn_id = await runtime.submit(session_id, "r1", "q")
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "Hel"},
            )
            assert await next_event(queue) == (
                "turn_end", {"turn_id": turn_id, "status": "failure"},
            )
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "assistant", "signal", "turn_settlement",
            ]
            assert json.loads(snapshot.entries[1][1]) == {
                "message": {
                    "role": "assistant",
                    "content": "Hel",
                    "tool_calls": [],
                    "tool_call_id": None,
                    "source": None,
                    "extra": {},
                },
                "usage": None,
            }
        finally:
            await storage.close()

    asyncio.run(run())


def test_shutdown_cancels_the_runner_and_settles_the_partial(tmp_path):
    async def run():
        recorder = Recorder()
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(first, rest, gate=gate),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        try:
            turn_id = await runtime.submit(session_id, "r1", "q")
            assert await next_event(queue) == (
                "delta", {"turn_id": turn_id, "text": "Hel"},
            )
            task = runtime._running[session_id].task
            assert task is not None
            await runtime.shutdown()
            assert task.cancelled()
            assert await next_event(queue) == (
                "turn_end", {"turn_id": turn_id, "status": "interrupted"},
            )
            assert runtime._running == {}
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "assistant", "signal", "turn_settlement",
            ]
            assert json.loads(snapshot.entries[1][1]) == {
                "message": {
                    "role": "assistant",
                    "content": "Hel",
                    "tool_calls": [],
                    "tool_call_id": None,
                    "source": None,
                    "extra": {},
                },
                "usage": None,
            }
            assert json.loads(snapshot.entries[3][1]) == {"status": "interrupted"}
        finally:
            gate.set()
            await storage.close()

    asyncio.run(run())


def test_settlement_fault_logs_only_the_category_and_waits_for_shutdown(
    tmp_path, caplog,
):
    async def run():
        recorder = Recorder()
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(text_stream(["Hel", "lo"])),
        )
        session_id = await storage.create_session()
        queue = runtime.subscribe(session_id)
        failed = asyncio.Event()
        state = {"faulted": False}
        original = storage.settle

        async def faulty(session_id, turn_id, status, assistant):
            if not state["faulted"]:
                state["faulted"] = True
                failed.set()
                raise ValueError("secret boom detail")
            return await original(session_id, turn_id, status, assistant)

        storage.settle = faulty
        try:
            turn_id = await runtime.submit(session_id, "r1", "q")
            await asyncio.wait_for(failed.wait(), 5)
            # The fault is logged by category and IDs only, the record and the
            # durable running state remain, and no end event was published.
            assert runtime._running[session_id].text == "Hello"
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id == turn_id
            assert [kind for kind, _ in snapshot.entries] == ["user"]
            assert queue.qsize() == 2  # the two deltas only

            # Managed shutdown still settles the record with its text.
            await runtime.shutdown()
            assert (await drain_to_end(queue))["status"] == "interrupted"
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "assistant", "signal", "turn_settlement",
            ]
            assert json.loads(snapshot.entries[1][1])["message"]["content"] == "Hello"
            assert json.loads(snapshot.entries[1][1])["usage"] is None
            return session_id, turn_id
        finally:
            await storage.close()

    with caplog.at_level(logging.ERROR):
        session_id, turn_id = asyncio.run(run())
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "ValueError" in messages[0]
    assert session_id in messages[0] and turn_id in messages[0]
    assert "secret boom" not in caplog.text


def test_shutdown_during_awaited_admission_returns_the_turn_without_spawn(tmp_path):
    async def run():
        recorder = Recorder()
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder, stream_step(text_stream(["ok"]))
        )
        session_id = await storage.create_session()
        entered = asyncio.Event()
        release = asyncio.Event()
        original = storage.admit

        async def blocked(session_id, request_id, text):
            entered.set()
            await release.wait()
            return await original(session_id, request_id, text)

        storage.admit = blocked
        try:
            task = asyncio.create_task(runtime.submit(session_id, "r1", "q"))
            await asyncio.wait_for(entered.wait(), 5)
            runtime._stopping = True
            release.set()
            turn_id = await asyncio.wait_for(task, 5)
            # The durably admitted turn is returned as an identity, and the
            # shutdown interrupt scan follows it in the worker queue.
            assert turn_id not in ("unknown", "busy", "closed")
            assert runtime._running == {}
            await runtime.shutdown()
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert [kind for kind, _ in snapshot.entries] == [
                "user", "signal", "turn_settlement",
            ]
            assert json.loads(snapshot.entries[1][1]) == {"status": "interrupted"}
        finally:
            release.set()
            await storage.close()

    asyncio.run(run())


def test_two_sessions_interleave_while_one_stream_is_held(tmp_path):
    async def run():
        recorder = Recorder()
        gate = asyncio.Event()
        first = sse_open(choice({"role": "assistant", "content": "Hel"}))
        rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())
        storage, runtime = await open_runtime(
            tmp_path / "db.sqlite3", recorder,
            stream_step(first, rest, gate=gate),
            stream_step(text_stream(["B"])),
        )
        first_session = await storage.create_session()
        second_session = await storage.create_session()
        first_queue = runtime.subscribe(first_session)
        second_queue = runtime.subscribe(second_session)
        try:
            first_turn = await runtime.submit(first_session, "r1", "q1")
            assert await next_event(first_queue) == (
                "delta", {"turn_id": first_turn, "text": "Hel"},
            )
            # The second session completes while the first stream is held.
            second_turn = await runtime.submit(second_session, "r1", "q2")
            assert (await drain_to_end(second_queue))["status"] == "success"
            snapshot = await storage.snapshot(first_session)
            assert snapshot is not None
            assert snapshot.current_turn_id == first_turn

            gate.set()
            assert await next_event(first_queue) == (
                "delta", {"turn_id": first_turn, "text": "lo"},
            )
            assert await next_event(first_queue) == (
                "turn_end", {"turn_id": first_turn, "status": "success"},
            )
            assert recorder.bodies()[0]["messages"][-1]["content"] == "q1"
            assert recorder.bodies()[1]["messages"][-1]["content"] == "q2"
            _, entries = dump(tmp_path / "db.sqlite3")
            assert [row[3] for row in entries] == [
                "user", "user", "assistant", "turn_settlement",
                "assistant", "turn_settlement",
            ]
        finally:
            gate.set()
            await storage.close()

    asyncio.run(run())