"""Public client: entry validation, credential resolution, retry, and cleanup."""

import asyncio
import json

import httpx
import pytest

from agent_qa.model import (
    Message,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
)
from agent_qa.providers._cc import encode_request
from agent_qa.providers.chat_completions import ChatCompletionsClient, ChatStream
from agent_qa.providers.config import ProvidersConfig

ENV = "UNIT_CREDENTIAL_ENV"
SECRET = "SENTINEL-cc-credential"
TARGET = ModelRef(provider="local", model="qwen")
OTHER_REF = ModelRef(provider="local", model="missing")
USER = Message(role="user", content="hello")

CONFIG = ProvidersConfig.model_validate(
    {
        "providers": {
            "local": {
                "base_url": "http://unit.test/v1",
                "api_key_env": ENV,
                "models": {"qwen": {"context_window": 8}},
            }
        }
    }
)
CONFIG_NO_KEY = ProvidersConfig.model_validate(
    {
        "providers": {
            "local": {
                "base_url": "http://unit.test/v1",
                "models": {"qwen": {"context_window": 8}},
            }
        }
    }
)


def request(*messages):
    return ModelRequest(messages=tuple(messages))


# --- scripted HTTP ------------------------------------------------------------


class Recording:
    """Captures requests plus the ordered close/sleep/emit events of a call."""

    def __init__(self):
        self.requests = []
        self.events = []

    def bodies(self):
        return [json.loads(item.content) for item in self.requests]

    def closes(self):
        return [value for kind, value in self.events if kind == "close"]

    def sleeps(self):
        return [value for kind, value in self.events if kind == "sleep"]

    def emits(self):
        return [value for kind, value in self.events if kind == "emit"]


class TrackedResponse(httpx.Response):
    """A response recording its first close.

    httpx closes a stream response automatically when its bytes are drained to
    EOF; a later explicit close is an idempotent no-op, so each opened response
    contributes exactly one close event.
    """

    def __init__(self, status_code, content, log):
        super().__init__(
            status_code, headers={"content-type": "text/event-stream"}, content=content
        )
        self._log = log
        self._recorded = False

    async def aclose(self):
        if not self._recorded:
            self._recorded = True
            self._log.events.append(("close", self.status_code))
        await super().aclose()


def make_client(log, *steps):
    remaining = list(steps)

    def handler(item):
        log.requests.append(item)
        return remaining.pop(0)()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatCompletionsClient(http)


def stream_step(log, *pieces, gate=None):
    """One HTTP 200 SSE response delivered as byte chunks.

    With a *gate*, the source pauses before the final piece until the gate is
    set, holding the stream open.
    """
    def step():
        async def source():
            head, tail = pieces[:-1], pieces[-1:]
            for piece in head:
                log.events.append(("emit", piece))
                yield piece
            if gate is not None:
                await gate.wait()
            for piece in tail:
                log.events.append(("emit", piece))
                yield piece

        return TrackedResponse(200, source(), log)

    return step


def status_step(log, code):
    def step():
        return TrackedResponse(code, b"", log)

    return step


def failure_step(error):
    def step():
        raise error

    return step


def record_sleeps(log, monkeypatch, action=None):
    """Replace asyncio.sleep with an instant recorder (and optional action)."""
    async def fake_sleep(delay):
        log.events.append(("sleep", delay))
        if action is not None:
            action()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


# --- stream fixtures ----------------------------------------------------------


def choice(delta, finish=None):
    event_choice = {"index": 0, "delta": delta}
    if finish is not None:
        event_choice["finish_reason"] = finish
    return {"choices": [event_choice]}


def usage_event():
    return {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}


def sse_open(*events):
    return b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )


def sse(*events):
    return sse_open(*events) + b"data: [DONE]\n\n"


def text_stream(fragments, usage=True):
    events = [choice({"role": "assistant", "content": piece}) for piece in fragments]
    events.append(choice({}, finish="stop"))
    if usage:
        events.append(usage_event())
    return sse(*events)


async def consume(stream, fragments):
    async with stream:
        async for fragment in stream:
            fragments.append(fragment)


# --- entry-surfaced validation (C2) -------------------------------------------


def test_invalid_request_is_surfaced_at_entry_without_http(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    bad_requests = [
        request(),
        request(Message(role="user", content="x", tool_call_id="call_1")),
    ]
    for bad in bad_requests:
        log = Recording()

        async def main(bad=bad, log=log):
            client = make_client(log, status_step(log, 200))
            with pytest.raises(ValueError):
                await client.stream(CONFIG, TARGET, bad).__aenter__()

        asyncio.run(main())
        assert log.requests == []


def test_unknown_model_reference_is_surfaced_at_entry(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        client = make_client(log, status_step(log, 200))
        with pytest.raises(ValueError):
            await client.stream(CONFIG, OTHER_REF, request(USER)).__aenter__()

    asyncio.run(main())
    assert log.requests == []


def test_missing_credential_fails_at_entry_without_http(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    log = Recording()

    async def main():
        client = make_client(log, status_step(log, 200))
        with pytest.raises(ValueError):
            await client.stream(CONFIG, TARGET, request(USER)).__aenter__()

    asyncio.run(main())
    assert log.requests == []


def test_empty_credential_fails_at_entry(monkeypatch):
    monkeypatch.setenv(ENV, "")
    log = Recording()

    async def main():
        client = make_client(log, status_step(log, 200))
        with pytest.raises(ValueError):
            await client.stream(CONFIG, TARGET, request(USER)).__aenter__()

    asyncio.run(main())
    assert log.requests == []


# --- credential runtime (C1) --------------------------------------------------


def test_configured_credential_is_sent_as_a_bearer_header(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["ok"])))
        fragments = []
        await consume(client.stream(CONFIG, TARGET, request(USER)), fragments)

    asyncio.run(main())
    sent = log.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://unit.test/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {SECRET}"
    assert sent.headers["content-type"] == "application/json"
    assert sent.headers["accept"] == "text/event-stream"
    body = json.loads(sent.content)
    assert body["model"] == "qwen" and body["stream"] is True
    assert SECRET not in sent.content.decode()


def test_absent_key_configuration_sends_no_authorization_header(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["ok"])))
        fragments = []
        await consume(client.stream(CONFIG_NO_KEY, TARGET, request(USER)), fragments)

    asyncio.run(main())
    assert "authorization" not in log.requests[0].headers


def test_env_change_does_not_alter_an_in_flight_retry(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    def rotate():
        monkeypatch.setenv(ENV, "ROTATED-credential")

    record_sleeps(log, monkeypatch, action=rotate)

    async def main():
        client = make_client(
            log, status_step(log, 429), stream_step(log, text_stream(["ok"]))
        )
        fragments = []
        await consume(client.stream(CONFIG, TARGET, request(USER)), fragments)

    asyncio.run(main())
    headers = [item.headers["authorization"] for item in log.requests]
    assert headers == [f"Bearer {SECRET}", f"Bearer {SECRET}"]


# --- caller misuse ------------------------------------------------------------


def test_stream_has_no_public_constructor():
    with pytest.raises(TypeError):
        ChatStream(None, None, None, None, _token=object())


def test_reentering_the_context_is_caller_misuse(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["ok"])))
        stream = client.stream(CONFIG, TARGET, request(USER))
        async with stream:
            pass
        with pytest.raises(ValueError):
            await stream.__aenter__()
        assert log.closes() == [200]

    asyncio.run(main())


def test_iteration_before_entry_is_caller_misuse():
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["ok"])))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ValueError):
            await stream.__anext__()
        assert stream.partial is None and stream.result is None
        assert log.requests == []

    asyncio.run(main())


def test_next_after_end_raises_stopiteration(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["ok"])))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    asyncio.run(main())


def test_failed_entry_consumes_the_stream(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    record_sleeps(log, monkeypatch)

    async def main():
        client = make_client(log, *(status_step(log, 429) for _ in range(4)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError, match="HTTP 429"):
            await stream.__aenter__()
        with pytest.raises(ValueError):
            await stream.__anext__()
        with pytest.raises(ValueError):
            await stream.__aenter__()

    asyncio.run(main())
    assert len(log.requests) == 4


# --- successful streams end to end (C6) ---------------------------------------


def test_text_stream_surfaces_fragments_and_finalizes_with_usage(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["Hel", "lo"])))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        async with stream as entered:
            assert entered is stream
            async for fragment in stream:
                fragments.append(fragment)
            assert stream.result == ModelResponse(
                message=Message(role="assistant", content="Hello", source=TARGET),
                usage={"prompt_tokens": 3, "completion_tokens": 5},
            )
        assert fragments == ["Hel", "lo"]
        assert stream.partial.message.source == TARGET

    asyncio.run(main())
    assert [kind for kind, _ in log.events] == ["emit", "close"]
    assert log.closes() == [200]


def test_absent_usage_stays_none(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        client = make_client(log, stream_step(log, text_stream(["ok"], usage=False)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        assert fragments == ["ok"]
        assert stream.result.usage is None

    asyncio.run(main())


def test_success_at_end_of_byte_stream_without_done(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    payload = sse_open(
        choice({"role": "assistant", "content": "ok"}),
        choice({}, finish="stop"),
        usage_event(),
    )

    async def main():
        client = make_client(log, stream_step(log, payload))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        assert fragments == ["ok"]
        assert stream.result == ModelResponse(
            message=Message(role="assistant", content="ok", source=TARGET),
            usage={"prompt_tokens": 3, "completion_tokens": 5},
        )

    asyncio.run(main())
    assert log.closes() == [200]


def test_tool_call_stream_projects_calls_with_content_none(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    events = (
        choice(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "look", "arguments": ""},
                    }
                ],
            }
        ),
        choice({"tool_calls": [{"index": 0, "function": {"arguments": '{"q":'}}]}),
        choice(
            {"tool_calls": [{"index": 0, "function": {"arguments": '1}'}}]},
            finish="tool_calls",
        ),
    )

    async def main():
        client = make_client(log, stream_step(log, sse(*events)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        assert fragments == []
        assert stream.result == ModelResponse(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=(ToolCall(id="call_1", name="look", arguments='{"q":1}'),),
                source=TARGET,
            ),
            usage=None,
        )

    asyncio.run(main())


def test_extras_only_response_is_a_success(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    events = (
        choice({"role": "assistant", "reasoning_content": "thought"}),
        choice({}, finish="stop"),
    )

    async def main():
        client = make_client(log, stream_step(log, sse(*events)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        assert fragments == []
        assert stream.result.message.content is None
        assert stream.result.message.extra == {"reasoning_content": "thought"}
        assert stream.result.usage is None

    asyncio.run(main())


def test_payload_less_response_fails_as_empty(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    events = (choice({"role": "assistant"}), choice({}, finish="stop"))

    async def main():
        client = make_client(log, stream_step(log, sse(*events)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError):
            await consume(stream, [])
        assert stream.result is None
        assert stream.partial is not None
        assert stream.partial.message.source == TARGET

    asyncio.run(main())


def test_stop_finish_with_calls_fails(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    events = (
        choice(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "look", "arguments": "{}"},
                    }
                ]
            }
        ),
        choice({}, finish="stop"),
    )

    async def main():
        client = make_client(log, stream_step(log, sse(*events)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError):
            await consume(stream, [])
        assert stream.result is None

    asyncio.run(main())


# --- failure and interruption (C7) --------------------------------------------


def test_truncated_stream_fails_with_partial_retained(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
    )

    async def main():
        client = make_client(log, stream_step(log, payload))
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        with pytest.raises(ProviderError):
            await consume(stream, fragments)
        assert fragments == ["Hel", "lo"]
        assert stream.partial.message.content == "Hello"
        assert stream.partial.message.source == TARGET
        assert stream.result is None

    asyncio.run(main())
    assert log.closes() == [200]


def test_cancellation_preserves_partial_and_a_second_call_proceeds(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    gate = asyncio.Event()
    first = sse_open(choice({"role": "assistant", "content": "Hel"}))
    rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())

    async def main():
        client = make_client(
            log,
            stream_step(log, first, rest, gate=gate),
            stream_step(log, text_stream(["B"])),
        )
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        seen = asyncio.Event()

        async def consume_a():
            async with stream:
                async for fragment in stream:
                    fragments.append(fragment)
                    seen.set()

        task = asyncio.create_task(consume_a())
        await seen.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        assert fragments == ["Hel"]
        assert stream.partial.message.content == "Hel"
        assert stream.partial.message.source == TARGET
        assert stream.result is None
        assert len(log.emits()) == 1
        assert log.closes() == [200]
        # A cancelled stream cannot block a second call.
        fragments_b = []
        await consume(client.stream(CONFIG, TARGET, request(USER)), fragments_b)
        assert fragments_b == ["B"]

    asyncio.run(main())


def test_cancellation_preserves_reasoning_tool_and_text_fragments(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    gate = asyncio.Event()
    first = sse_open(
        choice({"role": "assistant", "reasoning_content": "think"}),
        choice({"tool_calls": [{"index": 0, "id": "call_1",
                                "function": {"name": "look"}}]}),
        choice({"content": "Hel"}),
    )
    rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())

    async def main():
        client = make_client(
            log, stream_step(log, first, rest, gate=gate),
            stream_step(log, text_stream(["B"])),
        )
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        seen = asyncio.Event()

        async def consume_a():
            async with stream:
                async for fragment in stream:
                    fragments.append(fragment)
                    seen.set()

        task = asyncio.create_task(consume_a())
        await seen.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        assert fragments == ["Hel"]
        partial = stream.partial
        assert partial.message.content == "Hel"
        assert partial.message.extra == {"reasoning_content": "think"}
        assert partial.message.tool_calls == (
            ToolCall(id="call_1", name="look", arguments=""),
        )
        assert partial.message.source == TARGET
        assert stream.result is None


# --- HTTP retry (C8) ----------------------------------------------------------


def test_retryable_statuses_are_retried_with_exact_backoff(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    record_sleeps(log, monkeypatch)

    async def main():
        client = make_client(
            log,
            status_step(log, 429),
            status_step(log, 500),
            status_step(log, 503),
            stream_step(log, text_stream(["ok"])),
        )
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        assert fragments == ["ok"]

    asyncio.run(main())
    assert len(log.requests) == 4
    assert len({item.content for item in log.requests}) == 1
    significant = [event for event in log.events if event[0] != "emit"]
    assert significant == [
        ("close", 429), ("sleep", 2),
        ("close", 500), ("sleep", 4),
        ("close", 503), ("sleep", 8),
        ("close", 200),
    ]


def test_cancellation_during_backoff_starts_no_later_attempt(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    reached = asyncio.Event()
    hold = asyncio.Event()

    async def fake_sleep(delay):
        log.events.append(("sleep", delay))
        reached.set()
        await hold.wait()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def main():
        client = make_client(log, status_step(log, 429))
        stream = client.stream(CONFIG, TARGET, request(USER))
        task = asyncio.create_task(stream.__aenter__())
        await reached.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        assert len(log.requests) == 1
        assert log.sleeps() == [2]
        assert log.closes() == [429]

    asyncio.run(main())


def test_exhausted_retries_raise_with_the_last_status(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    record_sleeps(log, monkeypatch)

    async def main():
        client = make_client(log, *(status_step(log, 429) for _ in range(4)))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError, match="HTTP 429") as excinfo:
            await stream.__aenter__()
        assert SECRET not in str(excinfo.value)
        assert stream.partial is None

    asyncio.run(main())
    assert len(log.requests) == 4
    significant = [event for event in log.events if event[0] != "emit"]
    assert significant == [
        ("close", 429), ("sleep", 2),
        ("close", 429), ("sleep", 4),
        ("close", 429), ("sleep", 8),
        ("close", 429),
    ]


@pytest.mark.parametrize("code", [400, 401, 404])
def test_other_statuses_fail_immediately_without_retry(monkeypatch, code):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    record_sleeps(log, monkeypatch)

    async def main():
        client = make_client(log, status_step(log, code))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError, match=f"HTTP {code}") as excinfo:
            await stream.__aenter__()
        assert SECRET not in str(excinfo.value)
        # A failed entry still consumes the instance; re-entry cannot re-open
        # the bounded retry.
        with pytest.raises(ValueError):
            await stream.__aenter__()

    asyncio.run(main())
    assert len(log.requests) == 1
    assert log.requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert log.sleeps() == []
    assert log.closes() == [code]


def test_connection_failure_is_not_retried(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    record_sleeps(log, monkeypatch)

    async def main():
        client = make_client(log, failure_step(httpx.ConnectError("down")))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError, match="connection") as excinfo:
            await stream.__aenter__()
        assert SECRET not in str(excinfo.value)

    asyncio.run(main())
    assert len(log.requests) == 1
    assert log.sleeps() == []
    assert log.closes() == []


def test_accepted_stream_failure_is_not_retried(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    record_sleeps(log, monkeypatch)

    async def main():
        client = make_client(log, stream_step(log, b"data: not-json\n\n"))
        stream = client.stream(CONFIG, TARGET, request(USER))
        with pytest.raises(ProviderError) as excinfo:
            await consume(stream, [])
        assert "not-json" not in str(excinfo.value)

    asyncio.run(main())
    assert len(log.requests) == 1
    assert log.sleeps() == []
    assert log.closes() == [200]


# --- shared client and context exit (C9) --------------------------------------


def test_held_stream_does_not_block_a_second_call(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    gate = asyncio.Event()
    first = sse_open(choice({"role": "assistant", "content": "Hel"}))
    rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())

    async def main():
        client = make_client(
            log,
            stream_step(log, first, rest, gate=gate),
            stream_step(log, text_stream(["B"])),
        )
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        seen = asyncio.Event()

        async def consume_a():
            async with stream:
                async for fragment in stream:
                    fragments.append(fragment)
                    seen.set()

        task = asyncio.create_task(consume_a())
        await seen.wait()
        fragments_b = []
        await consume(client.stream(CONFIG, TARGET, request(USER)), fragments_b)
        assert fragments_b == ["B"]
        gate.set()
        await task
        assert fragments == ["Hel", "lo"]
        assert stream.result.message.content == "Hello"

    asyncio.run(main())


def test_concurrent_calls_are_independent(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()

    async def main():
        barrier = asyncio.Barrier(2)

        def staged(fragment):
            def step():
                async def source():
                    await barrier.wait()
                    yield text_stream([fragment])

                return TrackedResponse(200, source(), log)

            return step

        client = make_client(log, staged("one"), staged("two"))

        async def call(text):
            fragments = []
            stream = client.stream(CONFIG, TARGET, request(Message(role="user", content=text)))
            await consume(stream, fragments)
            return fragments

        one, two = await asyncio.gather(call("a"), call("b"))
        assert one == ["one"] and two == ["two"]

    asyncio.run(main())
    bodies = log.bodies()
    assert bodies[0]["messages"][0]["content"] == "a"
    assert bodies[1]["messages"][0]["content"] == "b"


def test_early_exit_closes_without_draining(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    first = sse_open(choice({"role": "assistant", "content": "Hel"}))
    rest = sse(choice({"content": "lo"}), choice({}, finish="stop"), usage_event())

    async def main():
        client = make_client(
            log, stream_step(log, first, rest), stream_step(log, text_stream(["B"]))
        )
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        async with stream:
            async for fragment in stream:
                fragments.append(fragment)
                break
        assert fragments == ["Hel"]
        assert len(log.emits()) == 1
        assert log.closes() == [200]
        assert stream.partial.message.content == "Hel"
        assert stream.result is None
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        fragments_b = []
        await consume(client.stream(CONFIG, TARGET, request(USER)), fragments_b)
        assert fragments_b == ["B"]

    asyncio.run(main())


# --- capture, assembly, and replay (C2-C5) ------------------------------------


def test_streamed_result_replays_extras_into_the_next_request(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    log = Recording()
    events = (
        choice(
            {
                "role": "assistant",
                "reasoning_content": "think",
                "extra_content": {"google": {"thought_signature": "sig"}},
            }
        ),
        choice({"content": "answer"}),
        choice({}, finish="stop"),
    )

    async def main():
        client = make_client(
            log, stream_step(log, sse(*events)), stream_step(log, text_stream(["ok"]))
        )
        stream = client.stream(CONFIG, TARGET, request(USER))
        fragments = []
        await consume(stream, fragments)
        result = stream.result
        assert result.message.source == TARGET
        assert result.message.extra == {
            "reasoning_content": "think",
            "extra_content": {"google": {"thought_signature": "sig"}},
        }
        follow_up = ModelRequest(messages=(USER, result.message))
        fragments_b = []
        await consume(client.stream(CONFIG, TARGET, follow_up), fragments_b)
        assert fragments_b == ["ok"]

    asyncio.run(main())
    replayed = log.bodies()[1]["messages"][-1]
    assert replayed == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "think",
        "extra_content": {"google": {"thought_signature": "sig"}},
    }
    assert "source" not in replayed
    assert log.bodies()[1] == encode_request(
        model_key="qwen", request=ModelRequest(
            messages=(USER, Message(role="assistant", content="answer",
                                    source=TARGET,
                                    extra={"reasoning_content": "think",
                                           "extra_content": {"google": {"thought_signature": "sig"}}}))),
        target=TARGET,
    )