"""Opt-in live gate: a real endpoint must stream a tool call and replay reasoning."""

import asyncio
import json
import os
import pathlib

import httpx
import pytest

from agent_qa.model import (
    Message,
    ModelRef,
    ModelRequest,
    ProviderError,
    ToolDefinition,
)
from agent_qa.providers.chat_completions import ChatCompletionsClient
from agent_qa.providers.config import load_config, resolve_model

_ENV = ("QA_LIVE_CONFIG", "QA_LIVE_PROVIDER", "QA_LIVE_MODEL")

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in _ENV),
    reason=(
        "opt-in live gate: set QA_LIVE_CONFIG, QA_LIVE_PROVIDER, and "
        "QA_LIVE_MODEL to run it; a skipped live gate is not a pass"
    ),
)

TOOL_NAME = "echo_value"
FIXED_VALUE = "agent-qa-live-gate"
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}

_CANONICAL_MESSAGE_FIELDS = frozenset({"role", "content", "tool_calls", "tool_call_id"})
_UNREACHABLE = "connection to the provider failed"


def replayed_extras(message):
    """The extension fields of one captured assistant message."""
    return {key: value for key, value in message.items() if key not in _CANONICAL_MESSAGE_FIELDS}


def assistant_messages(body):
    return [part for part in body["messages"] if part["role"] == "assistant"]


async def exchange(client, config, ref, request, captured, blocked=False):
    """Send one request to completion; return its fragments and final result.

    Only the first exchange opens before the endpoint has accepted a request,
    so only it reports a connection failure at entry as a blocked gate; every
    other failure, including later connection failures, fails the gate.
    """
    before = len(captured)
    stream = client.stream(config, ref, request)
    try:
        await stream.__aenter__()
    except ProviderError as error:
        if blocked and str(error) == _UNREACHABLE:
            pytest.skip(
                "live gate blocked: the configured endpoint is unreachable; "
                "a blocked gate is not a pass"
            )
        raise
    fragments = []
    try:
        async for fragment in stream:
            fragments.append(fragment)
    finally:
        await stream.__aexit__()
    assert len(captured) > before, "every exchange must capture its outgoing request body"
    return fragments, stream.result


def test_live_streamed_tool_call_and_reasoning_replay():
    config = load_config(pathlib.Path(os.environ["QA_LIVE_CONFIG"]))
    ref = ModelRef(os.environ["QA_LIVE_PROVIDER"], os.environ["QA_LIVE_MODEL"])
    _, model = resolve_model(config, ref)
    reasoning_field = model.reasoning_field
    assert reasoning_field, "the selected model must configure a reasoning_field"

    captured = []

    async def capture(request):
        captured.append(request.content)

    async def main():
        # A generous timeout: a shared local endpoint may queue before the first token.
        async with httpx.AsyncClient(
            event_hooks={"request": [capture]}, timeout=httpx.Timeout(60.0)
        ) as http:
            client = ChatCompletionsClient(http)

            first = ModelRequest(
                messages=(
                    Message(
                        role="user",
                        content=(
                            f"Call the {TOOL_NAME} function exactly once, passing "
                            f'"{FIXED_VALUE}" as its value, and do not answer '
                            "before calling it."
                        ),
                    ),
                ),
                tools=(
                    ToolDefinition(
                        name=TOOL_NAME,
                        description="Return the value it is given. A harmless test function.",
                        parameters=TOOL_PARAMETERS,
                    ),
                ),
            )
            _, response = await exchange(client, config, ref, first, captured, blocked=True)
            calls = response.message.tool_calls
            assert len(calls) == 1, "the model must request exactly the advertised call"
            call = calls[0]
            assert call.id, "the tool call must carry an id"
            assert call.name == TOOL_NAME, "the tool call must name the advertised function"
            arguments = json.loads(call.arguments)
            assert isinstance(arguments, dict)
            assert arguments.get("value") == FIXED_VALUE, "the call must pass the fixed test value"
            assert response.message.extra.get(reasoning_field), (
                f"the response must carry a nonempty {reasoning_field}"
            )

            second = ModelRequest(
                messages=first.messages
                + (
                    response.message,
                    Message(role="tool", content=FIXED_VALUE, tool_call_id=call.id),
                    Message(role="user", content="Now give your final answer."),
                )
            )
            fragments, final = await exchange(client, config, ref, second, captured)
            assert fragments, "the final answer must stream text"
            assert "".join(fragments) == final.message.content
            assert final.message.extra.get(reasoning_field), (
                f"the final answer must carry a nonempty {reasoning_field}"
            )

            body = json.loads(captured[-1])
            replayed = assistant_messages(body)
            assert len(replayed) == 1, "the retained tool-call response must be replayed"
            assert replayed_extras(replayed[0]) == response.message.extra, (
                "the next request must replay the captured reasoning and extension fields exactly"
            )

            third = ModelRequest(
                messages=second.messages
                + (
                    final.message,
                    Message(
                        role="user",
                        content="In one short sentence, repeat the value the tool returned.",
                    ),
                )
            )
            fragments, follow_up = await exchange(client, config, ref, third, captured)
            assert fragments, "the follow-up must stream text"
            assert "".join(fragments) == follow_up.message.content

            body = json.loads(captured[-1])
            replayed = assistant_messages(body)
            assert replayed_extras(replayed[-1]) == final.message.extra, (
                "the follow-up request must replay the captured reasoning and extension fields exactly"
            )

    asyncio.run(main())
