"""Chat Completions request encoding: envelope, tools, replay, and validation."""

import copy
import json

import pytest

from agent_qa.model import Message, ModelRef, ModelRequest, ToolCall, ToolDefinition
from agent_qa.providers._cc import encode_request

TARGET = ModelRef(provider="local", model="qwen")
OTHER = ModelRef(provider="local", model="other")
SECRET = "SENTINEL-7c1f-secret"

CALL = ToolCall(id="call_1", name="lookup", arguments='{"q": "x"}')
TOOL = ToolDefinition(
    name="lookup",
    description="Look things up.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


def encode(*messages, tools=(), target=TARGET, model_key="qwen"):
    request = ModelRequest(messages=tuple(messages), tools=tuple(tools))
    return encode_request(model_key=model_key, request=request, target=target)


def request(*messages, tools=()):
    return ModelRequest(messages=tuple(messages), tools=tuple(tools))


def assistant(**fields):
    """An assistant message at the default replay source with optional field replacements."""
    fields.setdefault("source", TARGET)
    return request(Message(role="assistant", **fields))


def call(**overrides):
    """A well-formed tool call with optional field replacements."""
    fields = {"id": "call_1", "name": "lookup", "arguments": "{}"}
    return ToolCall(**{**fields, **overrides})


def definition(**overrides):
    """A well-formed tool definition with optional field replacements."""
    fields = {"name": "lookup", "description": "d", "parameters": {}}
    return ToolDefinition(**{**fields, **overrides})


# --- fixed envelope ----------------------------------------------------------


def test_fixed_envelope_omits_tool_keys_and_output_cap_without_tools():
    body = encode(Message(role="user", content="hello"))
    assert body == {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "n": 1,
        "stream_options": {"include_usage": True},
    }


def test_fixed_envelope_with_tools_has_exact_shape():
    body = encode(Message(role="user", content="hi"), tools=(TOOL,))
    assert body == {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "n": 1,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look things up.",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "auto",
    }


def test_envelope_uses_the_configured_model_key():
    body = encode(Message(role="user", content="hello"), model_key="other-key")
    assert body["model"] == "other-key"


def test_tool_call_id_emitted_only_on_tool_messages():
    tool = encode(Message(role="tool", content="result", tool_call_id="call_1"))["messages"][0]
    assert tool == {"role": "tool", "content": "result", "tool_call_id": "call_1"}
    user = encode(Message(role="user", content="hi"))["messages"][0]
    assert "tool_call_id" not in user
    system = encode(Message(role="system", content="be brief"))["messages"][0]
    assert system == {"role": "system", "content": "be brief"}


# --- assistant content and call keys -----------------------------------------


def test_assistant_content_none_omits_the_key_even_with_tool_calls():
    part = encode(Message(role="assistant", tool_calls=(CALL,), source=TARGET))["messages"][0]
    assert "content" not in part
    assert part["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}
    ]


def test_assistant_empty_content_is_emitted_as_empty_string():
    part = encode(Message(role="assistant", content="", source=TARGET))["messages"][0]
    assert part["content"] == ""


def test_assistant_without_calls_omits_the_tool_calls_key():
    part = encode(Message(role="assistant", content="answer", source=TARGET))["messages"][0]
    assert part == {"role": "assistant", "content": "answer"}


def test_parallel_calls_keep_order_and_repeat_names():
    calls = (
        ToolCall(id="call_1", name="lookup", arguments="{}"),
        ToolCall(id="call_2", name="lookup", arguments='{"q": "b"}'),
    )
    part = encode(Message(role="assistant", tool_calls=calls, source=TARGET))["messages"][0]
    assert [call["id"] for call in part["tool_calls"]] == ["call_1", "call_2"]
    assert [call["function"]["name"] for call in part["tool_calls"]] == ["lookup", "lookup"]


# --- source-gated extras ------------------------------------------------------


def test_same_source_replay_emits_extras_at_every_scope():
    call = ToolCall(
        id="call_1",
        name="lookup",
        arguments='{"q": "x"}',
        extra={"call_ext": {"deep": [True, None, 1.5]}},
        function_extra={"fn_ext": "Ω"},
    )
    message = Message(
        role="assistant",
        content="answer",
        tool_calls=(call,),
        source=TARGET,
        extra={"reasoning_content": "միտք"},
    )
    assert encode(message) == {
        "model": "qwen",
        "messages": [
            {
                "role": "assistant",
                "content": "answer",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "x"}', "fn_ext": "Ω"},
                        "call_ext": {"deep": [True, None, 1.5]},
                    }
                ],
                "reasoning_content": "միտք",
            }
        ],
        "stream": True,
        "n": 1,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.parametrize("source", [None, OTHER], ids=["sourceless", "foreign"])
def test_non_matching_sources_emit_canonical_fields_only(source):
    call = ToolCall(
        id="call_1",
        name="lookup",
        arguments='{"q": "x"}',
        extra={"call_ext": 1},
        function_extra={"fn_ext": 2},
    )
    message = Message(
        role="assistant",
        content="",
        tool_calls=(call,),
        source=source,
        extra={"reasoning_content": "hidden"},
    )
    part = encode(message)["messages"][0]
    assert part == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}
        ],
    }


# --- lossless round trip without input mutation -------------------------------


def test_round_trip_preserves_opaque_values_without_mutating_input():
    extra = {"reasoning_content": "մտքեր թղթի վրա", "signature": "Σ|abc"}
    call = ToolCall(
        id="call_1",
        name="lookup",
        arguments='{"q": "տվյալ", "raw": "unbalanced {"}',
        extra={"call_ext": [1, 2.5, None, True, {"k": ["v"]}]},
        function_extra={"thought_signature": "Ω≈ç"},
    )
    message = Message(
        role="assistant",
        content=None,
        tool_calls=(call,),
        source=TARGET,
        extra=extra,
    )
    snapshot = {
        "extra": copy.deepcopy(extra),
        "call_extra": copy.deepcopy(call.extra),
        "function_extra": copy.deepcopy(call.function_extra),
    }
    body = encode(message)
    restored = json.loads(json.dumps(body))
    assert restored == {
        "model": "qwen",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"q": "տվյալ", "raw": "unbalanced {"}',
                            "thought_signature": "Ω≈ç",
                        },
                        "call_ext": [1, 2.5, None, True, {"k": ["v"]}],
                    }
                ],
                "reasoning_content": "մտքեր թղթի վրա",
                "signature": "Σ|abc",
            }
        ],
        "stream": True,
        "n": 1,
        "stream_options": {"include_usage": True},
    }
    assert {"extra": extra, "call_extra": call.extra, "function_extra": call.function_extra} == snapshot


def test_extras_are_copied_so_later_mutation_cannot_change_the_body():
    nested = {"deep": [1, 2]}
    call_ext = {"k": [True, None]}
    function_ext = {"fn_ext": "v"}
    extra = {"reasoning_content": "x", "nested": nested}
    call_extra = ToolCall(
        id="call_1",
        name="lookup",
        arguments="{}",
        extra={"call_ext": call_ext},
        function_extra=function_ext,
    )
    message = Message(
        role="assistant",
        content="answer",
        tool_calls=(call_extra,),
        source=TARGET,
        extra=extra,
    )
    body = encode(message)
    extra["reasoning_content"] = "mutated"
    nested["deep"].append(3)
    call_ext["k"].append(False)
    function_ext["fn_ext"] = "mutated"
    part = body["messages"][0]
    assert part["reasoning_content"] == "x"
    assert part["nested"] == {"deep": [1, 2]}
    assert part["tool_calls"][0]["call_ext"] == {"k": [True, None]}
    assert part["tool_calls"][0]["function"]["fn_ext"] == "v"


# --- invalid caller input ------------------------------------------------------

INVALID_REQUESTS = [
    ("no messages", lambda: request()),
    ("system content none", lambda: request(Message(role="system"))),
    ("user content none", lambda: request(Message(role="user"))),
    ("tool content none", lambda: request(Message(role="tool", tool_call_id="call_1"))),
    ("tool content not a string", lambda: request(Message(role="tool", content=["r"], tool_call_id="call_1"))),
    ("tool call id missing", lambda: request(Message(role="tool", content="r"))),
    ("tool call id empty", lambda: request(Message(role="tool", content="r", tool_call_id=""))),
    ("tool call id on user", lambda: request(Message(role="user", content="hi", tool_call_id="call_1"))),
    ("tool call id on assistant", lambda: request(Message(role="assistant", content="hi", tool_call_id="call_1"))),
    ("tool calls on user", lambda: request(Message(role="user", content="hi", tool_calls=(CALL,)))),
    ("source on user", lambda: request(Message(role="user", content="hi", source=TARGET))),
    ("extras on user", lambda: request(Message(role="user", content="hi", extra={"x": 1}))),
    ("assistant content not a string", lambda: assistant(content=5)),
    ("assistant extras collide with role", lambda: assistant(extra={"role": "user"})),
    ("assistant extras collide with content", lambda: assistant(extra={"content": "x"})),
    ("assistant extras collide with tool_calls", lambda: assistant(extra={"tool_calls": []})),
    ("assistant extras collide with tool_call_id", lambda: assistant(extra={"tool_call_id": "x"})),
    ("call id empty", lambda: assistant(tool_calls=(call(id=""),))),
    ("call name empty", lambda: assistant(tool_calls=(call(name=""),))),
    ("call arguments not a string", lambda: assistant(tool_calls=(call(arguments=5),))),
    ("duplicate call ids", lambda: assistant(tool_calls=(call(), call(id="call_1", name="other")))),
    ("call extra collides with index", lambda: assistant(tool_calls=(call(extra={"index": 0}),))),
    ("call extra collides with id", lambda: assistant(tool_calls=(call(extra={"id": "x"}),))),
    ("call extra collides with type", lambda: assistant(tool_calls=(call(extra={"type": "x"}),))),
    ("call extra collides with function", lambda: assistant(tool_calls=(call(extra={"function": {}}),))),
    ("function extra collides with name", lambda: assistant(tool_calls=(call(function_extra={"name": "x"}),))),
    (
        "function extra collides with arguments",
        lambda: assistant(tool_calls=(call(function_extra={"arguments": ""}),)),
    ),
    ("tool definition name empty", lambda: request(Message(role="user", content="hi"), tools=(definition(name=""),))),
    (
        "duplicate tool definition names",
        lambda: request(Message(role="user", content="hi"), tools=(TOOL, definition(description="again"))),
    ),
    ("message extra non-finite", lambda: assistant(extra={"x": float("inf")})),
    ("message extra non-finite nested", lambda: assistant(extra={"a": [{"b": float("nan")}]})),
    ("message extra non-string key", lambda: assistant(extra={1: "x"})),
    ("message extra unsupported object", lambda: assistant(extra={"x": object()})),
    ("message extra empty list", lambda: assistant(extra=[])),
    ("message extra none", lambda: assistant(extra=None)),
    ("message extra not a container", lambda: assistant(extra=5)),
    ("call extra non-finite", lambda: assistant(tool_calls=(call(extra={"x": float("nan")}),))),
    ("function extra non-finite", lambda: assistant(tool_calls=(call(function_extra={"x": float("-inf")}),))),
    (
        "tool parameters non-finite",
        lambda: request(Message(role="user", content="hi"), tools=(definition(parameters={"a": float("inf")}),)),
    ),
    (
        "tool parameters empty list",
        lambda: request(Message(role="user", content="hi"), tools=(definition(parameters=[]),)),
    ),
    ("extra outside domain on foreign source", lambda: assistant(source=OTHER, extra={"x": object()})),
]


@pytest.mark.parametrize(
    "name,build", INVALID_REQUESTS, ids=[row[0] for row in INVALID_REQUESTS]
)
def test_invalid_request_is_rejected(name, build):
    with pytest.raises(ValueError):
        encode_request(model_key="qwen", request=build(), target=TARGET)


def test_boundary_errors_hide_input_values():
    message = Message(role="assistant", source=TARGET, extra={SECRET: float("nan")})
    with pytest.raises(ValueError) as failure:
        encode(message)
    assert SECRET not in str(failure.value)
    with pytest.raises(ValueError) as failure:
        encode(Message(role="user", content="hi", extra={SECRET: 1}))
    assert SECRET not in str(failure.value)
