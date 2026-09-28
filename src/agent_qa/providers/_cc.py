"""Private Chat Completions request encoding."""

import math

from agent_qa.model import (
    JsonObject,
    JsonValue,
    Message,
    ModelRef,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)

_MESSAGE_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id"})
_CALL_KEYS = frozenset({"index", "id", "type", "function"})
_FUNCTION_KEYS = frozenset({"name", "arguments"})


def encode_request(*, model_key: str, request: ModelRequest, target: ModelRef) -> dict:
    """Build the complete JSON request body for one Chat Completions call.

    Validates every caller-supplied value and raises ``ValueError`` for invalid
    caller input. JSON data is copied out of the input values, so later mutation
    of caller-held extras cannot affect the returned body.
    """
    if not request.messages:
        raise ValueError("invalid request: at least one message is required")
    names = set()
    for definition in request.tools:
        if not definition.name:
            raise ValueError("invalid request: tool definitions require a nonempty name")
        if definition.name in names:
            raise ValueError("invalid request: tool definition names must be unique")
        names.add(definition.name)
    body = {
        "model": model_key,
        "messages": [_message_part(message, target) for message in request.messages],
        "stream": True,
        "n": 1,
        "stream_options": {"include_usage": True},
    }
    if request.tools:
        body["tools"] = [_tool_part(definition) for definition in request.tools]
        body["tool_choice"] = "auto"
    return body


def _message_part(message: Message, target: ModelRef) -> JsonObject:
    if message.role != "assistant":
        return _plain_part(message)
    return _assistant_part(message, target)


def _plain_part(message: Message) -> JsonObject:
    if message.source is not None:
        raise ValueError("invalid request: only assistant messages carry a source")
    if message.extra:
        raise ValueError("invalid request: only assistant messages carry extras")
    if message.tool_calls:
        raise ValueError("invalid request: only assistant messages carry tool calls")
    if message.tool_call_id is not None and message.role != "tool":
        raise ValueError("invalid request: tool_call_id is permitted only on tool messages")
    if not isinstance(message.content, str):
        raise ValueError("invalid request: non-assistant content must be a string")
    part: JsonObject = {"role": message.role, "content": message.content}
    if message.role == "tool":
        if not message.tool_call_id:
            raise ValueError("invalid request: tool messages require a nonempty call ID")
        part["tool_call_id"] = message.tool_call_id
    return part


def _assistant_part(message: Message, target: ModelRef) -> JsonObject:
    if message.content is not None and not isinstance(message.content, str):
        raise ValueError("invalid request: assistant content must be a string or None")
    if message.tool_call_id is not None:
        raise ValueError("invalid request: tool_call_id is permitted only on tool messages")
    extras = _copy_object(message.extra)
    for key in extras:
        if key in _MESSAGE_KEYS:
            raise ValueError("invalid request: message extras must not name canonical fields")
    part: JsonObject = {"role": "assistant"}
    if message.content is not None:
        part["content"] = message.content
    if message.tool_calls:
        seen = set()
        calls: list[JsonValue] = []
        for call in message.tool_calls:
            if call.id in seen:
                raise ValueError("invalid request: tool call IDs must be distinct within one message")
            seen.add(call.id)
            calls.append(_call_part(call, message.source == target))
        part["tool_calls"] = calls
    if message.source == target:
        part.update(extras)
    return part


def _call_part(call: ToolCall, same_source: bool) -> JsonObject:
    if not call.id:
        raise ValueError("invalid request: tool calls require a nonempty ID")
    if not call.name:
        raise ValueError("invalid request: tool calls require a nonempty name")
    if not isinstance(call.arguments, str):
        raise ValueError("invalid request: tool call arguments must be a string")
    call_extras = _copy_object(call.extra)
    function_extras = _copy_object(call.function_extra)
    for key in call_extras:
        if key in _CALL_KEYS:
            raise ValueError("invalid request: tool call extras must not name canonical fields")
    for key in function_extras:
        if key in _FUNCTION_KEYS:
            raise ValueError("invalid request: tool function extras must not name canonical fields")
    function: JsonObject = {"name": call.name, "arguments": call.arguments}
    if same_source:
        function.update(function_extras)
    part: JsonObject = {"id": call.id, "type": "function", "function": function}
    if same_source:
        part.update(call_extras)
    return part


def _tool_part(definition: ToolDefinition) -> JsonObject:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": _copy_object(definition.parameters),
        },
    }


def _json_copy(value: JsonValue) -> JsonValue:
    """Validate one JSON value and return an independent copy of it."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("invalid request: JSON values must be finite numbers")
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict):
        return _copy_object(value)
    raise ValueError("invalid request: values must belong to the JSON value domain")


def _copy_object(value: JsonObject) -> JsonObject:
    """Validate one JSON object and return an independent copy of it."""
    if not isinstance(value, dict):
        raise ValueError("invalid request: JSON objects must be dicts")
    copy: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("invalid request: JSON object keys must be strings")
        copy[key] = _json_copy(item)
    return copy
