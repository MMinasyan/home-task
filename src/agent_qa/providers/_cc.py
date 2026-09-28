"""Private Chat Completions request encoding and stream assembly."""

import copy
import json
import math

from agent_qa.model import (
    JsonObject,
    JsonValue,
    Message,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ProviderError,
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


def _accumulate(current: JsonValue, fragment: JsonValue) -> JsonValue:
    """One accumulation rule for streamed fragments of an opaque value.

    Strings concatenate and arrays append only when both sides are of that
    kind; a first value, a kind change, or a scalar/object fragment replaces
    the accumulated value.
    """
    if isinstance(current, str) and isinstance(fragment, str):
        return current + fragment
    if isinstance(current, list) and isinstance(fragment, list):
        return current + fragment
    return fragment


def _number(value: object) -> bool:
    """Whether the value is a finite nonnegative JSON number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _absorb(target: JsonObject, source: JsonObject, canonical: frozenset) -> None:
    """Accumulate non-canonical fragments from source into target."""
    for key, fragment in source.items():
        if key not in canonical:
            target[key] = _accumulate(target.get(key), fragment)


class StreamAssembler:
    """Assembles Chat Completions stream events into one response.

    Callers feed every dispatched event's data value in order, then call
    ``finalize``. ``feed`` returns each nonempty ``delta.content`` fragment to
    surface and raises ``ProviderError`` on malformed data; ``snapshot``
    returns a detached copy of everything assembled so far, or ``None``
    before any choice delta is accepted; ``finalize`` enforces the terminal
    rules and returns the completed response.
    """

    def __init__(self):
        self._seen_choice = False
        self._finish = None
        self._content = None
        self._extras: JsonObject = {}
        self._calls: dict[int | float, dict] = {}
        self._usage = None

    def feed(self, data: str) -> str | None:
        try:
            event = json.loads(data)
        except ValueError:
            raise ProviderError("invalid stream: event is not valid JSON") from None
        if not isinstance(event, dict):
            raise ProviderError("invalid stream: event is not a JSON object")
        if event.get("error") is not None:
            raise ProviderError("invalid stream: the provider reported an error")
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise ProviderError("invalid stream: choices is not an array")
        if choices and self._finish is not None:
            raise ProviderError("invalid stream: chunk after the finish marker carries choices")
        surfaced = ""
        for choice in choices:
            if not isinstance(choice, dict):
                raise ProviderError("invalid stream: choice is not an object")
            index = choice.get("index", 0)
            if not _number(index) or index != 0:
                raise ProviderError("invalid stream: choice index is outside n:1")
            self._seen_choice = True
            delta = choice.get("delta")
            if delta is None:
                delta = {}
            elif not isinstance(delta, dict):
                raise ProviderError("invalid stream: delta is not an object")
            finish = choice.get("finish_reason")
            if finish is not None:
                self._finish = finish
            role = delta.get("role")
            if role is not None and role != "assistant":
                raise ProviderError("invalid stream: delta role is not assistant")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ProviderError("invalid stream: content delta is not a string")
                self._content = content if self._content is None else self._content + content
                surfaced += content
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    raise ProviderError("invalid stream: tool_calls is not an array")
                for position, entry in enumerate(tool_calls):
                    self._feed_call(position, entry)
            _absorb(self._extras, delta, _MESSAGE_KEYS)
        usage = event.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ProviderError("invalid stream: usage is not an object")
            self._usage = usage
        return surfaced or None

    def _feed_call(self, position: int, entry: object) -> None:
        if not isinstance(entry, dict):
            raise ProviderError("invalid stream: tool call is not an object")
        index = entry.get("index", position)
        if not _number(index):
            raise ProviderError("invalid stream: tool call index is not a nonnegative number")
        call = self._calls.get(index)
        if call is None:
            call = {"id": "", "name": "", "arguments": "", "extra": {}, "function_extra": {}}
            self._calls[index] = call
        identifier = entry.get("id")
        if identifier is not None:
            if not isinstance(identifier, str):
                raise ProviderError("invalid stream: tool call id fragment is not a string")
            call["id"] += identifier
        kind = entry.get("type")
        if kind is not None and kind != "function":
            raise ProviderError("invalid stream: tool call type is not function")
        function = entry.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ProviderError("invalid stream: tool call function is not an object")
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str):
                    raise ProviderError("invalid stream: tool name fragment is not a string")
                call["name"] += name
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise ProviderError("invalid stream: tool arguments fragment is not a string")
                call["arguments"] += arguments
            _absorb(call["function_extra"], function, _FUNCTION_KEYS)
        _absorb(call["extra"], entry, _CALL_KEYS)

    def _project_calls(self) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(
                id=call["id"],
                name=call["name"],
                arguments=call["arguments"],
                extra=call["extra"],
                function_extra=call["function_extra"],
            )
            for call in (self._calls[index] for index in sorted(self._calls))
        )

    def _response(self) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role="assistant",
                content=self._content,
                tool_calls=self._project_calls(),
                extra=self._extras,
            ),
            usage=self._usage,
        )

    def snapshot(self) -> ModelResponse | None:
        if not self._seen_choice:
            return None
        return copy.deepcopy(self._response())

    def finalize(self) -> ModelResponse:
        if not self._seen_choice:
            raise ProviderError("invalid stream: response carried no choice")
        if self._finish not in ("stop", "tool_calls"):
            raise ProviderError("invalid stream: response ended without a successful finish reason")
        calls = self._project_calls()
        if calls:
            if self._finish == "stop":
                raise ProviderError("invalid stream: stop finish contradicts assembled tool calls")
            for call in calls:
                if not call.id or not call.name:
                    raise ProviderError("invalid stream: tool call lacks a complete identity")
            ids = {call.id for call in calls}
            if len(ids) != len(calls):
                raise ProviderError("invalid stream: tool call IDs must be distinct")
        if not self._content and not calls and not self._extras:
            raise ProviderError("invalid stream: response carries no payload")
        return self._response()
