"""Transport value types for model requests and responses."""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Union

# JSON-encodable data: null, bool, int, finite float, string, or a list or object
# recursively built from those values; object keys are strings.
JsonValue: TypeAlias = Union[
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]
]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class ModelRef:
    """The configured provider and model identity of a request or response."""

    provider: str
    model: str


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by an assistant message.

    ``arguments`` is an opaque string. ``extra`` holds provider extension fields at
    the call scope, ``function_extra`` at the function scope.
    """

    id: str
    name: str
    arguments: str
    extra: JsonObject = field(default_factory=dict)
    function_extra: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One conversation message.

    ``source`` is the configured provider/model that produced an assistant message,
    not the provider's reported model alias; it is provenance and never a wire field.
    Non-assistant messages leave it unset.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    source: ModelRef | None = None
    extra: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    """A function tool advertised to the model."""

    name: str
    description: str
    parameters: JsonObject


@dataclass(frozen=True)
class ModelRequest:
    """The messages to send, plus any advertised tools."""

    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True)
class ModelResponse:
    """A completed response: its message and the reported usage, if any."""

    message: Message
    usage: JsonObject | None


class ProviderError(Exception):
    """A provider failure; the message identifies the failure category."""
