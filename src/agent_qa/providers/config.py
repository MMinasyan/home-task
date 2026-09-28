"""Provider configuration: file loading and model reference resolution."""

import json
import pathlib
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_qa.model import ModelRef

_BASE_URL_ERROR = (
    "base_url must be an absolute HTTP(S) URL with a host, a valid port if "
    "present, and no credentials, query, or fragment"
)
_CANONICAL_MESSAGE_FIELDS = frozenset({"role", "content", "tool_calls", "tool_call_id"})

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)


class ModelConfig(BaseModel):
    """Per-model metadata for one configured model."""

    model_config = _MODEL_CONFIG

    context_window: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    reasoning_field: str | None = Field(default=None, min_length=1)

    @field_validator("reasoning_field")
    @classmethod
    def _reasoning_field_must_not_collide(cls, value: str | None) -> str | None:
        if value in _CANONICAL_MESSAGE_FIELDS:
            raise ValueError("reasoning_field must not name a canonical message field")
        return value


class ProviderConfig(BaseModel):
    """One provider endpoint and its models."""

    model_config = _MODEL_CONFIG

    base_url: str
    api_key_env: str | None = Field(default=None, min_length=1)
    models: dict[Annotated[str, Field(min_length=1)], ModelConfig] = Field(min_length=1)

    @field_validator("base_url")
    @classmethod
    def _base_url_must_be_absolute_http(cls, value: str) -> str:
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError:
            raise ValueError(_BASE_URL_ERROR) from None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(_BASE_URL_ERROR)
        if parts.username is not None or parts.password is not None:
            raise ValueError(_BASE_URL_ERROR)
        if parts.query or parts.fragment:
            raise ValueError(_BASE_URL_ERROR)
        return value


class ProvidersConfig(BaseModel):
    """The closed set of configured providers."""

    model_config = _MODEL_CONFIG

    providers: dict[Annotated[str, Field(min_length=1)], ProviderConfig] = Field(min_length=1)


def load_config(path: pathlib.Path) -> ProvidersConfig:
    """Read exactly the UTF-8 JSON file at *path* as a provider configuration.

    Every read, parse, or validation failure raises ValueError with a message built
    from error locations and categories only.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError("configuration file is not valid UTF-8") from None
    except OSError:
        raise ValueError("configuration file cannot be read") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("configuration file is not valid JSON") from None
    try:
        return ProvidersConfig.model_validate(data)
    except ValidationError as error:
        raise ValueError(_validation_message(error)) from None


def resolve_model(config: ProvidersConfig, ref: ModelRef) -> tuple[ProviderConfig, ModelConfig]:
    """Return the selected provider and model for *ref*."""
    provider = config.providers.get(ref.provider)
    if provider is None:
        raise ValueError("unknown provider or model")
    model = provider.models.get(ref.model)
    if model is None:
        raise ValueError("unknown provider or model")
    return provider, model


def _validation_message(error: ValidationError) -> str:
    """Describe validation failures by location and category only."""
    parts = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) or "''" for part in item["loc"]) or "<root>"
        parts.append(f"{location}: {item['type']}")
    return "invalid configuration: " + "; ".join(parts)
