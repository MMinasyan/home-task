"""Configuration loading, validation, model resolution, and value-contract checks."""

import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_qa.model import (
    Message,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)
from agent_qa.providers.config import load_config, resolve_model

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "config.example.json"
LOCAL_URL = "http://127.0.0.1:8080/v1"
SECRET = "SENTINEL-3f2a9c-secret"


def payload(base_url=LOCAL_URL, provider=None, model=None, top=None):
    """A one-provider, one-model document with optional field replacements."""
    provider_fields = {"base_url": base_url} if provider is None else provider
    model_fields = {"context_window": 32768} if model is None else model
    document = {"providers": {"local": {**provider_fields, "models": {"qwen": model_fields}}}}
    if top is not None:
        document.update(top)
    return document


def write_config(tmp_path, document):
    path = tmp_path / "config.json"
    if isinstance(document, bytes):
        path.write_bytes(document)
    elif isinstance(document, str):
        path.write_text(document, encoding="utf-8")
    else:
        path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --- valid configurations ----------------------------------------------------


def test_loads_valid_configuration(tmp_path):
    document = payload(
        provider={"base_url": LOCAL_URL, "api_key_env": "LOCAL_API_KEY"},
        model={
            "context_window": 32768,
            "max_output_tokens": 4096,
            "reasoning_field": "reasoning_content",
        },
    )
    config = load_config(write_config(tmp_path, document))
    provider = config.providers["local"]
    assert provider.base_url == LOCAL_URL
    assert provider.api_key_env == "LOCAL_API_KEY"
    model = provider.models["qwen"]
    assert model.context_window == 32768
    assert model.max_output_tokens == 4096
    assert model.reasoning_field == "reasoning_content"


def test_optional_configuration_fields_default_to_none(tmp_path):
    config = load_config(write_config(tmp_path, payload()))
    provider = config.providers["local"]
    assert provider.api_key_env is None
    model = provider.models["qwen"]
    assert model.max_output_tokens is None
    assert model.reasoning_field is None


def test_base_url_preserves_existing_path(tmp_path):
    url = "https://api.example.com/provider/v1/"
    config = load_config(write_config(tmp_path, payload(base_url=url)))
    assert config.providers["local"].base_url == url


def test_example_configuration_loads():
    config = load_config(EXAMPLE_PATH)
    (provider,) = config.providers.values()
    assert provider.api_key_env == "LOCAL_API_KEY"
    (model,) = provider.models.values()
    assert model.context_window > 0


# --- invalid configurations --------------------------------------------------

INVALID_CONFIGURATIONS = [
    ("unknown top-level field", payload(top={"extra": 1}), "extra"),
    ("literal api key field", payload(provider={"base_url": LOCAL_URL, "api_key": SECRET}), "api_key"),
    ("unknown model field", payload(model={"context_window": 1, "seed": 1}), "seed"),
    ("missing providers", {}, "providers"),
    ("empty providers", {"providers": {}}, "providers"),
    ("providers not a mapping", {"providers": [LOCAL_URL]}, "providers"),
    ("missing base_url", payload(provider={}), "base_url"),
    ("base_url wrong type", payload(provider={"base_url": 8080}), "base_url"),
    ("base_url wrong scheme", payload(base_url="ftp://127.0.0.1/v1"), "base_url"),
    ("base_url not absolute", payload(base_url="127.0.0.1:8080/v1"), "base_url"),
    ("base_url without host", payload(base_url="http:///v1"), "base_url"),
    ("base_url with credentials", payload(base_url="http://user:pass@127.0.0.1/v1"), "base_url"),
    ("base_url with query", payload(base_url=LOCAL_URL + "?q=1"), "base_url"),
    ("base_url with fragment", payload(base_url=LOCAL_URL + "#part"), "base_url"),
    ("base_url port out of range", payload(base_url="http://127.0.0.1:99999/v1"), "base_url"),
    ("empty api_key_env", payload(provider={"base_url": LOCAL_URL, "api_key_env": ""}), "api_key_env"),
    ("missing models", {"providers": {"local": {"base_url": LOCAL_URL}}}, "models"),
    ("empty models", {"providers": {"local": {"base_url": LOCAL_URL, "models": {}}}}, "models"),
    (
        "empty provider key",
        {"providers": {"": {"base_url": LOCAL_URL, "models": {"qwen": {"context_window": 1}}}}},
        "providers",
    ),
    (
        "empty model key",
        {"providers": {"local": {"base_url": LOCAL_URL, "models": {"": {"context_window": 1}}}}},
        "models",
    ),
    ("missing context_window", payload(model={}), "context_window"),
    ("context_window zero", payload(model={"context_window": 0}), "context_window"),
    ("context_window negative", payload(model={"context_window": -1}), "context_window"),
    ("context_window boolean", payload(model={"context_window": True}), "context_window"),
    ("context_window string", payload(model={"context_window": "32768"}), "context_window"),
    ("max_output_tokens zero", payload(model={"context_window": 1, "max_output_tokens": 0}), "max_output_tokens"),
    ("max_output_tokens string", payload(model={"context_window": 1, "max_output_tokens": "4096"}), "max_output_tokens"),
    ("empty reasoning_field", payload(model={"context_window": 1, "reasoning_field": ""}), "reasoning_field"),
    ("reasoning_field collides with role", payload(model={"context_window": 1, "reasoning_field": "role"}), "reasoning_field"),
    ("reasoning_field collides with content", payload(model={"context_window": 1, "reasoning_field": "content"}), "reasoning_field"),
    ("reasoning_field collides with tool_calls", payload(model={"context_window": 1, "reasoning_field": "tool_calls"}), "reasoning_field"),
    ("reasoning_field collides with tool_call_id", payload(model={"context_window": 1, "reasoning_field": "tool_call_id"}), "reasoning_field"),
]


@pytest.mark.parametrize(
    "name,document,expected", INVALID_CONFIGURATIONS, ids=[row[0] for row in INVALID_CONFIGURATIONS]
)
def test_invalid_configuration_is_rejected(tmp_path, name, document, expected):
    with pytest.raises(ValueError) as failure:
        load_config(write_config(tmp_path, document))
    assert expected in str(failure.value), name


# --- file-level failures -----------------------------------------------------

SECRET_DOCUMENTS = [
    ("invalid type value", payload(model={"context_window": SECRET})),
    ("literal api key value", payload(provider={"base_url": LOCAL_URL, "api_key": SECRET})),
    ("invalid base_url value", payload(base_url=SECRET)),
    ("invalid json", '{"providers": ' + SECRET),
    ("non-utf8 content", ('{"x": "' + SECRET + '"}').encode() + b"\xff"),
]


@pytest.mark.parametrize("name,document", SECRET_DOCUMENTS, ids=[row[0] for row in SECRET_DOCUMENTS])
def test_boundary_errors_hide_input_values(tmp_path, name, document):
    with pytest.raises(ValueError) as failure:
        load_config(write_config(tmp_path, document))
    assert SECRET not in str(failure.value), name


def test_missing_file_is_boundary_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(tmp_path / "absent.json")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_is_boundary_error(tmp_path):
    path = write_config(tmp_path, payload())
    path.chmod(0)
    with pytest.raises(ValueError):
        load_config(path)


def test_non_utf8_file_is_boundary_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, b"\xff\xfe not utf-8"))


def test_invalid_json_file_is_boundary_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, '{"providers":'))


# --- model resolution --------------------------------------------------------

def test_resolve_model_returns_selected_provider_and_model(tmp_path):
    config = load_config(write_config(tmp_path, payload()))
    provider, model = resolve_model(config, ModelRef(provider="local", model="qwen"))
    assert provider is config.providers["local"]
    assert model is provider.models["qwen"]


@pytest.mark.parametrize(
    "ref",
    [ModelRef(provider="absent", model="qwen"), ModelRef(provider="local", model="absent")],
    ids=["unknown provider", "unknown model"],
)
def test_resolve_model_rejects_unknown_reference(tmp_path, ref):
    config = load_config(write_config(tmp_path, payload()))
    with pytest.raises(ValueError) as failure:
        resolve_model(config, ref)
    assert str(failure.value) == "unknown provider or model"


# --- configuration models are frozen ----------------------------------------

def test_configuration_models_are_frozen(tmp_path):
    config = load_config(write_config(tmp_path, payload()))
    provider = config.providers["local"]
    model = provider.models["qwen"]
    with pytest.raises(ValidationError):
        config.providers = {}
    with pytest.raises(ValidationError):
        provider.api_key_env = "CHANGED"
    with pytest.raises(ValidationError):
        model.context_window = 1


# --- transport values --------------------------------------------------------

FROZEN_VALUES = [
    (ModelRef(provider="local", model="qwen"), "provider"),
    (ToolCall(id="1", name="lookup", arguments="{}"), "id"),
    (Message(role="user", content="hello"), "role"),
    (ToolDefinition(name="lookup", description="look things up", parameters={}), "name"),
    (ModelRequest(messages=(Message(role="user"),)), "messages"),
    (ModelResponse(message=Message(role="assistant"), usage=None), "message"),
]


@pytest.mark.parametrize(
    "value,attribute", FROZEN_VALUES, ids=[type(value).__name__ for value, _ in FROZEN_VALUES]
)
def test_values_are_frozen(value, attribute):
    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, None)


def test_default_maps_are_independently_allocated():
    first_call = ToolCall(id="1", name="lookup", arguments="{}")
    second_call = ToolCall(id="2", name="lookup", arguments="{}")
    assert first_call.extra is not second_call.extra
    assert first_call.function_extra is not second_call.function_extra
    assert first_call.extra is not first_call.function_extra
    first_call.extra["ext"] = True
    assert second_call.extra == {}
    assert "ext" not in first_call.function_extra

    first_message = Message(role="assistant")
    second_message = Message(role="assistant")
    assert first_message.extra is not second_message.extra
    first_message.extra["ext"] = True
    assert second_message.extra == {}


def test_message_optional_fields_default_to_unset():
    message = Message(role="user", content="hello")
    assert message.tool_calls == ()
    assert message.tool_call_id is None
    assert message.source is None
    assert message.extra == {}


def test_model_request_tools_default_to_empty():
    assert ModelRequest(messages=(Message(role="user"),)).tools == ()
