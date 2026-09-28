"""Chat Completions stream assembly: fragments, extras, snapshots, terminal rules."""

import asyncio
import json

import pytest

from agent_qa.model import Message, ModelResponse, ProviderError, ToolCall
from agent_qa.providers._cc import StreamAssembler
from agent_qa.providers._sse import iter_sse_data


def sse(*events):
    """Encode whole events with a trailing [DONE] terminator as SSE bytes."""
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return body + b"data: [DONE]\n\n"


def feed(assembler, *events):
    """Feed whole events and return the surfaced fragments."""
    surfaced = []
    for event in events:
        fragment = assembler.feed(json.dumps(event))
        if fragment is not None:
            surfaced.append(fragment)
    return surfaced


def assemble(*events):
    assembler = StreamAssembler()
    feed(assembler, *events)
    return assembler.finalize()


def chunk(delta, finish=None, **choice_fields):
    """One index-0 choice chunk; extra keyword fields land on the choice."""
    choice = {"index": 0, "delta": delta}
    if finish is not None:
        choice["finish_reason"] = finish
    choice.update(choice_fields)
    return {"choices": [choice]}


def call_delta(**entry_fields):
    fields = {"index": 0}
    fields.update(entry_fields)
    return {"tool_calls": [fields]}


def pipeline(chunks):
    """Run raw bytes through the framer into a fresh assembler; return the evidence."""

    async def source():
        for piece in chunks:
            yield piece

    async def run():
        assembler = StreamAssembler()
        surfaced = []
        async for text in iter_sse_data(source()):
            fragment = assembler.feed(text)
            if fragment is not None:
                surfaced.append(fragment)
        return surfaced, assembler

    return asyncio.run(run())


# A complete tool-call stream: split names/ids/arguments across interleaved
# indices, reasoning and opaque extension fields, trailing usage, [DONE].
FULL_STREAM = sse(
    chunk({"role": "assistant", "reasoning_content": "Մտածելու ընթացքում՝ "}),
    chunk({"reasoning_content": "ասում եմ", "refusal": None}),
    chunk({"content": "He", "tool_calls": [
        {"index": 0, "id": "call_", "type": "function", "function": {"name": "loo", "arguments": '{"q'}},
    ]}),
    chunk({"content": "y", "tool_calls": [
        {"index": 0, "id": "1", "function": {"name": "kup", "arguments": '": 1}'}},
    ]}),
    chunk({"tool_calls": [
        {"index": 1, "id": "call_2", "type": "function", "function": {"name": "echo", "arguments": "{}"}},
    ]}),
    chunk({"extra_content": {"google": {"thought_signature": "s1"}}, "path": ["a"]}),
    chunk({"extra_content": {"google": {"thought_signature": "s2"}}, "path": ["b"]}),
    chunk({}, finish="tool_calls"),
    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7, "cost": 0.5}},
)
FULL_RESULT = ModelResponse(
    message=Message(
        role="assistant",
        content="Hey",
        tool_calls=(
            ToolCall(id="call_1", name="lookup", arguments='{"q": 1}'),
            ToolCall(id="call_2", name="echo", arguments="{}"),
        ),
        extra={
            "reasoning_content": "Մտածելու ընթացքում՝ ասում եմ",
            "refusal": None,
            "extra_content": {"google": {"thought_signature": "s2"}},
            "path": ["a", "b"],
        },
    ),
    usage={"prompt_tokens": 3, "completion_tokens": 7, "cost": 0.5},
)


def test_text_stream_projects_fragments_and_trailing_usage():
    events = (
        chunk({"role": "assistant"}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 1}},
    )
    assembler = StreamAssembler()
    assert feed(assembler, *events) == ["Hel", "lo"]
    assert assembler.finalize() == ModelResponse(
        message=Message(role="assistant", content="Hello"),
        usage={"prompt_tokens": 1},
    )


def test_full_stream_assembles_calls_extras_and_usage_in_order():
    surfaced, assembler = pipeline([FULL_STREAM])
    assert surfaced == ["He", "y"]
    assert assembler.finalize() == FULL_RESULT


def outcome(chunks):
    surfaced, assembler = pipeline(chunks)
    return surfaced, assembler.finalize(), assembler.snapshot()


def test_full_stream_is_chunk_independent_at_fixed_splits():
    """Three composed-path splits: mid-data line, mid-event, and between events."""
    reference = outcome([FULL_STREAM])
    mid_line = FULL_STREAM.index(b"kup") + 1
    mid_event = FULL_STREAM.index(b"\n\n") + 1
    between_events = FULL_STREAM.index(b"\n\n") + 2
    for split in (mid_line, mid_event, between_events):
        assert outcome([FULL_STREAM[:split], FULL_STREAM[split:]]) == reference, split


def test_truncated_stream_cannot_finalize_but_keeps_partial():
    head = sse(
        chunk({"role": "assistant", "reasoning_content": "why"}),
        chunk({"content": "He", "tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": "{"}},
        ]}),
    )
    surfaced, assembler = pipeline([head])
    assert surfaced == ["He"]
    with pytest.raises(ProviderError):
        assembler.finalize()
    partial = assembler.snapshot()
    assert partial.message.content == "He"
    assert partial.message.extra == {"reasoning_content": "why"}
    assert partial.message.tool_calls[0].arguments == "{"
    assert partial.usage is None


def test_interleaved_indices_and_omitted_index():
    events = (
        chunk(call_delta(index=1, id="b", function={"name": "two", "arguments": "2a"})),
        chunk({"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "one", "arguments": "1a"}},
            {"index": 1, "function": {"arguments": "2b"}},
            {"id": "c", "function": {"name": "three", "arguments": "3"}},  # omitted index
        ]}),
        chunk(call_delta(index=0, function={"arguments": "1b"})),
        chunk({}, finish="tool_calls"),
    )
    first, second, third = assemble(*events).message.tool_calls
    assert (first.id, first.name, first.arguments) == ("a", "one", "1a1b")
    assert (second.id, second.name, second.arguments) == ("b", "two", "2a2b")
    assert (third.id, third.name, third.arguments) == ("c", "three", "3")


def test_argument_text_is_retained_opaque_without_parsing():
    result = assemble(
        chunk(call_delta(id="c", function={"name": "n", "arguments": "not {json"})),
        chunk({}, finish="tool_calls"),
    )
    assert result.message.tool_calls[0].arguments == "not {json"


def test_repeated_names_are_valid_but_duplicate_ids_fail():
    events = (
        chunk(call_delta(id="c1", function={"name": "same", "arguments": "{}"})),
        chunk(call_delta(index=1, id="c2", function={"name": "same", "arguments": "{}"})),
        chunk({}, finish="tool_calls"),
    )
    assert [call.name for call in assemble(*events).message.tool_calls] == ["same", "same"]
    duplicated = list(events)
    duplicated[1] = chunk(call_delta(index=1, id="c1", function={"name": "same", "arguments": "{}"}))
    with pytest.raises(ProviderError):
        assemble(*duplicated)


def test_partial_snapshot_is_none_before_any_choice_delta():
    assembler = StreamAssembler()
    assert assembler.snapshot() is None
    feed(assembler, {"choices": []})
    assert assembler.snapshot() is None


def test_snapshot_is_a_detached_copy_of_everything_so_far():
    assembler = StreamAssembler()
    feed(assembler, chunk({
        "content": "He",
        "reasoning_content": "why",
        "tool_calls": [{"index": 0, "id": "c", "function": {"name": "n"}}],
    }))
    before = assembler.snapshot()
    before.message.extra["reasoning_content"] = "mutated"
    before.message.tool_calls[0].extra["x"] = "mutated"
    expected = ModelResponse(
        message=Message(
            role="assistant",
            content="He",
            tool_calls=(ToolCall(id="c", name="n", arguments=""),),
            extra={"reasoning_content": "why"},
        ),
        usage=None,
    )
    assert assembler.snapshot() == expected
    assert assembler.snapshot() == expected


def test_snapshot_tracks_usage_and_call_fragments():
    assembler = StreamAssembler()
    feed(assembler, chunk(call_delta(id="c", function={"name": "n", "arguments": '{"'})))
    partial = assembler.snapshot()
    assert partial == ModelResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="c", name="n", arguments='{"'),),
        ),
        usage=None,
    )
    feed(assembler, {"choices": [], "usage": {"a": 1}})
    assert assembler.snapshot().usage == {"a": 1}


def test_call_and_function_extras_accumulate_independently():
    events = (
        chunk(call_delta(index=0, id="c", function={"name": "n", "arguments": "1", "signature": "s"}, priority=1)),
        chunk(call_delta(index=0, function={"arguments": "2", "signature": "t"}, priority=2)),
        chunk({}, finish="tool_calls"),
    )
    call = assemble(*events).message.tool_calls[0]
    assert call.arguments == "12"
    assert call.extra == {"priority": 2}
    assert call.function_extra == {"signature": "st"}


def test_extras_follow_the_one_accumulation_rule():
    cases = [
        (["x", "y"], "xy"),                     # strings concatenate
        ([["a"], ["b"]], ["a", "b"]),           # arrays append elementwise
        (["x", None], None),                    # kind change replaces
        (["x", {"a": 1}], {"a": 1}),            # object fragment replaces
        (["x", ["a"]], ["a"]),                  # kind change replaces
        ([{"a": 1}, {"b": 2}], {"b": 2}),       # objects are opaque, replaced
        ([["a"], "x"], "x"),                    # scalar fragment replaces
        ([1, 2], 2),                            # scalar replaces scalar
        ([""], ""),                             # empty first fragment retained
        ([[]], []),                             # empty first fragment retained
        ([{}], {}),                             # empty first fragment retained
    ]
    for fragments, expected in cases:
        assembler = StreamAssembler()
        for fragment in fragments:
            feed(assembler, chunk({"field": fragment}))
        feed(assembler, chunk({}, finish="stop"))
        result = assembler.finalize()
        assert result.message.extra["field"] == expected, fragments


def test_absent_key_never_erases_a_prior_value():
    assembler = StreamAssembler()
    feed(assembler, chunk({"reasoning_content": "keep"}))
    feed(assembler, chunk({"other": "x"}))
    feed(assembler, chunk({}, finish="stop"))
    assert assembler.finalize().message.extra["reasoning_content"] == "keep"


def test_usage_last_non_null_wins_and_is_never_summed():
    events = (
        chunk({"content": "x"}),
        {"choices": [], "usage": {"a": 1, "b": 2}},
        chunk({}, finish="stop"),
        {"choices": [], "usage": {"b": 3}},
    )
    assert assemble(*events).usage == {"b": 3}
    assert assemble(chunk({"content": "x"}), chunk({}, finish="stop")).usage is None


def test_null_usage_keeps_the_previous_snapshot():
    events = (
        chunk({"content": "x"}),
        {"choices": [], "usage": {"a": 1}},
        {"choices": [], "usage": None},
        chunk({}, finish="stop"),
    )
    assert assemble(*events).usage == {"a": 1}


def test_no_content_projects_content_none():
    events = (
        chunk({"role": "assistant"}),
        chunk({"reasoning_content": "silent"}, finish="stop"),
    )
    result = assemble(*events)
    assert result.message.content is None
    assert result.message.extra == {"reasoning_content": "silent"}


def test_empty_content_string_is_not_a_payload():
    with pytest.raises(ProviderError):
        assemble(chunk({"content": ""}), chunk({}, finish="stop"))


def test_stop_with_assembled_calls_is_contradictory():
    events = (
        chunk(call_delta(id="c", function={"name": "n", "arguments": "{}"})),
        chunk({}, finish="stop"),
    )
    with pytest.raises(ProviderError):
        assemble(*events)


def test_finalize_rejects_incomplete_call_identity():
    with pytest.raises(ProviderError):  # never receives an id
        assemble(
            chunk(call_delta(function={"name": "n", "arguments": "{}"})),
            chunk({}, finish="tool_calls"),
        )
    with pytest.raises(ProviderError):  # never receives a name
        assemble(
            chunk(call_delta(id="c", function={"arguments": "{}"})),
            chunk({}, finish="tool_calls"),
        )


def test_payload_rule_requires_content_calls_or_extras():
    with pytest.raises(ProviderError):
        assemble(chunk({"role": "assistant"}), chunk({}, finish="stop"))


def test_unsuccessful_or_missing_finish_fails():
    for finish in ("length", "content_filter"):
        with pytest.raises(ProviderError):
            assemble(chunk({"content": "x"}, finish=finish))
    with pytest.raises(ProviderError):
        assemble(chunk({"content": "x"}))  # never finished


def test_fragments_accompanying_a_failing_finish_still_reach_partial():
    events = (
        chunk({"content": "x", "function_call": {"name": "legacy", "arguments": "{}"}}, finish="length"),
        {"choices": [], "usage": {"a": 1}},
    )
    assembler = StreamAssembler()
    feed(assembler, *events)
    with pytest.raises(ProviderError):
        assembler.finalize()
    partial = assembler.snapshot()
    assert partial.message.content == "x"
    assert partial.message.extra["function_call"] == {"name": "legacy", "arguments": "{}"}
    assert partial.usage == {"a": 1}


def test_post_finish_chunks_must_carry_empty_choices():
    assembler = StreamAssembler()
    feed(assembler, chunk({}, finish="stop"))
    with pytest.raises(ProviderError):
        feed(assembler, chunk({"content": "late"}))
    with pytest.raises(ProviderError):
        feed(assembler, chunk({}, finish="stop"))  # repeated finish frame


def test_missing_choice_fails_finalization():
    assembler = StreamAssembler()
    feed(assembler, {"choices": [], "usage": {"a": 1}})
    with pytest.raises(ProviderError):
        assembler.finalize()
    assert assembler.snapshot() is None


MALFORMED_EVENTS = [
    {"choices": 0},                                      # choices not an array
    {"choices": None},                                   # choices not an array
    {"choices": ["nope"]},                               # choice not an object
    chunk({}, index=1),                                  # choice index outside n:1
    {"choices": [{"index": "0", "delta": {}}]},          # non-number choice index
    {"choices": [{"index": 0, "delta": "nope"}]},        # delta not an object
    chunk({"role": "user"}),                             # role drift
    chunk({"content": 5}),                               # content wrong type
    chunk({"tool_calls": "nope"}),                       # tool_calls not an array
    chunk({"tool_calls": ["nope"]}),                     # call not an object
    chunk(call_delta(id=5)),                             # id fragment not a string
    chunk(call_delta(index=-2)),                         # negative call index
    chunk(call_delta(index=True)),                       # boolean call index
    chunk(call_delta(type="custom")),                    # call type not function
    chunk(call_delta(function="nope")),                  # function not an object
    chunk(call_delta(function={"name": 5})),             # name fragment not a string
    chunk(call_delta(function={"arguments": 5})),        # arguments fragment not a string
    {"error": {"message": "boom"}},                      # top-level error object
    {"error": "boom"},                                   # top-level error scalar
]


@pytest.mark.parametrize("event", MALFORMED_EVENTS)
def test_malformed_events_raise_provider_error(event):
    assembler = StreamAssembler()
    with pytest.raises(ProviderError):
        assembler.feed(json.dumps(event))


def test_null_error_field_is_not_a_failure():
    event = chunk({"content": "x"})
    event["error"] = None
    assert assemble(event, chunk({}, finish="stop")).message.content == "x"


@pytest.mark.parametrize("data", ["", "[1, 2]"])
def test_invalid_event_data_raises_provider_error(data):
    assembler = StreamAssembler()
    with pytest.raises(ProviderError):
        assembler.feed(data)


def test_nonfinite_numbers_are_malformed():
    assembler = StreamAssembler()
    with pytest.raises(ProviderError):
        assembler.feed('{"choices":[{"index":NaN,"delta":{}}]}')
