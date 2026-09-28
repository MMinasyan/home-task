"""SSE framing: byte-level chunk independence, event dispatch, and termination."""

import asyncio

import pytest

from agent_qa.model import ProviderError
from agent_qa.providers._sse import iter_sse_data

# A fixture exercising CRLF and LF endings, a comment, a multi-data block, a
# colon-less data field, split multi-byte text, and the [DONE] terminator.
STREAM = (
    "data: first\r\n"
    "\r\n"
    ": keep-alive\n"
    "event: ping\n"
    "id: 7\n"
    "retry: 100\n"
    "\r\n"
    "data: մտա\n"
    "data: ծողություն\n"
    "\n"
    "data\n"
    "\n"
    "data: last"
    "\r\n"
    "\r\n"
    "data: [DONE]\r\n"
    "\r\n"
).encode("utf-8")
EVENTS = ["first", "մտա\nծողություն", "", "last"]


def collect(chunks):
    """Run one byte source through the framer and return the dispatched values."""

    async def source():
        for chunk in chunks:
            yield chunk

    async def run():
        return [event async for event in iter_sse_data(source())]

    return asyncio.run(run())


def test_every_single_split_point_is_chunk_independent():
    for split in range(len(STREAM) + 1):
        assert collect([STREAM[:split], STREAM[split:]]) == EVENTS, split


def test_one_byte_chunks_are_chunk_independent():
    assert collect([STREAM[i:i + 1] for i in range(len(STREAM))]) == EVENTS


def test_crlf_and_lf_both_terminate_lines():
    assert collect([b"data: a\r\ndata: b\n\ndata: c\r\n\r\n"]) == ["a\nb", "c"]


def test_multi_data_block_joins_with_newlines():
    assert collect([b"data: a\ndata: b\ndata: c\n\n"]) == ["a\nb\nc"]


def test_space_after_colon_removed_at_most_once():
    assert collect([b"data:x\n\ndata: x\n\ndata:  x\n\n"]) == ["x", "x", " x"]


def test_colonless_data_field_is_an_empty_value():
    assert collect([b"data\n\n"]) == [""]


def test_block_without_data_fields_dispatches_nothing():
    assert collect([b": comment\n\nevent: x\nid: 1\nretry: 5\n\ncustom\n\n"]) == []


def test_empty_data_value_dispatches():
    assert collect([b"data:\n\ndata: \n\n"]) == ["", ""]


def test_blank_lines_after_dispatch_start_no_new_event():
    assert collect([b"data: a\n\n\n\n"]) == ["a"]


def test_eof_dispatches_pending_data_including_unterminated_final_line():
    assert collect([b"data: a\n\ndata: b\ndata: c"]) == ["a", "b\nc"]
    assert collect([b"data: only"]) == ["only"]


def test_done_is_recognized_after_whitespace_removal():
    assert collect([b"data: [DONE]\n\n"]) == []
    assert collect([b"data:\t[DONE]  \n\n"]) == []
    assert collect([b"data: [DONE]"]) == []


def test_done_ends_the_stream_without_reading_further_bytes():
    pulled = []

    async def source():
        for chunk in (b"data: x\n\n", b"data: [DONE]\n\n", b"data: never\n\n"):
            pulled.append(chunk)
            yield chunk

    async def run():
        return [event async for event in iter_sse_data(source())]

    assert asyncio.run(run()) == ["x"]
    assert pulled == [b"data: x\n\n", b"data: [DONE]\n\n"]


def test_invalid_utf8_fails_the_stream():
    with pytest.raises(ProviderError):
        collect([b"data: ok\n\n", b"data: \xff\n\n"])
    with pytest.raises(ProviderError):
        collect([b"data: \xc3\n\n"])


def test_incomplete_multibyte_sequence_at_eof_fails_the_stream():
    with pytest.raises(ProviderError):
        collect(["data: մտա".encode(), b"\xc2"])


def test_error_event_dispatches_as_data_for_the_assembler_to_judge():
    event = '{"error": {"message": "boom"}}'
    assert collect([f"data: {event}\n\n".encode()]) == [event]


def test_done_terminates_only_when_the_whole_joined_event_matches():
    assert collect([b"data: x\ndata: [DONE]"]) == ["x\n[DONE]"]
