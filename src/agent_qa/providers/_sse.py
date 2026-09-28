"""Private SSE framing for Chat Completions streams."""

import codecs

from agent_qa.model import ProviderError


async def iter_sse_data(byte_chunks):
    """Yield each dispatched SSE event's joined ``data`` value as a string.

    Decodes bytes incrementally as UTF-8, so a multi-byte character may be
    split across chunks. Lines end with LF or CRLF. Each ``data`` field appends
    its value plus one newline to the event's data buffer, removing at most one
    space after ``:``; a blank line dispatches the buffer with its trailing
    newline removed, and a block without ``data`` fields dispatches nothing.
    Other fields carry no Chat Completions policy. A data event equal to
    ``[DONE]`` after surrounding-whitespace removal ends the stream: it is
    consumed, never yielded, and no further bytes are pulled. At EOF any
    pending data, including an unterminated final line, dispatches first.
    Invalid UTF-8 raises ``ProviderError``.
    """
    decode = codecs.getincrementaldecoder("utf-8")().decode
    pending = ""
    data = []
    async for chunk in byte_chunks:
        try:
            pending += decode(chunk)
        except UnicodeDecodeError:
            raise ProviderError("invalid stream: response is not valid UTF-8") from None
        while True:
            end = pending.find("\n")
            if end < 0:
                break
            line = pending[:end]
            pending = pending[end + 1:]
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                joined = _dispatch(data)
                if joined is not None:
                    if joined.strip() == "[DONE]":
                        return
                    yield joined
            else:
                _feed_line(data, line)
    try:
        pending += decode(b"", final=True)
    except UnicodeDecodeError:
        raise ProviderError("invalid stream: response is not valid UTF-8") from None
    if pending:
        _feed_line(data, pending)
    joined = _dispatch(data)
    if joined is not None and joined.strip() != "[DONE]":
        yield joined


def _feed_line(data, line):
    """Append one ``data`` field's value to the event buffer; ignore other fields."""
    name, _, value = line.partition(":")
    if name == "data":
        if value.startswith(" "):
            value = value[1:]
        data.append(value + "\n")


def _dispatch(data):
    """Remove the trailing newline, clear the buffer, and return the value.

    Returns ``None`` when the buffer holds no ``data`` fields, dispatching
    nothing for that block.
    """
    if not data:
        return None
    joined = "".join(data)[:-1]
    data.clear()
    return joined
