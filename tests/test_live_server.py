"""Opt-in live gate: the production server must stream and replay durable history."""

import asyncio
import json
import os
import pathlib
import socket
from urllib.parse import urlsplit

import httpx
import pytest

from agent_qa.app import create_app
from agent_qa.model import ModelRef
from agent_qa.providers.config import load_config, resolve_model
from test_app import Harness, frames_of

_ENV = ("QA_LIVE_CONFIG", "QA_LIVE_PROVIDER", "QA_LIVE_MODEL", "QA_LIVE_PROMPT")

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in _ENV),
    reason=(
        "opt-in live gate: set QA_LIVE_CONFIG, QA_LIVE_PROVIDER, QA_LIVE_MODEL, "
        "and QA_LIVE_PROMPT to run it; a skipped live gate is not a pass"
    ),
)

COOKIE = "agent_qa_session"
FIRST_QUESTION = "Reply with the single word: acknowledged."
FOLLOW_UP = "Repeat your previous reply exactly."
# The injected provider client uses the app's own profile: a 300 s total
# timeout with a 10 s connect bound; each turn's inline stream read is
# bounded to the same 300 s ceiling.
TURN_TIMEOUT = 300.0


def endpoint_reachable(base_url):
    """Probe the configured provider host and port with one bounded TCP connect."""
    parts = urlsplit(base_url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((parts.hostname, port), timeout=5):
            return True
    except OSError:
        return False


async def read_turn(response, turn_id):
    """Consume one live events response inline until its matching turn_end.

    The buffer keeps only the trailing incomplete frame, so every complete
    double-newline frame is decoded exactly once.
    """
    pieces = []
    remainder = b""
    async for chunk in response.aiter_bytes():
        remainder += chunk
        while b"\n\n" in remainder:
            frame, remainder = remainder.split(b"\n\n", 1)
            for name, data in frames_of(frame + b"\n\n"):
                if name == "delta" and data["turn_id"] == turn_id:
                    pieces.append(data["text"])
                elif name == "turn_end" and data["turn_id"] == turn_id:
                    return "".join(pieces), data
    raise AssertionError("the event stream closed before the turn ended")


def test_live_chat_continuity(tmp_path):
    config = load_config(pathlib.Path(os.environ["QA_LIVE_CONFIG"]))
    ref = ModelRef(os.environ["QA_LIVE_PROVIDER"], os.environ["QA_LIVE_MODEL"])
    provider, _ = resolve_model(config, ref)
    if not endpoint_reachable(provider.base_url):
        pytest.skip(
            "live gate blocked: the configured endpoint is unreachable; "
            "a blocked gate is not a pass"
        )
    system_prompt = pathlib.Path(os.environ["QA_LIVE_PROMPT"]).read_text(encoding="utf-8")

    captured = []

    async def capture(request):
        captured.append(request.content)

    async def run():
        # Caller-owned real client with the app's own timeout profile:
        # production provider traffic with an in-memory request hook; closed
        # here, never by the app.
        http = httpx.AsyncClient(
            event_hooks={"request": [capture]},
            timeout=httpx.Timeout(TURN_TIMEOUT, connect=10.0),
        )
        app = create_app(config, ref, tmp_path / "live.sqlite3", system_prompt, http=http)
        harness = Harness(app)
        url = await harness.start()
        try:
            async with httpx.AsyncClient(
                base_url=url, timeout=httpx.Timeout(TURN_TIMEOUT)
            ) as client:
                # One shared anonymous cookie jar for the whole scenario.
                session = await client.get("/api/session")
                assert session.status_code == 200
                assert session.json() == {"messages": [], "turn": None, "outcome": None}
                assert client.cookies.get(COOKIE), "the server must set one session cookie"

                # First turn: subscribe before admitting, read inline, and
                # close the events response before reading the snapshot.
                async with client.stream("GET", "/api/events") as events:
                    assert events.headers["content-type"].startswith("text/event-stream")
                    posted = await client.post(
                        "/api/message",
                        json={"request_id": "live-1", "text": FIRST_QUESTION},
                    )
                    assert posted.status_code == 200
                    turn_id = posted.json()["turn_id"]
                    answer, end = await asyncio.wait_for(
                        read_turn(events, turn_id), TURN_TIMEOUT
                    )
                assert answer, "the first answer must stream nonempty deltas"
                assert end == {"turn_id": turn_id, "status": "success"}

                settled = await client.get("/api/session")
                assert settled.json() == {
                    "messages": [
                        {"role": "user", "text": FIRST_QUESTION},
                        {"role": "assistant", "text": answer},
                    ],
                    "turn": None,
                    "outcome": "success",
                }

                # Follow-up: same client and cookie, with its own live
                # subscription opened before admission.
                async with client.stream("GET", "/api/events") as events:
                    posted = await client.post(
                        "/api/message",
                        json={"request_id": "live-2", "text": FOLLOW_UP},
                    )
                    assert posted.status_code == 200
                    follow_turn = posted.json()["turn_id"]
                    assert follow_turn != turn_id
                    follow_answer, follow_end = await asyncio.wait_for(
                        read_turn(events, follow_turn), TURN_TIMEOUT
                    )
                assert follow_answer, "the follow-up answer must stream nonempty deltas"
                assert follow_end == {"turn_id": follow_turn, "status": "success"}

                # Provider-level retries may add physical sends per logical
                # call, so only the logical turns are required and the last
                # captured body is the follow-up's final request.
                assert len(captured) >= 2, "both turns must reach the provider"
                body = json.loads(captured[-1])
                texts = [part["content"] for part in body["messages"]]
                replayed = (
                    system_prompt in texts
                    and FIRST_QUESTION in texts
                    and answer in texts
                    and FOLLOW_UP in texts
                )
                assert replayed, (
                    "the follow-up provider request must replay the first "
                    "turn's durable history"
                )

                snapshot = await client.get("/api/session")
                assert snapshot.json()["messages"] == [
                    {"role": "user", "text": FIRST_QUESTION},
                    {"role": "assistant", "text": answer},
                    {"role": "user", "text": FOLLOW_UP},
                    {"role": "assistant", "text": follow_answer},
                ]
                assert snapshot.json()["turn"] is None
                assert snapshot.json()["outcome"] == "success"

                # Retrying the original request id returns the existing turn
                # and causes no further provider work.
                sent_before = len(captured)
                retry = await client.post(
                    "/api/message", json={"request_id": "live-1", "text": "replacement"}
                )
                assert retry.status_code == 200
                assert retry.json() == {"turn_id": turn_id}
                assert len(captured) == sent_before, (
                    "a duplicate request id must cause no provider request"
                )
                after = await client.get("/api/session")
                assert after.json() == snapshot.json()
        finally:
            await harness.stop()
            await http.aclose()

    asyncio.run(run())
