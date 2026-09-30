"""HTTP session, admission, and streamed event endpoints over shared services."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from pydantic import BaseModel, ConfigDict, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from agent_qa.model import ModelRef
from agent_qa.providers.chat_completions import ChatCompletionsClient
from agent_qa.providers.config import ProvidersConfig
from agent_qa.storage import Snapshot, Storage
from agent_qa.turns import Runtime, client_messages

COOKIE_NAME = "agent_qa_session"
BODY_LIMIT = 1_048_576
REJECTION = {"error": "request rejected"}
REJECTIONS = {"unknown": 404, "busy": 409, "closed": 503}


class MessageInput(BaseModel):
    """One admission request body: strict, closed, and length-bounded."""

    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=50_000)


def _reject(status_code: int) -> JSONResponse:
    """One uniform rejection body; the status carries the cause."""
    return JSONResponse(REJECTION, status_code=status_code)


def _snapshot_body(snapshot: Snapshot) -> dict:
    """Project one committed revision: browser messages, identity, outcome."""
    outcome = None
    for kind, payload in reversed(snapshot.entries):
        if kind == "turn_settlement":
            outcome = json.loads(payload)["status"]
            break
    return {
        "messages": client_messages(snapshot),
        "turn": {"id": snapshot.current_turn_id}
        if snapshot.current_turn_id is not None
        else None,
        "outcome": outcome,
    }


async def _session(request: Request) -> JSONResponse:
    """Resolve or create the cookie session and return its committed snapshot."""
    storage: Storage = request.app.state.storage
    session_id = request.cookies.get(COOKIE_NAME)
    snapshot = await storage.snapshot(session_id) if session_id is not None else None
    if snapshot is None:
        session_id = await storage.create_session()
        response = JSONResponse({"messages": [], "turn": None, "outcome": None})
        response.set_cookie(
            COOKIE_NAME,
            session_id,
            httponly=True,
            samesite="lax",
            path="/",
            secure=request.url.scheme == "https",
        )
        return response
    return JSONResponse(_snapshot_body(snapshot))


async def _message(request: Request) -> JSONResponse:
    """Admit one length-capped, strict request and report its turn identity."""
    session_id = request.cookies.get(COOKIE_NAME)
    if session_id is None:
        return _reject(404)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > BODY_LIMIT:
            return _reject(400)
        body.extend(chunk)
    try:
        payload = MessageInput.model_validate(json.loads(body))
    except ValueError:
        # json.JSONDecodeError, pydantic's ValidationError, and
        # UnicodeDecodeError are all ValueError subclasses.
        return _reject(400)
    runtime: Runtime = request.app.state.runtime
    result = await runtime.submit(session_id, payload.request_id, payload.text)
    if result in REJECTIONS:
        return _reject(REJECTIONS[result])
    return JSONResponse({"turn_id": result})


async def frames(queue: asyncio.Queue):
    """Yield one server-sent-event frame per queued event."""
    while True:
        name, data = await queue.get()
        yield f"event: {name}\ndata: {json.dumps(data)}\n\n"


async def _events(request: Request):
    """Serve one live event stream for an existing cookie session."""
    session_id = request.cookies.get(COOKIE_NAME)
    if session_id is None:
        return _reject(404)
    storage: Storage = request.app.state.storage
    if await storage.snapshot(session_id) is None:
        return _reject(404)
    runtime: Runtime = request.app.state.runtime

    async def response(scope, receive, send):
        queue = runtime.subscribe(session_id)
        try:
            await StreamingResponse(
                frames(queue),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )(scope, receive, send)
        finally:
            runtime.unsubscribe(session_id, queue)

    return response


def create_app(
    config: ProvidersConfig,
    model: ModelRef,
    db_path: Path,
    system_prompt: str,
    http: httpx.AsyncClient | None = None,
) -> Starlette:
    """Compose one chat application over shared durable services.

    The lifespan opens one storage instance, recovers persisted running
    turns, owns one shared credential-free HTTP client unless the caller
    injects its own, and constructs the turn runtime once. Managed
    shutdown interrupts whatever still runs, then closes the owned client
    and the storage connection last; an injected client stays caller-owned.
    """

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            storage = await Storage.open(db_path)
        except Exception:
            raise RuntimeError("database startup failed") from None
        owned: httpx.AsyncClient | None = None
        http_client = http
        try:
            try:
                await storage.interrupt_all()
            except Exception:
                raise RuntimeError("database startup failed") from None
            if http_client is None:
                owned = httpx.AsyncClient(
                    timeout=httpx.Timeout(300.0, connect=10.0),
                    follow_redirects=False,
                )
                http_client = owned
            app.state.runtime = Runtime(
                storage,
                ChatCompletionsClient(http_client),
                config,
                model,
                system_prompt,
            )
            app.state.storage = storage
        except BaseException:
            if owned is not None:
                await owned.aclose()
            await storage.close()
            raise
        try:
            yield
        finally:
            try:
                await app.state.runtime.shutdown()
            finally:
                try:
                    if owned is not None:
                        await owned.aclose()
                finally:
                    await storage.close()

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/api/session", _session, methods=["GET"]),
            Route("/api/message", _message, methods=["POST"]),
            Route("/api/events", _events, methods=["GET"]),
        ],
    )
