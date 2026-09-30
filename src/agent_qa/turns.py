"""Durable-history projections and the server-owned turn runtime."""

import asyncio
import dataclasses
import json
import logging
from typing import Literal, cast

from agent_qa.model import (
    JsonObject,
    Message,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from agent_qa.providers.chat_completions import ChatCompletionsClient
from agent_qa.providers.config import ProvidersConfig
from agent_qa.storage import Snapshot, Status, Storage

logger = logging.getLogger(__name__)

_INTERRUPTED_NOTE = "The previous response ended before completion."

_Role = Literal["system", "user", "assistant", "tool"]


def encode_message(message: Message) -> JsonObject:
    """Return the six-field Message JSON object for durable storage."""
    value = dataclasses.asdict(message)
    value["tool_calls"] = list(value["tool_calls"])
    return value


def decode_message(value: JsonObject) -> Message:
    """Reconstruct one Message from its stored JSON object.

    Required-field access and typed reconstruction failures propagate; opaque
    maps and strings are passed through untouched.
    """
    calls = cast("list[JsonObject]", value["tool_calls"])
    source = value["source"]
    return Message(
        role=cast(_Role, value["role"]),
        content=cast("str | None", value["content"]),
        tool_calls=tuple(
            ToolCall(
                id=cast(str, call["id"]),
                name=cast(str, call["name"]),
                arguments=cast(str, call["arguments"]),
                extra=cast(JsonObject, call["extra"]),
                function_extra=cast(JsonObject, call["function_extra"]),
            )
            for call in calls
        ),
        tool_call_id=cast("str | None", value["tool_call_id"]),
        source=None if source is None else ModelRef(**cast("dict[str, str]", source)),
        extra=cast(JsonObject, value["extra"]),
    )


def model_context(snapshot: Snapshot, system_prompt: str) -> tuple[Message, ...]:
    """Project one committed session revision into the model message context.

    Settlements carry no conversation content; a signal becomes the system
    note that the previous response ended before completion.
    """
    messages = [Message(role="system", content=system_prompt)]
    for kind, payload in snapshot.entries:
        if kind == "user":
            messages.append(Message(role="user", content=json.loads(payload)["text"]))
        elif kind == "assistant":
            messages.append(decode_message(json.loads(payload)["message"]))
        elif kind == "signal":
            messages.append(Message(role="system", content=_INTERRUPTED_NOTE))
        elif kind != "turn_settlement":
            raise ValueError(f"unknown entry kind: {kind}")
    return tuple(messages)


def client_messages(snapshot: Snapshot) -> list[JsonObject]:
    """Project one committed session revision into browser answer messages.

    Only user and assistant answer text is surfaced; textless assistants are
    omitted.
    """
    messages = []
    for kind, payload in snapshot.entries:
        if kind == "user":
            messages.append({"role": "user", "text": json.loads(payload)["text"]})
        elif kind == "assistant":
            content = decode_message(json.loads(payload)["message"]).content
            if content is not None:
                messages.append({"role": "assistant", "text": content})
        elif kind not in ("signal", "turn_settlement"):
            raise ValueError(f"unknown entry kind: {kind}")
    return messages


class _Execution:
    """Mutable per-turn execution state: identity, spawned task, and live text."""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.task: asyncio.Task | None = None
        self.text = ""


class Runtime:
    """Server-owned turn execution and transient event delivery for one app.

    All runtime state lives on the event-loop thread; database work is
    delegated to the storage worker and model work to one detached task per
    admitted turn.
    """

    def __init__(
        self,
        storage: Storage,
        client: ChatCompletionsClient,
        config: ProvidersConfig,
        model: ModelRef,
        system_prompt: str,
    ) -> None:
        self._storage = storage
        self._client = client
        self._config = config
        self._model = model
        self._system_prompt = system_prompt
        self._running: dict[str, _Execution] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._stopping = False

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Register one per-session event queue and return it."""
        queue: asyncio.Queue = asyncio.Queue()
        # ponytail: unbounded subscriber queues; add bounded-drop delivery if
        # non-reading clients cause measured memory growth
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Drop one queue; idempotent, removing empty session sets."""
        queues = self._subscribers.get(session_id)
        if queues is not None:
            queues.discard(queue)
            if not queues:
                del self._subscribers[session_id]

    def publish(self, session_id: str, name: str, data: JsonObject) -> None:
        """Enqueue one event on every subscribed queue without blocking."""
        for queue in self._subscribers.get(session_id, ()):
            queue.put_nowait((name, data))

    async def submit(self, session_id: str, request_id: str, text: str) -> str:
        """Admit one user turn and return its turn ID, or a rejection word.

        Returns ``"unknown"``, ``"busy"``, or ``"closed"``. A duplicate
        request ID returns its existing turn ID and spawns nothing. An
        admission that commits while shutdown has begun returns its turn ID
        without spawning a runner; the shutdown interrupt scan follows it.
        """
        if self._stopping:
            return "closed"
        admitted = await self._storage.admit(session_id, request_id, text)
        if isinstance(admitted, str):
            return admitted
        turn_id, fresh = admitted
        if not fresh or self._stopping:
            return turn_id
        record = _Execution(turn_id)
        self._running[session_id] = record
        record.task = asyncio.create_task(self._run(session_id, record))
        return turn_id

    async def shutdown(self) -> None:
        """Stop admitting, cancel unfinished runners, and settle what remains.

        Each remaining execution record settles interrupted with its
        available text; the final storage interrupt scan then covers
        admissions that never spawned a runner and drained settlement jobs.
        """
        self._stopping = True
        tasks = [
            record.task for record in self._running.values() if record.task is not None
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for session_id, record in list(self._running.items()):
            await self._finish(session_id, record, "interrupted", None)
        await self._storage.interrupt_all()

    async def _run(self, session_id: str, record: _Execution) -> None:
        """Consume exactly one model request for the admitted turn."""
        cancelled = False
        status = "success"
        result: ModelResponse | None = None
        try:
            snapshot = await self._storage.snapshot(session_id)
            # Admission guarantees the session exists.
            assert snapshot is not None
            context = model_context(snapshot, self._system_prompt)
            stream = self._client.stream(
                self._config, self._model, ModelRequest(messages=context)
            )
            async with stream:
                async for fragment in stream:
                    record.text += fragment
                    self.publish(
                        session_id,
                        "delta",
                        {"turn_id": record.turn_id, "text": fragment},
                    )
            result = stream.result
            # Normal exhaustion always finalizes the response.
            assert result is not None
            if result.message.tool_calls:
                raise RuntimeError("the completed response requests tool calls")
        except asyncio.CancelledError:
            cancelled = True
            status = "interrupted"
        except Exception as error:
            status = "failure"
            logger.error(
                "turn failed: %s (session=%s, turn=%s)",
                type(error).__name__,
                session_id,
                record.turn_id,
            )
        await self._finish(session_id, record, status, result)
        if cancelled:
            raise asyncio.CancelledError

    async def _finish(
        self,
        session_id: str,
        record: _Execution,
        status: Status,
        result: ModelResponse | None,
    ) -> None:
        """Settle one terminal outcome, publish its end event, drop the record.

        The full result message is persisted on success; every non-success
        path persists the live text as a bare partial message when nonempty.
        A settlement or serialization failure is logged by category and IDs
        only and leaves the record and durable running state in place.
        """
        try:
            assistant = None
            if status == "success":
                assert result is not None
                assistant = {
                    "message": encode_message(result.message),
                    "usage": result.usage,
                }
            elif record.text:
                assistant = {
                    "message": encode_message(
                        Message(role="assistant", content=record.text)
                    ),
                    "usage": None,
                }
            settled = await self._storage.settle(
                session_id, record.turn_id, status, assistant
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "turn settlement failed: %s (session=%s, turn=%s)",
                type(error).__name__,
                session_id,
                record.turn_id,
            )
            return
        if settled:
            self.publish(
                session_id,
                "turn_end",
                {"turn_id": record.turn_id, "status": status},
            )
        self._running.pop(session_id, None)