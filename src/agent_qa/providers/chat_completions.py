"""Public Chat Completions client: HTTP lifecycle, retry, and stream surface."""

import asyncio
import dataclasses
import json
import os
from collections.abc import AsyncIterator

import httpx

from agent_qa.model import ModelRef, ModelRequest, ModelResponse, ProviderError
from agent_qa.providers._cc import StreamAssembler, encode_request
from agent_qa.providers._sse import iter_sse_data
from agent_qa.providers.config import ProvidersConfig, resolve_model

_RETRIES = 3
_BACKOFF = (2, 4, 8)

_TOKEN = object()


class ChatStream:
    """One logical streaming call; created only by ``ChatCompletionsClient.stream``.

    Enter the context (request validation, credential resolution, and HTTP
    opening), iterate the surfaced text fragments, and read ``partial`` and
    ``result``. Both properties remain readable after the context exits. The
    instance cannot be restarted.
    """

    def __init__(self, http, config, model, request, *, _token):
        if _token is not _TOKEN:
            raise TypeError("a stream is created by ChatCompletionsClient.stream")
        self._http = http
        self._config = config
        self._model = model
        self._request = request
        self._entered = False
        self._ended = False
        # The assembler is created only by a successful context entry; its
        # presence gates iteration and the partial snapshot.
        self._response: httpx.Response | None = None
        self._assembler: StreamAssembler | None = None
        self._events: AsyncIterator[str] | None = None
        self._result: ModelResponse | None = None

    async def __aenter__(self) -> "ChatStream":
        if self._entered:
            raise ValueError("the stream context cannot be re-entered")
        self._entered = True
        body = encode_request(
            model_key=self._model.model, request=self._request, target=self._model
        )
        content = json.dumps(body).encode("utf-8")
        provider, _ = resolve_model(self._config, self._model)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if provider.api_key_env is not None:
            credential = os.environ.get(provider.api_key_env)
            if not credential:
                raise ValueError(
                    "the selected credential environment variable is missing or empty"
                )
            headers["Authorization"] = f"Bearer {credential}"
        attempt = 0
        while True:
            response = await self._open(provider, headers, content)
            if 200 <= response.status_code < 300:
                self._response = response
                self._assembler = StreamAssembler()
                self._events = self._event_stream(response)
                return self
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            await response.aclose()
            if not retryable or attempt == _RETRIES:
                raise ProviderError(
                    f"provider rejected the request: HTTP {response.status_code}"
                )
            await asyncio.sleep(_BACKOFF[attempt])
            attempt += 1

    async def __aexit__(self, *_exc: object) -> None:
        self._ended = True
        await self._close()

    def __aiter__(self) -> "ChatStream":
        return self

    async def __anext__(self) -> str:
        assembler, events = self._assembler, self._events
        if assembler is None or events is None:
            raise ValueError("iteration requires entering the stream context")
        if self._ended:
            raise StopAsyncIteration
        while True:
            try:
                data = await events.__anext__()
                fragment = assembler.feed(data)
            except StopAsyncIteration:
                # Framer exhausted: apply the terminal rules, then end the stream.
                self._ended = True
                try:
                    self._result = self._stamp(assembler.finalize())
                except ProviderError:
                    await self._close()
                    raise
                await self._close()
                raise StopAsyncIteration from None
            except (ProviderError, asyncio.CancelledError):
                self._ended = True
                await self._close()
                raise
            if fragment:
                return fragment

    @property
    def partial(self) -> ModelResponse | None:
        """A detached snapshot of everything assembled so far, source-stamped."""
        if self._assembler is None:
            return None
        snapshot = self._assembler.snapshot()
        if snapshot is None:
            return None
        return self._stamp(snapshot)

    @property
    def result(self) -> ModelResponse | None:
        """The completed response, or ``None`` until a successful finalization."""
        return self._result

    async def _open(self, provider, headers, content):
        request = httpx.Request(
            "POST",
            provider.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            content=content,
        )
        try:
            return await self._http.send(request, stream=True)
        except httpx.HTTPError:
            raise ProviderError("connection to the provider failed") from None

    async def _event_stream(self, response: httpx.Response) -> AsyncIterator[str]:
        try:
            async for data in iter_sse_data(response.aiter_bytes()):
                yield data
        except httpx.HTTPError:
            raise ProviderError("connection to the provider failed") from None

    async def _close(self) -> None:
        if self._response is not None:
            response, self._response = self._response, None
            await response.aclose()

    def _stamp(self, response: ModelResponse) -> ModelResponse:
        message = dataclasses.replace(response.message, source=self._model)
        return dataclasses.replace(response, message=message)


class ChatCompletionsClient:
    """The Chat Completions transport over one caller-owned shared HTTP client."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    def stream(
        self, config: ProvidersConfig, model: ModelRef, request: ModelRequest
    ) -> ChatStream:
        return ChatStream(self._http, config, model, request, _token=_TOKEN)