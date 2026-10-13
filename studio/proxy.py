"""Relay requests to the engine, streaming replies through, so the page and tailnet clients talk to one origin.

Credentials have one owner each. The Studio's own token (the `x-studio-token` header or `?token=`) is consumed by
the server's gate and never travels to the engine. For the engine, the relay presents the key the engine actually
runs with whenever the Studio holds one, whatever a client sent: reaching the relay is what grants inference. A
client's own `Authorization` or `x-api-key` passes through only when the Studio has no key, for a remote engine
whose key it does not know.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate"}
STUDIO_HEADERS = {"x-studio-token"}
STUDIO_PARAMS = {"token"}
ENGINE_AUTH = {"authorization", "x-api-key"}


class Proxy:
    def __init__(self, engine_url_getter: Callable[[], str], api_key_getter: Callable[[], str | None] | None = None) -> None:
        self._url = engine_url_getter
        self._key = api_key_getter or (lambda: None)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))

    def upstream_headers(self, incoming: list[tuple[str, str]]) -> dict[str, str]:
        """The headers the engine gets: hop-by-hop and Studio credentials dropped, the engine's key enforced."""

        headers = {k: v for k, v in incoming if k.lower() not in HOP and k.lower() not in STUDIO_HEADERS}
        key = self._key()
        if key:
            headers = {k: v for k, v in headers.items() if k.lower() not in ENGINE_AUTH}
            headers["Authorization"] = f"Bearer {key}"        # the page never holds the engine's key itself
        return headers

    async def handle(self, request: Request, path: str) -> Response:
        url = self._url().rstrip("/") + "/" + path
        headers = self.upstream_headers(list(request.headers.items()))
        params = [(k, v) for k, v in request.query_params.multi_items() if k not in STUDIO_PARAMS]
        body = await request.body()
        try:
            upstream = self._client.build_request(request.method, url, headers=headers, content=body, params=params)
            resp = await self._client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            return Response(content=f'{{"error":{{"message":"engine unreachable: {exc}"}}}}', status_code=502,
                            media_type="application/json")
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}

        async def body_iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        # closed after the response whatever happened to the body (a client gone before its first chunk leaves the
        # generator never started); closing twice is harmless
        return StreamingResponse(body_iter(), status_code=resp.status_code, headers=out_headers,
                                 background=BackgroundTask(resp.aclose))
