"""FastAPI transport with constant-time authentication and bounded JSON bodies."""

from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import sqlite3

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .models import AddRequest, AddResponse, SearchRequest, SearchResponse
from .store import RequestConflict, TextStore
from .vectors import RetrievalUnavailable, configured_provider


VERSION = "text-http-hybrid-v8"
LEXICAL_VERSION = "text-http-lexical-v1"


@dataclass(frozen=True)
class Settings:
    database: Path
    key: str = field(repr=False)
    auth_scheme: str = "bearer"
    max_request_bytes: int = 8 * 1024 * 1024
    cache_users: int = 8
    retriever: str = "hybrid"
    embedding_config: Path | None = None
    embedding_mode: str = "cache_only"
    body_weight: float = 0.
    query_cache_entries: int = 16
    query_cache_bytes: int = 4 * 1024 * 1024
    window_context_chars: int = 48000
    session_frames: bool = True

    def __post_init__(self):
        if not self.key or any(char.isspace() for char in self.key):
            raise ValueError("A nonempty Memory System Key without whitespace is required")
        if self.auth_scheme not in {"bearer", "token", "x-api-key"}:
            raise ValueError("auth_scheme must be bearer, token or x-api-key")
        if not isinstance(self.max_request_bytes, int) or isinstance(self.max_request_bytes, bool) or self.max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        if self.retriever not in {"bm25", "hybrid", "lexical"}:
            raise ValueError("retriever must be bm25, hybrid or lexical")
        if self.embedding_mode not in {"cache_only", "remote"}:
            raise ValueError("embedding_mode must be cache_only or remote")
        if self.body_weight not in {0., .25}:
            raise ValueError("body_weight must be 0 or 0.25")
        for value in (self.query_cache_entries, self.query_cache_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Query cache limits must be nonnegative integers")
        if not isinstance(self.window_context_chars, int) or isinstance(self.window_context_chars, bool) or self.window_context_chars < 0:
            raise ValueError("window_context_chars must be a nonnegative integer")
        if not isinstance(self.session_frames, bool):
            raise ValueError("session_frames must be a boolean")

    @classmethod
    def from_environment(cls):
        key, key_file = os.environ.get("AML_MEMORY_SYSTEM_KEY"), os.environ.get("AML_MEMORY_SYSTEM_KEY_FILE")
        if key and key_file:
            raise ValueError("Configure one Memory System Key source")
        if key_file:
            key = Path(key_file).read_text(encoding="utf-8").rstrip("\r\n")
        if not key:
            raise ValueError("Set AML_MEMORY_SYSTEM_KEY or AML_MEMORY_SYSTEM_KEY_FILE")
        config = os.environ.get("AML_EMBEDDING_CONFIG")
        frames = os.environ.get("AML_SESSION_FRAMES", "1")
        if frames not in {"0", "1"}:
            raise ValueError("AML_SESSION_FRAMES must be 0 or 1")
        return cls(database=Path(os.environ.get("AML_TEXT_DATABASE", "var/text-memory.sqlite3")), key=key,
                   auth_scheme=os.environ.get("AML_AUTH_SCHEME", "bearer").lower(),
                   max_request_bytes=int(os.environ.get("AML_MAX_REQUEST_BYTES", 8 * 1024 * 1024)),
                   cache_users=int(os.environ.get("AML_CACHE_USERS", 8)),
                   retriever=os.environ.get("AML_RETRIEVER", "hybrid").lower(),
                   embedding_config=Path(config) if config else None,
                   embedding_mode=os.environ.get("AML_EMBEDDING_MODE", "cache_only"),
                   body_weight=float(os.environ.get("AML_BODY_LEXICAL_WEIGHT", "0")),
                   query_cache_entries=int(os.environ.get("AML_QUERY_CACHE_ENTRIES", "16")),
                   query_cache_bytes=int(os.environ.get("AML_QUERY_CACHE_BYTES", str(4 * 1024 * 1024))),
                   window_context_chars=int(os.environ.get("AML_WINDOW_CONTEXT_CHARS", "48000")),
                   session_frames=frames == "1")


class RequestGuard:
    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings

    async def reject(self, scope, receive, send, status, reason):
        headers = {"WWW-Authenticate": self.settings.auth_scheme} if status == 401 else None
        await JSONResponse({"detail": {"reason": reason}}, status_code=status, headers=headers)(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in {"/add", "/search"}:
            return await self.app(scope, receive, send)
        headers = scope["headers"]
        auth = [value for name, value in headers if name.lower() == b"authorization"]
        api = [value for name, value in headers if name.lower() == b"x-api-key"]
        credential = None
        if self.settings.auth_scheme == "x-api-key":
            if len(api) == 1 and not auth:
                credential = api[0]
        elif len(auth) == 1 and not api:
            scheme, separator, value = auth[0].partition(b" ")
            if separator and scheme.lower() == self.settings.auth_scheme.encode("ascii"):
                credential = value
        if credential is None or not hmac.compare_digest(credential, self.settings.key.encode("utf-8")):
            return await self.reject(scope, receive, send, 401, "invalid_authentication")
        lengths = [value for name, value in headers if name.lower() == b"content-length"]
        try:
            if len(lengths) > 1 or (lengths and int(lengths[0]) < 0):
                raise ValueError
            if lengths and int(lengths[0]) > self.settings.max_request_bytes:
                return await self.reject(scope, receive, send, 413, "request_body_too_large")
        except ValueError:
            return await self.reject(scope, receive, send, 400, "invalid_content_length")
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.settings.max_request_bytes:
                return await self.reject(scope, receive, send, 413, "request_body_too_large")
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, *, embedder=None,
               embedding_namespace: str | None = None, embedding_dimension: int | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    own_provider = False
    if settings.retriever == "hybrid" and embedder is None:
        if settings.embedding_config is None:
            raise ValueError("Hybrid mode requires AML_EMBEDDING_CONFIG; no BM25 fallback is used")
        embedder, embedding_namespace, embedding_dimension = configured_provider(
            settings.embedding_config, settings.embedding_mode)
        own_provider = True
    if settings.retriever in {"bm25", "lexical"} and embedder is not None:
        raise ValueError("BM25 and lexical modes do not consume an embedding provider")
    store = TextStore(settings.database, cache_users=settings.cache_users,
                      retriever=settings.retriever, provider=embedder, namespace=embedding_namespace,
                      dimension=embedding_dimension, body_weight=settings.body_weight,
                      query_cache_entries=settings.query_cache_entries, query_cache_bytes=settings.query_cache_bytes,
                      window_context_chars=settings.window_context_chars, session_frames=settings.session_frames)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if own_provider and hasattr(embedder, "close"):
                embedder.close()

    version = {"hybrid": VERSION, "bm25": "text-http-bm25-v1", "lexical": LEXICAL_VERSION}[settings.retriever]
    app = FastAPI(title="AML Text Memory", version=version, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.store = store
    app.add_middleware(RequestGuard, settings=settings)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Default validation payloads may echo raw memory content; omit input.
        return JSONResponse({"detail": [{key: entry[key] for key in ("type", "loc", "msg")}
                                        for entry in error.errors()]}, status_code=422)

    @app.exception_handler(RequestConflict)
    async def conflicting_request(request: Request, error: RequestConflict):
        return JSONResponse({"detail": {"reason": "request_id_payload_conflict"}}, status_code=409)

    @app.exception_handler(sqlite3.Error)
    async def database_unavailable(request: Request, error: sqlite3.Error):
        return JSONResponse({"detail": {"reason": "storage_unavailable"}}, status_code=503,
                            headers={"Retry-After": "1"})

    @app.exception_handler(RetrievalUnavailable)
    async def retrieval_unavailable(request: Request, error: RetrievalUnavailable):
        return JSONResponse({"detail": {"reason": "retrieval_temporarily_unavailable"}},
                            status_code=503, headers={"Retry-After": "1"})

    @app.get("/health")
    def health():
        store.healthy()
        return {"status": "ok", "version": version, "retriever": settings.retriever,
                "embedding_mode": settings.embedding_mode if settings.retriever == "hybrid" else "none"}

    @app.post("/add", response_model=AddResponse)
    def add(request: AddRequest):
        return store.add(request)

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        return store.search(request)

    return app
