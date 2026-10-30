"""Zero-model three-track transport with original ordered media evidence.

The existing textual service stays intact. This extension uses the same atomic
journal and evidence policy, adding a typed media side table in the same commit.
"""
import argparse
import asyncio
import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Annotated, Literal
import zlib

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, Field, model_validator

from ..adapters.coding.lexical import CodeLexicalRetriever, code_source
from ..adapters.multimodal.lexical import LexicalParameters, MultimodalLexicalRetriever
from ..core.models import Memory
from ..core.retrieval import BM25Retriever
from .app import RequestGuard, Settings
from .models import AddRequest, AddResponse, Envelope, Message, SearchRequest, Text, timestamp_iso
from .repository import PreparedAdd, RequestConflict, MESSAGE_COLUMNS
from .retrieval import project_rows
from .store import IndexedUser, TextStore
from .offline import loopback_only

VERSION = "multitrack-lexical-v1"
IMAGE_LIMIT = 10 * 1024 * 1024
MEDIA_LIMIT = 30 * 1024 * 1024


def image_bytes(url):
    header, separator, encoded = url.partition(",")
    if not separator or header not in {"data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"}:
        raise ValueError("Images require inline JPEG, PNG or WebP Base64 data URIs")
    if len(encoded) > 4 * ((IMAGE_LIMIT + 2) // 3):
        raise ValueError("Image exceeds 10 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        raise ValueError("Invalid Base64 image") from None
    valid = (header.startswith("data:image/png;") and raw.startswith(b"\x89PNG\r\n\x1a\n")
             or header.startswith("data:image/jpeg;") and raw.startswith(b"\xff\xd8\xff")
             or header.startswith("data:image/webp;") and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP")
    if not valid or len(raw) > IMAGE_LIMIT:
        raise ValueError("Invalid image signature or image exceeds 10 MiB")
    return raw


def valid_image(url):
    image_bytes(url)
    return url


class ImageURL(Envelope):
    url: Annotated[Text, AfterValidator(valid_image)]


class TextPart(Envelope):
    type: Literal["text"]
    text: Text


class ImagePart(Envelope):
    type: Literal["image_url"]
    image_url: ImageURL


Part = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = Text | Annotated[list[Part], Field(min_length=1)]


def parts(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else [p.model_dump() for p in content]


def media_size(content):
    return sum(len(image_bytes(p["image_url"]["url"])) for p in parts(content) if p["type"] == "image_url")


def text_content(content):
    return "\n".join(p["text"] for p in parts(content) if p["type"] == "text")


class MultiMessage(Message):
    content: Content


class MultiAdd(AddRequest):
    messages: Annotated[list[MultiMessage], Field(min_length=1)]

    @model_validator(mode="after")
    def media_budget(self):
        if sum(media_size(message.content) for message in self.messages) > MEDIA_LIMIT:
            raise ValueError("Add images exceed 30 MiB")
        return self


class MultiSearch(SearchRequest):
    query: Content

    @model_validator(mode="after")
    def media_budget(self):
        if media_size(self.query) > MEDIA_LIMIT:
            raise ValueError("Query images exceed 30 MiB")
        return self


class MultiItem(Envelope):
    id: str
    content: Content
    score: float
    created_at: str


class MultiResponse(Envelope):
    data: list[MultiItem]


class BoundedAdmission:
    """Queue before buffering bodies: at most two large media requests in RAM."""
    def __init__(self, app):
        self.app = app
        self.slots = asyncio.Semaphore(2)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] in {"/add", "/search"}:
            async with self.slots:
                return await self.app(scope, receive, send)
        return await self.app(scope, receive, send)


class RecordIndex:
    def __init__(self, user_id, rows, metadata, coding=False, code_candidate=False):
        self.chunks = project_rows(rows)
        self.memories = tuple(Memory(c.id, user_id, (c.source_id,), c.session_id, c.rendered) for c in self.chunks)
        self.by_id = {c.id: c for c in self.chunks}
        self.metadata = metadata
        self.coding = coding
        if coding:
            factory = CodeLexicalRetriever if code_candidate else BM25Retriever
            self.ranker = factory(self.memories)
        else:
            self.ranker = MultimodalLexicalRetriever(self.memories, parameters=LexicalParameters(neighbor_weight=.6))
        self.dates = {c.id: c.created_at for c in self.chunks}

    def ranked(self, query):
        text = text_content(query)
        ranked = self.ranker.search(text, 100) if text.strip() else []
        scores = {item.memory.id: float(item.score) for item in ranked if item.score > 0}
        hashes = {hashlib.sha256(image_bytes(p["image_url"]["url"])).hexdigest()
                  for p in parts(query) if p["type"] == "image_url"}
        if hashes:
            maximum = max(scores.values(), default=0.)
            # Exact original-image identity is a deterministic source match,
            # not a semantic image embedding or a learned similarity score.
            for chunk in self.chunks:
                if hashes.intersection(self.metadata.get(chunk.source_id, {}).get("hashes", ())):
                    scores[chunk.id] = maximum + 1. + scores.get(chunk.id, 0.)
        ordinals = {m.id: i for i, m in enumerate(self.memories)}
        return sorted(scores.items(), key=lambda pair: (-pair[1], ordinals[pair[0]]))


class MultiStore(TextStore):
    def __init__(self, path, *, code_candidate=False, **kwargs):
        super().__init__(path, retriever="lexical", **kwargs)
        self.code_candidate = code_candidate
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ordered_media (
                  memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                  payload BLOB NOT NULL, media_bytes INTEGER NOT NULL,
                  image_hashes TEXT NOT NULL,
                  FOREIGN KEY(memory_id) REFERENCES messages(memory_id));
                CREATE INDEX IF NOT EXISTS ordered_media_user ON ordered_media(user_id);
            """)

    def add(self, request):
        if all(isinstance(m.content, str) for m in request.messages):
            return super().add(AddRequest.model_validate(request.model_dump()))
        plain = AddRequest(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id,
                           messages=[Message(role=m.role, content=text_content(m.content) or "[image]", timestamp=m.timestamp)
                                     for m in request.messages])
        command = PreparedAdd.from_request(plain)
        canonical = json.dumps(request.model_dump(exclude_none=True), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        command = replace(command, signature=hashlib.sha256(canonical.encode()).hexdigest())
        media = []
        for row, message in zip(command.rows, request.messages):
            original = parts(message.content)
            hashes = [hashlib.sha256(image_bytes(p["image_url"]["url"])).hexdigest() for p in original if p["type"] == "image_url"]
            media.append((row["memory_id"], request.user_id, zlib.compress(json.dumps(original, ensure_ascii=False).encode()),
                          media_size(message.content), json.dumps(hashes)))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self.repository._existing(db, command)
            if previous is not None:
                return previous
            db.execute("INSERT OR IGNORE INTO users(user_id) VALUES(?)", (request.user_id,))
            db.execute("INSERT INTO requests VALUES(?,?,?,?,?)", (command.request_id, command.user_id, command.session_id,
                       command.signature, command.response_json))
            db.executemany("INSERT INTO messages (" + ",".join(MESSAGE_COLUMNS) + ") VALUES(" + ",".join("?" for _ in MESSAGE_COLUMNS) + ")",
                           [tuple(row[k] for k in MESSAGE_COLUMNS) for row in command.rows])
            db.executemany("INSERT INTO ordered_media VALUES(?,?,?,?,?)", media)
            db.execute("UPDATE users SET generation=generation+1 WHERE user_id=?", (request.user_id,))
            db.commit()
        return json.loads(command.response_json)

    def _index_snapshot(self, user_id):
        with self._lock:
            cached = self._cache.get(user_id)
        snapshot = self.repository.snapshot(user_id, cached_generation=cached.generation if cached else None)
        if snapshot is None:
            return None
        if snapshot.rows is None:
            with self._lock:
                if self._cache.get(user_id) is cached:
                    self._cache.move_to_end(user_id)
            return cached
        with self.connect() as db:
            metadata = {row["memory_id"]: {"bytes": row["media_bytes"], "hashes": json.loads(row["image_hashes"])}
                        for row in db.execute("SELECT memory_id,media_bytes,image_hashes FROM ordered_media WHERE user_id=?", (user_id,))}
        coding = not metadata and any(code_source(row["content"]) for row in snapshot.rows)
        if not metadata and not coding:
            return super()._index_snapshot(user_id)
        index = RecordIndex(user_id, snapshot.rows, metadata, coding=coding, code_candidate=self.code_candidate)
        built = IndexedUser(snapshot.generation, index, index.dates)
        self._remember(user_id, built)
        return built

    def search(self, request):
        indexed = self._index(request.user_id)
        if indexed is None:
            return {"data": []}
        index = indexed.retriever
        if not isinstance(index, RecordIndex):
            query = text_content(request.query)
            if not query.strip():
                return {"data": []}
            result = index.search(query, request.top_k)
            if isinstance(request.query, list):
                for record in result["data"]:
                    record["content"] = [{"type": "text", "text": record["content"]}]
            return result
        response, used, byte_count = [], set(), 0
        with self.connect() as db:
            for mid, score in index.ranked(request.query):
                chunk = index.by_id[mid]
                metadata = index.metadata.get(chunk.source_id)
                if metadata:
                    if chunk.source_id in used or byte_count + metadata["bytes"] > MEDIA_LIMIT:
                        continue
                    row = db.execute("SELECT payload FROM ordered_media WHERE user_id=? AND memory_id=?",
                                     (request.user_id, chunk.source_id)).fetchone()
                    if row is None:
                        raise sqlite3.DatabaseError("Original media source missing")
                    content = json.loads(zlib.decompress(row["payload"]))
                    identity = chunk.source_id
                    used.add(identity)
                    byte_count += metadata["bytes"]
                else:
                    content = chunk.rendered if index.coding else [{"type": "text", "text": chunk.rendered}]
                    identity = mid
                response.append({"id": identity, "content": content, "score": score, "created_at": chunk.created_at})
                if len(response) == request.top_k:
                    break
        return {"data": response}


def create_multitrack_app(settings=None, *, code_candidate=False):
    settings = settings or replace(Settings.from_environment(), max_request_bytes=48 * 1024 * 1024)
    if settings.retriever != "lexical":
        raise ValueError("Multitrack runtime requires zero-model lexical mode")
    store = MultiStore(settings.database, cache_users=settings.cache_users, code_candidate=code_candidate,
                       query_cache_entries=settings.query_cache_entries, query_cache_bytes=settings.query_cache_bytes,
                       window_context_chars=settings.window_context_chars, session_frames=settings.session_frames)
    app = FastAPI(title="EvidenceMemory", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.store = store
    app.add_middleware(RequestGuard, settings=settings)
    app.add_middleware(BoundedAdmission)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError):
        return JSONResponse({"detail": [{k: row[k] for k in ("type", "loc", "msg")} for row in error.errors()]}, status_code=422)

    @app.exception_handler(RequestConflict)
    async def conflict(request: Request, error: RequestConflict):
        return JSONResponse({"detail": {"reason": "request_id_payload_conflict"}}, status_code=409)

    @app.exception_handler(sqlite3.Error)
    async def unavailable(request: Request, error: sqlite3.Error):
        return JSONResponse({"detail": {"reason": "storage_unavailable"}}, status_code=503, headers={"Retry-After": "1"})

    @app.get("/health")
    def health():
        store.healthy()
        with store.connect() as db:
            db.execute("SELECT memory_id FROM ordered_media LIMIT 0")
        return {"status": "ok", "version": VERSION, "retriever": "lexical", "embedding_mode": "none",
                "tracks": ["textual", "coding", "multimodal"], "code_identifier_channel": code_candidate}

    @app.post("/add", response_model=AddResponse)
    def add(request: MultiAdd):
        return store.add(request)

    @app.post("/search", response_model=MultiResponse)
    def search(request: MultiSearch):
        return store.search(request)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    import uvicorn
    with loopback_only():
        uvicorn.run(create_multitrack_app(code_candidate=os.environ.get("AML_CODE_IDENTIFIERS", "0") == "1"),
                    host=args.host, port=args.port, access_log=False, limit_concurrency=16, timeout_keep_alive=15)


if __name__ == "__main__":
    main()
