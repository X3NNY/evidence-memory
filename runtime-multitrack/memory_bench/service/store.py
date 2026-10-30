"""Durable raw messages with isolated BM25 or v7-derived hybrid indexes."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading

from ..core.models import Memory
from ..core.retrieval import BM25Retriever
from .models import AddRequest, SearchRequest, timestamp_iso
from .repository import PreparedAdd, RequestConflict, SQLiteMemoryRepository
from .vectors import DurableVectors


@dataclass(frozen=True)
class IndexedUser:
    generation: int
    retriever: object
    created_at: dict[str, str]


class TextStore:
    def __init__(self, path: Path, *, cache_users: int = 8, timeout: float = 10,
                 retriever: str = "bm25", provider=None, namespace=None, dimension=None,
                 body_weight: float = 0., query_cache_entries=16, query_cache_bytes=4 * 1024 * 1024,
                 window_context_chars=48000, session_frames=True):
        if not isinstance(cache_users, int) or isinstance(cache_users, bool) or cache_users < 1:
            raise ValueError("cache_users must be positive")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout, self.cache_users = timeout, cache_users
        self._cache: OrderedDict[str, IndexedUser] = OrderedDict()
        self._lock = threading.Lock()
        self._write_locks = [threading.RLock() for _ in range(64)]
        self._index_locks = [threading.Lock() for _ in range(64)]
        if retriever not in {"bm25", "hybrid", "lexical"} or retriever == "hybrid" and provider is None:
            raise ValueError("Hybrid retrieval requires an explicit embedding provider")
        if retriever == "hybrid" and not namespace:
            raise ValueError("Hybrid retrieval requires an embedding namespace")
        self.retriever, self.body_weight = retriever, body_weight
        self.query_cache_entries, self.query_cache_bytes = query_cache_entries, query_cache_bytes
        self.window_context_chars = window_context_chars
        self.session_frames = session_frames
        self.vectors = None
        self.repository = SQLiteMemoryRepository(self.path, timeout=timeout,
            namespace=namespace if retriever == "hybrid" else None)
        if retriever == "hybrid":
            self.vectors = DurableVectors(self, provider, namespace, dimension=dimension)

    def connect(self):
        return self.repository.connect()

    def healthy(self):
        return self.repository.healthy()

    def add(self, request: AddRequest) -> dict:
        command = PreparedAdd.from_request(request)
        if self.vectors is None:
            return self.repository.append(command)
        stripe = int(hashlib.sha256(request.user_id.encode("utf-8")).hexdigest()[:8], 16) % len(self._write_locks)
        with self._write_locks[stripe]:
            previous = self.repository.existing(command)
            if previous is not None:
                return previous
            from .retrieval import project_rows
            chunks = project_rows(command.rows)
            # Only new bodies are embedded, outside a journal write transaction.
            # A failed journal commit can reuse persisted vectors on retry.
            self.vectors.prepare([c.raw.strip() for c in chunks], scope_id=request.user_id)
            return self.repository.append(command)

    def _remember(self, user_id, built):
        with self._lock:
            newer = self._cache.get(user_id)
            if newer is None or newer.generation <= built.generation:
                self._cache[user_id] = built
                self._cache.move_to_end(user_id)
            while len(self._cache) > self.cache_users:
                self._cache.popitem(last=False)

    def _index(self, user_id: str) -> IndexedUser | None:
        # Separate from write locks: a slow in-flight Add must not block reads
        # of the prior committed generation. Waiters re-read the generation.
        stripe = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8], 16) % len(self._index_locks)
        with self._index_locks[stripe]:
            return self._index_snapshot(user_id)

    def _index_snapshot(self, user_id: str) -> IndexedUser | None:
        with self._lock:
            cached = self._cache.get(user_id)
        snapshot = self.repository.snapshot(user_id, cached_generation=cached.generation if cached else None)
        if snapshot is None:
            return None
        generation, rows = snapshot.generation, snapshot.rows
        if rows is None:
            with self._lock:
                if self._cache.get(user_id) is cached:
                    self._cache.move_to_end(user_id)
            return cached
        if self.retriever in {"hybrid", "lexical"}:
            from .retrieval import HybridIndex, LexicalIndex, project_rows
            constructor = HybridIndex if self.retriever == "hybrid" else LexicalIndex
            vector_args = {"vectors": self.vectors} if self.retriever == "hybrid" else {}
            index = constructor(user_id, project_rows(rows), **vector_args, body_weight=self.body_weight,
                                query_cache_entries=self.query_cache_entries, query_cache_bytes=self.query_cache_bytes,
                                window_context_chars=self.window_context_chars, session_frames=self.session_frames)
            built = IndexedUser(generation, index, index.dates)
            self._remember(user_id, built)
            return built
        memories, dates = [], {}
        for row in rows:
            date = timestamp_iso(row["source_timestamp"]) if row["source_timestamp"] is not None else None
            # Explicit source metadata, not summaries or generated answers.
            text = (f"Conversation timestamp: {date}\n" if date else "") + f"{row['role']}: {row['content']}"
            memories.append(Memory(row["memory_id"], user_id, (), row["session_id"], text))
            dates[row["memory_id"]] = date or row["persisted_at"]
        built = IndexedUser(generation, BM25Retriever(memories), dates)
        self._remember(user_id, built)
        return built

    def search(self, request: SearchRequest) -> dict:
        index = self._index(request.user_id)
        if index is None:
            return {"data": []}
        if self.retriever in {"hybrid", "lexical"}:
            return index.retriever.search(request.query, request.top_k)
        # Options are accepted separately; the query is not replaced, and
        # choices are never ingested or mistaken for remembered facts.
        ranked = index.retriever.search(request.query, request.top_k)
        return {"data": [{"id": item.memory.id, "content": item.memory.text, "score": item.score,
                          "created_at": index.created_at[item.memory.id]}
                         for item in ranked if item.score > 0]}
