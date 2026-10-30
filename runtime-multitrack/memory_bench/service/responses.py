"""Bounded, immutable results for one user and one index generation."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading


@dataclass(frozen=True)
class CachedRecord:
    id: str
    content: str
    score: float
    created_at: str


@dataclass(frozen=True)
class SourceRecord(CachedRecord):
    # Immutable provenance lives and expires with the bounded response cache.
    source_json: str


class ResponseCache:
    def __init__(self, *, max_entries=16, max_bytes=4 * 1024 * 1024):
        for value in (max_entries, max_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Result cache limits must be nonnegative integers")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._flights = [threading.Lock() for _ in range(64)]
        self._stats = {"hits": 0, "computations": 0, "evictions": 0, "oversize_skips": 0}

    @property
    def stats(self):
        with self._lock:
            return {**self._stats, "entries": len(self._entries), "estimated_bytes": self._bytes}

    def peek(self, query):
        with self._lock:
            value = self._entries.get(query)
            return value[0] if value is not None else None

    def _lookup(self, query):
        with self._lock:
            value = self._entries.get(query)
            if value is not None:
                self._entries.move_to_end(query)
                self._stats["hits"] += 1
                return value[0]
        return None

    def _compute(self, build):
        with self._lock:
            self._stats["computations"] += 1
        return tuple(build())

    def get(self, query, build):
        if not self.max_entries or not self.max_bytes:
            return self._compute(build)
        cached = self._lookup(query)
        if cached is not None:
            return cached
        stripe = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16) % len(self._flights)
        with self._flights[stripe]:
            cached = self._lookup(query)
            if cached is not None:
                return cached
            records = self._compute(build)  # Exceptions are never cached.
            # Charge conservative Unicode storage, keys and record metadata.
            # Strings may share index storage; charging the full text is safer.
            cost = 4 * len(query) + 256 + sum(4 * (len(r.id) + len(r.content) + len(r.created_at) + len(getattr(r, "source_json", ""))) + 256
                                            for r in records)
            with self._lock:
                if cost > self.max_bytes:
                    self._stats["oversize_skips"] += 1
                    return records
                while self._entries and (len(self._entries) >= self.max_entries or self._bytes + cost > self.max_bytes):
                    _, (_, previous_cost) = self._entries.popitem(last=False)
                    self._bytes -= previous_cost
                    self._stats["evictions"] += 1
                self._entries[query] = (records, cost)
                self._bytes += cost
            return records


def response(records, top_k):
    # Fresh dictionaries prevent a caller from mutating another request's result.
    return {"data": [{"id": r.id, "content": r.content, "score": r.score, "created_at": r.created_at}
                     for r in records[:top_k]]}
