"""User-isolated, durable vectors. Remote access requires explicit opt-in.

Only raw document bodies and original queries reach the supplied provider.
The service cache is separate from benchmark caches and survives restarts.
"""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import threading

import numpy as np


class RetrievalUnavailable(RuntimeError):
    """A retryable retrieval failure; never expose provider payloads."""


VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS service_vectors (
 user_id TEXT NOT NULL, namespace TEXT NOT NULL, text_type TEXT NOT NULL,
 text_sha256 TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
 PRIMARY KEY(user_id, namespace, text_type, text_sha256)
);
CREATE TABLE IF NOT EXISTS service_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
VECTOR_READ_BATCH = 200  # Below SQLite's conservative 999-bind limit.


def validated_vectors(values, count, dimension=None):
    try:
        array = np.asarray(values, dtype=np.float64)
        if (array.ndim != 2 or len(array) != count or array.shape[1] < 1
                or dimension is not None and array.shape[1] != dimension
                or not np.isfinite(array).all()):
            raise ValueError
        norms = np.linalg.norm(array, axis=1)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError
        return (array / norms[:, None]).astype("<f4")
    except (ValueError, TypeError, OverflowError):
        raise RetrievalUnavailable("invalid_embedding_vectors") from None


class DurableVectors:
    def __init__(self, store, provider, namespace: str, *, dimension=None):
        if not namespace:
            raise ValueError("An embedding namespace is required")
        self.store, self.provider = store, provider
        self.namespace, self.dimension = namespace, dimension
        self._locks = [threading.RLock() for _ in range(64)]
        self._stats_lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "provider_texts": 0}

    def _encode(self, texts, *, text_type, scope_id, allow_provider):
        if text_type not in {"document", "query"} or not scope_id:
            raise ValueError("Embedding needs a scope and query/document type")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding text must be nonempty")
        if not texts:
            return np.empty((0, self.dimension or 0), dtype=np.float32)
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        # Ordered stripes prevent duplicate concurrent queries within a worker.
        stripes = sorted({int(key[:8], 16) % len(self._locks) for key in keys})
        with ExitStack() as stack:
            for stripe in stripes:
                stack.enter_context(self._locks[stripe])
            found, missing = {}, {}
            with self.store.connect() as db:
                db.execute("BEGIN")
                unique = tuple(dict.fromkeys(keys))
                for offset in range(0, len(unique), VECTOR_READ_BATCH):
                    batch = unique[offset:offset + VECTOR_READ_BATCH]
                    marks = ",".join("?" for _ in batch)
                    rows = db.execute(f"""SELECT text_sha256,dimension,vector FROM service_vectors
                        WHERE user_id=? AND namespace=? AND text_type=? AND text_sha256 IN ({marks})""",
                        (scope_id, self.namespace, text_type, *batch)).fetchall()
                    for row in rows:
                        try:
                            vector = np.frombuffer(row["vector"], dtype="<f4").copy()
                        except (TypeError, ValueError):
                            raise RetrievalUnavailable("corrupt_service_vector") from None
                        if (vector.ndim != 1 or len(vector) != row["dimension"]
                                or self.dimension is not None and len(vector) != self.dimension
                                or not np.isfinite(vector).all()
                                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)):
                            raise RetrievalUnavailable("corrupt_service_vector")
                        found[row["text_sha256"]] = vector
            missing = {key: text for key, text in zip(keys, texts) if key not in found}
            with self._stats_lock:
                self.stats["hits"] += len(texts) - len(missing)
                self.stats["misses"] += len(missing)
            if missing:
                if not allow_provider:
                    raise RetrievalUnavailable("document_index_not_ready")
                provider_texts, slices = [], []
                for text in missing.values():
                    begin = len(provider_texts)
                    if text_type == "query" and len(text.encode("utf-8")) > 6000:
                        from .retrieval import raw_ranges
                        parts = [text[a:b].strip() for a, b in raw_ranges(text) if text[a:b].strip()]
                    else:
                        parts = [text]
                    provider_texts.extend(parts)
                    slices.append((begin, len(provider_texts)))
                try:
                    encoded = self.provider.encode(provider_texts, text_type=text_type,
                                                   scope_id=scope_id)
                except Exception:
                    raise RetrievalUnavailable("embedding_provider_unavailable") from None
                parts = validated_vectors(encoded, len(provider_texts), self.dimension)
                if all(b - a == 1 for a, b in slices):
                    encoded = parts
                else:
                    pooled = [np.average(parts[a:b], axis=0,
                                         weights=[len(t.encode("utf-8")) for t in provider_texts[a:b]])
                              for a, b in slices]
                    encoded = validated_vectors(pooled, len(missing), self.dimension)
                if found and any(len(v) != encoded.shape[1] for v in found.values()):
                    raise RetrievalUnavailable("embedding_dimension_changed")
                with self.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    for key, vector in zip(missing, encoded):
                        db.execute("INSERT OR IGNORE INTO service_vectors VALUES(?,?,?,?,?,?)",
                                   (scope_id, self.namespace, text_type, key, len(vector), vector.tobytes()))
                        found[key] = vector
                    db.commit()
                with self._stats_lock:
                    self.stats["provider_texts"] += len(provider_texts)
            try:
                return np.stack([found[key] for key in keys])
            except ValueError:
                raise RetrievalUnavailable("embedding_dimension_changed") from None

    def prepare(self, texts, *, scope_id):
        return self._encode(texts, text_type="document", scope_id=scope_id, allow_provider=True)

    def encode(self, texts, *, text_type, scope_id):
        # A Search may embed its query, but must never re-embed documents.
        return self._encode(texts, text_type=text_type, scope_id=scope_id,
                            allow_provider=text_type == "query")


def configured_provider(path: Path, mode: str):
    """Construct a cache reader or an explicitly enabled native v4 client.

    Construction never sends a model request. Cache readers are opened inside
    encode, so SQLite connections remain on the request's own thread.
    """
    import os
    from ..embeddings import (DashScopeEmbeddingClient, ReadOnlyEmbeddingCache,
                              _validate_endpoint, embedding_config)

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = embedding_config(raw.get("embedding", raw))
    endpoint = _validate_endpoint(config["endpoint"] or os.environ.get(config["endpoint_env"], ""))
    namespace = hashlib.sha256(json.dumps(
        ["raw-body-v1", endpoint, config["model"], config["dimension"]],
        separators=(",", ":")).encode()).hexdigest()
    if mode == "remote":
        if os.environ.get("AML_ALLOW_EMBEDDING_API") != "1":
            raise ValueError("Remote embeddings require AML_ALLOW_EMBEDDING_API=1")
        provider = DashScopeEmbeddingClient(config)
    elif mode == "cache_only":
        class CacheProvider:
            def __init__(self):
                self.stats = {"cache_hits": 0, "cache_misses": 0, "api_requests": 0}
                self._lock = threading.Lock()

            def encode(self, texts, *, text_type, scope_id):
                with ReadOnlyEmbeddingCache(config) as cache:
                    try:
                        return cache.encode(texts, text_type=text_type, scope_id=scope_id)
                    finally:
                        with self._lock:
                            for name in self.stats:
                                self.stats[name] += cache.stats[name]
        # Validate the file without ever consulting model credentials.
        with ReadOnlyEmbeddingCache(config):
            pass
        provider = CacheProvider()
    else:
        raise ValueError("embedding_mode must be cache_only or remote")
    return provider, namespace, config["dimension"]
