"""DashScope text-embedding-v4 adapter with an isolated, persistent vector cache."""

from __future__ import annotations

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import requests


DEFAULT_EMBEDDING = {
    "model": "text-embedding-v4",
    "endpoint": None,
    "endpoint_env": "DASHSCOPE_EMBEDDING_URL",
    "api_key_env": "DASHSCOPE_API_KEY",
    "dimension": 1024,
    "batch_size": 10,
    "timeout_seconds": 60,
    "max_retries": 2,
    "max_input_tokens": None,
    "max_api_requests": None,
    "min_request_interval_seconds": 0,
    "workers": 1,
    "cache_dir": "runs/embedding_cache",
}
DIMENSIONS = {64, 128, 256, 512, 768, 1024, 1536, 2048}


class EmbeddingError(ValueError):
    """A provider/configuration error whose message contains no credentials."""


def _validate_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str):
        raise EmbeddingError("Embedding endpoint must be a URL")
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password and not parsed.query
                 and not parsed.fragment and not re.search(r"[\s{}<>]", endpoint))
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            valid = False
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise EmbeddingError("Embedding endpoint must use HTTPS (HTTP allowed on localhost), without credentials, query, fragment, or placeholders")
    return endpoint.rstrip("/")


def embedding_config(raw: dict | None = None) -> dict:
    if raw is not None and (not isinstance(raw, dict) or set(raw) - set(DEFAULT_EMBEDDING)):
        raise EmbeddingError("Unsupported embedding configuration fields; credentials belong in environment variables")
    config = dict(DEFAULT_EMBEDDING, **(raw or {}))
    if config["model"] != "text-embedding-v4":
        raise EmbeddingError("This adapter requires model=text-embedding-v4")
    for name in ("endpoint_env", "api_key_env"):
        if not isinstance(config[name], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config[name]):
            raise EmbeddingError(f"embedding.{name} must name an environment variable")
    if config["endpoint"] is not None:
        config["endpoint"] = _validate_endpoint(config["endpoint"])
    for name in ("dimension", "batch_size", "max_retries", "workers"):
        if not isinstance(config[name], int) or isinstance(config[name], bool):
            raise EmbeddingError(f"embedding.{name} must be an integer")
    if config["dimension"] not in DIMENSIONS:
        raise EmbeddingError("Unsupported text-embedding-v4 dimension")
    if not 1 <= config["batch_size"] <= 10 or not 0 <= config["max_retries"] <= 5:
        raise EmbeddingError("Embedding batch_size must be 1..10 and max_retries must be 0..5")
    if not 1 <= config["workers"] <= 8:
        raise EmbeddingError("Embedding workers must be 1..8")
    for name in ("max_input_tokens", "max_api_requests"):
        value = config[name]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise EmbeddingError(f"embedding.{name} must be null or a positive integer")
    timeout = config["timeout_seconds"]
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise EmbeddingError("Embedding timeout_seconds must be positive and finite")
    interval = config["min_request_interval_seconds"]
    if (not isinstance(interval, (int, float)) or isinstance(interval, bool)
            or not math.isfinite(interval) or interval < 0):
        raise EmbeddingError("Embedding min_request_interval_seconds must be nonnegative and finite")
    if not isinstance(config["cache_dir"], str) or not config["cache_dir"].strip():
        raise EmbeddingError("Embedding cache_dir must be a nonempty path")
    return config


def _cache_keys(config: dict, endpoint: str, texts: list[str], text_type: str,
                scope_id: str) -> list[str]:
    identity = [1, endpoint, config["model"], config["dimension"], scope_id, text_type]
    prefix = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return [hashlib.sha256((prefix + "\0" + text).encode("utf-8")).hexdigest() for text in texts]


class ReadOnlyEmbeddingCache:
    """Offline reader: no credentials, HTTP client, cache writes, or fallback.

    A missing vector is an error even when provider credentials happen to exist
    in the environment. Use this for experiments which must incur zero cost.
    """

    def __init__(self, config: dict):
        self.config = embedding_config(config)
        endpoint = self.config["endpoint"] or os.environ.get(self.config["endpoint_env"], "")
        if not endpoint:
            raise EmbeddingError("The original endpoint identity is required to read cached vectors")
        self.endpoint = _validate_endpoint(endpoint.strip())
        path = (Path(self.config["cache_dir"]) / "vectors.sqlite3").resolve()
        if not path.is_file():
            raise EmbeddingError("Embedding cache does not exist; offline mode cannot create it")
        self._connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        self._connection.execute("PRAGMA query_only=ON")
        self.stats = {"api_requests": 0, "retries": 0, "embedded_texts": 0,
                      "cache_hits": 0, "input_tokens": 0, "cache_misses": 0}

    def encode(self, texts: list[str], *, text_type: str, scope_id: str,
               use_cache: bool = True) -> np.ndarray:
        if not use_cache:
            raise EmbeddingError("Offline mode requires the cache; network fallback is disabled")
        if text_type not in {"document", "query"} or not isinstance(scope_id, str) or not scope_id:
            raise EmbeddingError("Embedding requires text_type=query/document and a nonempty isolated scope")
        if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise EmbeddingError("Embedding inputs must be nonempty text strings")
        keys = _cache_keys(self.config, self.endpoint, texts, text_type, scope_id)
        vectors = []
        for key in keys:
            row = self._connection.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.stats["cache_misses"] += 1
                raise EmbeddingError(f"Cache miss for {text_type} in {scope_id}; network fallback is disabled")
            try:
                vector = np.frombuffer(row[0], dtype="<f4").copy()
            except (ValueError, TypeError):
                raise EmbeddingError("Invalid vector in read-only embedding cache") from None
            if (vector.shape != (self.config["dimension"],) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)):
                raise EmbeddingError("Invalid vector in read-only embedding cache")
            vectors.append(vector)
            self.stats["cache_hits"] += 1
        return np.stack(vectors) if vectors else np.empty((0, self.config["dimension"]), dtype=np.float32)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DashScopeEmbeddingClient:
    """Native HTTP API preserves the recommended query/document distinction.

    Only memory text or a question is sent to the provider. Scope IDs are used
    solely to partition the local cache and never appear in an API request.
    """

    def __init__(self, config: dict, *, require_credentials: bool = True):
        self.config = embedding_config(config)
        self.endpoint = (self.config["endpoint"] or os.environ.get(self.config["endpoint_env"], "")).strip()
        if self.endpoint:
            self.endpoint = _validate_endpoint(self.endpoint)
        self._api_key = os.environ.get(self.config["api_key_env"], "").strip()
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._sessions = []
        self._pool = ThreadPoolExecutor(max_workers=self.config["workers"]) if self.config["workers"] > 1 else None
        self._next_request_started = 0.0
        self._reserved_input_bytes = 0
        self.stats = {"api_requests": 0, "retries": 0, "embedded_texts": 0,
                      "cache_hits": 0, "input_tokens": 0}
        if require_credentials:
            self.require_ready()

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True, cancel_futures=True)
        for session in self._sessions:
            session.close()

    def _session(self):
        if not hasattr(self._thread_local, "session"):
            session = requests.Session()
            self._thread_local.session = session
            with self._lock:
                self._sessions.append(session)
        return self._thread_local.session

    def require_ready(self) -> None:
        missing = []
        if not self._api_key:
            missing.append(self.config["api_key_env"])
        if not self.endpoint:
            missing.append(self.config["endpoint_env"])
        if missing:
            raise EmbeddingError("Missing embedding configuration: " + ", ".join(missing))

    def status(self) -> dict:
        return {"model": self.config["model"], "dimension": self.config["dimension"],
                "endpoint": self.endpoint or None, "endpoint_configured": bool(self.endpoint),
                "api_key_configured": bool(self._api_key), "cache_dir": self.config["cache_dir"]}

    def _validate_vector(self, value: object) -> np.ndarray:
        try:
            if (not isinstance(value, list) or len(value) != self.config["dimension"]
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
                raise ValueError
            vector = np.asarray(value, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise EmbeddingError("Embedding provider returned an invalid, zero, or wrong-dimension vector") from None
        return (vector / norm).astype("<f4")

    def _request(self, texts: list[str], text_type: str) -> list[np.ndarray]:
        self.require_ready()
        token_cap = self.config["max_input_tokens"]
        reserved = sum(len(t.encode("utf-8")) for t in texts)
        # Include in-flight reservations so parallel requests share one budget.
        with self._lock:
            if token_cap is not None and self.stats["input_tokens"] + self._reserved_input_bytes + reserved > token_cap:
                raise EmbeddingError("Embedding input-token budget reached; no further request was sent")
            self._reserved_input_bytes += reserved
        try:
            return self._perform_request(texts, text_type)
        finally:
            with self._lock:
                self._reserved_input_bytes -= reserved

    def _perform_request(self, texts: list[str], text_type: str) -> list[np.ndarray]:
        payload = {"model": self.config["model"], "input": {"texts": texts},
                   "parameters": {"text_type": text_type, "dimension": self.config["dimension"],
                                  "output_type": "dense"}}
        response = None
        for attempt in range(self.config["max_retries"] + 1):
            with self._lock:
                if self.config["max_api_requests"] is not None and self.stats["api_requests"] >= self.config["max_api_requests"]:
                    raise EmbeddingError("Embedding request budget reached; no further request was sent")
                scheduled = max(time.monotonic(), self._next_request_started)
                self._next_request_started = scheduled + self.config["min_request_interval_seconds"]
                self.stats["api_requests"] += 1
            delay = scheduled - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            try:
                response = self._session().post(self.endpoint, json=payload,
                                         headers={"Authorization": f"Bearer {self._api_key}"},
                                         timeout=self.config["timeout_seconds"], allow_redirects=False)
            except requests.RequestException:
                if attempt == self.config["max_retries"]:
                    raise EmbeddingError("Embedding provider network request failed") from None
            else:
                if 200 <= response.status_code < 300:
                    break
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.config["max_retries"]:
                    raise EmbeddingError(f"Embedding provider returned HTTP {response.status_code}")
            with self._lock:
                self.stats["retries"] += 1
            time.sleep(min(2 ** attempt, 8))
        try:
            body = response.json()
            if body.get("code") not in {None, "", "OK"} or body.get("status_code", 200) != 200:
                raise ValueError
            entries = body["output"]["embeddings"]
            if not isinstance(entries, list) or len(entries) != len(texts):
                raise ValueError
            ordered: dict[int, np.ndarray] = {}
            for item in entries:
                index = item["text_index"]
                if (not isinstance(index, int) or isinstance(index, bool)
                        or index not in range(len(texts)) or index in ordered):
                    raise ValueError
                ordered[index] = self._validate_vector(item["embedding"])
            vectors = [ordered[index] for index in range(len(texts))]
        except EmbeddingError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError):
            raise EmbeddingError("Embedding provider returned malformed or incomplete output") from None
        usage = body.get("usage")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        with self._lock:
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                self.stats["input_tokens"] += tokens
            elif self.config["max_input_tokens"] is not None:
                raise EmbeddingError("Embedding provider omitted token usage; refusing to continue under a token budget")
            self.stats["embedded_texts"] += len(texts)
        return vectors

    def encode(self, texts: list[str], *, text_type: str, scope_id: str, use_cache: bool = True) -> np.ndarray:
        if text_type not in {"document", "query"} or not isinstance(scope_id, str) or not scope_id:
            raise EmbeddingError("Embedding requires text_type=query/document and a nonempty isolated scope")
        if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise EmbeddingError("Embedding inputs must be nonempty text strings")
        if not texts:
            return np.empty((0, self.config["dimension"]), dtype=np.float32)
        # A UTF-8 byte bound is conservative relative to the 8,192-token limit.
        # Reject instead of silently truncating and breaking source attribution.
        if any(len(text.encode("utf-8")) > 8192 for text in texts):
            raise EmbeddingError("Embedding input exceeds conservative 8192-byte limit; reduce chunk_words")
        if not use_cache:
            return np.stack([vector for start in range(0, len(texts), self.config["batch_size"])
                             for vector in self._request(texts[start:start + self.config["batch_size"]], text_type)])
        keys = _cache_keys(self.config, self.endpoint, texts, text_type, scope_id)
        cache_dir = Path(self.config["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectors: dict[str, np.ndarray] = {}
        pending: dict[str, str] = {}
        with closing(sqlite3.connect(cache_dir / "vectors.sqlite3")) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            for key, text in zip(keys, texts):
                if key in vectors or key in pending:
                    self.stats["cache_hits"] += 1
                    continue
                row = connection.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
                if row is None:
                    pending[key] = text
                    continue
                try:
                    vector = np.frombuffer(row[0], dtype="<f4").copy()
                except (TypeError, ValueError):
                    raise EmbeddingError("Embedding cache is corrupt; use a new cache_dir") from None
                if (vector.shape != (self.config["dimension"],) or not np.isfinite(vector).all()
                        or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)):
                    raise EmbeddingError("Embedding cache is corrupt; use a new cache_dir")
                vectors[key] = vector
                self.stats["cache_hits"] += 1
            items = list(pending.items())
            batches = [items[start:start + self.config["batch_size"]]
                       for start in range(0, len(items), self.config["batch_size"])]

            def commit_batch(batch, encoded):
                with connection:
                    for (key, _), vector in zip(batch, encoded):
                        vectors[key] = vector
                        connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, vector.tobytes()))

            if self._pool and len(batches) > 1:
                futures = {self._pool.submit(self._request, [text for _, text in batch], text_type): batch
                           for batch in batches}
                failure = None
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    try:
                        encoded = future.result()
                    except Exception as error:
                        if failure is None:
                            failure = error
                            for pending_future in futures:
                                pending_future.cancel()
                    else:
                        # Drain and save successful in-flight batches even if
                        # another batch failed; avoid paying for them twice.
                        commit_batch(futures[future], encoded)
                if failure is not None:
                    raise failure
            else:
                for batch in batches:
                    commit_batch(batch, self._request([text for _, text in batch], text_type))
        return np.stack([vectors[key] for key in keys])
