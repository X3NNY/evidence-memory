"""SQLite journal: atomic raw writes, idempotency and consistent snapshots."""

from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .models import AddRequest
from .vectors import RetrievalUnavailable, VECTOR_SCHEMA


class RequestConflict(ValueError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
 user_id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS requests (
 request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL, response_json TEXT NOT NULL,
 FOREIGN KEY(user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS messages (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL UNIQUE,
 user_id TEXT NOT NULL, session_id TEXT NOT NULL, request_id TEXT NOT NULL,
 position INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
 source_timestamp INTEGER, persisted_at TEXT NOT NULL,
 UNIQUE(request_id, position),
 FOREIGN KEY(user_id) REFERENCES users(user_id),
 FOREIGN KEY(request_id) REFERENCES requests(request_id)
);
CREATE INDEX IF NOT EXISTS messages_user_order ON messages(user_id, sequence);
"""
MESSAGE_COLUMNS = ("memory_id", "user_id", "session_id", "request_id", "position",
                   "role", "content", "source_timestamp", "persisted_at")


@dataclass(frozen=True)
class PreparedAdd:
    request_id: str
    user_id: str
    session_id: str
    signature: str
    response_json: str = field(repr=False)
    rows: tuple[dict, ...] = field(repr=False)

    @classmethod
    def from_request(cls, request: AddRequest):
        canonical = json.dumps(request.model_dump(exclude_none=True), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        response = {"success": True, "request_id": request.request_id,
                    "user_id": request.user_id, "session_id": request.session_id}
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rows = []
        for position, message in enumerate(request.messages):
            identity = json.dumps([request.user_id, request.request_id, position], ensure_ascii=False)
            rows.append(dict(zip(MESSAGE_COLUMNS, (
                "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                request.user_id, request.session_id, request.request_id, position,
                message.role, message.content, message.timestamp, now))))
        return cls(request.request_id, request.user_id, request.session_id, signature,
                   json.dumps(response, ensure_ascii=False), tuple(rows))


@dataclass(frozen=True)
class UserSnapshot:
    generation: int
    # None denotes a matching generation, so no raw messages were loaded.
    rows: tuple[sqlite3.Row, ...] | None


class SQLiteMemoryRepository:
    def __init__(self, path: Path, *, timeout=10, namespace=None):
        self.path, self.timeout, self.namespace = Path(path).resolve(), timeout, namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.path), timeout=timeout)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            if db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise ValueError("Unsupported text service database schema")
            db.executescript(SCHEMA)
            db.executescript(VECTOR_SCHEMA)
            if namespace is not None:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute("SELECT value FROM service_metadata WHERE key='embedding_namespace'").fetchone()
                if previous is None:
                    if db.execute("SELECT 1 FROM messages LIMIT 1").fetchone() is not None:
                        raise ValueError("Use a fresh database for hybrid mode; existing BM25 memories require explicit migration")
                    db.execute("INSERT INTO service_metadata VALUES('embedding_namespace',?)", (namespace,))
                elif previous[0] != namespace:
                    raise ValueError("Embedding namespace changed; use a new versioned database")
            db.execute("PRAGMA user_version=1")
            db.commit()

    @contextmanager
    def connect(self):
        # Do not silently recreate a removed database during a request.
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=self.timeout)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            if db.in_transaction:
                db.rollback()
            db.close()

    @staticmethod
    def _existing(db, command):
        row = db.execute("SELECT payload_sha256,response_json FROM requests WHERE request_id=?",
                         (command.request_id,)).fetchone()
        if row is None:
            return None
        if row["payload_sha256"] != command.signature:
            raise RequestConflict("request_id_payload_conflict")
        return json.loads(row["response_json"])

    def existing(self, command):
        with self.connect() as db:
            return self._existing(db, command)

    def append(self, command: PreparedAdd):
        # Both retrieval modes publish through this transaction. Hybrid callers
        # prepare vectors first, outside the write transaction.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._existing(db, command)
            if previous is not None:
                return previous
            db.execute("INSERT OR IGNORE INTO users(user_id) VALUES(?)", (command.user_id,))
            db.execute("INSERT INTO requests VALUES(?,?,?,?,?)", (command.request_id, command.user_id,
                       command.session_id, command.signature, command.response_json))
            db.executemany("""INSERT INTO messages
                (memory_id,user_id,session_id,request_id,position,role,content,source_timestamp,persisted_at)
                VALUES(?,?,?,?,?,?,?,?,?)""", [tuple(row[k] for k in MESSAGE_COLUMNS) for row in command.rows])
            db.execute("UPDATE users SET generation=generation+1 WHERE user_id=?", (command.user_id,))
            db.commit()
        return json.loads(command.response_json)

    def snapshot(self, user_id, *, cached_generation=None):
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT generation FROM users WHERE user_id=?", (user_id,)).fetchone()
            if row is None:
                return None
            generation = row["generation"]
            rows = None if generation == cached_generation else tuple(db.execute(
                "SELECT * FROM messages WHERE user_id=? ORDER BY sequence", (user_id,)).fetchall())
            return UserSnapshot(generation, rows)

    def healthy(self):
        with self.connect() as db:
            db.execute("SELECT generation FROM users LIMIT 1").fetchall()
            db.execute("SELECT memory_id FROM messages LIMIT 0")
            db.execute("SELECT request_id FROM requests LIMIT 0")
            if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise sqlite3.DatabaseError("schema unavailable")
            if self.namespace is not None:
                db.execute("SELECT namespace FROM service_vectors LIMIT 0")
                row = db.execute("SELECT value FROM service_metadata WHERE key='embedding_namespace'").fetchone()
                if row is None or row[0] != self.namespace:
                    raise RetrievalUnavailable("embedding_namespace_unavailable")
