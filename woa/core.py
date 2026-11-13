import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def data_directory():
    path = Path(os.environ.get("WOA_DATA_DIR", Path.home() / ".woa-awe-oo"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_checkpoint_path(default_path=None):
    configured = os.environ.get("WOA_MODEL_PATH")
    if configured:
        return Path(configured)
    if default_path is not None:
        return Path(default_path)
    return data_directory() / "model.pt"


class Store:
    def __init__(self, path=None):
        self.path = Path(path) if path else data_directory() / "woa.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_by_conversation
                    ON messages(conversation_id, created_at, id);
                CREATE TABLE IF NOT EXISTS training_events (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (source, content_hash, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS knowledge_chunks_by_source
                    ON knowledge_chunks(source);
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    source UNINDEXED,
                    chunk_index UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                );
                """
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 10000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_conversation(self, title="New conversation"):
        conversation_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_messages(self, conversation_id, limit=None):
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("Message limit must be a positive integer.")
        with self.connect() as db:
            if limit is None:
                rows = db.execute(
                    "SELECT id, conversation_id, role, content, created_at "
                    "FROM messages WHERE conversation_id = ? "
                    "ORDER BY created_at, rowid",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id, conversation_id, role, content, created_at "
                    "FROM ("
                    "SELECT id, conversation_id, role, content, created_at, rowid "
                    "FROM messages WHERE conversation_id = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?"
                    ") ORDER BY created_at, rowid",
                    (conversation_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, conversation_id, role, content):
        message = {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": utc_now(),
        }
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, created_at) "
                "VALUES (:id, :conversation_id, :role, :content, :created_at)",
                message,
            )
            if cursor.rowcount != 1:
                raise RuntimeError("The message was not saved.")
            if role == "user":
                db.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (message["created_at"], conversation_id),
                )
                db.execute(
                    "UPDATE conversations SET title = ? "
                    "WHERE id = ? AND title = 'New conversation'",
                    (content[:60].strip() or "New conversation", conversation_id),
                )
        return message


    def record_training_event(self, event_type, payload):
        event = {
            "id": str(uuid.uuid4()),
            "created_at": utc_now(),
            "event_type": event_type,
            "payload": json.dumps(payload),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO training_events (id, created_at, event_type, payload) "
                "VALUES (:id, :created_at, :event_type, :payload)",
                event,
            )
        return event

    def list_training_events(self, limit=100):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, created_at, event_type, payload "
                "FROM training_events ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]
