from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
import uuid

from rwkv_search.text import search_tokens


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    session_id: str
    content: str
    created_at: str
    score: float = 0.0


@dataclass(frozen=True)
class MessageRecord:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


class MemoryStore:
    """Durable session transcript and explicit long-term Agent memories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_session_created "
                "ON memories(session_id, created_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS messages_session_id "
                "ON messages(session_id, id DESC)"
            )

    @staticmethod
    def _session_id(session_id: str) -> str:
        return str(session_id or "default").strip() or "default"

    def save(self, content: str, *, session_id: str) -> MemoryRecord:
        value = " ".join(str(content or "").split()).strip()
        if not value:
            raise ValueError("memory content must not be empty")
        if len(value) > 4000:
            raise ValueError("memory content exceeds 4000 characters")
        normalized_session = self._session_id(session_id)
        record = MemoryRecord(
            id="MEM-" + uuid.uuid4().hex[:16],
            session_id=normalized_session,
            content=value,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO memories(id, session_id, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    record.id,
                    record.session_id,
                    record.content,
                    record.created_at,
                ),
            )
        return record

    def search(
        self,
        query: str,
        *,
        session_id: str,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        normalized_session = self._session_id(session_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id, content, created_at FROM memories "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT 500",
                (normalized_session,),
            ).fetchall()
        query_tokens = set(search_tokens(query))
        query_folded = str(query or "").casefold().strip()
        scored: list[MemoryRecord] = []
        for row in rows:
            content = str(row["content"])
            content_tokens = set(search_tokens(content))
            overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            substring = 1.0 if query_folded and query_folded in content.casefold() else 0.0
            score = max(overlap, substring)
            if score <= 0.0:
                continue
            scored.append(
                MemoryRecord(
                    id=str(row["id"]),
                    session_id=str(row["session_id"]),
                    content=content,
                    created_at=str(row["created_at"]),
                    score=round(score, 6),
                )
            )
        scored.sort(key=lambda item: (item.score, item.created_at), reverse=True)
        return scored[: max(1, int(limit))]

    def recent(
        self,
        *,
        session_id: str,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        normalized_session = self._session_id(session_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id, content, created_at FROM memories "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (normalized_session, max(1, int(limit))),
            ).fetchall()
        return [
            MemoryRecord(
                id=str(row["id"]),
                session_id=str(row["session_id"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                score=0.0,
            )
            for row in rows
        ]

    def append_exchange(
        self,
        *,
        session_id: str,
        user: str,
        assistant: str,
    ) -> tuple[MessageRecord, MessageRecord]:
        normalized_session = self._session_id(session_id)
        user_text = str(user or "").strip()
        assistant_text = str(assistant or "").strip()
        if not user_text or not assistant_text:
            raise ValueError("conversation messages must not be empty")
        if len(user_text) > 200_000 or len(assistant_text) > 200_000:
            raise ValueError("conversation message exceeds 200000 characters")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            user_cursor = connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?, 'user', ?, ?)",
                (normalized_session, user_text, created_at),
            )
            assistant_cursor = connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?, 'assistant', ?, ?)",
                (normalized_session, assistant_text, created_at),
            )
        return (
            MessageRecord(
                id=int(user_cursor.lastrowid),
                session_id=normalized_session,
                role="user",
                content=user_text,
                created_at=created_at,
            ),
            MessageRecord(
                id=int(assistant_cursor.lastrowid),
                session_id=normalized_session,
                role="assistant",
                content=assistant_text,
                created_at=created_at,
            ),
        )

    def history(
        self,
        *,
        session_id: str,
        limit: int = 12,
    ) -> list[MessageRecord]:
        normalized_session = self._session_id(session_id)
        bounded_limit = min(100, max(1, int(limit)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id, role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (normalized_session, bounded_limit),
            ).fetchall()
        return [
            MessageRecord(
                id=int(row["id"]),
                session_id=str(row["session_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
            )
            for row in reversed(rows)
        ]

    @staticmethod
    def to_dict(record: MemoryRecord) -> dict:
        return asdict(record)
