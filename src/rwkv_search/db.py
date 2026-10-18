from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .text import content_hash, indexed_text, simhash64


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    search_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    simhash TEXT NOT NULL,
    published_at TEXT,
    fetched_at REAL NOT NULL,
    etag TEXT,
    last_modified TEXT,
    content_type TEXT,
    language TEXT,
    source_type TEXT NOT NULL DEFAULT 'web',
    authority REAL NOT NULL DEFAULT 0.5,
    http_status INTEGER NOT NULL DEFAULT 200,
    response_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_fetched ON documents(fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    doc_id UNINDEXED,
    title,
    search_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS frontier (
    url TEXT PRIMARY KEY,
    source_url TEXT,
    depth INTEGER NOT NULL DEFAULT 0,
    priority REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_fetch_at REAL NOT NULL DEFAULT 0,
    inserted_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_frontier_schedule
    ON frontier(status, next_fetch_at, priority DESC, inserted_at);

CREATE TABLE IF NOT EXISTS robots_cache (
    origin TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    status INTEGER NOT NULL,
    fetched_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS crawl_events (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    url TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT
);
"""


class SearchDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def enqueue(self, url: str, source_url: Optional[str] = None, depth: int = 0, priority: float = 0.0) -> bool:
        now = time.time()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO frontier(url, source_url, depth, priority, status, inserted_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    priority = MAX(frontier.priority, excluded.priority),
                    source_url = COALESCE(frontier.source_url, excluded.source_url),
                    depth = MIN(frontier.depth, excluded.depth),
                    updated_at = excluded.updated_at
                """,
                (url, source_url, depth, priority, now, now),
            )
            return cursor.rowcount > 0

    def lease_frontier(self, limit: int) -> List[Dict[str, Any]]:
        now = time.time()
        with self.transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT * FROM frontier
                WHERE status IN ('queued', 'retry') AND next_fetch_at <= ?
                ORDER BY priority DESC, inserted_at ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
            if rows:
                connection.executemany(
                    "UPDATE frontier SET status='fetching', updated_at=? WHERE url=?",
                    [(now, row["url"]) for row in rows],
                )
            return [dict(row) for row in rows]

    def complete_frontier(self, url: str, detail: str = "") -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "UPDATE frontier SET status='done', updated_at=?, last_error=NULL WHERE url=?", (now, url)
            )
            self._event(connection, url, "done", detail)

    def skip_frontier(self, url: str, reason: str) -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "UPDATE frontier SET status='skipped', updated_at=?, last_error=? WHERE url=?",
                (now, reason[:1000], url),
            )
            self._event(connection, url, "skipped", reason)

    def fail_frontier(self, url: str, error: str, max_attempts: int = 3) -> None:
        now = time.time()
        with self.connect() as connection:
            row = connection.execute("SELECT attempts FROM frontier WHERE url=?", (url,)).fetchone()
            attempts = int(row["attempts"] if row else 0) + 1
            status = "failed" if attempts >= max_attempts else "retry"
            backoff = min(3600.0, 15.0 * (2 ** max(0, attempts - 1)))
            connection.execute(
                """
                UPDATE frontier
                SET status=?, attempts=?, next_fetch_at=?, updated_at=?, last_error=?
                WHERE url=?
                """,
                (status, attempts, now + backoff, now, error[:1000], url),
            )
            self._event(connection, url, status, error)

    def _event(self, connection: sqlite3.Connection, url: str, event: str, detail: str) -> None:
        connection.execute(
            "INSERT INTO crawl_events(created_at, url, event, detail) VALUES (?, ?, ?, ?)",
            (time.time(), url, event, detail[:2000]),
        )

    def get_document_by_url(self, canonical_url: str) -> Optional[Dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE canonical_url=?", (canonical_url,)
            ).fetchone()
            return dict(row) if row else None

    def upsert_document(
        self,
        *,
        url: str,
        canonical_url: str,
        title: str,
        content: str,
        published_at: Optional[str],
        fetched_at: float,
        etag: Optional[str],
        last_modified: Optional[str],
        content_type: Optional[str],
        language: Optional[str],
        source_type: str = "web",
        authority: float = 0.5,
        http_status: int = 200,
        response_bytes: int = 0,
    ) -> tuple[int, bool]:
        digest = content_hash(content)
        searchable = indexed_text(title, content)
        near_digest = simhash64(content)
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT id, content_hash FROM documents WHERE canonical_url=?", (canonical_url,)
            ).fetchone()
            if existing and existing["content_hash"] == digest:
                connection.execute(
                    """
                    UPDATE documents SET url=?, fetched_at=?, etag=?, last_modified=?, http_status=?,
                        response_bytes=?, content_type=COALESCE(?, content_type)
                    WHERE id=?
                    """,
                    (
                        url,
                        fetched_at,
                        etag,
                        last_modified,
                        http_status,
                        response_bytes,
                        content_type,
                        existing["id"],
                    ),
                )
                return int(existing["id"]), False

            if existing:
                doc_id = int(existing["id"])
                connection.execute(
                    """
                    UPDATE documents SET
                        url=?, title=?, content=?, search_text=?, content_hash=?, simhash=?,
                        published_at=?, fetched_at=?, etag=?, last_modified=?, content_type=?, language=?,
                        source_type=?, authority=?, http_status=?, response_bytes=?
                    WHERE id=?
                    """,
                    (
                        url,
                        title,
                        content,
                        searchable,
                        digest,
                        near_digest,
                        published_at,
                        fetched_at,
                        etag,
                        last_modified,
                        content_type,
                        language,
                        source_type,
                        max(0.0, min(1.0, authority)),
                        http_status,
                        response_bytes,
                        doc_id,
                    ),
                )
                connection.execute("DELETE FROM documents_fts WHERE doc_id=?", (str(doc_id),))
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO documents(
                        url, canonical_url, title, content, search_text, content_hash, simhash,
                        published_at, fetched_at, etag, last_modified, content_type, language,
                        source_type, authority, http_status, response_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        url,
                        canonical_url,
                        title,
                        content,
                        searchable,
                        digest,
                        near_digest,
                        published_at,
                        fetched_at,
                        etag,
                        last_modified,
                        content_type,
                        language,
                        source_type,
                        max(0.0, min(1.0, authority)),
                        http_status,
                        response_bytes,
                    ),
                )
                doc_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO documents_fts(doc_id, title, search_text) VALUES (?, ?, ?)",
                (str(doc_id), " ".join([title] + title.split()), searchable),
            )
            return doc_id, True

    def search_fts(self, tokens: Sequence[str], limit: int = 100) -> List[Dict[str, Any]]:
        clean = [token.replace('"', '""') for token in tokens if token]
        if not clean:
            return []
        match_query = " OR ".join(f'"{token}"' for token in clean[:32])
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT d.*, bm25(documents_fts, 0.0, 6.0, 1.0) AS bm25_rank
                FROM documents_fts
                JOIN documents d ON d.id = CAST(documents_fts.doc_id AS INTEGER)
                WHERE documents_fts MATCH ?
                ORDER BY bm25_rank ASC
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_documents(self, ids: Iterable[int]) -> List[Dict[str, Any]]:
        values = list(dict.fromkeys(int(value) for value in ids))
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM documents WHERE id IN ({placeholders})", values
            ).fetchall()
            mapping = {int(row["id"]): dict(row) for row in rows}
            return [mapping[value] for value in values if value in mapping]

    def get_robots(self, origin: str) -> Optional[Dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM robots_cache WHERE origin=?", (origin,)).fetchone()
            return dict(row) if row else None

    def set_robots(self, origin: str, body: str, status: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO robots_cache(origin, body, status, fetched_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(origin) DO UPDATE SET
                    body=excluded.body, status=excluded.status, fetched_at=excluded.fetched_at
                """,
                (origin, body, status, time.time()),
            )

    def stats(self) -> Dict[str, Any]:
        with self.connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            frontier = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM frontier GROUP BY status"
                ).fetchall()
            }
            size = sum(
                candidate.stat().st_size
                for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"))
                if candidate.exists()
            )
            return {"documents": int(documents), "frontier": frontier, "database_bytes": size}
