from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, Optional


_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class DebugTrace:
    """One buffered JSONL trace for a single chat request."""

    def __init__(self, path: Path, request: Any) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file = path.open("a", encoding="utf-8", buffering=64 * 1024)
        self._closed = False
        self.write(
            "trace_started",
            {
                "schema_version": getattr(request, "schema_version", "1.0"),
                "request_id": getattr(request, "request_id", ""),
                "conversation_id": getattr(request, "conversation_id", ""),
                "message_id": getattr(request, "message_id", ""),
                "query": getattr(request, "query", ""),
                "history": getattr(request, "history", []),
                "search_mode": getattr(request, "search_mode", "auto"),
                "research_depth": getattr(request, "research_depth", "fast"),
                "source_scope": getattr(request, "source_scope", "auto"),
                "use_finewiki": bool(getattr(request, "use_finewiki", False)),
                "timezone": getattr(request, "timezone", ""),
                "locale": getattr(request, "locale", ""),
            },
        )

    def write(self, category: str, payload: Dict[str, Any]) -> None:
        if self._closed:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": category,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")

    def flush(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._file.flush()

    def close(self, status: str = "completed", error: Optional[str] = None) -> None:
        if self._closed:
            return
        self.write(
            "trace_finished",
            {"status": status, "error": error},
        )
        with self._lock:
            self._file.flush()
            self._file.close()
            self._closed = True


class DebugTraceStore:
    """Backend-only trace store controlled by RWKV_SEARCH_DEBUG_DIR."""

    def __init__(self, directory: Optional[str] = None, *, max_files: int = 50) -> None:
        value = directory if directory is not None else os.environ.get(
            "RWKV_SEARCH_DEBUG_DIR", ""
        )
        self.directory = Path(value).expanduser() if value else None
        self.max_files = max(1, int(os.environ.get("RWKV_SEARCH_DEBUG_MAX_FILES", max_files)))
        self.enabled = self.directory is not None
        self._lock = threading.Lock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def open(self, request: Any) -> Optional[DebugTrace]:
        if self.directory is None:
            return None
        now = datetime.now(timezone.utc)
        request_id = _SAFE_ID.sub("_", str(getattr(request, "request_id", "request")))[:128]
        filename = f"{now:%Y%m%dT%H%M%S.%fZ}_{request_id}.jsonl"
        path = self.directory / filename
        trace = DebugTrace(path, request)
        with self._lock:
            latest = self.directory / "latest.jsonl"
            temporary = self.directory / ".latest.jsonl.tmp"
            try:
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(path.name)
                temporary.replace(latest)
            except OSError:
                pass
            self._prune_locked()
        return trace

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "directory": str(self.directory) if self.directory is not None else None,
            "max_files": self.max_files,
        }

    def _prune_locked(self) -> None:
        if self.directory is None:
            return
        files = sorted(
            (
                path
                for path in self.directory.glob("*.jsonl")
                if path.name != "latest.jsonl" and not path.is_symlink()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files[self.max_files :]:
            try:
                path.unlink()
            except OSError:
                pass
