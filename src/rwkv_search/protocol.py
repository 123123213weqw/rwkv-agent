from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import threading
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SEARCH_MODES = {"auto", "always", "never"}
_RESEARCH_DEPTHS = {"fast", "deep"}
_SOURCE_SCOPES = {"auto", "local", "web"}


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, *, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field:
            value["field"] = self.field
        return value


@dataclass(frozen=True)
class ChatRequest:
    schema_version: str
    request_id: str
    conversation_id: str
    message_id: str
    query: str
    history: List[Dict[str, str]]
    search_mode: str
    research_depth: str
    source_scope: str
    use_finewiki: bool
    timezone: str
    locale: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ChatRequest":
        schema_version = str(payload.get("schema_version") or SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ProtocolError(
                "UNSUPPORTED_SCHEMA_VERSION",
                f"schema_version must be {SCHEMA_VERSION}",
                field="schema_version",
            )
        query = " ".join(str(payload.get("query") or "").strip().split())
        if not query:
            raise ProtocolError("INVALID_REQUEST", "query is required", field="query")
        if len(query) > 20000:
            raise ProtocolError(
                "INVALID_REQUEST", "query exceeds 20000 characters", field="query"
            )

        request_id = cls._id(payload.get("request_id"), "request_id", "req")
        conversation_id = cls._id(
            payload.get("conversation_id"), "conversation_id", "conv"
        )
        message_id = cls._id(payload.get("message_id"), "message_id", "msg")
        search_mode = str(payload.get("search_mode") or "auto")
        research_depth = str(payload.get("research_depth") or "fast")
        source_scope = str(payload.get("source_scope") or "auto")
        use_finewiki = payload.get("use_finewiki", False)
        if search_mode not in _SEARCH_MODES:
            raise ProtocolError(
                "INVALID_REQUEST",
                "search_mode must be auto, always, or never",
                field="search_mode",
            )
        if research_depth not in _RESEARCH_DEPTHS:
            raise ProtocolError(
                "INVALID_REQUEST",
                "research_depth must be fast or deep",
                field="research_depth",
            )
        if source_scope not in _SOURCE_SCOPES:
            raise ProtocolError(
                "INVALID_REQUEST",
                "source_scope must be auto, local, or web",
                field="source_scope",
            )
        if not isinstance(use_finewiki, bool):
            raise ProtocolError(
                "INVALID_REQUEST",
                "use_finewiki must be a boolean",
                field="use_finewiki",
            )
        return cls(
            schema_version=schema_version,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=message_id,
            query=query,
            history=normalize_history(payload.get("history")),
            search_mode=search_mode,
            research_depth=research_depth,
            source_scope=source_scope,
            use_finewiki=use_finewiki,
            timezone=str(payload.get("timezone") or "Asia/Shanghai")[:128],
            locale=str(payload.get("locale") or "zh-CN")[:32],
        )

    @staticmethod
    def _id(value: Any, field: str, prefix: str) -> str:
        identifier = str(value or f"{prefix}_{uuid4().hex}")
        if not _ID_PATTERN.fullmatch(identifier):
            raise ProtocolError(
                "INVALID_REQUEST",
                f"{field} contains unsupported characters or is too long",
                field=field,
            )
        return identifier


def normalize_history(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, str]] = []
    character_budget = 12000
    for item in value[-12:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = " ".join(str(item.get("content") or "").split())[:3000]
        if not content:
            continue
        content = content[:character_budget]
        normalized.append({"role": str(item["role"]), "content": content})
        character_budget -= len(content)
        if character_budget <= 0:
            break
    return normalized


class EventFactory:
    def __init__(self, request: ChatRequest) -> None:
        self.request = request
        self.sequence = 0

    def make(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        self.sequence += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "type": event_type,
            "request_id": self.request.request_id,
            "conversation_id": self.request.conversation_id,
            "message_id": self.request.message_id,
            "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }


class RequestRegistry:
    """Thread-safe cancellation registry for active streaming requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[str, threading.Event] = {}

    def register(self, request_id: str) -> threading.Event:
        with self._lock:
            if request_id in self._requests:
                raise ProtocolError(
                    "DUPLICATE_REQUEST_ID", "request_id is already active", field="request_id"
                )
            event = threading.Event()
            self._requests[request_id] = event
            return event

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._requests.get(request_id)
            if event is None:
                return False
            event.set()
            return True

    def finish(self, request_id: str) -> None:
        with self._lock:
            self._requests.pop(request_id, None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._requests)


def chunk_text(text: str, target_size: int = 72) -> Iterable[str]:
    """Create display-sized deltas without splitting surrogate-independent code points."""
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + target_size)
        if end < len(text):
            boundary = max(
                text.rfind("。", start, end),
                text.rfind("！", start, end),
                text.rfind("？", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if boundary >= start + max(12, target_size // 3):
                end = boundary + 1
        yield text[start:end]
        start = end
