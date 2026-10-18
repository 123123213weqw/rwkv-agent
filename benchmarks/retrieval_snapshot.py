"""Deterministic, content-light snapshots for Web benchmark retrieval stages.

The live search engines are intentionally outside the replay boundary.  A
snapshot freezes their URL/title/snippet candidates and the later fetch and
evidence decisions, so ranking and loss-funnel analysis can be repeated without
issuing another network request.  Page bodies and credentials are never stored.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Mapping


SNAPSHOT_SCHEMA_VERSION = "rwkv-agent-retrieval-snapshot.v1"
CONFIG_SNAPSHOT_SCHEMA_VERSION = "rwkv-agent-config-snapshot.v1"
FUNNEL_SCHEMA_VERSION = "rwkv-agent-retrieval-funnel.v1"

_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?key|access_?key|private_?key|token|secret|password|"
    r"authorization|cookie|credential)(?:$|_)",
    re.I,
)
_SECRET_VALUE = re.compile(
    r"^(?:tvly-|ghp_|github_pat_|sk-|Bearer\s+)[A-Za-z0-9._-]{8,}$",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def sanitize_config(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe config copy with secret material removed.

    Environment-variable *names* such as ``tavily_api_key_env`` are needed for
    reproducibility and are not credentials, so ``*_env`` fields remain visible.
    """

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            name = str(raw_key)
            if not name.casefold().endswith("_env") and _SECRET_KEY.search(name):
                output[name] = "<redacted>" if child not in (None, "") else child
            else:
                output[name] = sanitize_config(child, key=name)
        return output
    if isinstance(value, list):
        return [sanitize_config(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [sanitize_config(item, key=key) for item in value]
    if isinstance(value, str) and _SECRET_VALUE.match(value.strip()):
        return "<redacted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def freeze_config_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """Freeze a sanitized configuration and return its manifest binding."""

    source = source.expanduser().resolve()
    decoded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("configuration root must be an object")
    snapshot = {
        "schema_version": CONFIG_SNAPSHOT_SCHEMA_VERSION,
        "source": {"name": source.name, "sha256": sha256(source)},
        "config": sanitize_config(decoded),
    }
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != snapshot:
            raise RuntimeError(f"config snapshot mismatch: {destination}")
    else:
        _json_dump(destination, snapshot)
    return {
        "artifact": destination.name,
        "sha256": sha256(destination),
        "source_sha256": snapshot["source"]["sha256"],
    }


def _candidate_rows(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    output: list[dict[str, Any]] = []
    for position, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            continue
        item = {
            str(key): child
            for key, child in value.items()
            if str(key)
            in {
                "url",
                "title",
                "snippet",
                "engine",
                "rank",
                "position",
                "published_hint",
                "rrf_score",
                "candidate_score",
                "engine_score",
                "engines",
                "positions",
                "matched_queries",
                "query_positions",
                "source_channels",
                "discovery_stage",
                "discovery_stages",
                "parent_url",
                "score_components",
                "rejection_reasons",
            }
        }
        item.setdefault("position", position)
        output.append(item)
    return output


def _result_rows(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    output: list[dict[str, Any]] = []
    for position, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            continue
        item = {
            str(key): child
            for key, child in value.items()
            if str(key)
            in {
                "url",
                "uri",
                "title",
                "snippet",
                "source",
                "source_type",
                "published_at",
                "score",
                "content_length",
                "retrieval_mode",
            }
        }
        item.setdefault("position", position)
        output.append(item)
    return output


def _evidence_rows(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        {
            "id": str(value.get("id") or ""),
            "uri": str(value.get("uri") or ""),
            "title": str(value.get("title") or "")[:500],
            "source": str(value.get("source") or ""),
            "score": float(value.get("score") or 0.0),
        }
        for value in values
        if isinstance(value, Mapping)
    ]


def snapshot_call(
    public: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    call_index: int,
) -> dict[str, Any]:
    """Serialize one tool call without copying fetched page content."""

    return {
        "call_index": int(call_index),
        "status": str(trace.get("status") or public.get("status") or ""),
        "query": str(trace.get("query") or ""),
        "original_query": str(trace.get("original_query") or ""),
        "effective_query": str(trace.get("effective_query") or ""),
        "execution_queries": [
            str(value)
            for value in trace.get("execution_queries") or ()
            if str(value).strip()
        ],
        "original_query_lane": str(trace.get("original_query_lane") or ""),
        "compiled_query": sanitize_config(trace.get("compiled_query") or {}),
        "scope_root": str(trace.get("scope_root") or ""),
        "scope_mode": str(trace.get("scope_mode") or ""),
        "scope_rejected": sanitize_config(trace.get("scope_rejected") or {}),
        "raw_candidates": _candidate_rows(trace.get("raw_candidates") or ()),
        "initial_candidates": _candidate_rows(
            trace.get("initial_candidates") or ()
        ),
        "post_pivot_candidates": _candidate_rows(
            trace.get("post_pivot_candidates") or ()
        ),
        "candidates": _candidate_rows(trace.get("candidates") or ()),
        "rejected_candidates": _candidate_rows(
            trace.get("rejected_candidates") or ()
        ),
        "results": _result_rows(trace.get("results") or ()),
        "evidence": _evidence_rows(public.get("evidence") or ()),
        "fetches": sanitize_config(list(trace.get("fetches") or ())),
        "warnings": sanitize_config(list(trace.get("warnings") or ())),
        "stats": sanitize_config(trace.get("stats") or {}),
        "latency_ms": round(float(trace.get("latency_ms") or 0.0), 3),
        "evidence_stage": str(trace.get("evidence_stage") or ""),
    }


def validate_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported retrieval snapshot schema_version")
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("retrieval snapshot case_id must be non-empty")
    calls = value.get("calls")
    if not isinstance(calls, list):
        raise ValueError("retrieval snapshot calls must be a list")
    indexes: list[int] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError("retrieval snapshot call must be an object")
        index = call.get("call_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("retrieval snapshot call_index must be positive")
        indexes.append(index)
        for key in (
            "raw_candidates",
            "initial_candidates",
            "post_pivot_candidates",
            "candidates",
            "rejected_candidates",
            "results",
            "evidence",
            "fetches",
            "execution_queries",
        ):
            if not isinstance(call.get(key, []), list):
                raise ValueError(f"retrieval snapshot {key} must be a list")
    if indexes != list(range(1, len(indexes) + 1)):
        raise ValueError("retrieval snapshot call indexes must be contiguous")
    return dict(value)


class RetrievalSnapshotRecorder:
    """Concurrency-safe per-case recorder with atomic checkpoint files."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.case_dir = output_path.with_suffix("")
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, list[dict[str, Any]]] = {}
        self._write_lock = threading.Lock()

    @contextmanager
    def capture_case(self, case_id: str):
        case_id = str(case_id)
        with self._write_lock:
            if case_id in self._active:
                raise RuntimeError(f"retrieval snapshot case already active: {case_id}")
            self._active[case_id] = []
        try:
            yield self
        finally:
            with self._write_lock:
                calls = self._active.pop(case_id)
            snapshot = validate_snapshot(
                {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "case_id": case_id,
                    "captured_at": utc_now(),
                    "calls": calls,
                }
            )
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
            with self._write_lock:
                _json_dump(self.case_dir / f"{safe_name}.json", snapshot)

    def observe(
        self,
        case_id: str,
        public: Mapping[str, Any],
        trace: Mapping[str, Any],
    ) -> None:
        with self._write_lock:
            calls = self._active.get(str(case_id))
            if calls is None:
                raise RuntimeError("retrieval trace observed outside capture_case")
            calls.append(
                snapshot_call(
                    public,
                    trace,
                    call_index=len(calls) + 1,
                )
            )

    def finalize(self) -> Path:
        rows = []
        for path in sorted(self.case_dir.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append(validate_snapshot(value))
        rows.sort(key=lambda row: str(row["case_id"]))
        _jsonl_dump(self.output_path, rows)
        return self.output_path


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
            rows.append(validate_snapshot(decoded))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    ids = [str(row["case_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval snapshots contain duplicate case_id values")
    return rows
