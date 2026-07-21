from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from .candidate_index import CandidateIndexClient
from .config import ShadowSearchConfig
from .evidence import Evidence
from .search import SearchResult
from .text import best_snippet


_TITLE_SPACE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


class FineWikiShadowSearch:
    """Run FineWiki beside the live retriever and record, but never publish, it."""

    def __init__(
        self,
        config: Optional[ShadowSearchConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or ShadowSearchConfig()
        self.client = client or CandidateIndexClient(
            self.config.endpoint, timeout=self.config.timeout_seconds
        )
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._last_error: Optional[str] = None
        self._closed = False

    def start(self, query: str, route: Mapping[str, Any]) -> Optional[Future[Any]]:
        if not self.config.enabled or self._closed or not query.strip():
            return None
        rate = min(1.0, max(0.0, float(self.config.sample_rate)))
        if rate <= 0.0 or (rate < 1.0 and random.random() >= rate):
            return None
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, int(self.config.max_workers)),
                    thread_name_prefix="finewiki-shadow",
                )
            executor = self._executor
            self._submitted += 1
        route_copy = dict(route)
        return executor.submit(self._retrieve, query, route_copy)

    def attach(
        self,
        future: Optional[Future[Any]],
        *,
        primary_results: Sequence[SearchResult],
        visible_results: Sequence[SearchResult],
        primary_latency_ms: float,
        query: str = "",
        route: Optional[Mapping[str, Any]] = None,
        visible_output_changed: bool = False,
    ) -> None:
        if future is None:
            return
        primary = self._serialize_primary(primary_results)
        visible = self._serialize_primary(visible_results)

        def completed(done: Future[Any]) -> None:
            try:
                shadow = done.result()
                record = {
                    "schema_version": "shadow-search.v1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": shadow["query"],
                    "route": shadow["route"],
                    "visible_output_changed": bool(visible_output_changed),
                    "primary": {
                        "latency_ms": round(float(primary_latency_ms), 3),
                        "count": len(primary),
                        "evidence": primary,
                    },
                    "visible_retrieval": {
                        "count": len(visible),
                        "evidence": visible,
                    },
                    "shadow": {
                        "latency_ms": shadow["latency_ms"],
                        "analyzed_query": shadow["analyzed_query"],
                        "count": len(shadow["evidence"]),
                        "evidence": shadow["evidence"],
                    },
                    "comparison": self._compare(primary, shadow["evidence"]),
                    "error": None,
                }
                self._append(record)
                with self._lock:
                    self._completed += 1
                    self._last_error = None
            except Exception as exc:
                self.record_failure(
                    "retrieve_or_record", exc, query=query, route=route
                )

        future.add_done_callback(completed)

    def record_failure(
        self,
        stage: str,
        error: Exception,
        *,
        query: str = "",
        route: Optional[Mapping[str, Any]] = None,
    ) -> None:
        error_value = f"{stage}: {type(error).__name__}: {str(error)[:500]}"
        with self._lock:
            self._failed += 1
            self._last_error = error_value
        try:
            self._append(
                {
                    "schema_version": "shadow-search.v1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query,
                    "route": dict(route or {}),
                    "visible_output_changed": False,
                    "error": error_value,
                }
            )
        except Exception:
            # An unwritable observability path remains isolated from chat.
            pass

    def _retrieve(self, query: str, route: Mapping[str, Any]) -> Dict[str, Any]:
        analysis, hits, elapsed_ms = self.client.search(
            query,
            index=self.config.index,
            channel_size=max(1, int(self.config.channel_size)),
            limit=max(1, int(self.config.limit)),
        )
        analyzed = analysis.to_dict() if hasattr(analysis, "to_dict") else {}
        evidence = [
            Evidence.from_candidate_hit(hit, evidence_id=f"W{index}").to_dict()
            for index, hit in enumerate(hits, start=1)
        ]
        return {
            "query": query,
            "route": dict(route),
            "analyzed_query": analyzed,
            "latency_ms": round(float(elapsed_ms), 3),
            "evidence": evidence,
            "hits": hits,
        }

    def live_results(
        self, future: Future[Any]
    ) -> tuple[list[SearchResult], Dict[str, Any]]:
        """Reuse the already-running Shadow request as an explicit live source."""

        payload = future.result(timeout=max(0.1, self.config.timeout_seconds + 0.25))
        query = str(payload.get("query") or "")
        hits = list(payload.get("hits") or [])
        fetched_at = time.time()
        output: list[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            candidate_score = float(getattr(hit, "score", 0.0) or 0.0)
            # CandidateIndex uses RRF values around 0.01-0.07.  Calibrate them
            # into the same broad range as the current hybrid/web rankers while
            # preserving the CandidateIndex ordering.
            live_score = 0.42 + min(0.28, candidate_score * 4.0) + 0.04 / rank
            output.append(
                SearchResult(
                    document_id=-rank,
                    url=str(getattr(hit, "url", "")),
                    title=str(getattr(hit, "title", "")),
                    snippet=best_snippet(
                        str(getattr(hit, "text", "")), query, limit=900
                    ),
                    content=str(getattr(hit, "text", "")),
                    published_at=None,
                    fetched_at=fetched_at,
                    source_type="finewiki",
                    authority=0.85,
                    score=live_score,
                    score_components={
                        "candidate_rrf": candidate_score,
                        "finewiki_rank": 1.0 / rank,
                        "passage_selection": float(
                            getattr(hit, "passage_score", 0.0) or 0.0
                        ),
                        "passage_candidates": float(
                            getattr(hit, "candidate_chunk_count", 1) or 1
                        ),
                    },
                    source_id=str(getattr(hit, "doc_id", "") or "") or None,
                    updated_at=str(getattr(hit, "modified_at", "") or "") or None,
                    matched_channels=tuple(
                        dict.fromkeys(getattr(hit, "channels", ()) or ())
                    ),
                )
            )
        return output, {
            "enabled": True,
            "used": True,
            "count": len(output),
            "latency_ms": float(payload.get("latency_ms") or 0.0),
            "index": self.config.index,
        }

    @staticmethod
    def _serialize_primary(results: Sequence[SearchResult]) -> list[Dict[str, Any]]:
        output = []
        for index, result in enumerate(results, start=1):
            text = best_snippet(result.content, result.title, limit=1200)
            output.append(
                Evidence.from_search_result(
                    result, evidence_id=f"P{index}", text=text
                ).to_dict()
            )
        return output

    @classmethod
    def _compare(
        cls,
        primary: Sequence[Mapping[str, Any]],
        shadow: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        primary_urls = {cls._url_key(str(item.get("url") or "")) for item in primary}
        shadow_urls = {cls._url_key(str(item.get("url") or "")) for item in shadow}
        primary_titles = {cls._title_key(str(item.get("title") or "")) for item in primary}
        shadow_titles = {cls._title_key(str(item.get("title") or "")) for item in shadow}
        primary_urls.discard("")
        shadow_urls.discard("")
        primary_titles.discard("")
        shadow_titles.discard("")
        url_overlap = primary_urls & shadow_urls
        title_overlap = primary_titles & shadow_titles
        denominator = max(1, min(len(primary), len(shadow)))
        return {
            "url_overlap_count": len(url_overlap),
            "title_overlap_count": len(title_overlap),
            "overlap_at_k": round(
                max(len(url_overlap), len(title_overlap)) / denominator, 4
            ),
        }

    @staticmethod
    def _url_key(value: str) -> str:
        if not value:
            return ""
        parts = urlsplit(value.strip())
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, ""))

    @staticmethod
    def _title_key(value: str) -> str:
        return _TITLE_SPACE.sub("", value).casefold()

    def _append(self, record: Mapping[str, Any]) -> None:
        path = Path(self.config.log_path).expanduser()
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if path.exists() and path.stat().st_size >= max(
                    1024, int(self.config.max_log_bytes)
                ):
                    rotated = path.with_suffix(path.suffix + ".1")
                    rotated.unlink(missing_ok=True)
                    path.replace(rotated)
            except OSError:
                pass
            with path.open("a", encoding="utf-8", buffering=64 * 1024) as handle:
                handle.write(line)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self.config.enabled),
                "ready": bool(self.config.enabled and not self._closed),
                "mode": "shadow_only",
                "visible_output_changed": False,
                "endpoint": self.config.endpoint,
                "index": self.config.index,
                "log_path": self.config.log_path,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "last_error": self._last_error,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
