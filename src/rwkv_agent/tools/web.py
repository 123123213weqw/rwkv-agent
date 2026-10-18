from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import socket
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from rwkv_search.config import AppConfig
from rwkv_search.analysis.query import QueryAnalyzer
from rwkv_search.p4_search import resolve_model_query_constraints
from rwkv_search.passage_selection import select_page_passages
from rwkv_search.pipeline.query_compiler import QueryCompiler, QueryHints
from rwkv_search.realtime import RealtimeSearchEngine
from rwkv_search.realtime.cache import TTLByteCache
from rwkv_search.realtime.types import DiscoveredURL
from rwkv_search.realtime.precision_discovery import build_original_query_lane
from rwkv_search.semantic_selection import PairScorer, select_diverse_items
from rwkv_search.text import canonicalize_url


ENHANCED_FEATURES = (
    "candidate_admission_enabled",
    "query_compaction_enabled",
    "source_channels_enabled",
    "domain_pivot_enabled",
    "one_hop_link_expansion_enabled",
)

PROFILE_FEATURES = {
    "legacy": frozenset(),
    # Keep relevance admission, compact queries and bounded domain pivots, but
    # avoid the source-channel and one-hop fan-out that pushed Web P95 above
    # the FitGen release budget in the Enhanced arm.
    "balanced": frozenset(
        {
            "candidate_admission_enabled",
            "query_compaction_enabled",
            "domain_pivot_enabled",
        }
    ),
    "enhanced": frozenset(ENHANCED_FEATURES),
}

# The visible Legacy request and its background Enhanced Shadow must observe
# the same general-discovery response. Besides making the A/B comparison fair,
# this halves normal metasearch traffic and reduces upstream rate-limit noise.
# The cache is process-local, TTL-bounded by RealtimeSearchConfig, and stores
# discovery metadata only (never fetched page bodies).
_SHARED_DISCOVERY_CACHE = TTLByteCache[list[DiscoveredURL]](16 * 1024 * 1024)
_STRUCTURED_ENGINES = frozenset({"crossref", "github", "mediawiki"})


@dataclass(frozen=True)
class WebEvidence:
    id: str
    title: str
    content: str
    uri: str
    source: str
    published_at: str | None
    score: float
    discovery_stage: str = ""


def _structured_stage_representatives(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the strongest record for each provider-declared capability stage."""

    output: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    for source in items:
        item = dict(source)
        stage = str(item.get("discovery_stage") or "").strip()
        key = (
            f"{item.get('source') or ''}:{stage}"
            if stage
            else (
                canonicalize_url(str(item.get("uri") or ""))
                or str(item.get("uri") or "").casefold()
            )
        )
        if not key:
            continue
        index = indexes.get(key)
        if index is None:
            indexes[key] = len(output)
            output.append(item)
            continue
        current = output[index]
        current_key = (
            str(current.get("published_at") or ""),
            float(current.get("_upstream_score") or 0.0),
            len(str(current.get("content") or "")),
        )
        item_key = (
            str(item.get("published_at") or ""),
            float(item.get("_upstream_score") or 0.0),
            len(str(item.get("content") or "")),
        )
        if item_key > current_key:
            output[index] = item
    return output


def _structured_preference(item: Mapping[str, Any]) -> float:
    """Prefer high-information structured record shapes, independent of topic."""

    stage = str(item.get("discovery_stage") or "").casefold()
    value = min(1.0, len(str(item.get("content") or "")) / 500.0)
    if any(kind in stage for kind in ("profile", "repository_index", "latest_commit")):
        value += 2.0
    elif any(kind in stage for kind in ("latest_release", "primary_repository")):
        value += 1.0
    return value


def _configure(
    config_path: str,
    *,
    profile: str,
    fallback_engines: Sequence[str] | None = None,
    api_providers: Sequence[str] | None = None,
) -> AppConfig:
    config = (
        AppConfig.load(config_path)
        if config_path and Path(config_path).is_file()
        else AppConfig()
    )
    config.realtime_search.enabled = True
    config.realtime_search.force_ipv4 = True
    if profile not in PROFILE_FEATURES:
        raise ValueError(f"unsupported web profile: {profile}")
    for name in ENHANCED_FEATURES:
        setattr(config.realtime_search, name, name in PROFILE_FEATURES[profile])
    searxng = config.realtime_search.searxng_url.rstrip("/")
    parsed = urlsplit(searxng)
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        try:
            with socket.create_connection(
                (parsed.hostname or "127.0.0.1", parsed.port or 80),
                timeout=0.2,
            ):
                pass
        except OSError:
            config.realtime_search.searxng_url = ""
    if not config.realtime_search.searxng_url.rstrip("/"):
        config.realtime_search.bing_base_url = "https://cn.bing.com"
    if fallback_engines is None:
        environment_fallbacks = os.getenv(
            "RWKV_AGENT_WEB_FALLBACK_ENGINES", ""
        ).strip()
        if environment_fallbacks:
            fallback_engines = environment_fallbacks.split(",")
    if fallback_engines is not None:
        values = list(dict.fromkeys(str(value).strip() for value in fallback_engines))
        invalid = [
            value
            for value in values
            if value not in {"bing", "baidu", "so360", "wikipedia"}
        ]
        if invalid:
            raise ValueError(f"unsupported fallback web engines: {invalid}")
        if not values:
            raise ValueError("at least one fallback web engine is required")
        config.realtime_search.fallback_engines = values
    if api_providers is None:
        environment_providers = os.getenv(
            "RWKV_AGENT_WEB_API_PROVIDERS", ""
        ).strip()
        if environment_providers:
            api_providers = environment_providers.split(",")
    if api_providers is not None:
        values = list(dict.fromkeys(str(value).strip() for value in api_providers))
        invalid = [
            value
            for value in values
            if value not in {"tavily", "github", "crossref", "mediawiki"}
        ]
        if invalid:
            raise ValueError(f"unsupported API discovery providers: {invalid}")
        config.realtime_search.api_discovery_providers = values
    return config


def _candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(item).items()
        if key
        in {
            "url",
            "title",
            "snippet",
            "engine",
            "rank",
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


def _result(item: Any) -> dict[str, Any]:
    content = str(
        (
            item.get("content")
            if isinstance(item, Mapping)
            else getattr(item, "content", "")
        )
        or ""
    )
    if hasattr(item, "to_dict"):
        value = dict(item.to_dict(include_content=False))
    elif isinstance(item, Mapping):
        value = dict(item)
        content = str(value.pop("content", "") or content)
    else:
        value = {
            "title": str(getattr(item, "title", "") or ""),
            "url": str(getattr(item, "url", "") or ""),
            "snippet": str(getattr(item, "snippet", "") or ""),
            "source_type": str(getattr(item, "source_type", "web") or "web"),
            "published_at": getattr(item, "published_at", None),
            "score": float(getattr(item, "score", 0.0) or 0.0),
        }
    value["content_length"] = len(content)
    return value


def _item_url(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("url") or item.get("uri") or "")
    return str(getattr(item, "url", "") or getattr(item, "uri", "") or "")


def _within_scope(item: Any, scope_host: str) -> bool:
    """Enforce an explicitly bound site on final URLs, not search syntax alone."""

    if not scope_host:
        return True
    host = (urlsplit(_item_url(item)).hostname or "").casefold().strip(".")
    expected = str(scope_host or "").casefold().strip(".")
    host = host[4:] if host.startswith("www.") else host
    expected = expected[4:] if expected.startswith("www.") else expected
    return bool(host and expected and (host == expected or host.endswith("." + expected)))


def _public_evidence(
    results: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    query: str = "",
    scorer: PairScorer | None = None,
) -> tuple[list[WebEvidence], str]:
    pool: list[dict[str, Any]] = []
    for position, result in enumerate(results[:12], start=1):
        if isinstance(result, Mapping):
            title = str(result.get("title") or "")
            snippet = str(result.get("snippet") or "")
            full_content = str(result.get("content") or snippet)
            uri = str(result.get("url") or "")
            source = str(result.get("source_type") or "web")
            published_at = (
                str(result.get("published_at"))
                if result.get("published_at")
                else None
            )
            score = float(result.get("score") or 0.0)
            discovery_stage = str(result.get("discovery_stage") or "")
        else:
            title = str(getattr(result, "title", "") or "")
            snippet = str(getattr(result, "snippet", "") or "")
            full_content = str(getattr(result, "content", "") or snippet)
            uri = str(getattr(result, "url", "") or "")
            source = str(getattr(result, "source_type", "web") or "web")
            published = getattr(result, "published_at", None)
            published_at = str(published) if published else None
            score = float(getattr(result, "score", 0.0) or 0.0)
            discovery_stage = str(getattr(result, "discovery_stage", "") or "")
        normalized_content = " ".join(full_content.split())
        if query and len(normalized_content) > 900:
            selection = select_page_passages(
                query,
                title,
                full_content,
                max_passages=3,
                max_chars=900,
                target_chars=300,
                hard_max_chars=380,
                scorer=scorer,
            )
            content = selection.text or snippet or normalized_content
        else:
            content = normalized_content or snippet
        if not uri:
            continue
        pool.append(
            {
                "title": title[:500],
                "content": " ".join(content.split())[:900],
                "uri": uri,
                "source": source,
                "published_at": published_at,
                "score": score,
                "origin": "fetched",
                "discovery_stage": discovery_stage,
                "_best_position": position,
                "_upstream_score": score,
            }
        )
    for position, candidate in enumerate(candidates, start=1):
        uri = str(candidate.get("url") or "")
        if not uri:
            continue
        source = str(candidate.get("engine") or "web_discovery")
        candidate_score = float(
            candidate.get("candidate_score")
            or candidate.get("rrf_score")
            or candidate.get("engine_score")
            or 0.0
        )
        pool.append(
            {
                "title": str(candidate.get("title") or "")[:500],
                "content": " ".join(
                    str(candidate.get("snippet") or "").split()
                )[:900],
                "uri": uri,
                "source": source,
                "published_at": (
                    str(candidate.get("published_hint"))
                    if candidate.get("published_hint")
                    else None
                ),
                "score": candidate_score,
                "origin": (
                    "structured" if source in _STRUCTURED_ENGINES else "discovery"
                ),
                "discovery_stage": str(candidate.get("discovery_stage") or ""),
                "_best_position": position,
                "_upstream_score": candidate_score,
            }
        )

    structured = _structured_stage_representatives(
        [item for item in pool if item.get("origin") == "structured"]
    )
    for item in structured:
        item["_preference_score"] = _structured_preference(item)
    selection_pool = [
        item for item in pool if item.get("origin") != "structured"
    ] + structured
    reservable_structured = [
        item for item in structured if _structured_preference(item) > 1.0
    ]
    reserved = select_diverse_items(
        query,
        (),
        reservable_structured,
        # Reserve only records whose provider declared a high-information
        # capability stage (profile, repository index, commit, release).  A
        # source label alone no longer forces a weak structured result into the
        # final packet; it still competes in the normal global selection.
        limit=min(4, len(reservable_structured)),
        scorer=scorer,
        preference_weight=0.18,
    ).items
    reserved_keys = {
        canonicalize_url(str(item.get("uri") or ""))
        or str(item.get("uri") or "").casefold()
        for item in reserved
    }
    remainder = [
        item
        for item in selection_pool
        if (
            canonicalize_url(str(item.get("uri") or ""))
            or str(item.get("uri") or "").casefold()
        )
        not in reserved_keys
    ]
    selected = [
        *reserved,
        *select_diverse_items(
            query,
            (),
            remainder,
            limit=max(0, 8 - len(reserved)),
            scorer=scorer,
        ).items,
    ]
    evidence = [
        WebEvidence(
            id=f"W{index}",
            title=str(item.get("title") or "")[:500],
            content=str(item.get("content") or "")[:900],
            uri=str(item.get("uri") or ""),
            source=str(item.get("source") or "web"),
            published_at=(
                str(item.get("published_at"))
                if item.get("published_at")
                else None
            ),
            score=round(float(item.get("score") or 0.0), 6),
            discovery_stage=str(item.get("discovery_stage") or ""),
        )
        for index, item in enumerate(selected, start=1)
    ]
    origins = {str(item.get("origin") or "") for item in selected}
    if "fetched" in origins and len(origins) > 1:
        return evidence, "mixed"
    return evidence, "fetched" if origins == {"fetched"} else "discovery"


class WebSearchAdapter:
    """Bounded Agent adapter over legacy or enhanced realtime web retrieval."""

    def __init__(
        self,
        config_path: str = "configs/default.json",
        *,
        engine: Any | None = None,
        profile: str = "legacy",
        shadow: Any | None = None,
        fallback_engines: Sequence[str] | None = None,
        api_providers: Sequence[str] | None = None,
        semantic_scorer: PairScorer | None = None,
        query_compiler: QueryCompiler | None = None,
        trace_observer: Any | None = None,
    ) -> None:
        self.profile = profile
        self.semantic_scorer = semantic_scorer
        self.query_compiler = query_compiler or QueryCompiler()
        self.query_analyzer = QueryAnalyzer(max_queries=1)
        self.trace_observer = trace_observer
        if engine is None:
            config = _configure(
                config_path,
                profile=profile,
                fallback_engines=fallback_engines,
                api_providers=api_providers,
            )
            engine = RealtimeSearchEngine(
                config.realtime_search,
                config.search,
                discovery_cache=_SHARED_DISCOVERY_CACHE,
                semantic_scorer=semantic_scorer,
            )
        self.engine = engine
        self._scope_root = ""
        self._scope_original_query = ""
        self._trace_case_id = ""
        self._query_lane_lock = threading.Lock()
        self._original_query_lane_claimed = False
        if shadow is None and profile == "legacy":
            shadow = build_web_shadow_from_env(config_path)
        self.shadow = shadow

    @contextmanager
    def scoped(self, root_url: str, *, original_query: str = ""):
        """Bind one leased controller request to a public crawl root."""

        previous = (self._scope_root, self._scope_original_query)
        with self._query_lane_lock:
            previous_claimed = self._original_query_lane_claimed
            self._original_query_lane_claimed = False
        self._scope_root = str(root_url or "").strip()
        self._scope_original_query = str(original_query or "").strip()
        try:
            yield self
        finally:
            self._scope_root, self._scope_original_query = previous
            with self._query_lane_lock:
                self._original_query_lane_claimed = previous_claimed

    def _claim_original_query_lane(self) -> bool:
        """Claim the request-level complementary lane exactly once."""

        if not self._scope_root:
            return False
        with self._query_lane_lock:
            if self._original_query_lane_claimed:
                return False
            self._original_query_lane_claimed = True
            return True

    @contextmanager
    def capture_trace_case(self, case_id: str):
        """Bind benchmark-only trace callbacks to one leased controller case."""

        previous = self._trace_case_id
        self._trace_case_id = str(case_id or "")
        try:
            yield self
        finally:
            self._trace_case_id = previous

    def execute_with_trace(
        self,
        query: str,
        *,
        original_query: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        value = str(query or "").strip()
        if not value:
            result = {
                "status": "invalid",
                "evidence": [],
                "message": "web_search requires a non-empty query.",
            }
            return result, {
                "schema_version": "agent-web-trace.v1",
                "status": "invalid",
                "profile": self.profile,
                "query": value,
                "candidates": [],
                "results": [],
                "fetches": [],
                "warnings": [],
                "stats": {},
                "latency_ms": 0.0,
            }
        started = time.perf_counter()
        original = str(original_query or value).strip()
        query_resolution = resolve_model_query_constraints(original, value)
        resolved_query = str(query_resolution["effective_query"] or original)
        scope_root = self._scope_root
        scope_host = (urlsplit(scope_root).hostname or "").removeprefix("www.")
        hints = QueryHints(
            freshness="latest",
            sites=(scope_host,) if scope_host else (),
            depth="single",
        )
        compiled = self.query_compiler.compile(
            original,
            resolved_query,
            hints=hints,
        )
        effective_query = compiled.execution_queries[0]
        execution_queries = [effective_query]
        original_query_lane = ""
        engine_config = getattr(self.engine, "config", None)
        if (
            scope_host
            and bool(
                getattr(
                    engine_config,
                    "original_query_lane_enabled",
                    False,
                )
            )
        ):
            candidate_lane = build_original_query_lane(
                self._scope_original_query or original,
                effective_query,
                site=scope_host,
                analyzer=self.query_analyzer,
            )
            if candidate_lane and self._claim_original_query_lane():
                original_query_lane = candidate_lane
                execution_queries.append(candidate_lane)
        ranking_query = (
            compiled.raw_query if original_query is not None else effective_query
        )
        seed_urls = (
            (scope_root,)
            if scope_root
            and "one_hop_link_expansion_enabled"
            in PROFILE_FEATURES[self.profile]
            else ()
        )
        results: list[Any] = []
        raw_candidates: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        initial_candidates: list[dict[str, Any]] = []
        post_pivot_candidates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        fetches: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        warnings: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for event in self.engine.search_events(
            ranking_query,
            execution_queries,
            freshness=compiled.freshness,
            depth=compiled.depth,
            source_preference=compiled.source_preference,
            seed_urls=seed_urls,
            include_candidates=True,
        ):
            event_type = str(event.get("type") or "")
            events.append(
                {
                    "type": event_type,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    ),
                }
            )
            if event_type == "realtime_result":
                results = list(event.get("results") or ())
                stats = dict(event.get("stats") or {})
            elif event_type == "discovery_progress":
                progress = event.get("progress") or {}
                if isinstance(progress, Mapping):
                    for error in progress.get("errors") or ():
                        if not isinstance(error, Mapping):
                            continue
                        warning = {
                            "code": "DISCOVERY_PROVIDER_FAILED",
                            "message": (
                                f"{error.get('engine') or 'discovery'}: "
                                f"{error.get('error_type') or 'Error'}: "
                                f"{error.get('message') or ''}"
                            )[:300],
                        }
                        if warning not in warnings:
                            warnings.append(warning)
                    raw = progress.get("candidates") or ()
                    raw_candidates = [
                        _candidate(item)
                        for item in progress.get("raw_candidates") or raw
                        if isinstance(item, Mapping)
                    ]
                    candidates = [
                        _candidate(item)
                        for item in raw
                        if isinstance(item, Mapping)
                    ]
                    initial = progress.get("initial_candidates") or raw
                    initial_candidates = [
                        _candidate(item)
                        for item in initial
                        if isinstance(item, Mapping)
                    ]
                    post_pivot_candidates = list(candidates)
                    rejected_candidates = [
                        _candidate(item)
                        for item in progress.get("rejected_candidates") or ()
                        if isinstance(item, Mapping)
                    ]
            elif event_type == "discovery_enrichment":
                progress = event.get("progress") or {}
                if isinstance(progress, Mapping) and progress.get("candidates"):
                    candidates = [
                        _candidate(item)
                        for item in progress["candidates"]
                        if isinstance(item, Mapping)
                    ]
            elif event_type == "fetch_progress":
                progress = event.get("progress") or {}
                if isinstance(progress, Mapping):
                    fetch = progress.get("fetch")
                    if isinstance(fetch, Mapping):
                        fetches.append(dict(fetch))
            elif event_type == "search_warning":
                warnings.append(
                    {
                        "code": str(event.get("code") or ""),
                        "message": str(event.get("message") or "")[:300],
                    }
                )
        scope_rejected = {
            "results": sum(not _within_scope(item, scope_host) for item in results),
            "raw_candidates": sum(
                not _within_scope(item, scope_host) for item in raw_candidates
            ),
            "candidates": sum(
                not _within_scope(item, scope_host) for item in candidates
            ),
            "initial_candidates": sum(
                not _within_scope(item, scope_host) for item in initial_candidates
            ),
            "post_pivot_candidates": sum(
                not _within_scope(item, scope_host)
                for item in post_pivot_candidates
            ),
        }
        if scope_host:
            results = [item for item in results if _within_scope(item, scope_host)]
            raw_candidates = [
                item for item in raw_candidates if _within_scope(item, scope_host)
            ]
            candidates = [
                item for item in candidates if _within_scope(item, scope_host)
            ]
            initial_candidates = [
                item
                for item in initial_candidates
                if _within_scope(item, scope_host)
            ]
            post_pivot_candidates = [
                item
                for item in post_pivot_candidates
                if _within_scope(item, scope_host)
            ]
            rejected_candidates = [
                item
                for item in rejected_candidates
                if _within_scope(item, scope_host)
            ]
        evidence, evidence_stage = _public_evidence(
            results,
            candidates,
            query=ranking_query,
            scorer=self.semantic_scorer,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        public = {
            "status": "ok" if evidence else "empty",
            "query": value,
            "original_query": original,
            "effective_query": effective_query,
            "execution_queries": execution_queries,
            "original_query_lane": original_query_lane,
            "query_resolution": query_resolution,
            "compiled_query": compiled.to_dict(),
            "scope_root": scope_root,
            "scope_mode": "strict" if scope_host else "open",
            "scope_rejected": scope_rejected,
            "evidence": [asdict(item) for item in evidence],
            "warnings": warnings,
            "retrieval": {
                "latency_ms": round(latency_ms, 3),
                "returned": len(evidence),
                "evidence_stage": evidence_stage,
                "stats": stats,
                "profile": self.profile,
            },
            "message": (
                "Use only the supplied live-web evidence."
                if evidence
                else "No live-web evidence was found."
            ),
        }
        trace = {
            "schema_version": "agent-web-trace.v1",
            "status": public["status"],
            "profile": self.profile,
            "query": value,
            "original_query": original,
            "effective_query": effective_query,
            "execution_queries": execution_queries,
            "original_query_lane": original_query_lane,
            "query_resolution": query_resolution,
            "compiled_query": compiled.to_dict(),
            "scope_root": scope_root,
            "scope_mode": "strict" if scope_host else "open",
            "scope_rejected": scope_rejected,
            "raw_candidates": raw_candidates,
            "initial_candidates": initial_candidates,
            "post_pivot_candidates": post_pivot_candidates,
            "candidates": candidates,
            "rejected_candidates": rejected_candidates,
            "results": [_result(item) for item in results],
            "fetches": fetches,
            "warnings": warnings,
            "stats": stats,
            "events": events,
            "latency_ms": round(latency_ms, 3),
            "evidence_stage": evidence_stage,
        }
        return public, trace

    def execute(
        self,
        query: str,
        *,
        original_query: str | None = None,
    ) -> dict[str, Any]:
        public, trace = self.execute_with_trace(
            query,
            original_query=original_query,
        )
        if public.get("status") != "invalid" and self.shadow not in (None, False):
            public["retrieval"]["shadow"] = self.shadow.submit(
                str(original_query or query or "").strip(),
                legacy_trace=trace,
                legacy_evidence=public["evidence"],
            )
        if self.trace_observer is not None:
            self.trace_observer(self._trace_case_id, public, trace)
        return public

    def close(self) -> None:
        close = getattr(self.shadow, "close", None)
        if callable(close):
            close()
        self.engine.close()


class EnhancedWebShadow:
    def __init__(
        self,
        adapter: WebSearchAdapter,
        *,
        log_path: str = "",
        max_pending: int = 2,
        sample_rate: float = 1.0,
        sampler: Callable[[], float] | None = None,
        log_mode: str = "full",
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.adapter = adapter
        self.log_path = Path(log_path) if log_path else None
        rate = float(sample_rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("shadow sample_rate must be between 0 and 1")
        mode = str(log_mode or "").strip().casefold()
        if mode not in {"full", "metrics"}:
            raise ValueError("shadow log_mode must be 'full' or 'metrics'")
        self.sample_rate = rate
        self._sampler = sampler or random.SystemRandom().random
        self.log_mode = mode
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="enhanced-web-shadow",
        )
        self._slots = threading.BoundedSemaphore(max(1, int(max_pending)))
        self._write_lock = threading.Lock()

    def compare(
        self,
        query: str,
        *,
        legacy_trace: Mapping[str, Any],
        legacy_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            enhanced, trace = self.adapter.execute_with_trace(query)
            enhanced_evidence = [
                dict(item) for item in enhanced.get("evidence", ())
            ]
            legacy_values = [dict(item) for item in legacy_evidence]
            fallback_used = not enhanced_evidence and bool(legacy_values)
            effective_evidence = (
                legacy_values if fallback_used else enhanced_evidence
            )
            row = {
                "schema_version": "agent-web-shadow.v1",
                "status": (
                    "fallback_legacy_evidence" if fallback_used else "ok"
                ),
                "query": query,
                "legacy_urls": [
                    str(item.get("uri") or "") for item in legacy_evidence
                ],
                "enhanced_urls": [
                    str(item.get("uri") or "")
                    for item in enhanced_evidence
                ],
                "effective_urls": [
                    str(item.get("uri") or "")
                    for item in effective_evidence
                ],
                "fallback_used": fallback_used,
                "fallback_reason": (
                    "enhanced_evidence_empty" if fallback_used else ""
                ),
                "legacy_trace": dict(legacy_trace),
                "enhanced_trace": trace,
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            }
        except Exception as exc:
            row = {
                "schema_version": "agent-web-shadow.v1",
                "status": "fallback_legacy",
                "query": query,
                "legacy_urls": [
                    str(item.get("uri") or "") for item in legacy_evidence
                ],
                "enhanced_urls": [],
                "effective_urls": [
                    str(item.get("uri") or "") for item in legacy_evidence
                ],
                "fallback_used": bool(legacy_evidence),
                "fallback_reason": "enhanced_runtime_error",
                "legacy_trace": dict(legacy_trace),
                "enhanced_trace": {},
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            }
        self._write(row)
        return row

    def submit(
        self,
        query: str,
        *,
        legacy_trace: Mapping[str, Any],
        legacy_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if self.sample_rate <= 0.0 or (
            self.sample_rate < 1.0 and self._sampler() >= self.sample_rate
        ):
            return {
                "enabled": True,
                "submitted": False,
                "reason": "sample_not_selected",
                "sample_rate": self.sample_rate,
                "visible_strategy": "legacy",
            }
        if not self._slots.acquire(blocking=False):
            return {
                "enabled": True,
                "submitted": False,
                "reason": "queue_full",
                "sample_rate": self.sample_rate,
                "visible_strategy": "legacy",
            }
        try:
            future = self._executor.submit(
                self.compare,
                query,
                legacy_trace=dict(legacy_trace),
                legacy_evidence=[dict(item) for item in legacy_evidence],
            )
        except RuntimeError:
            self._slots.release()
            return {
                "enabled": True,
                "submitted": False,
                "reason": "shadow_closed",
                "sample_rate": self.sample_rate,
                "visible_strategy": "legacy",
            }
        future.add_done_callback(lambda _future: self._slots.release())
        return {
            "enabled": True,
            "submitted": True,
            "sample_rate": self.sample_rate,
            "visible_strategy": "legacy",
        }

    @staticmethod
    def _trace_metrics(trace: Mapping[str, Any]) -> dict[str, Any]:
        fetches = [
            item
            for item in trace.get("fetches") or ()
            if isinstance(item, Mapping)
        ]
        return {
            "status": str(trace.get("status") or ""),
            "candidate_count": len(trace.get("candidates") or ()),
            "result_count": len(trace.get("results") or ()),
            "fetch_count": len(fetches),
            "fetch_success_count": sum(
                str(item.get("status") or "").casefold()
                in {"ok", "success", "succeeded"}
                for item in fetches
            ),
            "warning_count": len(trace.get("warnings") or ()),
            "latency_ms": float(trace.get("latency_ms") or 0.0),
            "evidence_stage": str(trace.get("evidence_stage") or ""),
        }

    @classmethod
    def _metrics_row(cls, row: Mapping[str, Any]) -> dict[str, Any]:
        def urls(name: str) -> set[str]:
            output: set[str] = set()
            for value in row.get(name) or ():
                canonical = canonicalize_url(str(value))
                if canonical:
                    output.add(canonical)
            return output

        legacy_urls = urls("legacy_urls")
        enhanced_urls = urls("enhanced_urls")
        effective_urls = urls("effective_urls")
        error = str(row.get("error") or "")
        return {
            "schema_version": "agent-web-shadow-metrics.v1",
            "status": str(row.get("status") or ""),
            "query_chars": len(str(row.get("query") or "")),
            "legacy_evidence_count": len(legacy_urls),
            "enhanced_evidence_count": len(enhanced_urls),
            "effective_evidence_count": len(effective_urls),
            "evidence_url_overlap_count": len(legacy_urls & enhanced_urls),
            "fallback_used": bool(row.get("fallback_used")),
            "fallback_reason": str(row.get("fallback_reason") or ""),
            "legacy": cls._trace_metrics(dict(row.get("legacy_trace") or {})),
            "enhanced": cls._trace_metrics(
                dict(row.get("enhanced_trace") or {})
            ),
            "error_type": error.split(":", 1)[0] if error else "",
            "elapsed_ms": float(row.get("elapsed_ms") or 0.0),
        }

    def _write(self, row: Mapping[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        output = self._metrics_row(row) if self.log_mode == "metrics" else dict(row)
        if self.log_mode == "metrics":
            output["recorded_at"] = datetime.now(timezone.utc).isoformat()
            output["sample_rate"] = self.sample_rate
        value = json.dumps(
            output,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._write_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(value + "\n")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.adapter.close()


def build_web_shadow_from_env(
    config_path: str,
) -> EnhancedWebShadow | None:
    enabled = os.getenv("RWKV_AGENT_WEB_SHADOW", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    log_path = os.getenv(
        "RWKV_AGENT_WEB_SHADOW_LOG",
        "var/web-shadow.jsonl",
    )
    sample_rate_text = os.getenv(
        "RWKV_AGENT_WEB_SHADOW_SAMPLE_RATE",
        "0.10",
    ).strip()
    max_pending_text = os.getenv(
        "RWKV_AGENT_WEB_SHADOW_MAX_PENDING",
        "2",
    ).strip()
    try:
        sample_rate = float(sample_rate_text)
        max_pending = int(max_pending_text)
    except ValueError as exc:
        raise ValueError("invalid Web Shadow sampling configuration") from exc
    if max_pending < 1:
        raise ValueError("Web Shadow max pending must be positive")
    log_mode = os.getenv(
        "RWKV_AGENT_WEB_SHADOW_LOG_MODE",
        "metrics",
    ).strip().casefold()
    if not math.isfinite(sample_rate) or not 0.0 <= sample_rate <= 1.0:
        raise ValueError("Web Shadow sample rate must be between 0 and 1")
    if log_mode not in {"full", "metrics"}:
        raise ValueError("Web Shadow log mode must be 'full' or 'metrics'")
    return EnhancedWebShadow(
        WebSearchAdapter(
            config_path,
            profile="enhanced",
            shadow=False,
        ),
        log_path=log_path,
        max_pending=max_pending,
        sample_rate=sample_rate,
        log_mode=log_mode,
    )
