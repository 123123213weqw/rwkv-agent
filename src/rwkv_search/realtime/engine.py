from __future__ import annotations

import asyncio
from collections import Counter
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit

from ..config import RealtimeSearchConfig, SearchConfig
from .candidate_ranker import (
    CandidateAdmission,
    admit_candidates,
    candidate_rejection_reasons,
)
from .discovery import URLDiscovery
from .extractor import extract_page
from .fetcher import AsyncPageFetcher
from .precision_discovery import (
    build_pivot_queries,
    discover_one_hop_links,
    merge_candidate_groups,
    organization_domain,
    select_pivot_domains,
    select_source_channels,
)
from .ranker import rank_documents, to_search_results
from .types import DiscoveredURL, RealtimeDocument


@dataclass
class _FetchOutcome:
    candidate: DiscoveredURL
    document: Optional[RealtimeDocument]
    elapsed_ms: float
    error_type: str = ""
    error_message: str = ""

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "requested_url": self.candidate.url,
            "final_url": self.document.url if self.document else "",
            "status": "succeeded" if self.document else "failed",
            "elapsed_ms": round(self.elapsed_ms, 3),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class RealtimeSearchEngine:
    """Persistent asynchronous I/O runtime exposed as a synchronous event stream."""

    def __init__(
        self,
        config: Optional[RealtimeSearchConfig] = None,
        search_config: Optional[SearchConfig] = None,
    ) -> None:
        self.config = config or RealtimeSearchConfig()
        self.search_config = search_config or SearchConfig()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._startup_error: Optional[str] = None
        self._session: Any = None
        self._discovery: Optional[URLDiscovery] = None
        self._fetcher: Optional[AsyncPageFetcher] = None
        self._start_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": bool(self._ready.is_set() and not self._startup_error),
            "searxng_url": self.config.searxng_url,
            "fallback_engines": list(self.config.fallback_engines),
            "network_family": "ipv4" if self.config.force_ipv4 else "auto",
            "source_channels_enabled": self.config.source_channels_enabled,
            "domain_pivot_enabled": self.config.domain_pivot_enabled,
            "one_hop_link_expansion_enabled": self.config.one_hop_link_expansion_enabled,
            "error": self._startup_error,
        }

    def search_events(
        self,
        query: str,
        queries: Sequence[str],
        *,
        freshness: str,
        depth: str,
        cancel_event: Optional[threading.Event] = None,
        include_candidates: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if not self.enabled:
            yield {"type": "realtime_result", "results": []}
            return
        self._ensure_started()
        if self._startup_error or not self._loop:
            yield {
                "type": "search_warning",
                "code": "REALTIME_SEARCH_UNAVAILABLE",
                "message": self._startup_error or "实时搜索运行时未就绪",
            }
            yield {"type": "realtime_result", "results": []}
            return

        events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._search(
                query,
                queries,
                freshness,
                depth,
                cancel_event,
                events,
                include_candidates,
            ),
            self._loop,
        )
        while True:
            if cancel_event and cancel_event.is_set() and not future.done():
                future.cancel()
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                if future.done():
                    try:
                        future.result()
                    except (asyncio.CancelledError, Exception) as exc:
                        if not (cancel_event and cancel_event.is_set()):
                            yield {
                                "type": "search_warning",
                                "code": "REALTIME_SEARCH_FAILED",
                                "message": f"{type(exc).__name__}: {str(exc)[:240]}",
                            }
                    break
                continue
            yield event
            if event.get("type") == "realtime_result":
                break

    async def _search(
        self,
        query: str,
        queries: Sequence[str],
        freshness: str,
        depth: str,
        cancel_event: Optional[threading.Event],
        events: "queue.Queue[Dict[str, Any]]",
        include_candidates: bool,
    ) -> None:
        assert self._discovery is not None and self._fetcher is not None
        deep = depth == "multi"
        max_queries = (
            self.config.deep_max_queries if deep else self.config.fast_max_queries
        )
        max_candidates = (
            self.config.deep_max_candidates if deep else self.config.fast_max_candidates
        )
        max_fetch = (
            self.config.deep_max_fetch_pages
            if deep
            else self.config.fast_max_fetch_pages
        )
        deadline_seconds = (
            self.config.deep_deadline_seconds
            if deep
            else self.config.fast_deadline_seconds
        )
        selected_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[
            :max_queries
        ] or [query]
        started = time.monotonic()
        discovery_started = time.monotonic()
        discovery_errors: List[Dict[str, str]] = []
        discovery_limit = max_candidates
        if self.config.candidate_admission_enabled:
            discovery_limit *= max(1, self.config.candidate_pool_multiplier)
        source_channels: Sequence[str] = ()
        if self.config.source_channels_enabled:
            source_channels = select_source_channels(query, selected_queries)
        initial_discovery_request_count = len(selected_queries)
        if self.config.searxng_url.rstrip("/") and len(source_channels) > 1:
            initial_discovery_request_count *= len(source_channels)
        initial_raw_candidates = await asyncio.wait_for(
            self._discovery.discover(
                selected_queries,
                freshness=freshness,
                max_candidates=discovery_limit,
                diagnostics=discovery_errors if include_candidates else None,
                source_channels=source_channels,
            ),
            timeout=min(
                deadline_seconds, max(0.2, self.config.discovery_timeout_seconds + 0.5)
            ),
        )
        initial_admission = CandidateAdmission(admitted=list(initial_raw_candidates))
        if self.config.candidate_admission_enabled:
            initial_admission = admit_candidates(
                query,
                selected_queries,
                initial_raw_candidates,
                max_candidates=max_candidates,
                per_domain_limit=self.config.candidate_per_domain_limit,
            )
        pivot_selection = initial_admission.admitted
        if (
            self.config.domain_pivot_enabled
            and not self.config.candidate_admission_enabled
        ):
            pivot_selection = admit_candidates(
                query,
                selected_queries,
                initial_raw_candidates,
                max_candidates=max_candidates,
                per_domain_limit=self.config.candidate_per_domain_limit,
            ).admitted
        pivot_domains = select_pivot_domains(
            query,
            selected_queries,
            pivot_selection,
            max_domains=(
                self.config.domain_pivot_max_domains
                if self.config.domain_pivot_enabled
                or self.config.one_hop_link_expansion_enabled
                else 0
            ),
        )
        pivot_queries: List[str] = []
        pivot_candidates: List[DiscoveredURL] = []
        if self.config.domain_pivot_enabled and pivot_domains:
            pivot_queries = build_pivot_queries(selected_queries[0], pivot_domains)
            remaining = deadline_seconds - (time.monotonic() - started)
            if remaining > 0.2 and pivot_queries:
                try:
                    pivot_candidates = await asyncio.wait_for(
                        self._discovery.discover(
                            pivot_queries,
                            freshness=freshness,
                            max_candidates=min(
                                discovery_limit,
                                max(1, self.config.domain_pivot_max_candidates),
                            ),
                            diagnostics=(
                                discovery_errors if include_candidates else None
                            ),
                            source_channels=("general",) if source_channels else (),
                        ),
                        timeout=min(
                            remaining,
                            max(0.2, self.config.domain_pivot_timeout_seconds),
                        ),
                    )
                    allowed_pivot_domains = {
                        organization_domain(value) for value in pivot_domains
                    }
                    pivot_candidates = [
                        item
                        for item in pivot_candidates
                        if organization_domain(item.url) in allowed_pivot_domains
                    ]
                except asyncio.TimeoutError:
                    if include_candidates:
                        discovery_errors.append(
                            {
                                "query": " | ".join(pivot_queries),
                                "engine": "domain_pivot",
                                "source_channels": ",".join(source_channels),
                                "error_type": "TimeoutError",
                                "message": "domain pivot exceeded its bounded timeout",
                            }
                        )
        raw_candidates = merge_candidate_groups(
            initial_raw_candidates,
            pivot_candidates,
            max_candidates=discovery_limit,
        )
        admission = CandidateAdmission(admitted=list(raw_candidates))
        if self.config.candidate_admission_enabled:
            admission = admit_candidates(
                query,
                selected_queries,
                raw_candidates,
                max_candidates=max_candidates,
                per_domain_limit=self.config.candidate_per_domain_limit,
            )
        candidates = admission.admitted
        discovery_elapsed_ms = round((time.monotonic() - discovery_started) * 1000.0, 3)
        discovery_progress: Dict[str, Any] = {
            "candidate_count": len(candidates),
            "raw_candidate_count": len(raw_candidates),
            "rejected_candidate_count": len(admission.rejected),
            "rejection_counts": admission.rejection_counts,
            "candidate_admission_enabled": self.config.candidate_admission_enabled,
            "source_channels_enabled": self.config.source_channels_enabled,
            "source_channels": list(source_channels),
            "domain_pivot_enabled": self.config.domain_pivot_enabled,
            "pivot_domains": pivot_domains,
            "pivot_queries": pivot_queries,
            "pivot_candidate_count": len(pivot_candidates),
            "discovery_request_count": initial_discovery_request_count
            + len(pivot_queries),
            "query_count": len(selected_queries),
            "queries": selected_queries,
            "engines": sorted(
                {
                    engine
                    for item in candidates
                    for engine in (item.engines or [item.engine])
                }
            ),
            "elapsed_ms": discovery_elapsed_ms,
            "message": f"发现 {len(candidates)} 个网页候选",
        }
        if include_candidates:
            discovery_progress["errors"] = discovery_errors
            discovery_progress["initial_candidates"] = [
                self._candidate_debug_dict(item, position)
                for position, item in enumerate(initial_admission.admitted, 1)
            ]
            discovery_progress["candidates"] = [
                self._candidate_debug_dict(item, position)
                for position, item in enumerate(candidates, 1)
            ]
            discovery_progress["rejected_candidates"] = [
                {
                    "url": item.url,
                    "title": item.title,
                    "snippet": item.snippet,
                    "engine": item.engine,
                    "rank": item.rank,
                    "engines": item.engines,
                    "positions": item.positions,
                    "matched_queries": item.matched_queries,
                    "query_positions": item.query_positions,
                    "rejection_reasons": item.rejection_reasons,
                }
                for item in admission.rejected
            ]
        events.put({"type": "discovery_progress", "progress": discovery_progress})
        if not candidates or (cancel_event and cancel_event.is_set()):
            events.put(
                {
                    "type": "realtime_result",
                    "results": [],
                    "stats": {
                        "candidates": len(candidates),
                        "initial_candidates": len(initial_admission.admitted),
                        "pivot_candidates": len(pivot_candidates),
                        "one_hop_candidates": 0,
                        "source_channels": list(source_channels),
                        "pivot_domains": pivot_domains,
                        "pivot_queries": pivot_queries,
                        "discovery_request_count": initial_discovery_request_count
                        + len(pivot_queries),
                        "raw_candidates": len(raw_candidates),
                        "rejected_candidates": len(admission.rejected),
                        "rejection_counts": admission.rejection_counts,
                        "attempted": 0,
                        "completed": 0,
                        "fetched": 0,
                        "usable": 0,
                        "failed": 0,
                        "cancelled": 0,
                        "selected": 0,
                        "fetch_success_rate": 0.0,
                        "discovery_elapsed_ms": discovery_elapsed_ms,
                        "fetch_elapsed_ms": 0.0,
                        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                    },
                }
            )
            return

        documents: List[RealtimeDocument] = []
        scheduled = 0
        completed = 0
        failed = 0
        cancelled = 0
        post_fetch_rejected = 0
        post_fetch_rejection_counts: Counter[str] = Counter()
        target_candidates = candidates[:max_fetch]
        expanded_parent_urls: set[str] = set()
        one_hop_candidate_count = 0
        batch_size = min(4, max(1, self.config.global_concurrency))
        offset = 0
        fetch_started = time.monotonic()
        while offset < len(target_candidates):
            if cancel_event and cancel_event.is_set():
                break
            remaining = deadline_seconds - (time.monotonic() - started)
            if remaining <= 0.1:
                break
            batch = target_candidates[offset : offset + batch_size]
            tasks = [
                asyncio.create_task(self._fetch_extract_outcome(item)) for item in batch
            ]
            scheduled += len(tasks)
            try:
                for task in asyncio.as_completed(tasks, timeout=remaining):
                    if cancel_event and cancel_event.is_set():
                        break
                    try:
                        outcome = await task
                    except asyncio.CancelledError:
                        continue
                    completed += 1
                    rejection_reasons: List[str] = []
                    if outcome.document is None:
                        failed += 1
                    elif self.config.candidate_admission_enabled:
                        rejection_reasons = candidate_rejection_reasons(
                            query,
                            DiscoveredURL(
                                url=outcome.document.url,
                                title=outcome.document.title,
                                snippet=outcome.document.text[:500],
                                engine="fetched_page",
                            ),
                        )
                        if rejection_reasons:
                            post_fetch_rejected += 1
                            post_fetch_rejection_counts.update(rejection_reasons)
                        else:
                            documents.append(outcome.document)
                    else:
                        documents.append(outcome.document)
                    progress: Dict[str, Any] = {
                        "attempted": completed,
                        "scheduled": scheduled,
                        "succeeded": completed - failed,
                        "usable": len(documents),
                        "rejected": post_fetch_rejected,
                        "failed": failed,
                        "total": len(target_candidates),
                        "message": f"已提取 {len(documents)}/{completed} 个可用网页",
                    }
                    if include_candidates:
                        progress["fetch"] = outcome.to_debug_dict()
                        if rejection_reasons:
                            progress["fetch"]["admission_rejection_reasons"] = (
                                rejection_reasons
                            )
                    events.put(
                        {
                            "type": "fetch_progress",
                            "progress": progress,
                        }
                    )
            except asyncio.TimeoutError:
                pass
            finally:
                pending = [task for task in tasks if not task.done()]
                cancelled += len(pending)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if include_candidates:
                    for task, candidate in zip(tasks, batch):
                        if task in pending:
                            events.put(
                                {
                                    "type": "fetch_progress",
                                    "progress": {
                                        "attempted": completed,
                                        "scheduled": scheduled,
                                        "succeeded": completed - failed,
                                        "usable": len(documents),
                                        "rejected": post_fetch_rejected,
                                        "failed": failed,
                                        "cancelled": cancelled,
                                        "total": len(target_candidates),
                                        "message": f"已取消 {cancelled} 个超时网页",
                                        "fetch": {
                                            "requested_url": candidate.url,
                                            "final_url": "",
                                            "status": "cancelled",
                                            "elapsed_ms": 0.0,
                                            "error_type": "TimeoutOrCancellation",
                                            "error_message": "fetch did not finish before the request deadline",
                                        },
                                    },
                                }
                            )
            # A fetched official landing page often exposes a much more
            # precise version/release URL than the search result page. Replace
            # lower-ranked remaining candidates with those links without
            # increasing the fetch budget.
            expandable_documents = [
                document
                for document in documents
                if document.url not in expanded_parent_urls
            ]
            expanded_parent_urls.update(
                document.url for document in expandable_documents
            )
            precision: List[DiscoveredURL] = []
            if self.config.one_hop_link_expansion_enabled:
                precision = discover_one_hop_links(
                    query,
                    selected_queries,
                    expandable_documents,
                    allowed_domains=pivot_domains,
                    seen_urls=(item.url for item in candidates),
                    max_links=max(
                        0, self.config.one_hop_max_links - one_hop_candidate_count
                    ),
                )
                if precision and self.config.candidate_admission_enabled:
                    precision = admit_candidates(
                        query,
                        selected_queries,
                        precision,
                        max_candidates=len(precision),
                        per_domain_limit=max(1, self.config.one_hop_max_links),
                    ).admitted
            if precision:
                processed_count = offset + len(batch)
                processed = target_candidates[:processed_count]
                remainder = target_candidates[processed_count:]
                rebuilt_fetch_queue: List[DiscoveredURL] = []
                seen = set()
                for item in [*processed, *precision, *remainder]:
                    if item.url in seen:
                        continue
                    seen.add(item.url)
                    rebuilt_fetch_queue.append(item)
                target_candidates = rebuilt_fetch_queue[:max_fetch]

                # Fetch the promising child links promptly, but do not let
                # them displace already discovered domains/pages from the
                # observable top 10. This keeps recall attribution stable
                # while still allowing one-hop pages to contribute at @20.
                protected_prefix = min(10, len(candidates))
                rebuilt_candidates: List[DiscoveredURL] = []
                seen = set()
                for item in [
                    *candidates[:protected_prefix],
                    *precision,
                    *candidates[protected_prefix:],
                ]:
                    if item.url in seen:
                        continue
                    seen.add(item.url)
                    rebuilt_candidates.append(item)
                candidates = rebuilt_candidates[:max_candidates]
                one_hop_candidate_count += len(precision)
                if include_candidates:
                    events.put(
                        {
                            "type": "discovery_enrichment",
                            "progress": {
                                "stage": "one_hop_link",
                                "new_candidate_count": len(precision),
                                "one_hop_candidate_count": one_hop_candidate_count,
                                "parent_count": len(expandable_documents),
                                "candidates": [
                                    self._candidate_debug_dict(item, position)
                                    for position, item in enumerate(candidates, 1)
                                ],
                                "new_candidates": [
                                    self._candidate_debug_dict(item, position)
                                    for position, item in enumerate(precision, 1)
                                ],
                            },
                        }
                    )
            offset += len(batch)
            # Early evidence sufficiency: enough pages and enough independent hosts.
            domains = {
                (urlsplit(item.url).hostname or "").casefold() for item in documents
            }
            if not deep and len(documents) >= 5 and len(domains) >= 3:
                break

        for document in documents:
            if document.rrf_score > 0:
                continue
            candidate = next(
                (item for item in candidates if item.url == document.url), None
            )
            if candidate is None:
                candidate = next(
                    (
                        item
                        for item in candidates
                        if (urlsplit(item.url).hostname or "").casefold()
                        == (urlsplit(document.url).hostname or "").casefold()
                    ),
                    None,
                )
            if candidate:
                document.rrf_score = candidate.rrf_score
        ranked = rank_documents(
            query,
            documents,
            freshness_mode=freshness,
            limit=self.search_config.result_limit,
            per_domain_limit=self.search_config.per_domain_limit,
        )
        results = to_search_results(query, ranked)
        fetch_elapsed_ms = round((time.monotonic() - fetch_started) * 1000.0, 3)
        events.put(
            {
                "type": "realtime_result",
                "results": results,
                "stats": {
                    "candidates": len(candidates),
                    "initial_candidates": len(initial_admission.admitted),
                    "pivot_candidates": len(pivot_candidates),
                    "one_hop_candidates": one_hop_candidate_count,
                    "source_channels": list(source_channels),
                    "pivot_domains": pivot_domains,
                    "pivot_queries": pivot_queries,
                    "discovery_request_count": initial_discovery_request_count
                    + len(pivot_queries),
                    "raw_candidates": len(raw_candidates),
                    "rejected_candidates": len(admission.rejected),
                    "rejection_counts": admission.rejection_counts,
                    "post_fetch_rejected": post_fetch_rejected,
                    "post_fetch_rejection_counts": dict(
                        sorted(post_fetch_rejection_counts.items())
                    ),
                    "attempted": scheduled,
                    "completed": completed,
                    "fetched": len(documents) + post_fetch_rejected,
                    "usable": len(documents),
                    "failed": failed,
                    "cancelled": cancelled,
                    "selected": len(results),
                    "fetch_success_rate": round(len(documents) / max(1, scheduled), 4),
                    "discovery_elapsed_ms": discovery_elapsed_ms,
                    "fetch_elapsed_ms": fetch_elapsed_ms,
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                },
            }
        )

    @staticmethod
    def _candidate_debug_dict(item: DiscoveredURL, position: int) -> Dict[str, Any]:
        return {
            "position": position,
            "url": item.url,
            "title": item.title,
            "snippet": item.snippet,
            "engine": item.engine,
            "rank": item.rank,
            "published_hint": item.published_hint,
            "rrf_score": item.rrf_score,
            "engine_score": item.engine_score,
            "engines": item.engines,
            "positions": item.positions,
            "matched_queries": item.matched_queries,
            "query_positions": item.query_positions,
            "source_channels": item.source_channels,
            "discovery_stage": item.discovery_stage,
            "discovery_stages": item.discovery_stages,
            "parent_url": item.parent_url,
            "candidate_score": item.candidate_score,
            "score_components": item.score_components,
        }

    async def _fetch_extract(
        self, candidate: DiscoveredURL
    ) -> Optional[RealtimeDocument]:
        assert self._fetcher is not None
        page = await self._fetcher.fetch(candidate.url)
        document = extract_page(page)
        if document:
            document.rrf_score = candidate.rrf_score
            if (
                not document.title or document.title == document.url
            ) and candidate.title:
                document.title = candidate.title
        return document

    async def _fetch_extract_outcome(self, candidate: DiscoveredURL) -> _FetchOutcome:
        started = time.monotonic()
        try:
            document = await self._fetch_extract(candidate)
            if document is None:
                return _FetchOutcome(
                    candidate=candidate,
                    document=None,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                    error_type="ExtractionError",
                    error_message="page fetched but no usable document was extracted",
                )
            return _FetchOutcome(
                candidate=candidate,
                document=document,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
            )
        except Exception as exc:
            return _FetchOutcome(
                candidate=candidate,
                document=None,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            )

    def _ensure_started(self) -> None:
        if self._thread and self._thread.is_alive():
            self._ready.wait(timeout=5.0)
            return
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop, daemon=True, name="rwkv-realtime-search"
            )
            self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._initialize())
            self._ready.set()
            loop.run_forever()
        except Exception as exc:
            self._startup_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self._ready.set()
        finally:
            if self._session is not None:
                try:
                    loop.run_until_complete(self._session.close())
                except Exception:
                    pass
            loop.close()

    async def _initialize(self) -> None:
        try:
            import aiohttp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("aiohttp 未安装，请安装 realtime 可选依赖") from exc
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self.config.connect_timeout_seconds,
            sock_connect=self.config.connect_timeout_seconds,
            sock_read=self.config.page_timeout_seconds,
        )
        connector = aiohttp.TCPConnector(
            limit=max(1, self.config.global_concurrency),
            family=socket.AF_INET if self.config.force_ipv4 else socket.AF_UNSPEC,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            auto_decompress=False,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        self._discovery = URLDiscovery(self.config, self._session)
        self._fetcher = AsyncPageFetcher(self.config, self._session)

    def close(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
