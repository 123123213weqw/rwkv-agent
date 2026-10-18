from __future__ import annotations

import asyncio
import re
from typing import Mapping, Protocol, Sequence

from ..candidate_index import CandidateHit, CandidateIndexClient
from ..config import RealtimeSearchConfig
from .types import DiscoveredURL


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_SITE_OPERATOR_RE = re.compile(r"(?<!\w)site:", re.IGNORECASE)
_CALENDAR_YEAR_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:18|19|20|21)\d{2}(?![A-Za-z0-9])"
)
_SEARCH_TERM_RE = re.compile(r"[A-Za-z\u3400-\u9fff][A-Za-z0-9_\-\u3400-\u9fff]*")


class CandidateSearchClient(Protocol):
    def search(
        self,
        query_text: str,
        *,
        index: str,
        channel_size: int,
        limit: int,
    ) -> tuple[object, Sequence[CandidateHit], float]: ...


def local_index_for_query(
    query: str,
    indexes: Mapping[str, str],
) -> str:
    """Choose a configured language index using only script information."""

    language = "zh" if _CJK_RE.search(query) else "en"
    return str(indexes.get(language) or indexes.get("default") or "").strip()


def stable_local_query(query: str, *, strip_calendar_years: bool) -> str:
    """Remove standalone year scaffolding without damaging model/version IDs.

    This only applies to the stable local corpus. It deliberately preserves
    embedded identifiers such as ``Python3`` or ``CE16808`` and keeps the
    original query when a year is the only meaningful search term.
    """

    value = str(query or "").strip()
    if not strip_calendar_years or not _CALENDAR_YEAR_RE.search(value):
        return value
    cleaned = " ".join(_CALENDAR_YEAR_RE.sub(" ", value).split())
    if len(_SEARCH_TERM_RE.findall(cleaned)) < 2:
        return value
    return cleaned


class LocalIndexDiscovery:
    """Expose an existing page index as a bounded URL-discovery source."""

    def __init__(
        self,
        config: RealtimeSearchConfig,
        *,
        client: CandidateSearchClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or CandidateIndexClient(
            config.local_discovery_endpoint,
            timeout=config.local_discovery_timeout_seconds,
        )

    async def discover(
        self,
        query: str,
        *,
        freshness: str = "stable",
        diagnostics: list[dict[str, str]] | None = None,
    ) -> list[DiscoveredURL]:
        if not self.config.local_discovery_enabled:
            return []
        if self.config.local_discovery_stable_only and (
            str(freshness).strip().casefold() in {"latest", "realtime"}
        ):
            return []
        # A local general index cannot enforce an arbitrary host constraint.
        # Returning unrelated pages would weaken an explicit scope request.
        if _SITE_OPERATOR_RE.search(query):
            return []
        index = local_index_for_query(
            query,
            self.config.local_discovery_indexes,
        )
        if not index:
            return []
        timeout = max(0.25, float(self.config.local_discovery_timeout_seconds))
        index_query = stable_local_query(
            query,
            strip_calendar_years=self.config.local_discovery_strip_calendar_years,
        )
        try:
            _, hits, _ = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.search,
                    index_query,
                    index=index,
                    channel_size=max(
                        1, int(self.config.local_discovery_channel_size)
                    ),
                    limit=max(1, int(self.config.local_discovery_limit)),
                ),
                timeout=timeout,
            )
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "query": query,
                        "engine": "local_index",
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                )
            return []

        output: list[DiscoveredURL] = []
        for rank, hit in enumerate(hits, 1):
            if not hit.url or not hit.title:
                continue
            output.append(
                DiscoveredURL(
                    url=hit.url,
                    title=hit.title[:500],
                    snippet=" ".join(hit.text.split())[:1800],
                    engine="local_index",
                    rank=rank,
                    published_hint=hit.modified_at or None,
                    # CandidateIndex scores are meaningful within the local
                    # index but are not calibrated to web-engine RRF scores.
                    # URLDiscovery adds the comparable rank contribution.
                    rrf_score=0.0,
                    # Passage scores only compare chunks inside one local
                    # index. They are not calibrated against web-engine
                    # scores and must not become a cross-source prior.
                    engine_score=0.0,
                    engines=["local_index"],
                    positions=[rank],
                    discovery_stage="local_index",
                    discovery_stages=["local_index"],
                    cached_text=hit.text[:400000],
                    cached_text_mode="local_index",
                )
            )
        return output
