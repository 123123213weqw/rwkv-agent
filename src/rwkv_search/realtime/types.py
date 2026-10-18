from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DiscoveredURL:
    url: str
    title: str = ""
    snippet: str = ""
    engine: str = "unknown"
    rank: int = 0
    published_hint: Optional[str] = None
    rrf_score: float = 0.0
    candidate_score: float = 0.0
    engine_score: float = 0.0
    engines: List[str] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    matched_queries: List[str] = field(default_factory=list)
    query_positions: Dict[str, int] = field(default_factory=dict)
    source_channels: List[str] = field(default_factory=list)
    discovery_stage: str = "initial"
    discovery_stages: List[str] = field(default_factory=list)
    parent_url: str = ""
    score_components: Dict[str, float] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)
    # Trusted text already returned by a read-only local index or structured
    # source adapter. It avoids a redundant page GET; the mode remains visible
    # without exposing the text in candidate debug traces.
    cached_text: str = ""
    cached_text_mode: str = ""


@dataclass
class FetchedPage:
    requested_url: str
    final_url: str
    status: int
    content_type: str
    body: bytes
    fetched_at: float
    elapsed_ms: float
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class RealtimeDocument:
    url: str
    title: str
    text: str
    published_at: Optional[str]
    fetched_at: float
    source_type: str
    authority: float
    extraction_quality: float
    relevance: float = 0.0
    freshness: float = 0.0
    score: float = 0.0
    rrf_score: float = 0.0
    candidate_score: float = 0.0
    simhash: str = ""
    links: List[str] = field(default_factory=list)
    retrieval_mode: str = "web_fetch"
