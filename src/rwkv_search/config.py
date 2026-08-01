from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CrawlConfig:
    user_agent: str = "RWKVLocalSearchBot/0.1 (+local-research)"
    global_concurrency: int = 8
    per_host_concurrency: int = 2
    per_host_delay_seconds: float = 0.75
    timeout_seconds: float = 15.0
    max_response_bytes: int = 4 * 1024 * 1024
    max_pages_per_run: int = 200
    max_depth: int = 2
    max_links_per_page: int = 200
    robots_cache_seconds: int = 86400
    allow_private_networks: bool = False


@dataclass
class SearchConfig:
    candidate_limit: int = 100
    result_limit: int = 10
    per_domain_limit: int = 4
    evidence_limit: int = 8
    evidence_character_budget: int = 18000
    passage_selection_enabled: bool = True
    passage_min_document_chars: int = 700
    passage_target_chars: int = 900
    passage_hard_max_chars: int = 1400
    passage_max_per_document: int = 3
    passage_max_chars_per_evidence: int = 3200


@dataclass
class ModelConfig:
    enabled: bool = False
    path: str = ""
    label: str = "RWKV"
    device: str = "cuda"
    dtype: str = "fp16"
    native_model: bool = False
    max_input_tokens: int = 8192
    max_new_tokens: int = 640
    repair_once: bool = True
    warmup: bool = True
    session_cache_enabled: bool = True
    session_ttl_seconds: int = 1800
    session_max_entries: int = 12
    session_cpu_offload: bool = True


@dataclass
class RealtimeSearchConfig:
    """Bounded query-time web discovery and extraction.

    This path deliberately does not write fetched pages into the local index.
    It also does not perform HEAD or robots.txt requests, keeping the fast path
    to one discovery request plus the minimum number of document GETs.
    """

    enabled: bool = False
    searxng_url: str = "http://127.0.0.1:8888"
    searxng_engines: List[str] = field(
        default_factory=lambda: ["dogpile", "naver"]
    )
    # Optional additive lanes selected from the query script. This keeps the
    # stable metasearch core small while adding a regional engine only where
    # it is useful. Supported keys: zh, ja, ko, and default.
    searxng_language_engines: Dict[str, List[str]] = field(default_factory=dict)
    fallback_engines: List[str] = field(default_factory=lambda: ["bing"])
    # Optional structured discovery providers. They remain disabled by default
    # and preserve the single model-facing ``web_search(query)`` contract.
    api_discovery_providers: List[str] = field(default_factory=list)
    api_provider_timeout_seconds: float = 4.5
    api_provider_max_results: int = 10
    tavily_api_key_env: str = "TAVILY_API_KEY"
    github_token_env: str = "GITHUB_TOKEN"
    crossref_mailto: str = ""
    # Optional read-only local discovery source.  This is deliberately
    # separate from the persistent knowledge tool: when enabled, indexed
    # public pages participate in URL discovery and are fused with live web
    # candidates, but query-time web fetches are still never written back.
    local_discovery_enabled: bool = False
    local_discovery_endpoint: str = "http://127.0.0.1:19220"
    local_discovery_indexes: Dict[str, str] = field(
        default_factory=lambda: {
            "zh": "rwkv-finewiki-zh-full-v1",
            "en": "rwkv-finewiki-en-full-v1",
        }
    )
    local_discovery_timeout_seconds: float = 3.5
    local_discovery_limit: int = 20
    local_discovery_channel_size: int = 50
    local_discovery_evidence_limit: int = 3
    local_discovery_evidence_min_score: float = 0.22
    local_discovery_evidence_min_entity_coverage: float = 0.5
    # Calendar years are often "as of" scaffolding in stable knowledge
    # questions. The candidate index otherwise treats them as high-weight
    # exact terms and can rank generic year pages above named entities.
    # Frozen FRAMES A/B accepted this as the local-discovery default. It has
    # no effect while local discovery itself remains disabled.
    local_discovery_strip_calendar_years: bool = True
    # Monthly/offline corpora must not be mistaken for a source of changing
    # facts. Stable-only also avoids unnecessary index I/O on live queries.
    local_discovery_stable_only: bool = True
    bing_base_url: str = "https://www.bing.com"
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 RWKVSearch/0.2"
    )
    global_concurrency: int = 8
    per_host_concurrency: int = 2
    force_ipv4: bool = False
    connect_timeout_seconds: float = 1.5
    page_timeout_seconds: float = 5.0
    discovery_timeout_seconds: float = 3.5
    max_redirects: int = 5
    response_header_max_bytes: int = 32 * 1024
    max_compressed_bytes: int = 2 * 1024 * 1024
    max_decompressed_bytes: int = 4 * 1024 * 1024
    fast_max_queries: int = 2
    fast_max_candidates: int = 30
    fast_max_fetch_pages: int = 8
    fast_deadline_seconds: float = 8.0
    deep_max_queries: int = 4
    deep_max_candidates: int = 60
    deep_max_fetch_pages: int = 20
    deep_deadline_seconds: float = 30.0
    candidate_admission_enabled: bool = False
    candidate_pool_multiplier: int = 2
    candidate_per_domain_limit: int = 3
    query_compaction_enabled: bool = False
    # Experimental scoped-search lane.  When enabled, the first web call in
    # one scoped Agent request executes one additional deterministic query
    # derived from the user's original wording.  It complements rather than
    # replaces the model Tool query and is disabled by default until the
    # frozen discovery benchmark accepts it.
    original_query_lane_enabled: bool = False
    source_channels_enabled: bool = False
    domain_pivot_enabled: bool = False
    domain_pivot_max_domains: int = 2
    domain_pivot_max_candidates: int = 20
    domain_pivot_timeout_seconds: float = 3.5
    one_hop_link_expansion_enabled: bool = False
    one_hop_max_links: int = 8
    # A search-result snippet is weaker than fetched page text, but is still
    # useful when the origin blocks or times out.  Keep this fallback explicit
    # and conservative so answer generation can distinguish it from full-page
    # evidence.
    snippet_fallback_enabled: bool = False
    snippet_fallback_min_chars: int = 96
    snippet_fallback_min_cjk_chars: int = 48
    snippet_fallback_min_candidate_score: float = 0.18
    snippet_fallback_min_entity_coverage: float = 0.5
    # Disabled above 1.0 by default. Experiments may let a strong composite
    # admission score substitute for brittle lexical entity overlap.
    snippet_fallback_entity_bypass_score: float = 1.1
    # Experimental gate: a successful HTTP response can still be unusable
    # when a shell/JS page yields no extractable正文.  Reusing the already
    # validated SERP excerpt costs no extra request, but remains separately
    # switchable until its retrieval A/B is accepted.
    snippet_fallback_on_extraction_error: bool = False
    search_cache_ttl_seconds: int = 120
    page_cache_ttl_seconds: int = 300
    cache_max_bytes: int = 64 * 1024 * 1024
    allow_private_networks: bool = False


@dataclass
class ShadowSearchConfig:
    """Read-only candidate-index evaluation that never changes user output."""

    enabled: bool = False
    endpoint: str = "http://127.0.0.1:19220"
    index: str = "rwkv-finewiki-zh-full-v1"
    timeout_seconds: float = 3.0
    limit: int = 10
    channel_size: int = 50
    sample_rate: float = 1.0
    max_workers: int = 2
    log_path: str = "data/shadow/finewiki-shadow-v1.jsonl"
    max_log_bytes: int = 64 * 1024 * 1024
    passage_hydration_enabled: bool = False
    passage_max_pages: int = 8
    passage_chunks_per_page: int = 12
    passage_max_chars: int = 3200
    passage_model: str = "BAAI/bge-reranker-v2-m3"
    passage_device: str = "auto"
    passage_batch_size: int = 16
    passage_max_length: int = 512
    passage_fp16: bool = True
    passage_local_files_only: bool = True


@dataclass
class AppConfig:
    database: str = "data/search.db"
    crawler: CrawlConfig = field(default_factory=CrawlConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    realtime_search: RealtimeSearchConfig = field(default_factory=RealtimeSearchConfig)
    shadow_search: ShadowSearchConfig = field(default_factory=ShadowSearchConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        if path is None:
            return cls()
        data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            database=str(data.get("database", "data/search.db")),
            crawler=CrawlConfig(**data.get("crawler", {})),
            search=SearchConfig(**data.get("search", {})),
            model=ModelConfig(**data.get("model", {})),
            realtime_search=RealtimeSearchConfig(**data.get("realtime_search", {})),
            shadow_search=ShadowSearchConfig(**data.get("shadow_search", {})),
        )
