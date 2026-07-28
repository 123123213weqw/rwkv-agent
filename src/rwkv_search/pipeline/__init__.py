"""Composable, domain-neutral layers for the RWKV search path.

The package deliberately separates *when* to search, *what* to query,
*where* to discover, *how* to rank, and *how* to expose an answer.  Semantic
decisions are injected through model/scorer interfaces; deterministic code is
limited to protocol, URL, budget, and page-quality constraints.
"""

from .answer_policy import AnswerPolicy
from .discovery import DiscoveryLayer
from .query_compiler import CompiledQuery, QueryCompiler, QueryHints
from .reranker import RetrievalReranker
from .search_need import SearchNeedDecision, SearchNeedGate
from .source_selector import SourceCapability, SourceSelector

__all__ = [
    "AnswerPolicy",
    "CompiledQuery",
    "DiscoveryLayer",
    "QueryCompiler",
    "QueryHints",
    "RetrievalReranker",
    "SearchNeedDecision",
    "SearchNeedGate",
    "SourceCapability",
    "SourceSelector",
]
