"""Evidence compaction and cross-query-view selection for State research."""

from __future__ import annotations

from typing import Any

from rwkv_search.semantic_selection import PairScorer, select_diverse_items
from rwkv_search.text import canonicalize_url


PRIMARY_EVIDENCE_SOURCES = frozenset(
    {
        "company_filing",
        "github",
        "government",
        "mediawiki",
        "official_docs",
        "official_repository",
        "regulator",
    }
)

def _compact_branch_observation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "evidence": [
            {
                "id": item.get("id"),
                "title": str(item.get("title") or "")[:160],
                "content": str(item.get("content") or "")[:700],
                "uri": item.get("uri"),
                "source": item.get("source"),
                "published_at": item.get("published_at"),
                "score": item.get("score"),
                "discovery_stage": item.get("discovery_stage"),
            }
            for item in list(result.get("evidence") or [])[:5]
            if isinstance(item, dict)
        ],
    }


def _merge_evidence(
    results: list[dict[str, Any]],
    *,
    question: str = "",
    limit: int = 12,
    scorer: PairScorer | None = None,
    preserve_query_views: bool = False,
) -> list[dict[str, Any]]:
    """Select complementary evidence from model-generated query views.

    The branch queries are the information-needs representation.  Selection is
    therefore independent of topic words, domains, URL shapes, and fixed source
    records; it uses the shared query-view MMR selector instead.
    """

    candidates: list[dict[str, Any]] = []
    query_views: list[str] = []
    grouped_candidates: dict[str, list[dict[str, Any]]] = {}
    candidate_view_ids: dict[str, set[str]] = {}
    for result_index, result in enumerate(results):
        result_query = " ".join(str(result.get("query") or "").split()).strip()
        if result_query:
            query_views.append(result_query)
        view_id = str(result.get("_query_view_id") or "").strip()
        if not view_id:
            view_id = result_query.casefold() or f"result-{result_index}"
        group = grouped_candidates.setdefault(view_id, [])
        for position, item in enumerate(result.get("evidence") or [], start=1):
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "").strip()
            content = str(item.get("content") or "").strip()
            if not uri and not content:
                continue
            candidate = {
                "title": str(item.get("title") or "")[:240],
                "content": content[:1800],
                "uri": uri,
                "source": str(item.get("source") or ""),
                "published_at": item.get("published_at"),
                "discovery_stage": str(item.get("discovery_stage") or ""),
                "_best_position": position,
                "_upstream_score": float(item.get("score") or 0.0),
            }
            candidates.append(candidate)
            group.append(candidate)
            key = canonicalize_url(uri) or uri.casefold()
            candidate_view_ids.setdefault(key, set()).add(view_id)
    primary_candidates = [
        item
        for item in candidates
        if str(item.get("source") or "") in PRIMARY_EVIDENCE_SOURCES
    ]
    primary_by_stage: dict[str, dict[str, Any]] = {}
    primary_without_stage: list[dict[str, Any]] = []
    for item in primary_candidates:
        stage = str(item.get("discovery_stage") or "").strip()
        if not stage:
            primary_without_stage.append(item)
            continue
        key = f"{item.get('source') or ''}:{stage}"
        current = primary_by_stage.get(key)
        if current is None or (
            str(item.get("published_at") or ""),
            float(item.get("_upstream_score") or 0.0),
            len(str(item.get("content") or "")),
        ) > (
            str(current.get("published_at") or ""),
            float(current.get("_upstream_score") or 0.0),
            len(str(current.get("content") or "")),
        ):
            primary_by_stage[key] = item
    primary_candidates = [*primary_by_stage.values(), *primary_without_stage]
    reserved = list(
        select_diverse_items(
            question,
            query_views,
            primary_candidates,
            # Keep enough primary records to cover compound questions (identity,
            # collection/index and latest event) while leaving half of the normal
            # twelve-item budget for independent corroboration.
            limit=min(6, int(limit), len(primary_candidates)),
            scorer=scorer,
        ).items
    )
    reserved_keys = {
        canonicalize_url(str(item.get("uri") or ""))
        or str(item.get("uri") or "").casefold()
        for item in reserved
    }
    remainder = [
        item
        for item in candidates
        if (
            canonicalize_url(str(item.get("uri") or ""))
            or str(item.get("uri") or "").casefold()
        )
        not in reserved_keys
    ]
    selected = [
        *reserved,
        *select_diverse_items(
            question,
            query_views,
            remainder,
            limit=max(0, int(limit) - len(reserved)),
            scorer=scorer,
        ).items,
    ]
    if preserve_query_views and selected:
        # Preserve the accepted control Top-K byte-for-byte, then append a
        # bounded set of query-view representatives.  This is intentionally
        # non-destructive: a complementary view cannot evict an item selected
        # by the frozen global MMR baseline.
        selected_keys = {
            canonicalize_url(str(item.get("uri") or ""))
            or str(item.get("uri") or "").casefold()
            for item in selected
        }
        representatives: list[dict[str, Any]] = []
        for group in grouped_candidates.values():
            if not group:
                continue
            # Each tool result is already ordered by the retrieval pipeline.
            # Preserve its first admitted item rather than globally reranking
            # the lane a second time and losing its local winner.
            representative = group[0]
            key = (
                canonicalize_url(str(representative.get("uri") or ""))
                or str(representative.get("uri") or "").casefold()
            )
            # One-off lane winners are too noisy to expand the answer context.
            # Append only pages independently observed by at least two
            # generated query views; the existing global Top-K remains intact.
            if key not in selected_keys and len(candidate_view_ids.get(key, ())) >= 2:
                representatives.append(representative)
        extra_budget = min(len(representatives), max(0, int(limit) // 2))
        selected.extend(
            select_diverse_items(
                question,
                query_views,
                representatives,
                limit=extra_budget,
                scorer=scorer,
            ).items
        )
    output = [
        {
            "id": f"W{index}",
            "title": str(item["title"]),
            "content": str(item["content"]),
            "uri": str(item["uri"]),
            **(
                {"source": str(item["source"])}
                if item.get("source")
                else {}
            ),
            **(
                {"published_at": item["published_at"]}
                if item.get("published_at")
                else {}
            ),
            **(
                {"discovery_stage": str(item["discovery_stage"])}
                if item.get("discovery_stage")
                else {}
            ),
        }
        for index, item in enumerate(selected, start=1)
    ]
    return output


def compact_answer_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_source: int = 900,
) -> list[dict[str, Any]]:
    """Keep one question-relevant bounded span per evidence source.

    This is a Gold-blind context-budget operation.  It preserves source IDs and
    URIs while ensuring answer-stage training and inference see the same compact
    evidence shape instead of silently left-truncating several whole documents.
    """

    from .tools.long_text import chunk_text, rank_chunks

    if max_chars_per_source < 256:
        raise ValueError("max_chars_per_source must be at least 256")
    output: list[dict[str, Any]] = []
    for item in evidence:
        value = dict(item)
        content = str(value.get("content") or "").strip()
        if content:
            chunks = chunk_text(
                content,
                max_chars=max_chars_per_source,
                overlap_chars=min(80, max_chars_per_source // 4),
            )
            selected = rank_chunks(
                f"{question} {value.get('title') or ''}",
                chunks,
                top_k=1,
            )
            if selected:
                value["content"] = selected[0][1].text[:max_chars_per_source]
            else:
                value["content"] = content[:max_chars_per_source]
        output.append(value)
    return output
