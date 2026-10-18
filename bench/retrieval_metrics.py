from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import unquote, urlsplit


SEARCH_ENGINE_DOMAINS = {
    "baidu.com",
    "bing.com",
    "google.com",
    "searx.space",
    "so.com",
    "sogou.com",
}
DICTIONARY_DOMAINS = {
    "baike.baidu.com",
    "dictionary.com",
    "hanyuguoxue.com",
    "iciba.com",
    "merriam-webster.com",
    "zdic.net",
}
DICTIONARY_TERMS = ("字典", "词典", "词霸", "dictionary definition")
ERROR_TERMS = (
    "403 forbidden",
    "404 not found",
    "access denied",
    "page not found",
    "页面不存在",
)
LOGIN_TERMS = (
    "captcha",
    "verify you are human",
    "sign in to continue",
    "login to continue",
    "安全验证",
    "验证码",
    "登录后查看",
)


def normalized_domain(url: str) -> str:
    value = (urlsplit(str(url or "")).hostname or "").casefold().strip(".")
    return value[4:] if value.startswith("www.") else value


def domain_matches(actual: str, expected: str) -> bool:
    actual, expected = actual.casefold().strip("."), expected.casefold().strip(".")
    if actual.startswith("www."):
        actual = actual[4:]
    if expected.startswith("www."):
        expected = expected[4:]
    return bool(
        actual and expected and (actual == expected or actual.endswith("." + expected))
    )


def organization_domain(domain: str) -> str:
    """Return a conservative organization-level parent for partial source credit.

    Strict domain recall remains unchanged.  This extra relation distinguishes a
    useful organization homepage (``usgs.gov``) from the task-specific service
    domain (``earthquake.usgs.gov``) without calling it an exact-domain hit.
    """
    labels = domain.casefold().strip(".").split(".")
    if len(labels) <= 2:
        return ".".join(labels)
    common_second_level = {"ac", "co", "com", "edu", "gov", "net", "org"}
    keep = 3 if len(labels[-1]) == 2 and labels[-2] in common_second_level else 2
    return ".".join(labels[-keep:])


def organization_domain_matches(actual: str, expected: str) -> bool:
    return organization_domain(actual) == organization_domain(expected)


def target_page_matches(
    url: str, expected_domains: Sequence[str], patterns: Sequence[str]
) -> bool:
    parsed = urlsplit(str(url or ""))
    domain = normalized_domain(url)
    if not any(domain_matches(domain, expected) for expected in expected_domains):
        return False
    if not patterns:
        return True
    searchable = unquote(parsed.path or "/")
    if parsed.query:
        searchable += "?" + unquote(parsed.query)
    folded = searchable.casefold()
    for pattern in patterns:
        wanted = unquote(str(pattern)).casefold()
        if wanted.endswith("/"):
            # Dataset path patterns describe a complete path segment sequence,
            # not necessarily a root prefix.  Official sites commonly insert a
            # locale prefix (``/zh-cn/blog/...`` or ``/cn/newsroom/...``).
            # Padding both sides with '/' keeps that valid while still rejecting
            # lookalikes such as ``/downloads-malware``.
            marker = "/" + wanted.strip("/") + "/"
            path_only = folded.split("?", 1)[0]
            padded_path = "/" + path_only.strip("/") + "/"
            if marker in padded_path:
                return True
        elif wanted in folded:
            return True
    return False


def classify_garbage_types(item: Mapping[str, Any]) -> List[str]:
    url = str(item.get("url") or "")
    parsed = urlsplit(url)
    domain = normalized_domain(url)
    path = (parsed.path or "/").casefold()
    title = str(item.get("title") or "").casefold()
    snippet = str(item.get("snippet") or "").casefold()
    visible = f"{title} {snippet}"
    kinds = set()

    if any(domain_matches(domain, value) for value in SEARCH_ENGINE_DOMAINS):
        if path in {"", "/", "/search", "/s", "/web"} or "search" in path:
            kinds.add("search_homepage")
    if any(domain_matches(domain, value) for value in DICTIONARY_DOMAINS) or any(
        term in visible for term in DICTIONARY_TERMS
    ):
        kinds.add("dictionary")

    status = item.get("status_code", item.get("status"))
    try:
        if int(status) >= 400:
            kinds.add("error_page")
    except (TypeError, ValueError):
        pass
    if any(term in visible for term in ERROR_TERMS):
        kinds.add("error_page")
    if any(term in f"{url.casefold()} {visible}" for term in LOGIN_TERMS):
        kinds.add("login_or_captcha")

    if "content_length" in item:
        try:
            if int(item.get("content_length") or 0) <= 0:
                kinds.add("empty_content")
        except (TypeError, ValueError):
            kinds.add("empty_content")
    elif not title.strip() and not snippet.strip():
        kinds.add("empty_content")
    return sorted(kinds)


def is_garbage_result(item: Mapping[str, Any], case: Mapping[str, Any]) -> bool:
    forbidden = set(str(value) for value in case.get("forbidden_result_types", ()))
    return bool(forbidden.intersection(classify_garbage_types(item)))


def _domain_hit(
    items: Sequence[Mapping[str, Any]], expected: Sequence[str], k: int
) -> bool:
    return any(
        domain_matches(normalized_domain(str(item.get("url") or "")), domain)
        for item in items[:k]
        for domain in expected
    )


def _organization_domain_hit(
    items: Sequence[Mapping[str, Any]], expected: Sequence[str], k: int
) -> bool:
    return any(
        organization_domain_matches(
            normalized_domain(str(item.get("url") or "")), domain
        )
        for item in items[:k]
        for domain in expected
    )


def _target_page_hit(
    items: Sequence[Mapping[str, Any]],
    expected_domains: Sequence[str],
    patterns: Sequence[str],
    k: int,
) -> bool:
    return any(
        target_page_matches(str(item.get("url") or ""), expected_domains, patterns)
        for item in items[:k]
    )


def evaluate_candidate_stage(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate one observable discovery stage without fetched-result metrics."""
    expected = [str(value) for value in case.get("expected_domains_any", ())]
    patterns = [str(value) for value in case.get("target_url_patterns_any", ())]
    return {
        "candidate_count": len(candidates),
        "candidate_domain_hit_at_5": _domain_hit(candidates, expected, 5),
        "candidate_domain_hit_at_10": _domain_hit(candidates, expected, 10),
        "candidate_domain_hit_at_20": _domain_hit(candidates, expected, 20),
        "candidate_organization_domain_hit_at_10": _organization_domain_hit(
            candidates, expected, 10
        ),
        "candidate_organization_domain_hit_at_20": _organization_domain_hit(
            candidates, expected, 20
        ),
        "candidate_target_page_hit_at_10": _target_page_hit(
            candidates, expected, patterns, 10
        ),
        "candidate_target_page_hit_at_20": _target_page_hit(
            candidates, expected, patterns, 20
        ),
    }


def evaluate_case(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    stats = stats or {}
    expected = [str(value) for value in case.get("expected_domains_any", ())]
    patterns = [str(value) for value in case.get("target_url_patterns_any", ())]
    candidate_metrics = evaluate_candidate_stage(case, candidates)
    garbage_types = Counter(
        kind for item in results for kind in classify_garbage_types(item)
    )
    garbage_count = sum(is_garbage_result(item, case) for item in results)
    attempted = int(stats.get("attempted", 0) or 0)
    succeeded = int(
        stats.get(
            "origin_fetch_succeeded",
            stats.get("fetched", stats.get("succeeded", 0)),
        )
        or 0
    )
    failed = int(stats.get("failed", 0) or 0)
    cancelled = int(stats.get("cancelled", 0) or 0)
    return {
        **candidate_metrics,
        "result_count": len(results),
        "nonempty_result": bool(results),
        "result_domain_hit_at_5": _domain_hit(results, expected, 5),
        "result_domain_hit_at_10": _domain_hit(results, expected, 10),
        "result_domain_hit_at_20": _domain_hit(results, expected, 20),
        "result_organization_domain_hit_at_10": _organization_domain_hit(
            results, expected, 10
        ),
        "result_organization_domain_hit_at_20": _organization_domain_hit(
            results, expected, 20
        ),
        "result_target_page_hit_at_10": _target_page_hit(
            results, expected, patterns, 10
        ),
        "result_target_page_hit_at_20": _target_page_hit(
            results, expected, patterns, 20
        ),
        "garbage_result_count": garbage_count,
        "garbage_result_rate": round(garbage_count / max(1, len(results)), 4),
        "garbage_type_counts": dict(sorted(garbage_types.items())),
        "fetch_attempted": attempted,
        "fetch_succeeded": succeeded,
        "fetch_failed": failed,
        "fetch_cancelled": cancelled,
        "fetch_success_rate": round(succeeded / max(1, attempted), 4),
    }


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def aggregate(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    metric_names = (
        "candidate_domain_hit_at_5",
        "candidate_domain_hit_at_10",
        "candidate_domain_hit_at_20",
        "candidate_organization_domain_hit_at_10",
        "candidate_organization_domain_hit_at_20",
        "result_domain_hit_at_5",
        "result_domain_hit_at_10",
        "result_domain_hit_at_20",
        "result_organization_domain_hit_at_10",
        "result_organization_domain_hit_at_20",
        "candidate_target_page_hit_at_10",
        "candidate_target_page_hit_at_20",
        "result_target_page_hit_at_10",
        "result_target_page_hit_at_20",
    )

    def summarize(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {"total": 0}
        output: Dict[str, Any] = {"total": len(group)}
        for name in metric_names:
            output[name + "_rate"] = round(
                sum(bool(row["metrics"].get(name)) for row in group) / len(group), 4
            )
        output["nonempty_result_rate"] = round(
            sum(bool(row["metrics"].get("nonempty_result")) for row in group)
            / len(group),
            4,
        )
        candidate_count = sum(
            int(row["metrics"].get("candidate_count", 0)) for row in group
        )
        result_count = sum(int(row["metrics"].get("result_count", 0)) for row in group)
        garbage_count = sum(
            int(row["metrics"].get("garbage_result_count", 0)) for row in group
        )
        attempted = sum(int(row["metrics"].get("fetch_attempted", 0)) for row in group)
        succeeded = sum(int(row["metrics"].get("fetch_succeeded", 0)) for row in group)
        failed = sum(int(row["metrics"].get("fetch_failed", 0)) for row in group)
        cancelled = sum(int(row["metrics"].get("fetch_cancelled", 0)) for row in group)
        elapsed = [float(row.get("total_elapsed_ms", 0.0)) for row in group]
        output.update(
            {
                "candidate_count": candidate_count,
                "result_count": result_count,
                "average_candidate_count": round(candidate_count / len(group), 3),
                "average_result_count": round(result_count / len(group), 3),
                "garbage_result_count": garbage_count,
                "garbage_result_rate": round(garbage_count / max(1, result_count), 4),
                "fetch_attempted": attempted,
                "fetch_succeeded": succeeded,
                "fetch_failed": failed,
                "fetch_cancelled": cancelled,
                "fetch_success_rate": round(succeeded / max(1, attempted), 4),
                "average_total_elapsed_ms": round(sum(elapsed) / len(group), 3),
                "p95_total_elapsed_ms": round(_percentile_95(elapsed), 3),
            }
        )
        return output

    def summarize_candidate_stage(
        group: Sequence[Mapping[str, Any]], stage: str
    ) -> Dict[str, Any]:
        values = [
            row.get("candidate_stage_metrics", {}).get(stage, {}) for row in group
        ]
        values = [value for value in values if isinstance(value, Mapping)]
        if not values:
            return {"total": 0}
        names = (
            "candidate_domain_hit_at_5",
            "candidate_domain_hit_at_10",
            "candidate_domain_hit_at_20",
            "candidate_organization_domain_hit_at_10",
            "candidate_organization_domain_hit_at_20",
            "candidate_target_page_hit_at_10",
            "candidate_target_page_hit_at_20",
        )
        output: Dict[str, Any] = {"total": len(values)}
        for name in names:
            output[name + "_rate"] = round(
                sum(bool(value.get(name)) for value in values) / len(values), 4
            )
        count = sum(int(value.get("candidate_count", 0)) for value in values)
        output["candidate_count"] = count
        output["average_candidate_count"] = round(count / len(values), 3)
        return output

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"language:{row.get('language', 'unknown')}"].append(row)
        groups[f"category:{row.get('category', 'unknown')}"].append(row)
        groups[f"source_policy:{row.get('source_policy', 'unknown')}"].append(row)
    candidate_stages = sorted(
        {
            str(stage)
            for row in rows
            for stage in (row.get("candidate_stage_metrics", {}) or {})
        }
    )
    return {
        "schema_version": "realtime-retrieval-bench.v1",
        "overall": summarize(rows),
        "groups": {key: summarize(value) for key, value in sorted(groups.items())},
        "candidate_stages": {
            stage: summarize_candidate_stage(rows, stage) for stage in candidate_stages
        },
    }
