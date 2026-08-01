"""Capability-driven search of a website's own public search endpoint.

This module does not contain domain or topic routing.  It discovers search
forms from the site's HTML, preserves their hidden configuration, and supports
ordinary HTML forms plus the public Webplus AJAX protocol used by many sites.
It is currently consumed only by an isolated benchmark.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlsplit

from selectolax.parser import HTMLParser

from ..analysis.query import QueryAnalyzer
from ..text import canonicalize_url, search_tokens


_WEBPLUS_API = re.compile(
    r"url\s*:\s*['\"]([^'\"]*searchCon/create\.rst[^'\"]*)['\"]",
    re.I,
)
_AOP_OWNER = re.compile(r"\bappOwner\s*=\s*['\"]([^'\"]+)['\"]", re.I)
_AOP_URL_PREFIX = re.compile(r"\burlPrefix\s*=\s*['\"]([^'\"]+)['\"]", re.I)
_AOP_RESULT_PAGE = re.compile(
    r"['\"]([^'\"]*/aop_views/search/modules/[^'\"]*/soso\.html)['\"]",
    re.I,
)
_SITE_OPERATOR = re.compile(r"(?:^|\s)site:[^\s]+", re.I)
_SEARCH_NAME = re.compile(r"(?:^|[_-])(?:q|query|keyword|key|search|text)(?:$|[_-])", re.I)
_SEARCH_HINT = re.compile(r"search|query|keyword|搜索|检索", re.I)
_SCRIPT_SCHEME = re.compile(r"^(?:javascript|mailto|tel):", re.I)
_CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+.#'-]{1,}(?:\s+|$)){2,4}"
)
_QUERY_NOISE = frozenset(
    {
        "according",
        "and",
        "answer",
        "are",
        "date",
        "find",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "official",
        "or",
        "question",
        "search",
        "source",
        "the",
        "time",
        "to",
        "website",
        "what",
        "when",
        "where",
        "which",
        "who",
        "为什么",
        "什么",
        "什么时候",
        "何时",
        "内容",
        "分别",
        "哪位",
        "哪些",
        "哪个",
        "多少",
        "如何",
        "如果",
        "官网",
        "官方",
        "时间",
        "日期",
        "是什么",
        "活动",
        "以及",
        "其中",
        "同时",
        "参与",
        "进行",
        "结果",
        "进行",
        "在",
        "的",
        "与",
        "和",
        "或",
        "并",
        "由",
        "中",
        "年",
        "月",
        "日",
        "了",
        "是",
        "吗",
        "哪",
        "来",
        "从",
        "到",
    }
)


@dataclass(frozen=True)
class InternalSearchForm:
    action: str
    method: str
    query_field: str
    hidden_fields: Mapping[str, str] = field(default_factory=dict)
    protocol: str = "html_form"


@dataclass(frozen=True)
class InternalSearchCandidate:
    url: str
    title: str
    snippet: str = ""
    query: str = ""
    protocol: str = "html_form"


@dataclass(frozen=True)
class InternalSearchResult:
    candidates: tuple[InternalSearchCandidate, ...]
    queries: tuple[str, ...]
    forms: tuple[InternalSearchForm, ...]
    requests: tuple[Mapping[str, Any], ...]
    error: str = ""


def _decode_html(body: bytes) -> str:
    # HTMLParser handles byte input, but decoded text is also needed for the
    # Webplus JavaScript capability descriptor.  UTF-8 covers the tested sites;
    # replacement is safer than guessing from untrusted page content.
    return body.decode("utf-8", "replace")


def _host(value: str) -> str:
    return (urlsplit(value).hostname or "").casefold().removeprefix("www.")


def _same_host_or_subdomain(url: str, expected: str) -> bool:
    actual = _host(url)
    return bool(
        actual
        and expected
        and (actual == expected or actual.endswith("." + expected))
    )


def extract_internal_search_forms(
    html: bytes | str,
    *,
    base_url: str,
    limit: int = 2,
) -> tuple[InternalSearchForm, ...]:
    """Discover bounded search capabilities from form semantics."""

    parser = HTMLParser(html)
    ranked: list[tuple[int, int, InternalSearchForm]] = []
    for order, node in enumerate(parser.css("form")):
        action = urljoin(base_url, str(node.attributes.get("action") or base_url))
        if not _same_host_or_subdomain(action, _host(base_url)):
            continue
        method = str(node.attributes.get("method") or "get").casefold()
        method = "post" if method == "post" else "get"
        hidden: dict[str, str] = {}
        text_inputs: list[tuple[int, str]] = []
        for child in node.css("input"):
            name = str(child.attributes.get("name") or "").strip()
            if not name:
                continue
            input_type = str(child.attributes.get("type") or "text").casefold()
            if input_type == "hidden":
                hidden[name] = str(child.attributes.get("value") or "")
                continue
            if input_type not in {"text", "search"}:
                continue
            visible = " ".join(
                (
                    name,
                    str(child.attributes.get("id") or ""),
                    str(child.attributes.get("class") or ""),
                    str(child.attributes.get("placeholder") or ""),
                )
            )
            score = 4 if _SEARCH_NAME.search(name) else 0
            score += 2 if _SEARCH_HINT.search(visible) else 0
            text_inputs.append((score, name))
        if not text_inputs:
            continue
        text_inputs.sort(key=lambda value: (-value[0], value[1]))
        input_score, query_field = text_inputs[0]
        visible_form = " ".join(
            (
                action,
                str(node.attributes.get("id") or ""),
                str(node.attributes.get("class") or ""),
                str(node.attributes.get("portletmode") or ""),
            )
        )
        score = input_score + (3 if _SEARCH_HINT.search(visible_form) else 0)
        if score < 3:
            continue
        if "/_web/_search/api/search/new.rst" in action:
            protocol = "webplus_ajax"
        elif any("lucenenewssearchkey" in name.casefold() for name in hidden):
            protocol = "vsb_lucene"
        else:
            protocol = "html_form"
        ranked.append(
            (
                score,
                order,
                InternalSearchForm(
                    action=action,
                    method=method,
                    query_field=query_field,
                    hidden_fields=hidden,
                    protocol=protocol,
                ),
            )
        )
    ranked.sort(key=lambda value: (-value[0], value[1]))
    output: list[InternalSearchForm] = []
    seen: set[tuple[str, str]] = set()
    for _, _, form in ranked:
        key = (form.action, form.query_field)
        if key in seen:
            continue
        seen.add(key)
        output.append(form)
        if len(output) >= max(0, int(limit)):
            break
    if len(output) < max(0, int(limit)):
        for node in parser.css("input[value]"):
            action = canonicalize_url(
                urljoin(base_url, str(node.attributes.get("value") or ""))
            )
            if (
                not action
                or "/_web/_search/api/search/new.rst" not in action
                or not _same_host_or_subdomain(action, _host(base_url))
            ):
                continue
            query_input = (
                parser.css_first("input[name='keyword']")
                or parser.css_first("input[id='keyword']")
                or parser.css_first("input[name='showkeycode']")
            )
            query_field = (
                str(query_input.attributes.get("name") or "keyword")
                if query_input is not None
                else "keyword"
            )
            key = (action, query_field)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                InternalSearchForm(
                    action=action,
                    method="get",
                    query_field=query_field,
                    hidden_fields={},
                    protocol="webplus_ajax",
                )
            )
            if len(output) >= max(0, int(limit)):
                break
    if len(output) < max(0, int(limit)):
        source = _decode_html(html)
        owner = _AOP_OWNER.search(source)
        prefix = _AOP_URL_PREFIX.search(source)
        result_page = _AOP_RESULT_PAGE.search(source)
        if owner and prefix and result_page:
            raw_prefix = prefix.group(1).strip().rstrip("/")
            endpoint = urljoin(
                base_url,
                raw_prefix + "/webber/search/search/search/queryPage",
            )
            key = (endpoint, "keyWord")
            if _same_host_or_subdomain(endpoint, _host(base_url)) and key not in seen:
                output.append(
                    InternalSearchForm(
                        action=endpoint,
                        method="post",
                        query_field="keyWord",
                        hidden_fields={
                            "owner": owner.group(1).strip(),
                            "token": "tourist",
                            "urlPrefix": prefix.group(1).strip(),
                            "lang": (
                                "i18n_en_US"
                                if "i18n_en_US" in source and "i18n_zh_CN" not in source
                                else "i18n_zh_CN"
                            ),
                        },
                        protocol="aop_search",
                    )
                )
    return tuple(output)


def build_internal_search_queries(
    original_query: str,
    execution_queries: Sequence[str] = (),
    *,
    max_queries: int = 3,
    visible_site_text: str = "",
) -> tuple[str, ...]:
    """Compile a few label-free queries for a site's own search index.

    When the root page is available, terms already visible in its chrome are
    down-ranked.  This is a generic query-likelihood signal: institution names
    and navigation labels usually occur on every page, while the event, person,
    version, or document sought by the user is rarer.
    """

    analyzer = QueryAnalyzer(max_queries=max(1, max_queries))
    original = " ".join(str(original_query or "").split()).strip()
    analysis = analyzer.analyze(original)
    site_counts = Counter(
        value.casefold() for value in search_tokens(visible_site_text)
    )
    visible_folded = str(visible_site_text or "").casefold()

    def site_frequency(value: str) -> int:
        folded = value.casefold()
        return max(site_counts.get(folded, 0), visible_folded.count(folded))
    values: list[str] = [
        value
        for value in analysis.exact_terms
        if not re.fullmatch(r"[\d\s./:-]+", value)
    ]
    values.extend(
        match.group(0).strip()
        for match in _CAPITALIZED_PHRASE.finditer(original)
    )
    rare_tokens = [
        token
        for token in analysis.tokens
        if token.kind in {"word", "exact"}
        and token.weight >= 0.5
        and len(token.normalized) >= 2
        and token.normalized.casefold() not in _QUERY_NOISE
        and not re.fullmatch(r"\d+(?:[./:-]\d+)*", token.normalized)
    ]
    rare_tokens.sort(
        key=lambda token: (
            0 if token.kind == "exact" else 1,
            -len(token.normalized),
            site_frequency(token.normalized),
            -float(token.weight),
            token.start,
        )
    )
    phrase_values: list[tuple[int, int, int, str]] = []
    phrase_tokens = sorted(
        (
            token
            for token in analysis.tokens
            if token.kind in {"word", "exact"}
            and token.weight >= 0.5
            and token.normalized.casefold() not in _QUERY_NOISE
            and not re.fullmatch(r"\d+(?:[./:-]\d+)*", token.normalized)
        ),
        key=lambda token: token.start,
    )
    for size in (3, 2):
        for start in range(0, len(phrase_tokens) - size + 1):
            group = phrase_tokens[start : start + size]
            if any(
                not re.fullmatch(r"\s*", original[left.end : right.start])
                for left, right in zip(group, group[1:])
            ):
                continue
            phrase = original[group[0].start : group[-1].end].strip()
            compact = re.sub(r"\s+", " ", phrase)
            if len(compact.replace(" ", "")) >= 4:
                phrase_values.append(
                    (
                        sum(
                            site_frequency(token.normalized)
                            for token in group
                        ),
                        -len(compact.replace(" ", "")),
                        group[0].start,
                        compact,
                    )
                )
    phrase_values.sort()
    values.extend(value for _, _, _, value in phrase_values[:3])
    values.extend(token.surface for token in rare_tokens)
    for raw in execution_queries:
        cleaned = _SITE_OPERATOR.sub(" ", str(raw or ""))
        cleaned = " ".join(cleaned.split()).strip(' "“”‘’\'')
        if cleaned:
            values.append(cleaned)
    values.extend(analysis.search_queries)
    values.append(original)

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split()).strip()
        folded = cleaned.casefold()
        if not cleaned or folded in seen:
            continue
        if any(
            folded in existing.casefold() or existing.casefold() in folded
            for existing in output
        ):
            continue
        seen.add(folded)
        output.append(cleaned[:300])
        if len(output) >= max(0, int(max_queries)):
            break
    return tuple(output)


def _parse_links(
    html: bytes | str,
    *,
    base_url: str,
    scope_host: str,
    query: str,
    protocol: str,
) -> list[InternalSearchCandidate]:
    parser = HTMLParser(html)
    output: list[InternalSearchCandidate] = []
    seen: set[str] = set()
    for node in parser.css("a[href]"):
        raw_url = str(node.attributes.get("href") or "").strip()
        if not raw_url or raw_url.startswith("#") or _SCRIPT_SCHEME.match(raw_url):
            continue
        url = canonicalize_url(urljoin(base_url, raw_url))
        if not url or not _same_host_or_subdomain(url, scope_host):
            continue
        path = urlsplit(url).path.casefold()
        if path in {"", "/"} or "search" in path or url in seen:
            continue
        title = " ".join(node.text(separator=" ").split()).strip()
        if not title:
            title = str(node.attributes.get("title") or "").strip()
        if not title:
            continue
        seen.add(url)
        parent = node.parent
        context = (
            " ".join(parent.text(separator=" ").split())[:600]
            if parent is not None
            else ""
        )
        output.append(
            InternalSearchCandidate(
                url=url,
                title=title[:500],
                snippet=context,
                query=query,
                protocol=protocol,
            )
        )
    return output


def _webplus_endpoint(search_page: str, *, base_url: str) -> str:
    match = _WEBPLUS_API.search(search_page)
    return urljoin(base_url, match.group(1)) if match else ""


def _webplus_payload(query: str) -> str:
    values = [
        {"field": "pageIndex", "value": 1},
        {"field": "group", "value": 0},
        {"field": "searchType", "value": ""},
        {"field": "keyword", "value": query},
        {"field": "recommend", "value": ""},
        *({"field": index, "value": ""} for index in range(4, 8)),
    ]
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode("ascii")


def build_aop_search_payload(
    form: InternalSearchForm,
    query: str,
) -> dict[str, Any]:
    """Build the public JSON request declared by the AOP search component."""

    fields = dict(form.hidden_fields)
    owner = str(fields.get("owner") or "")
    token = str(fields.get("token") or "tourist")
    lang = str(fields.get("lang") or "i18n_zh_CN")
    return {
        "aliasName": "article_data,open_data",
        "keyWord": str(query),
        "lastkeyWord": str(query),
        "searchKeyWord": False,
        "orderType": "date",
        "searchType": "text",
        "searchScope": "1",
        "searchOperator": 1,
        "language": "english" if lang == "i18n_en_US" else "chinese",
        "showId": "",
        "auditing": ["1", "5"],
        "owner": owner,
        "token": token,
        "urlPrefix": str(fields.get("urlPrefix") or ""),
        "filterTerm": {"wbsourceid": ["0", "6"]},
        "page": {"current": 0, "size": 20},
        "advance": False,
        "advanceKeyWord": "",
        "lang": lang,
    }


def parse_aop_search_results(
    body: bytes | str,
    *,
    base_url: str,
    scope_host: str,
    query: str,
) -> tuple[InternalSearchCandidate, ...]:
    """Parse the bounded result records returned by an AOP search component."""

    try:
        value = json.loads(_decode_html(body) if isinstance(body, bytes) else body)
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, Mapping) or str(value.get("code") or "") != "0000":
        return ()
    data = value.get("data")
    page = data.get("page") if isinstance(data, Mapping) else None
    records = page.get("records") if isinstance(page, Mapping) else None
    if not isinstance(records, list):
        return ()
    output: list[InternalSearchCandidate] = []
    seen: set[str] = set()
    for record in records[:100]:
        if not isinstance(record, Mapping):
            continue
        url = canonicalize_url(urljoin(base_url, str(record.get("url") or "")))
        if not url or url in seen or not _same_host_or_subdomain(url, scope_host):
            continue
        seen.add(url)
        title = re.sub(r"<[^>]+>", " ", str(record.get("title") or ""))
        snippet = re.sub(
            r"<[^>]+>",
            " ",
            str(record.get("intro") or record.get("content") or ""),
        )
        output.append(
            InternalSearchCandidate(
                url=url,
                title=" ".join(title.split())[:500],
                snippet=" ".join(snippet.split())[:600],
                query=query,
                protocol="aop_search",
            )
        )
    return tuple(output)


def build_html_form_payload(
    form: InternalSearchForm,
    query: str,
) -> dict[str, str]:
    """Materialize an HTML form submission from its declared capability."""

    data = {**dict(form.hidden_fields), form.query_field: str(query)}
    if form.protocol == "vsb_lucene":
        for name in tuple(data):
            if "lucenenewssearchkey" in name.casefold():
                data[name] = str(query)
    return data


async def discover_internal_site_search(
    session: Any,
    *,
    root_url: str,
    queries: Sequence[str],
    original_query: str = "",
    execution_queries: Sequence[str] = (),
    timeout_seconds: float = 8.0,
    max_results: int = 40,
) -> InternalSearchResult:
    """Use a detected first-party search capability with a bounded request plan."""

    requests: list[dict[str, Any]] = []
    candidates: list[InternalSearchCandidate] = []
    scope_host = _host(root_url)

    async def exchange(method: str, url: str, **kwargs: Any) -> tuple[bytes, str, int]:
        started = __import__("time").perf_counter()
        try:
            async with session.request(
                method,
                url,
                allow_redirects=True,
                timeout=timeout_seconds,
                **kwargs,
            ) as response:
                body = await response.read()
                final_url = str(response.url)
                requests.append(
                    {
                        "method": method.upper(),
                        "url": url,
                        "final_url": final_url,
                        "status": int(response.status),
                        "bytes": len(body),
                        "elapsed_ms": round(
                            (__import__("time").perf_counter() - started) * 1000.0,
                            3,
                        ),
                        "error": "",
                    }
                )
                if response.status < 200 or response.status >= 300:
                    return b"", final_url, int(response.status)
                return body, final_url, int(response.status)
        except Exception as exc:
            requests.append(
                {
                    "method": method.upper(),
                    "url": url,
                    "final_url": "",
                    "status": 0,
                    "bytes": 0,
                    "elapsed_ms": round(
                        (__import__("time").perf_counter() - started) * 1000.0,
                        3,
                    ),
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
            return b"", url, 0

    root_body, root_final, _ = await exchange("get", root_url)
    if not root_body:
        return InternalSearchResult((), tuple(queries), (), tuple(requests), "root_fetch_failed")
    forms = extract_internal_search_forms(root_body, base_url=root_final or root_url)
    if not forms:
        return InternalSearchResult((), tuple(queries), (), tuple(requests), "no_search_form")

    if original_query:
        root_parser = HTMLParser(root_body)
        for node in root_parser.css("script,style,noscript"):
            node.decompose()
        root_text = " ".join(root_parser.text(separator=" ").split())[:100_000]
        refined = build_internal_search_queries(
            original_query,
            execution_queries,
            max_queries=max(1, len(queries)),
            visible_site_text=root_text,
        )
        if refined:
            queries = refined

    for form in forms:
        if form.protocol == "aop_search":
            owner = str(form.hidden_fields.get("owner") or "")
            token = str(form.hidden_fields.get("token") or "tourist")
            for query in queries:
                body, final_url, _ = await exchange(
                    "post",
                    form.action,
                    json=build_aop_search_payload(form, str(query)),
                    headers={
                        "Authorization": token,
                        "owner": owner,
                        "appId": "app-search",
                    },
                )
                candidates.extend(
                    parse_aop_search_results(
                        body,
                        base_url=final_url or form.action,
                        scope_host=scope_host,
                        query=str(query),
                    )
                )
        elif form.protocol == "webplus_ajax":
            seed_query = str(queries[0]) if queries else ""
            seed_data = {**dict(form.hidden_fields), form.query_field: seed_query}
            page_body, page_url, _ = await exchange(
                form.method,
                form.action,
                **({"data": seed_data} if form.method == "post" else {"params": seed_data}),
            )
            endpoint = _webplus_endpoint(
                _decode_html(page_body),
                base_url=page_url or form.action,
            )
            if not endpoint:
                continue
            for query in queries:
                body, final_url, _ = await exchange(
                    "post",
                    endpoint,
                    data={"searchInfo": _webplus_payload(str(query))},
                    headers={"Referer": page_url or form.action},
                )
                try:
                    value = json.loads(_decode_html(body))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                data = str(value.get("data") or "") if isinstance(value, Mapping) else ""
                candidates.extend(
                    _parse_links(
                        data,
                        base_url=final_url or endpoint,
                        scope_host=scope_host,
                        query=str(query),
                        protocol=form.protocol,
                    )
                )
        else:
            for query in queries:
                data = build_html_form_payload(form, str(query))
                body, final_url, _ = await exchange(
                    form.method,
                    form.action,
                    **({"data": data} if form.method == "post" else {"params": data}),
                )
                candidates.extend(
                    _parse_links(
                        body,
                        base_url=final_url or form.action,
                        scope_host=scope_host,
                        query=str(query),
                        protocol=form.protocol,
                    )
                )

    def relevance(candidate: InternalSearchCandidate) -> tuple[float, int]:
        query_terms = {
            value.casefold()
            for value in search_tokens(candidate.query)
            if value.casefold() not in _QUERY_NOISE
        }
        document = f"{candidate.title} {candidate.snippet}".casefold()
        document_terms = set(search_tokens(document))
        overlap = query_terms & document_terms
        phrase = " ".join(candidate.query.casefold().split())
        score = (
            2.0 * float(bool(phrase and phrase in document))
            + len(overlap) / max(1, len(query_terms))
            + 0.05 * len(overlap)
        )
        return score, len(candidate.title)

    candidates.sort(key=relevance, reverse=True)
    output: list[InternalSearchCandidate] = []
    seen_urls: set[str] = set()
    for candidate in candidates:
        if candidate.url in seen_urls:
            continue
        seen_urls.add(candidate.url)
        output.append(candidate)
        if len(output) >= max(0, int(max_results)):
            break
    return InternalSearchResult(
        tuple(output),
        tuple(queries),
        forms,
        tuple(requests),
        "" if output else "no_internal_results",
    )
