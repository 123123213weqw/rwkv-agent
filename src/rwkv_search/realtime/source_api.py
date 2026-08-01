from __future__ import annotations

import asyncio
import copy
import html
import json
import os
import re
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from ..config import RealtimeSearchConfig
from ..pipeline.source_selector import SourceCapability, SourceSelector
from ..semantic_selection import PairScorer
from ..text import canonicalize_url
from .types import DiscoveredURL


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._+-]{1,63}")
_GITHUB_NOISE = {
    "author", "commit", "commits", "creator", "find", "founder", "from",
    "github", "issue", "issues", "latest", "maintainer", "newest",
    "official", "owner", "project", "projects", "recent", "release",
    "releases", "repo", "repository", "search", "show", "the", "update",
    "updates", "what", "which", "who",
}
_ACADEMIC_NOISE = {
    "author", "authors", "citation", "citations", "doi", "find", "latest",
    "original", "paper", "papers", "publication", "publications", "search",
}
_PROVIDER_CAPABILITIES = {
    "tavily": SourceCapability(
        "tavily",
        "General live public web discovery across websites and news. 通用实时网页搜索。",
        always=True,
    ),
    "github": SourceCapability(
        "github",
        "GitHub public code hosting: GitHub users, organizations, source-code "
        "repositories, repos, commits, issues and pull requests. GitHub 用户、组织、"
        "开源代码、代码仓库、项目、提交和问题记录。",
    ),
    "crossref": SourceCapability(
        "crossref",
        "Crossref scholarly metadata: research papers, journal articles, authors, "
        "publications, citations and DOI records. 学术论文、期刊、作者、引用和 DOI 元数据。",
    ),
    "mediawiki": SourceCapability(
        "mediawiki",
        "Wikipedia encyclopedia background: concepts, people, organizations, "
        "founders, biographies, definitions and history. "
        "维基百科概念、人物、组织、创始人、简介、定义和历史背景。",
    ),
}
_GITHUB_DETAIL_CAPABILITIES = {
    "profile": SourceCapability(
        "profile",
        "User profile, creator, founder, author, maintainer or identity. "
        "用户、创始人、作者、维护者和身份。",
    ),
    "repos": SourceCapability(
        "repos",
        "All projects, repositories and project list owned by a user or "
        "organization. 用户或组织拥有的全部项目、仓库和项目列表。",
    ),
    "releases": SourceCapability(
        "releases",
        "Software release, releases, version, versions, stable version, tag and "
        "changelog. 软件发布、正式版、版本、标签和发布记录。",
    ),
    "commits": SourceCapability(
        "commits",
        "Commit, commits, update, updates, recent changes and development activity. "
        "提交、代码更新、改动、变更和开发活动。",
    ),
}


def _structured_evidence(title: str, snippet: str) -> str:
    """Keep bounded normalized text already returned by a source API."""

    return " ".join(f"{title}. {snippet}".split())[:4000]


def select_source_providers(
    query: str,
    configured: Sequence[str],
    *,
    scorer: PairScorer | None = None,
    max_specialized: int = 2,
) -> tuple[str, ...]:
    """Select APIs by capability descriptions, not intent keyword branches."""

    allowed = tuple(dict.fromkeys(value.strip().casefold() for value in configured))
    # ``site:`` queries are internally generated domain pivots (or explicit
    # search-engine constraints). Repeating structured APIs for every pivot
    # adds no new scope and can consume the whole discovery deadline.
    if re.search(r"(?<!\w)site:", query, re.I):
        return ()
    return SourceSelector(scorer=scorer).select(
        query,
        allowed,
        _PROVIDER_CAPABILITIES,
        max_optional=max_specialized,
    )


def select_github_detail_scopes(query: str) -> tuple[str, ...]:
    """Select only the GitHub endpoints required by the visible request."""

    # Endpoint selection intentionally uses the declarative lexical fallback.
    # Unlike a general reranker, it returns no scope when the query contains no
    # evidence for that endpoint, preventing four core API calls per repo hit.
    return SourceSelector().select(
        query,
        tuple(_GITHUB_DETAIL_CAPABILITIES),
        _GITHUB_DETAIL_CAPABILITIES,
        max_optional=3,
    )


def parse_tavily_results(value: Mapping[str, Any]) -> list[DiscoveredURL]:
    output: list[DiscoveredURL] = []
    for rank, item in enumerate(value.get("results") or (), 1):
        if not isinstance(item, Mapping):
            continue
        url = canonicalize_url(str(item.get("url") or ""))
        title = " ".join(str(item.get("title") or "").split())
        if not url or not title:
            continue
        output.append(
            DiscoveredURL(
                url=url,
                title=title[:500],
                snippet=" ".join(str(item.get("content") or "").split())[:1800],
                engine="tavily",
                rank=rank,
                published_hint=str(item.get("published_date") or "") or None,
                engine_score=float(item.get("score") or 0.0),
                engines=["tavily"],
                positions=[rank],
            )
        )
    return output


def parse_github_repositories(value: Mapping[str, Any]) -> list[DiscoveredURL]:
    output: list[DiscoveredURL] = []
    for rank, item in enumerate(value.get("items") or (), 1):
        if not isinstance(item, Mapping):
            continue
        url = canonicalize_url(str(item.get("html_url") or ""))
        name = str(item.get("full_name") or item.get("name") or "").strip()
        if not url or not name:
            continue
        description = " ".join(str(item.get("description") or "").split())
        language = str(item.get("language") or "").strip()
        stars = int(item.get("stargazers_count") or 0)
        pushed = str(item.get("pushed_at") or "")
        details = [description]
        if language:
            details.append(f"Language: {language}")
        details.append(f"Stars: {stars}")
        if pushed:
            details.append(f"Last push: {pushed}")
        title = name[:500]
        snippet = ". ".join(value for value in details if value)[:1800]
        output.append(
            DiscoveredURL(
                url=url,
                title=title,
                snippet=snippet,
                engine="github",
                rank=rank,
                published_hint=pushed or None,
                engine_score=min(1.0, 0.2 + stars / 100_000.0),
                engines=["github"],
                positions=[rank],
                cached_text=_structured_evidence(title, snippet),
                cached_text_mode="structured_api",
            )
        )
    return output


def parse_github_release(item: Mapping[str, Any]) -> DiscoveredURL | None:
    url = canonicalize_url(str(item.get("html_url") or ""))
    tag = str(item.get("tag_name") or item.get("name") or "").strip()
    if not url or not tag:
        return None
    body = " ".join(str(item.get("body") or "").split())
    published = str(item.get("published_at") or item.get("created_at") or "")
    title = f"GitHub Release {tag}"[:500]
    snippet = body[:1800]
    return DiscoveredURL(
        url=url,
        title=title,
        snippet=snippet,
        engine="github",
        published_hint=published or None,
        rrf_score=0.02,
        engine_score=1.0,
        engines=["github"],
        cached_text=_structured_evidence(title, snippet),
        cached_text_mode="structured_api",
    )


def parse_github_commits(
    values: Sequence[Mapping[str, Any]],
) -> list[DiscoveredURL]:
    output: list[DiscoveredURL] = []
    for rank, item in enumerate(values, 1):
        url = canonicalize_url(str(item.get("html_url") or ""))
        commit = item.get("commit")
        commit = commit if isinstance(commit, Mapping) else {}
        message = " ".join(str(commit.get("message") or "").split())
        author = commit.get("author")
        author = author if isinstance(author, Mapping) else {}
        date = str(author.get("date") or "")
        sha = str(item.get("sha") or "")[:12]
        if not url or not message:
            continue
        title = f"Commit {sha}: {message.splitlines()[0]}"[:500]
        snippet = message[:1800]
        output.append(
            DiscoveredURL(
                url=url,
                title=title,
                snippet=snippet,
                engine="github",
                rank=rank,
                published_hint=date or None,
                rrf_score=0.03,
                engine_score=0.9,
                engines=["github"],
                positions=[rank],
                cached_text=_structured_evidence(title, snippet),
                cached_text_mode="structured_api",
            )
        )
    return output


def parse_crossref_results(value: Mapping[str, Any]) -> list[DiscoveredURL]:
    message = value.get("message")
    items = message.get("items", ()) if isinstance(message, Mapping) else ()
    output: list[DiscoveredURL] = []
    for rank, item in enumerate(items, 1):
        if not isinstance(item, Mapping):
            continue
        titles = item.get("title") or ()
        title = str(titles[0] if isinstance(titles, list) and titles else "").strip()
        doi = str(item.get("DOI") or "").strip()
        url = canonicalize_url(str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")))
        if not title or not url:
            continue
        authors: list[str] = []
        for author in item.get("author") or ():
            if not isinstance(author, Mapping):
                continue
            name = " ".join(
                value for value in (str(author.get("given") or ""), str(author.get("family") or ""))
                if value
            )
            if name:
                authors.append(name)
        containers = item.get("container-title") or ()
        container = str(containers[0] if isinstance(containers, list) and containers else "")
        abstract = re.sub(r"<[^>]+>", " ", str(item.get("abstract") or ""))
        snippet = ". ".join(
            value for value in (
                f"DOI: {doi}" if doi else "",
                f"Authors: {', '.join(authors[:6])}" if authors else "",
                f"Published in: {container}" if container else "",
                " ".join(html.unescape(abstract).split()),
            ) if value
        )
        published = _crossref_date(item)
        title = title[:500]
        snippet = snippet[:1800]
        output.append(
            DiscoveredURL(
                url=url,
                title=title,
                snippet=snippet,
                engine="crossref",
                rank=rank,
                published_hint=published,
                engine_score=max(0.0, float(item.get("score") or 0.0)),
                engines=["crossref"],
                positions=[rank],
                cached_text=_structured_evidence(title, snippet),
                cached_text_mode="structured_api",
            )
        )
    return output


def _crossref_date(item: Mapping[str, Any]) -> str | None:
    for key in ("published-print", "published-online", "published", "created"):
        value = item.get(key)
        parts = value.get("date-parts") if isinstance(value, Mapping) else None
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        numbers = [int(part) for part in parts[0][:3] if str(part).isdigit()]
        if numbers:
            return "-".join(
                [f"{numbers[0]:04d}", *[f"{part:02d}" for part in numbers[1:]]]
            )
    return None


def parse_mediawiki_pages(
    value: Mapping[str, Any],
    *,
    language: str,
) -> list[DiscoveredURL]:
    query = value.get("query")
    pages = query.get("pages", ()) if isinstance(query, Mapping) else ()
    if isinstance(pages, Mapping):
        pages = list(pages.values())
    output: list[DiscoveredURL] = []
    for rank, item in enumerate(pages if isinstance(pages, list) else (), 1):
        if not isinstance(item, Mapping):
            continue
        title = " ".join(str(item.get("title") or "").split())
        url = canonicalize_url(str(item.get("fullurl") or ""))
        if not url and title:
            url = f"https://{language}.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        extract = " ".join(str(item.get("extract") or "").split())
        revisions = item.get("revisions") or ()
        timestamp = ""
        if isinstance(revisions, list) and revisions and isinstance(revisions[0], Mapping):
            timestamp = str(revisions[0].get("timestamp") or "")
        if title and url:
            title = title[:500]
            snippet = extract[:1800]
            output.append(
                DiscoveredURL(
                    url=url,
                    title=title,
                    snippet=snippet,
                    engine="mediawiki",
                    rank=rank,
                    published_hint=timestamp or None,
                    engine_score=0.8,
                    engines=["mediawiki"],
                    positions=[rank],
                    cached_text=_structured_evidence(title, snippet),
                    cached_text_mode="structured_api",
                )
            )
    return output


class SourceAPIDiscovery:
    def __init__(
        self,
        config: RealtimeSearchConfig,
        session: object,
        *,
        semantic_scorer: PairScorer | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.semantic_scorer = semantic_scorer
        self._github_cache: dict[
            str,
            tuple[float, list[DiscoveredURL]],
        ] = {}
        self._github_locks: dict[str, asyncio.Lock] = {}
        self._github_repository_cache: dict[
            str,
            tuple[float, list[DiscoveredURL]],
        ] = {}
        self._github_repository_locks: dict[str, asyncio.Lock] = {}

    async def discover(
        self,
        query: str,
        *,
        diagnostics: list[dict[str, str]] | None = None,
    ) -> list[DiscoveredURL]:
        providers = select_source_providers(
            query,
            self.config.api_discovery_providers,
        )
        tasks = {
            provider: asyncio.create_task(self._one(provider, query, diagnostics))
            for provider in providers
        }
        if not tasks:
            return []
        # One unreachable provider must not erase results already returned by
        # another provider when the outer discovery stage reaches its budget.
        # Keep a small margin for merging and publishing the partial result.
        total_timeout = min(
            self.config.api_provider_timeout_seconds,
            max(0.25, self.config.discovery_timeout_seconds - 0.25),
        )
        done, pending = await asyncio.wait(
            set(tasks.values()),
            timeout=total_timeout,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        groups = {
            provider: task.result()
            for provider, task in tasks.items()
            if task in done and not task.cancelled() and task.exception() is None
        }
        for provider, task in tasks.items():
            if task in pending:
                self._diagnose(
                    diagnostics,
                    query,
                    provider,
                    TimeoutError(f"provider exceeded {total_timeout:.2f}s budget"),
                )
        output: list[DiscoveredURL] = []
        for provider in providers:
            output.extend(groups.get(provider, ()))
        return output

    async def _one(
        self,
        provider: str,
        query: str,
        diagnostics: list[dict[str, str]] | None,
    ) -> list[DiscoveredURL]:
        try:
            if provider == "tavily":
                return await self._tavily(query)
            if provider == "github":
                return await self._github(query)
            if provider == "crossref":
                return await self._crossref(query)
            if provider == "mediawiki":
                return await self._mediawiki(query)
            return []
        except Exception as exc:
            self._diagnose(diagnostics, query, provider, exc)
            return []

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | list[Any]:
        # Fixed structured API endpoints are low-volume.  The stdlib client is
        # more reliable than aiohttp across the direct and CONNECT-proxy paths
        # on both experiment hosts, and it automatically honors proxy env.
        return await asyncio.wait_for(
            asyncio.to_thread(
                self._request_json_via_urllib,
                method,
                url,
                params,
                headers,
                payload,
                self.config.api_provider_timeout_seconds,
            ),
            timeout=self.config.api_provider_timeout_seconds,
        )

    @staticmethod
    def _request_json_via_urllib(
        method: str,
        url: str,
        params: Mapping[str, str] | None,
        headers: Mapping[str, str] | None,
        payload: Mapping[str, Any] | None,
        timeout: float,
    ) -> Mapping[str, Any] | list[Any]:
        query = urlencode(dict(params or {}))
        target = f"{url}{'&' if '?' in url else '?'}{query}" if query else url
        body = None
        request_headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(dict(payload)).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request = Request(
            target,
            data=body,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urlopen(request, timeout=max(0.25, timeout)) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}") from exc
        if len(raw) > 4 * 1024 * 1024:
            raise RuntimeError("API response exceeds 4 MiB")
        value = json.loads(raw.decode("utf-8", "replace"))
        if not isinstance(value, (Mapping, list)):
            raise RuntimeError("API response is not JSON object/list")
        return value

    async def _tavily(self, query: str) -> list[DiscoveredURL]:
        key = os.getenv(self.config.tavily_api_key_env, "").strip()
        if not key:
            return []
        value = await self._request_json(
            "post",
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {key}"},
            payload={
                "query": query,
                "search_depth": "basic",
                "topic": "general",
                "max_results": self.config.api_provider_max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
        )
        return parse_tavily_results(value if isinstance(value, Mapping) else {})

    async def _github(self, query: str) -> list[DiscoveredURL]:
        tokens = []
        for token in _ASCII_TOKEN_RE.findall(query):
            folded = token.casefold()
            if folded in _GITHUB_NOISE or folded in {value.casefold() for value in tokens}:
                continue
            tokens.append(token)
            if len(tokens) >= 4:
                break
        if not tokens:
            return []
        # GitHub's ``readme`` scope overwhelmingly promotes generic popular
        # repositories that merely mention the entity.  Prefer the strongest
        # product/repository token and search repository names, then expand
        # the winning owner below.
        repository_token = max(
            tokens,
            key=lambda value: (
                int(any(char in value for char in ".-_+")) * 15 + len(value),
                -tokens.index(value),
            ),
        )
        detail_scopes = select_github_detail_scopes(query)
        repository_key = repository_token.casefold()
        cache_key = repository_key + "\0" + ",".join(detail_scopes)
        cached = self._github_cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return copy.deepcopy(cached[1])
        lock = self._github_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._github_cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                return copy.deepcopy(cached[1])
            output = await self._github_uncached(repository_token, detail_scopes)
            if output:
                ttl = max(30.0, float(self.config.search_cache_ttl_seconds))
                self._github_cache[cache_key] = (
                    time.monotonic() + ttl,
                    copy.deepcopy(output),
                )
            return output

    async def _github_repositories(
        self,
        repository_token: str,
        headers: Mapping[str, str],
    ) -> list[DiscoveredURL]:
        cache_key = repository_token.casefold()
        cached = self._github_repository_cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return copy.deepcopy(cached[1])
        lock = self._github_repository_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._github_repository_cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                return copy.deepcopy(cached[1])
            value = await self._request_json(
                "get",
                "https://api.github.com/search/repositories",
                headers=headers,
                params={
                    "q": repository_token + " in:name",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": str(self.config.api_provider_max_results),
                },
            )
            repositories = parse_github_repositories(
                value if isinstance(value, Mapping) else {}
            )
            if repositories:
                ttl = max(30.0, float(self.config.search_cache_ttl_seconds))
                self._github_repository_cache[cache_key] = (
                    time.monotonic() + ttl,
                    copy.deepcopy(repositories),
                )
            return repositories

    async def _github_uncached(
        self,
        repository_token: str,
        detail_scopes: Sequence[str],
    ) -> list[DiscoveredURL]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rwkv-search",
        }
        auth_token = os.getenv(self.config.github_token_env, "").strip()
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        repositories = await self._github_repositories(repository_token, headers)
        if not repositories:
            return []
        parsed = urlsplit(repositories[0].url).path.strip("/").split("/")
        if len(parsed) < 2:
            return repositories
        owner, repo = parsed[:2]
        requests: dict[str, Any] = {}
        selected_scopes = set(detail_scopes)
        if "profile" in selected_scopes:
            requests["profile"] = self._request_json(
                "get",
                f"https://api.github.com/users/{quote(owner)}",
                headers=headers,
            )
        if "releases" in selected_scopes:
            requests["release"] = self._request_json(
                "get",
                f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/releases/latest",
                headers=headers,
            )
        if "commits" in selected_scopes:
            requests["commits"] = self._request_json(
                "get",
                f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/commits",
                headers=headers,
                params={"per_page": "3"},
            )
        if "repos" in selected_scopes:
            requests["owner_repos"] = self._request_json(
                "get",
                f"https://api.github.com/users/{quote(owner)}/repos",
                headers=headers,
                params={"sort": "pushed", "direction": "desc", "per_page": "100"},
            )
        details: dict[str, Any] = {}
        if requests:
            values = await asyncio.gather(*requests.values(), return_exceptions=True)
            details = dict(zip(requests, values))
        profile = details.get("profile")
        release = details.get("release")
        commits = details.get("commits")
        owner_repos = details.get("owner_repos")
        primary = repositories[0]
        primary.discovery_stage = "github_primary_repository"
        primary.discovery_stages = [primary.discovery_stage]
        output = [primary]
        if isinstance(profile, Mapping):
            login = str(profile.get("login") or owner).strip()
            name = str(profile.get("name") or "").strip()
            bio = " ".join(str(profile.get("bio") or "").split())
            public_repos = int(profile.get("public_repos") or 0)
            profile_url = canonicalize_url(
                str(profile.get("html_url") or f"https://github.com/{owner}")
            )
            if profile_url:
                profile_result = DiscoveredURL(
                    url=profile_url,
                    title=(f"{name} (@{login})" if name else f"@{login}")[:500],
                    snippet=". ".join(
                        value
                        for value in (
                            bio,
                            f"GitHub user: {login}",
                            f"Primary repository owner: {owner}",
                            f"Public repositories: {public_repos}" if public_repos else "",
                        )
                        if value
                    )[:1800],
                    engine="github",
                    published_hint=str(profile.get("updated_at") or "") or None,
                    rrf_score=0.04,
                    engine_score=0.95,
                    engines=["github"],
                    discovery_stage="github_owner_profile",
                    discovery_stages=["github_owner_profile"],
                )
                profile_result.cached_text = _structured_evidence(
                    profile_result.title, profile_result.snippet
                )
                profile_result.cached_text_mode = "structured_api"
                output.append(profile_result)
        parsed_owner_repos: list[DiscoveredURL] = []
        if isinstance(owner_repos, list):
            parsed_owner_repos = parse_github_repositories(
                {"items": [item for item in owner_repos if isinstance(item, Mapping)]}
            )
            for item in parsed_owner_repos:
                item.discovery_stage = "github_owner_repository"
                item.discovery_stages = [item.discovery_stage]
            if parsed_owner_repos:
                latest_push = next(
                    (item.published_hint for item in parsed_owner_repos if item.published_hint),
                    None,
                )
                repo_names = ", ".join(item.title for item in parsed_owner_repos)
                index_snippet = (
                    f"GitHub owner: {owner}. Public repositories "
                    f"({len(parsed_owner_repos)}): {repo_names}."
                )
                if latest_push:
                    index_snippet += (
                        f" Most recently pushed: {parsed_owner_repos[0].title} "
                        f"at {latest_push}."
                    )
                index_result = DiscoveredURL(
                    url=f"https://github.com/{quote(owner)}?tab=repositories",
                    title=f"{owner} GitHub repositories",
                    snippet=index_snippet[:1800],
                    engine="github",
                    published_hint=latest_push,
                    rrf_score=0.04,
                    engine_score=1.0,
                    engines=["github"],
                    discovery_stage="github_owner_repository_index",
                    discovery_stages=["github_owner_repository_index"],
                )
                index_result.cached_text = _structured_evidence(
                    index_result.title, index_result.snippet
                )
                index_result.cached_text_mode = "structured_api"
                output.append(index_result)
        if isinstance(release, Mapping):
            parsed_release = parse_github_release(release)
            if parsed_release is not None:
                parsed_release.discovery_stage = "github_latest_release"
                parsed_release.discovery_stages = [parsed_release.discovery_stage]
                output.append(parsed_release)
        if isinstance(commits, list):
            parsed_commits = parse_github_commits(
                [item for item in commits if isinstance(item, Mapping)]
            )
            for item in parsed_commits:
                item.discovery_stage = "github_latest_commit"
                item.discovery_stages = [item.discovery_stage]
            output.extend(parsed_commits)
        seen_urls = {item.url for item in output}
        output.extend(item for item in parsed_owner_repos if item.url not in seen_urls)
        seen_urls.update(item.url for item in output)
        for item in repositories[1:]:
            item.discovery_stage = "github_repository_search"
            item.discovery_stages = [item.discovery_stage]
            if item.url not in seen_urls:
                output.append(item)
                seen_urls.add(item.url)
        return output[: max(20, self.config.api_provider_max_results * 3)]

    async def _crossref(self, query: str) -> list[DiscoveredURL]:
        doi = _DOI_RE.search(query)
        params = {
            "rows": str(self.config.api_provider_max_results),
            "mailto": self.config.crossref_mailto,
        }
        if doi:
            url = "https://api.crossref.org/v1/works/" + quote(doi.group(0), safe="")
        else:
            url = "https://api.crossref.org/v1/works"
            academic_tokens = [
                token
                for token in _ASCII_TOKEN_RE.findall(query)
                if token.casefold() not in _ACADEMIC_NOISE
            ]
            params["query.title"] = " ".join(academic_tokens) or query
        value = await self._request_json(
            "get",
            url,
            params={key: value for key, value in params.items() if value},
            headers={"User-Agent": "rwkv-search/0.3"},
        )
        if doi and isinstance(value, Mapping):
            message = value.get("message")
            value = {"message": {"items": [message] if isinstance(message, Mapping) else []}}
        return parse_crossref_results(value if isinstance(value, Mapping) else {})

    async def _mediawiki(self, query: str) -> list[DiscoveredURL]:
        language = "zh" if _CJK_RE.search(query) else "en"
        value = await self._request_json(
            "get",
            f"https://{language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": str(self.config.api_provider_max_results),
                "prop": "extracts|info|revisions",
                "explaintext": "1",
                "exintro": "1",
                "inprop": "url",
                "rvprop": "timestamp",
                "format": "json",
                "formatversion": "2",
                "utf8": "1",
            },
            headers={"User-Agent": "rwkv-search/0.3"},
        )
        return parse_mediawiki_pages(
            value if isinstance(value, Mapping) else {},
            language=language,
        )

    @staticmethod
    def _diagnose(
        diagnostics: list[dict[str, str]] | None,
        query: str,
        provider: str,
        exc: BaseException,
    ) -> None:
        if diagnostics is not None:
            diagnostics.append(
                {
                    "query": query,
                    "engine": provider,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:300],
                }
            )
