from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import queue
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import RealtimeSearchConfig, SearchConfig, ShadowSearchConfig
from .db import SearchDatabase
from .evidence import Evidence, EvidenceBuilder
from .router import RouteDecision, RuleRouter
from .search import HybridSearcher, SearchResult
from .text import search_tokens


class Answerer(Protocol):
    def answer(
        self,
        query: str,
        route: RouteDecision,
        evidence: List[Evidence],
        *,
        as_of: str,
        timezone: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        ...


class SearchService:
    def __init__(
        self,
        database: SearchDatabase,
        *,
        search_config: Optional[SearchConfig] = None,
        answerer: Optional[Answerer] = None,
        realtime_config: Optional[RealtimeSearchConfig] = None,
        realtime_engine: Optional[Any] = None,
        shadow_config: Optional[ShadowSearchConfig] = None,
        shadow_search: Optional[Any] = None,
    ) -> None:
        self.database = database
        self.router = RuleRouter()
        self.searcher = HybridSearcher(database, search_config)
        self.evidence_builder = EvidenceBuilder(search_config)
        self.answerer = answerer
        if realtime_engine is not None:
            self.realtime_engine = realtime_engine
        elif realtime_config and realtime_config.enabled:
            from .realtime import RealtimeSearchEngine

            self.realtime_engine = RealtimeSearchEngine(realtime_config, search_config)
        else:
            self.realtime_engine = None
        if shadow_search is not None:
            self.shadow_search = shadow_search
        elif shadow_config and shadow_config.enabled:
            from .shadow_search import FineWikiShadowSearch

            self.shadow_search = FineWikiShadowSearch(shadow_config)
        else:
            self.shadow_search = None

    def ask_events(
        self,
        query: str,
        *,
        user_timezone: str = "Asia/Shanghai",
        mode: str = "fast",
        history: Optional[List[Dict[str, str]]] = None,
        search_mode: str = "auto",
        source_scope: str = "auto",
        use_finewiki: bool = False,
        cancel_event: Optional[threading.Event] = None,
        conversation_id: Optional[str] = None,
        debug: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        query = " ".join(query.strip().split())
        if not query:
            yield {"type": "error", "message": "query is empty"}
            return
        history = history or []
        retrieval_query = self._contextual_query(query, history)
        # Classify only the current turn. Previous messages may enrich a search
        # query, but must never change “你是谁” into a time query merely because
        # the previous turn asked for the weekday.
        route = self.router.route(query, user_timezone)
        route = self._apply_search_preferences(
            route,
            query,
            search_mode=search_mode,
            source_scope=source_scope,
        )
        if route.queries and retrieval_query != query:
            route.queries = list(
                dict.fromkeys([retrieval_query, *route.queries])
            )[:4]
        if mode == "deep" and route.depth != "direct":
            route.depth = "multi"
        yield {"type": "route", "route": route.to_dict()}
        if debug:
            yield {
                "type": "debug",
                "debug": {
                    "kind": "request_context",
                    "query": query,
                    "retrieval_query": retrieval_query,
                    "history": history,
                    "history_count": len(history),
                    "route": route.to_dict(),
                    "use_finewiki": use_finewiki,
                },
            }

        if cancel_event and cancel_event.is_set():
            yield {"type": "cancelled"}
            return

        if route.intent == "time":
            yield {
                "type": "answer",
                "answer": self._time_answer(route, user_timezone, query),
            }
            yield {"type": "done"}
            return
        evidence: List[Evidence] = []
        needs_retrieval = bool(route.queries and self._has_retrieval_tool(route.tools))
        if needs_retrieval:
            merged: List[SearchResult] = []
            wants_local = "local_search" in route.tools
            wants_web = self._has_web_tool(route.tools)
            uses_local = wants_local or not (wants_web and self.realtime_engine)
            shadow_future = None
            if self.shadow_search and (uses_local or use_finewiki):
                try:
                    shadow_future = self.shadow_search.start(
                        route.queries[0], route.to_dict()
                    )
                except Exception as exc:
                    # Shadow is an observability path.  It must never make the
                    # live retriever fail or alter its response.
                    report = getattr(self.shadow_search, "record_failure", None)
                    if callable(report):
                        try:
                            report(
                                "start",
                                exc,
                                query=route.queries[0],
                                route=route.to_dict(),
                            )
                        except Exception:
                            pass
                    shadow_future = None
            primary_local_results: List[SearchResult] = []
            primary_started = time.perf_counter()
            # If query-time web search is not configured, preserve the existing
            # local-index fallback instead of silently returning no evidence.
            if uses_local:
                primary_local_results = self._multi_query_search(route)
                merged.extend(primary_local_results)
            primary_latency_ms = (time.perf_counter() - primary_started) * 1000.0
            realtime_stats: Dict[str, Any] = {}
            if wants_web and self.realtime_engine:
                for realtime_event in self.realtime_engine.search_events(
                    route.queries[0],
                    route.queries,
                    freshness=route.freshness,
                    depth=route.depth,
                    cancel_event=cancel_event,
                ):
                    kind = str(realtime_event.get("type") or "")
                    if kind == "realtime_result":
                        merged.extend(realtime_event.get("results") or [])
                        realtime_stats = dict(realtime_event.get("stats") or {})
                    else:
                        yield realtime_event
            finewiki_used = False
            if use_finewiki:
                live_results = getattr(self.shadow_search, "live_results", None)
                if shadow_future is not None and callable(live_results):
                    try:
                        finewiki_results, finewiki_stats = live_results(shadow_future)
                        merged.extend(finewiki_results)
                        realtime_stats["finewiki"] = finewiki_stats
                        finewiki_used = bool(finewiki_results)
                    except Exception as exc:
                        yield {
                            "type": "search_warning",
                            "code": "FINEWIKI_UNAVAILABLE",
                            "message": f"FineWiki 暂时不可用，继续使用其他搜索来源：{type(exc).__name__}",
                        }
                else:
                    yield {
                        "type": "search_warning",
                        "code": "FINEWIKI_UNAVAILABLE",
                        "message": "FineWiki 尚未配置，继续使用其他搜索来源",
                    }
            merged = self._merge_results(merged)
            merged = self._filter_results_for_route(route, merged)
            if self.shadow_search:
                try:
                    self.shadow_search.attach(
                        shadow_future,
                        primary_results=primary_local_results,
                        visible_results=merged,
                        primary_latency_ms=primary_latency_ms,
                        query=route.queries[0],
                        route=route.to_dict(),
                        visible_output_changed=finewiki_used,
                    )
                except Exception as exc:
                    report = getattr(self.shadow_search, "record_failure", None)
                    if callable(report):
                        try:
                            report(
                                "attach",
                                exc,
                                query=route.queries[0],
                                route=route.to_dict(),
                            )
                        except Exception:
                            pass
            if cancel_event and cancel_event.is_set():
                yield {"type": "cancelled"}
                return
            public_results = [item.to_dict() for item in merged]
            yield {
                "type": "sources",
                "sources": public_results,
                "count": len(public_results),
                "stats": realtime_stats,
            }
            evidence = self.evidence_builder.build(query, merged)
            yield {
                "type": "evidence",
                "evidence": [item.to_dict() for item in evidence],
            }

            if cancel_event and cancel_event.is_set():
                yield {"type": "cancelled"}
                return

        # For current financial facts, absence of primary evidence is itself
        # the answer. Do not ask RWKV to fill the gap from model memory.
        if route.intent == "finance" and not evidence:
            as_of = datetime.now(timezone.utc).isoformat()
            yield {
                "type": "answer",
                "answer": self._extractive_fallback(
                    query, evidence, as_of, intent=route.intent
                ),
            }
            yield {"type": "done"}
            return

        as_of = datetime.now(timezone.utc).isoformat()
        answer: Dict[str, Any]
        model_meta: Optional[Dict[str, Any]] = None
        if self.answerer:
            try:
                answer_kwargs: Dict[str, Any] = {
                    "as_of": as_of,
                    "timezone": user_timezone,
                    "history": history,
                }
                if getattr(self.answerer, "supports_cancellation", False):
                    answer_kwargs["cancel_event"] = cancel_event
                if getattr(self.answerer, "supports_sessions", False):
                    answer_kwargs["conversation_id"] = conversation_id
                if getattr(self.answerer, "supports_streaming", False):
                    stream_queue: "queue.Queue[Any]" = queue.Queue()

                    def on_delta(value: str) -> None:
                        if value:
                            stream_queue.put(("delta", value))

                    def on_debug(value: Dict[str, Any]) -> None:
                        if debug and value:
                            stream_queue.put(("debug", value))

                    def generation_worker() -> None:
                        try:
                            result = self.answerer.answer(
                                query,
                                route,
                                evidence,
                                on_delta=on_delta,
                                **(
                                    {"on_debug": on_debug}
                                    if debug
                                    and getattr(self.answerer, "supports_debug", False)
                                    else {}
                                ),
                                **answer_kwargs,
                            )
                            stream_queue.put(("result", result))
                        except Exception as worker_error:
                            stream_queue.put(("error", worker_error))

                    worker = threading.Thread(
                        target=generation_worker,
                        daemon=True,
                        name="rwkv-token-stream",
                    )
                    worker.start()
                    generated = None
                    while generated is None:
                        try:
                            stream_kind, stream_value = stream_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if stream_kind == "delta":
                            if not (cancel_event and cancel_event.is_set()):
                                yield {"type": "answer_delta", "delta": stream_value}
                        elif stream_kind == "debug":
                            if not (cancel_event and cancel_event.is_set()):
                                yield {"type": "debug", "debug": stream_value}
                        elif stream_kind == "error":
                            raise stream_value
                        elif stream_kind == "result":
                            generated = stream_value
                else:
                    generated = self.answerer.answer(
                        query, route, evidence, **answer_kwargs
                    )
                if cancel_event and cancel_event.is_set():
                    yield {"type": "cancelled"}
                    return
                if getattr(generated, "answer", None):
                    answer = generated.answer
                    model_meta = {
                        "used": True,
                        "latency_ms": generated.latency_ms,
                        "new_tokens": generated.new_tokens,
                        "repaired": generated.repaired,
                    }
                else:
                    model_error = getattr(generated, "error", None) or "model output rejected"
                    raw = str(getattr(generated, "raw", "") or "")
                    if not evidence and route.freshness == "stable" and raw:
                        answer = self._unstructured_model_answer(raw, as_of, query)
                        model_meta = {
                            "used": True,
                            "structured": False,
                            "latency_ms": getattr(generated, "latency_ms", 0.0),
                            "new_tokens": getattr(generated, "new_tokens", 0),
                            "error": model_error,
                        }
                    elif not evidence and route.intent == "chat":
                        answer = self._chat_fallback(query, as_of)
                        model_meta = {"used": False, "error": model_error}
                    else:
                        answer = self._extractive_fallback(
                            query,
                            evidence,
                            as_of,
                            model_error=model_error,
                            intent=route.intent,
                        )
                        model_meta = {
                            "used": False,
                            "latency_ms": getattr(generated, "latency_ms", 0.0),
                            "new_tokens": getattr(generated, "new_tokens", 0),
                            "repaired": getattr(generated, "repaired", False),
                            "error": model_error,
                        }
            except Exception as exc:
                model_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                answer = (
                    self._chat_fallback(query, as_of)
                    if not evidence and route.intent == "chat"
                    else self._extractive_fallback(
                        query,
                        evidence,
                        as_of,
                        model_error=model_error,
                        intent=route.intent,
                    )
                )
                model_meta = {"used": False, "error": model_error}
        else:
            answer = self._extractive_fallback(
                query, evidence, as_of, intent=route.intent
            )
        event: Dict[str, Any] = {"type": "answer", "answer": answer}
        if model_meta:
            event["model"] = model_meta
        yield event
        yield {"type": "done"}

    @staticmethod
    def _apply_search_preferences(
        route: RouteDecision,
        query: str,
        *,
        search_mode: str,
        source_scope: str,
    ) -> RouteDecision:
        """Apply explicit UI controls after safety/time routing has completed."""
        if search_mode == "never" and route.intent != "time":
            return RouteDecision(
                intent="chat",
                tools=[],
                freshness="stable",
                depth="direct",
                needs_clarification=False,
                queries=[],
                missing_context=[],
                reason="user disabled retrieval for this request",
            )
        if search_mode == "never":
            route.tools = [tool for tool in route.tools if tool == "clock"]
            route.queries = []
            route.reason += "; user disabled retrieval, so current evidence may be unavailable"
        elif search_mode == "always" and route.intent == "chat":
            # The single UI search switch is an explicit user instruction.
            # It is one of only two retrieval triggers; the other is Router
            # recognition from the user's own words.
            route = RouteDecision(
                intent="search",
                tools=["local_search", "web_search"],
                freshness="stable",
                depth="single",
                needs_clarification=False,
                queries=RuleRouter._queries(query),
                missing_context=[],
                reason="user enabled the search switch for this request",
            )

        if source_scope == "local":
            searchable = any(tool.endswith("search") for tool in route.tools)
            route.tools = [
                tool for tool in route.tools if tool == "clock" or tool == "local_search"
            ]
            if searchable and "local_search" not in route.tools:
                route.tools.append("local_search")
            route.reason += "; source scope is local"
        elif source_scope == "web":
            route.tools = [tool for tool in route.tools if tool != "local_search"]
            route.reason += "; source scope is web"
        return route

    @staticmethod
    def _unstructured_model_answer(
        raw: str, as_of: str, query: str = ""
    ) -> Dict[str, Any]:
        # Small/base RWKV checkpoints may answer conversational prompts well
        # while failing the strict JSON envelope. Accept only the first pass;
        # repair instructions and any citations remain excluded.
        text = raw.split("<REPAIR>", 1)[0].strip()
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^(?:Assistant|助手)\s*:\s*", "", text, flags=re.I)
        if text.startswith("{"):
            match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)', text)
            if match:
                try:
                    text = json.loads('"' + match.group(1) + '"')
                except json.JSONDecodeError:
                    pass
        text = re.sub(
            r"<(?:think|analysis)>[\s\S]*?</(?:think|analysis)>\s*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"^\s*<(?:think|analysis)>[\s\S]*$", "", text, flags=re.I)
        text = re.split(
            r"\n\s*(?:User|Assistant|用户|助手)\s*:", text, maxsplit=1, flags=re.I
        )[0]
        text = text.strip()[:6000]
        # Final display boundary: a backend may have produced an unusable role
        # marker or continuation after an unfinished phrase.  Do not expose a
        # visibly broken long sentence in the chat UI.
        if len(text) >= 120 and not re.search(r"[。！？.!?}\])]$", text):
            boundaries = list(re.finditer(r"[。！？.!?}\])]", text))
            if boundaries:
                boundary = boundaries[-1].end()
                if boundary >= max(40, len(text) // 2):
                    text = text[:boundary].rstrip()
        if re.search(r"(你是谁|介绍你自己|你是什么|who are you)", query, re.I) and re.search(
            r"OpenAI|ChatGPT", text, re.I
        ):
            text = "我是 RWKV Search，由本地 RWKV-7 模型驱动，可以结合检索证据回答问题。"
        abuse = re.search(
            r"(我是你(?:爹|爸|爷)|你妈(?:死|逼)|傻[逼比]|蠢货|滚蛋|操你|fuck\s+you)",
            query,
            re.I,
        )
        if abuse and re.sub(r"\s+", "", text) == re.sub(r"\s+", "", query):
            text = "我不会和你对骂。你可以直接说需要我解决什么问题。"
        if not text:
            text = "模型没有生成可显示的内容。"
        return {
            "answer": text,
            "citations": [],
            "data_time": as_of,
            "insufficient_evidence": False,
            "needs_clarification": False,
        }

    @staticmethod
    def _contextual_query(query: str, history: List[Dict[str, str]]) -> str:
        """Resolve short follow-ups for retrieval without treating chat as evidence."""
        if not history:
            return query
        follow_up = re.search(
            r"(它|他|她|这个|那个|这点|上述|前面提到|继续说|再说说|其优点|其缺点)",
            query,
        ) or re.search(
            r"\b(it|this|that|they|them|those|continue)\b", query, re.I
        )
        if not follow_up:
            return query
        previous_user = next(
            (
                str(item.get("content", "")).strip()
                for item in reversed(history)
                if item.get("role") == "user" and item.get("content")
            ),
            "",
        )
        return f"{previous_user}；追问：{query}" if previous_user else query

    def search(self, query: str, *, freshness: str = "stable", limit: int = 10) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.searcher.search(query, freshness=freshness, limit=limit)]

    def _multi_query_search(self, route: RouteDecision) -> List[SearchResult]:
        merged: Dict[int, SearchResult] = {}
        max_queries = 4 if route.depth == "multi" else 2
        for query_index, query in enumerate(route.queries[:max_queries]):
            for rank, result in enumerate(
                self.searcher.search(query, freshness=route.freshness, limit=20), start=1
            ):
                bonus = 1.0 / (60.0 + rank) + 0.002 / (query_index + 1)
                existing = merged.get(result.document_id)
                if existing:
                    existing.score += bonus
                    existing.score_components["multi_query"] = existing.score_components.get("multi_query", 0.0) + bonus
                else:
                    result.score += bonus
                    result.score_components["multi_query"] = bonus
                    merged[result.document_id] = result
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return self.searcher._diversify(ordered, self.searcher.config.result_limit)

    def _merge_results(self, results: List[SearchResult]) -> List[SearchResult]:
        by_url: Dict[str, SearchResult] = {}
        for item in results:
            existing = by_url.get(item.url)
            if existing:
                existing.score = max(existing.score, item.score) + 0.01
                existing.score_components["multi_source"] = 0.01
                if len(item.content) > len(existing.content):
                    item.score = existing.score
                    item.score_components.update(existing.score_components)
                    by_url[item.url] = item
            else:
                by_url[item.url] = item
        ordered = sorted(by_url.values(), key=lambda item: item.score, reverse=True)
        return self.searcher._diversify(ordered, self.searcher.config.result_limit)

    @staticmethod
    def _filter_results_for_route(
        route: RouteDecision, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Keep generic portal pages out of high-risk finance evidence."""
        if route.intent != "finance":
            return results
        primary_hosts = (
            "sec.gov",
            "cninfo.com.cn",
            "hkexnews.hk",
            "sse.com.cn",
            "szse.cn",
            "bse.cn",
            "hkex.com.hk",
            "csrc.gov.cn",
            "sfc.hk",
            "finra.org",
        )
        finance_title = re.compile(
            r"(股票|股市|A股|港股|美股|基金|ETF|行情|指数|上市公司|公告|财报|业绩|"
            r"stock|equity|market|NASDAQ|NYSE|filing|earnings)",
            re.I,
        )
        filtered: List[SearchResult] = []
        for item in results:
            host = (urlsplit(item.url).hostname or "").casefold()
            primary_domain = any(
                host == value or host.endswith("." + value)
                for value in primary_hosts
            )
            primary = (
                item.source_type in {"regulator", "company_filing"}
                or primary_domain
            ) and bool(finance_title.search(f"{item.title} {item.content[:3000]}"))
            verified_news = item.source_type == "news" and bool(
                finance_title.search(item.title)
            )
            if primary or verified_news:
                filtered.append(item)
        return filtered

    @staticmethod
    def _has_web_tool(tools: List[str]) -> bool:
        return bool(
            set(tools)
            & {
                "web_search",
                "news_search",
                "social_search",
                "financial_news",
                "company_filings",
                "market_data",
            }
        )

    @classmethod
    def _has_retrieval_tool(cls, tools: List[str]) -> bool:
        return "local_search" in tools or cls._has_web_tool(tools)

    def close(self) -> None:
        shadow_close = getattr(self.shadow_search, "close", None)
        if callable(shadow_close):
            shadow_close()
        close = getattr(self.realtime_engine, "close", None)
        if callable(close):
            close()

    def shadow_status(self) -> Dict[str, Any]:
        status = getattr(self.shadow_search, "status", None)
        if callable(status):
            return dict(status())
        return {
            "enabled": False,
            "ready": False,
            "mode": "shadow_only",
            "visible_output_changed": False,
            "error": None,
        }

    @staticmethod
    def _time_answer(
        route: RouteDecision, user_timezone: str, query: str = ""
    ) -> Dict[str, Any]:
        if route.needs_clarification:
            return {
                "answer": "请提供你的时区或所在城市。",
                "citations": [],
                "data_time": datetime.now(timezone.utc).isoformat(),
                "insufficient_evidence": True,
                "needs_clarification": True,
            }
        try:
            zone = ZoneInfo(user_timezone)
        except ZoneInfoNotFoundError:
            return {
                "answer": f"无法识别时区 {user_timezone}，请提供 IANA 时区名称。",
                "citations": [],
                "data_time": datetime.now(timezone.utc).isoformat(),
                "insufficient_evidence": True,
                "needs_clarification": True,
            }
        current = datetime.now(zone)
        target = current
        relative = "现在"
        if "明天" in query:
            target = current + timedelta(days=1)
            relative = "明天"
        elif "昨天" in query:
            target = current - timedelta(days=1)
            relative = "昨天"
        weekday = "一二三四五六日"[target.weekday()]
        return {
            "answer": f"{relative}是 {target:%Y-%m-%d %H:%M:%S}，星期{weekday}（{user_timezone}）。",
            "citations": [],
            "data_time": current.isoformat(),
            "insufficient_evidence": False,
            "needs_clarification": False,
        }

    @staticmethod
    def _extractive_fallback(
        query: str,
        evidence: List[Evidence],
        as_of: str,
        model_error: Optional[str] = None,
        intent: str = "",
    ) -> Dict[str, Any]:
        if intent == "finance":
            citations = [item.id for item in evidence[:3]]
            if evidence:
                text = (
                    "已找到相关官方公告或监管来源，但本次没有生成可靠的带引用摘要。"
                    "为避免误导，不直接拼接网页原文，请查看下方来源。"
                )
            else:
                text = "暂未抓取到足以支持回答的官方市场公告或监管信息。"
            return {
                "answer": text,
                "citations": citations,
                "data_time": as_of,
                "insufficient_evidence": True,
                "needs_clarification": False,
            }
        if not evidence:
            return {
                "answer": "本地索引中没有找到足够证据。请添加相关站点种子或扩大抓取范围。",
                "citations": [],
                "data_time": as_of,
                "insufficient_evidence": True,
                "needs_clarification": False,
            }
        query_tokens = set(search_tokens(query))
        entities = {
            token
            for token in query_tokens
            if len(token) >= 2 and token.isascii() and token.isalnum()
        }
        definition_query = bool(
            re.search(r"(什么是|是什么|指什么|定义|what is\b)", query, re.I)
        )
        candidates: List[tuple[int, float, str, str]] = []
        for evidence_rank, item in enumerate(evidence[:5]):
            sentences = re.split(r"(?<=[。！？!?])|(?<=\.)\s+|[\r\n]+", item.text)
            for sentence_index, sentence in enumerate(sentences[:80]):
                clean = " ".join(sentence.split()).strip(" -•\t")
                if len(clean) < 12 or len(clean) > 360:
                    continue
                if clean.endswith(("？", "?")):
                    continue
                tokens = set(search_tokens(clean))
                if entities and not (entities & tokens):
                    continue
                overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
                score = 2.0 * overlap + 0.18 * item.authority
                score += 0.3 / (1 + evidence_rank) + 0.08 / (1 + sentence_index)
                direct_definition = bool(definition_query and re.search(
                    r"(?:Python|RWKV|[A-Za-z0-9_.+#-]{2,}|[\u4e00-\u9fff]{2,})"
                    r".{0,20}(?:是一种|是一个|是由|指的是|属于).{0,30}"
                    r"(?:语言|模型|框架|工具|系统|协议|方法|技术|平台|软件|项目|"
                    r"组织|公司|人物|概念|架构|程序|库)",
                    clean,
                    re.I,
                ))
                if direct_definition:
                    score += 2.5
                candidates.append((int(direct_definition), score, clean, item.id))
        candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
        chosen: List[tuple[str, str]] = []
        source_counts: Dict[str, int] = {}
        normalized = set()
        for _, _, sentence, source_id in candidates:
            key = re.sub(r"\W+", "", sentence.casefold())[:120]
            if key in normalized:
                continue
            if source_counts.get(source_id, 0) >= 2:
                continue
            normalized.add(key)
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            chosen.append((sentence, source_id))
            if len(chosen) >= 3:
                break
        if not chosen:
            chosen = [(item.text[:240].strip(), item.id) for item in evidence[:2]]
        lines = ["根据当前可核查证据："]
        citations = []
        for sentence, source_id in chosen:
            sentence = sentence.rstrip()
            if sentence and sentence[-1] not in "。！？.!?":
                sentence += "。"
            lines.append(f"- {sentence} [{source_id}]")
            citations.append(source_id)
        return {
            "answer": "\n".join(lines),
            "citations": citations,
            "data_time": as_of,
            "insufficient_evidence": False,
            "needs_clarification": False,
        }

    @staticmethod
    def _chat_fallback(query: str, as_of: str) -> Dict[str, Any]:
        if re.search(r"^(你好|您好|hello|hi)[！!。,.，\s]*$", query, re.I):
            text = "你好！有什么我可以帮助你的吗？"
        elif re.search(r"(你是谁|介绍你自己|who are you)", query, re.I):
            text = "我是 RWKV Search，由本地 RWKV 模型驱动。"
        else:
            text = "我在。请直接告诉我你希望我帮你解决什么问题。"
        return {
            "answer": text,
            "citations": [],
            "data_time": as_of,
            "insufficient_evidence": False,
            "needs_clarification": False,
        }
