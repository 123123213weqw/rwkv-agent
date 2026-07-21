from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RouteDecision:
    intent: str
    tools: List[str]
    freshness: str
    depth: str
    needs_clarification: bool
    queries: List[str]
    missing_context: List[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleRouter:
    TIME = re.compile(r"(当前时间|现在时间|几点|星期几|周几|礼拜几|几号|什么日期|time\b|date\b|weekday)", re.I)
    LOCAL_SOURCE = re.compile(
        r"(本地资料|本地知识库|我们的知识库|内部资料|内部文档|会议纪要|项目文档|文件里|SLA)",
        re.I,
    )
    REALTIME = re.compile(
        r"(今天|今日|现在|当前|实时|刚刚|此刻|截至目前|today\b|now\b|realtime|current)",
        re.I,
    )
    LATEST = re.compile(
        r"(最新|最近|近期|目前|进展|新版本|更新了什么|release|latest|recent)",
        re.I,
    )
    SOURCE_REQUEST = re.compile(
        r"(给出来源|提供来源|引用来源|附上来源|官方来源|链接|出处|官方资料|官网资料|原始论文|"
        r"公告|报道|新闻|网友怎么|网友如何|社区怎么|论坛怎么|reddit|知乎|"
        r"sources?\b|official source|look up)",
        re.I,
    )
    RECOMMENDATION = re.compile(
        r"(买什么|推荐(?:一下|一些|几个)?|哪个好|哪一个好|怎么选|如何选择|值不值得|值得买吗|"
        r"should i buy|recommend|which one)",
        re.I,
    )
    RESEARCH = re.compile(
        r"(调研|研究报告|深度研究|深入了解|综合比较|方案选型|发展怎么样|deep research)",
        re.I,
    )
    EXPLICIT_SEARCH = re.compile(
        r"(搜(?:索)?(?:一)?下|搜(?:索)?|查(?:询)?(?:一)?下|查查|查询|帮我查找|查找(?:资料|信息|网页|来源)|"
        r"帮我找(?:资料|信息|网页|来源|新闻)|找一下(?:资料|信息|网页|来源|新闻)|检索|联网|网上查|找资料|找来源|"
        r"给出来源|提供来源|引用来源|官网资料|原始论文|search\b|look up|find sources?)",
        re.I,
    )
    NO_SEARCH_TASK = re.compile(
        r"(翻译|改写|润色|根据以下|总结这|写代码|写一个?函数|写脚本|"
        r"(?:写|实现|编写).{0,20}(?:代码|函数|脚本|程序)|SQL|正则|算法|"
        r"计算|解方程|证明|讲个笑话|写故事|写文案|写邮件|新闻稿|扮演|模拟|"
        r"translate\b|rewrite\b|calculate\b|write code)",
        re.I,
    )
    PERSONAL_CHAT = re.compile(
        r"^(?:我|我们|咱们).*(?:心情|感觉|觉得|想|希望|喜欢|讨厌|累|困|难过|开心|焦虑)",
        re.I,
    )
    CHAT_TASK = re.compile(
        r"(你好|您好|谢谢|再见|你是谁|介绍你自己|聊聊|讲个|笑话|故事|写一|写个|写段|"
        r"写份|改写|润色|翻译|总结这|根据以下|代码|函数|脚本|SQL|正则|算法|计算|解方程|"
        r"证明|创作|文案|邮件|计划|头脑风暴|扮演|模拟|hello\b|thanks?\b|translate\b|"
        r"rewrite\b|write\b|code\b|calculate\b)",
        re.I,
    )
    KNOWLEDGE = re.compile(
        r"(是什么|什么是|是做什么的|指什么|是谁|在哪里|哪一年|何时|什么时候|多少|有哪些|"
        r"什么区别|区别是什么|原理|定义|历史|起源|介绍一下|解释一下|为什么会|如何工作|"
        r"what is\b|who is\b|where is\b|when did\b|how does\b)",
        re.I,
    )
    DEFINITION = re.compile(r"(什么是|是什么(?:东西)?|指什么|定义)", re.I)

    def route(self, query: str, timezone: Optional[str] = "Asia/Shanghai") -> RouteDecision:
        clean = " ".join(query.strip().split())
        clean = self._normalize_initialisms(clean)
        pure_clock = re.fullmatch(r"[？?。.!！\s]*(现在|当前|此刻)[？?。.!！\s]*", clean, re.I)
        if self.TIME.search(clean) or pure_clock:
            missing = [] if timezone and timezone != "unknown" else ["timezone"]
            return RouteDecision(
                intent="time",
                tools=["clock"],
                freshness="realtime",
                depth="direct",
                needs_clarification=bool(missing),
                queries=[],
                missing_context=missing,
                reason="time/date must use a clock and explicit timezone",
            )

        # Explicit transformations and personal conversation remain model-only
        # even when their text happens to contain words such as “今天” or “最新”.
        if self.NO_SEARCH_TASK.search(clean) or self.PERSONAL_CHAT.search(clean):
            return RouteDecision(
                intent="chat",
                tools=[],
                freshness="stable",
                depth="direct",
                needs_clarification=False,
                queries=[],
                missing_context=[],
                reason="transformation or personal conversation does not require retrieval",
            )

        if self.LOCAL_SOURCE.search(clean):
            return RouteDecision(
                intent="search",
                tools=["local_search"],
                freshness="stable",
                depth="single",
                needs_clarification=False,
                queries=[clean],
                missing_context=[],
                reason="query explicitly refers to local/internal material",
            )

        explicit = bool(self.EXPLICIT_SEARCH.search(clean))
        realtime = bool(self.REALTIME.search(clean))
        latest = bool(self.LATEST.search(clean))
        source_request = bool(self.SOURCE_REQUEST.search(clean))
        recommendation = bool(self.RECOMMENDATION.search(clean))
        research = bool(self.RESEARCH.search(clean))
        citation_request = bool(
            re.search(r"(来源|引用|链接|出处|官网资料|官方资料|原始论文)", clean, re.I)
        )
        if (
            self.DEFINITION.search(clean)
            and not explicit
            and not realtime
            and not latest
            and not citation_request
        ):
            return RouteDecision(
                intent="chat",
                tools=[],
                freshness="stable",
                depth="direct",
                needs_clarification=False,
                queries=[],
                missing_context=[],
                reason="stable definition uses normal model chat unless retrieval is requested",
            )
        if explicit or realtime or latest or source_request or recommendation or research:
            freshness = "realtime" if realtime else "latest" if latest or recommendation else "stable"
            return RouteDecision(
                intent="search",
                tools=["local_search", "web_search"],
                freshness=freshness,
                depth="multi" if research else "single",
                needs_clarification=False,
                queries=self._queries(clean),
                missing_context=[],
                reason="query requests retrieval, fresh information, external evidence, or a current recommendation",
            )

        if self.CHAT_TASK.search(clean):
            return RouteDecision(
                intent="chat",
                tools=[],
                freshness="stable",
                depth="direct",
                needs_clarification=False,
                queries=[],
                missing_context=[],
                reason="ordinary conversation or generation task uses the model directly",
            )

        if self.KNOWLEDGE.search(clean):
            return RouteDecision(
                intent="chat",
                tools=[],
                freshness="stable",
                depth="direct",
                needs_clarification=False,
                queries=[],
                missing_context=[],
                reason="stable knowledge uses normal model chat unless the user requests search",
            )

        return RouteDecision(
            intent="chat",
            tools=[],
            freshness="stable",
            depth="direct",
            needs_clarification=False,
            queries=[],
            missing_context=[],
            reason="default to ordinary model conversation when search need is not explicit",
        )

    @staticmethod
    def _queries(query: str, **_: Any) -> List[str]:
        """Return one domain-neutral first-round query.

        Additional rewrites belong to the later evidence-feedback loop, not to
        the Router. ``**_`` keeps older callers source-compatible while the API
        migrates away from ``latest``/``official`` planning flags.
        """
        core = query.strip()
        core = re.sub(
            r"^(?:请|麻烦)?\s*(?:你)?\s*(?:帮我)?\s*"
            r"(?:联网搜索|帮我查找|查找|搜(?:索)?(?:一)?下|搜(?:索)?|"
            r"查(?:询)?(?:一)?下|查(?:询)?|检索)\s*",
            "",
            core,
            flags=re.I,
        )
        core = re.sub(
            r"[,，;；\s]*(?:请)?\s*(?:给出|提供|附上|标注)\s*(?:可靠|权威|相关)?\s*"
            r"(?:的)?\s*(?:来源|引用|链接|出处).*$",
            "",
            core,
            flags=re.I,
        )
        core = core.strip(" ？?。.!！,，;；") or query.strip(" ？?")
        # Search engines need the entity/topic, not the conversational wrapper
        # (“什么是 …” / “… 是什么”). Keeping the wrapper overweights generic
        # Chinese dictionary pages and can bury the intended named entity.
        core = RuleRouter._knowledge_subject(core)
        return [core]

    @staticmethod
    def _normalize_initialisms(query: str) -> str:
        """Collapse user-spelled initialisms such as ``r w k v`` to RWKV."""
        return re.sub(
            r"(?<![A-Za-z])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])",
            lambda match: re.sub(r"\s+", "", match.group(0)).upper(),
            query,
        )

    @staticmethod
    def _knowledge_subject(query: str) -> str:
        value = query.strip(" ？?。.!！")
        prefix = re.fullmatch(r"什么是\s*(.+)", value, flags=re.I)
        if prefix:
            return prefix.group(1).strip()
        suffix = re.fullmatch(
            r"(.+?)\s*(?:是什么(?:东西)?|是做什么的|指什么)",
            value,
            flags=re.I,
        )
        return suffix.group(1).strip() if suffix else value
