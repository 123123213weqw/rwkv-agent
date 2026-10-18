"""Generic page-shape policy for live Web evidence and feedback queries.

The policy deliberately classifies document shapes rather than subject areas.
It therefore applies equally to software, finance, education, policy, and
general factual questions without maintaining topic or preferred-domain lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit


_LEXICAL_INTENT = re.compile(
    r"(?:是什么意思|什么含义|词义|释义|怎么读|如何翻译|翻译成|词典|字典|音标|读音|"
    r"\bdefinition\b|\bmeaning\s+of\b|\bwhat\s+does\b.{0,80}\bmean\b|"
    r"\btranslate\b|\btranslation\b|\bpronunciation\b|\bdictionary\b)",
    re.I,
)
_DICTIONARY = re.compile(
    r"(?:字典|词典|汉典|释义|是什么意思|拼音|部首|笔顺|音标|读音|例句|"
    r"英文名|英文名字|名字含义|名字寓意|"
    r"\bdictionary\b|\bdefinition\b|\bpronunciation\b|\bthesaurus\b|"
    r"(?:^|[./_-])(?:[a-z0-9-]*dictionary|dict|zidian)(?:[./_?=-]|$)|"
    r"(?:iciba|eudic|merriam-webster|collinsdictionary|dictionary\.cambridge))",
    re.I,
)
_TRANSLATION = re.compile(
    r"(?:在线翻译|英语翻译|中文翻译|英汉|汉英|翻译_用法|"
    r"\btranslate\b|\btranslation\b|\btranslator\b)",
    re.I,
)
_SEARCH_PAGE = re.compile(
    r"(?:^|\.)(?:bing|google|baidu|sogou|so|duckduckgo)\.[a-z.]+$|"
    r"/(?:search|s|web)(?:/|\?|$)",
    re.I,
)
_LOGIN = re.compile(
    r"(?:^|[/_.-])(?:login|signin|sign-in|captcha|verify)(?:[/_.?&=-]|$)|"
    r"登录|验证码|安全验证|verify you are human|sign in to continue",
    re.I,
)
_ERROR = re.compile(
    r"(?:^|[/_.-])(?:404|403|500|error)(?:[/_.?&=-]|$)|"
    r"page not found|access denied|forbidden|页面不存在|访问被拒绝",
    re.I,
)
_GENERIC_TITLE = re.compile(
    r"^(?:home|homepage|welcome|official site|index|首页|主页|官网|网站首页)$",
    re.I,
)
_NAVIGATION_TEXT = re.compile(
    r"(?:首页\s+新闻\s+产品|登录\s+注册|关于我们\s+联系我们|"
    r"home\s+about\s+contact|skip to (?:content|navigation))",
    re.I,
)


@dataclass(frozen=True)
class PageQualityDecision:
    page_type: str
    evidence_allowed: bool
    pivot_allowed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lexical_information_requested(question: str) -> bool:
    """Return whether dictionaries/translations directly answer the request."""

    return bool(_LEXICAL_INTENT.search(" ".join(str(question or "").split())))


def classify_page_quality(
    question: str,
    item: Mapping[str, Any],
) -> PageQualityDecision:
    """Classify a result using only generic URL/title/content page features."""

    title = " ".join(str(item.get("title") or "").split()).strip()
    content = " ".join(
        str(item.get("content") or item.get("snippet") or "").split()
    ).strip()
    uri = str(item.get("uri") or item.get("url") or "").strip()
    parsed = urlsplit(uri)
    host = (parsed.hostname or "").casefold()
    path = unquote(parsed.path or "/").casefold()
    shape = f"{host} {path} {title}".strip()

    if not title and not content:
        return PageQualityDecision("empty", False, False, ("empty_page",))
    if _ERROR.search(f"{shape} {content[:240]}"):
        return PageQualityDecision("error", False, False, ("error_page",))
    if _LOGIN.search(f"{shape} {content[:240]}"):
        return PageQualityDecision(
            "login_or_captcha", False, False, ("login_or_captcha",)
        )
    if _SEARCH_PAGE.search(f"{host}{path}"):
        return PageQualityDecision(
            "search_homepage", False, False, ("search_homepage",)
        )
    if _DICTIONARY.search(f"{shape} {content[:400]}"):
        allowed = lexical_information_requested(question)
        return PageQualityDecision(
            "dictionary",
            allowed,
            False,
            (() if allowed else ("dictionary_not_requested",)),
        )
    if _TRANSLATION.search(f"{shape} {content[:400]}"):
        allowed = lexical_information_requested(question)
        return PageQualityDecision(
            "translation",
            allowed,
            False,
            (() if allowed else ("translation_not_requested",)),
        )

    root_like = path in {"", "/", "/index.html", "/index.htm"}
    generic_title = bool(_GENERIC_TITLE.fullmatch(title.strip()))
    navigation_boilerplate = bool(_NAVIGATION_TEXT.search(content[:500]))
    if (root_like and generic_title and len(content) < 700) or (
        navigation_boilerplate and len(content) < 500
    ):
        return PageQualityDecision(
            "navigation", False, False, ("navigation_only",)
        )

    return PageQualityDecision("content", True, bool(title), ())
