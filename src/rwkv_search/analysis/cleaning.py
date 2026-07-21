from __future__ import annotations

import re
from dataclasses import dataclass


_INITIALISM_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])")
_OPERATION_PREFIX_RE = re.compile(
    r"^(?:请|麻烦)?\s*(?:你)?\s*(?:"
    r"帮我\s*(?:搜索|搜|查询|查|查找|检索)|"
    r"联网搜索|帮我查找|查找(?:资料|信息|网页|来源)|"
    r"搜索一下|搜一下|查询一下|查一下|检索一下|"
    r"搜索(?=\s)|查询(?=\s)|检索(?=\s)"
    r")\s*",
    re.I,
)
_SOURCE_SUFFIX_RE = re.compile(
    r"[,，;；\s]*(?:请)?\s*(?:给出|提供|附上|标注)\s*(?:可靠|权威|相关)?\s*"
    r"(?:的)?\s*(?:来源|引用|链接|出处)[。.!！\s]*$",
    re.I,
)
_DEFINITION_PREFIX_RE = re.compile(r"^什么是\s*(.+)$", re.I)
_DEFINITION_SUFFIX_RE = re.compile(
    r"^(.+?)\s*(?:"
    r"是什么(?:东西|国家|组织|机构|公司|项目|产品|技术|语言|疾病|人物)?|"
    r"是做什么的|指什么"
    r")[？?。.!！\s]*$",
    re.I,
)
_OVERVIEW_SUFFIX_RE = re.compile(
    r"^(.{2,}?)\s*(?:的)?(?:基本情况|基本信息|简介|概况|介绍(?:一下)?)[？?。.!！\s]*$",
    re.I,
)


@dataclass(frozen=True)
class CleanedQuery:
    original: str
    text: str
    changed: bool
    operations: tuple[str, ...]


def clean_query_surface(value: str) -> CleanedQuery:
    """Remove retrieval-irrelevant conversational syntax without an LLM."""

    original = value
    current = value.strip()
    operations: list[str] = []

    collapsed = _INITIALISM_RE.sub(
        lambda match: re.sub(r"\s+", "", match.group(0)).upper(),
        current,
    )
    if collapsed != current:
        operations.append("collapse_initialism")
        current = collapsed

    stripped = _OPERATION_PREFIX_RE.sub("", current).strip()
    if stripped and stripped != current:
        operations.append("strip_operation_prefix")
        current = stripped

    stripped = _SOURCE_SUFFIX_RE.sub("", current).strip()
    if stripped and stripped != current:
        operations.append("strip_source_suffix")
        current = stripped

    candidate = current.strip(" ？?。.!！")
    prefix = _DEFINITION_PREFIX_RE.fullmatch(candidate)
    suffix = _DEFINITION_SUFFIX_RE.fullmatch(candidate)
    subject = prefix.group(1).strip() if prefix else suffix.group(1).strip() if suffix else ""
    if subject:
        operations.append("extract_definition_subject")
        current = subject

    candidate = current.strip(" ？?。.!！")
    overview = _OVERVIEW_SUFFIX_RE.fullmatch(candidate)
    if overview and overview.group(1).strip():
        operations.append("strip_overview_shell")
        current = overview.group(1).strip()

    current = current.strip() or original.strip()
    return CleanedQuery(
        original=original,
        text=current,
        changed=current != original,
        operations=tuple(operations),
    )
