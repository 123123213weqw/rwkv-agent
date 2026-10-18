"""Citation token parsing shared by the Agent and benchmark metrics."""

from __future__ import annotations

import re
from typing import Iterable


_CITATION_ID = r"[A-Za-z][A-Za-z0-9_-]*"
_CITATION_GROUP = re.compile(
    rf"\[\s*({_CITATION_ID}(?:\s*[,;]\s*{_CITATION_ID})*)\s*\]"
)
_CITATION_ITEM = re.compile(_CITATION_ID)


def extract_citation_ids(
    value: str,
    *,
    prefixes: Iterable[str] | None = None,
) -> list[str]:
    """Return citation IDs from single or comma/semicolon grouped brackets.

    ``[W1]``, ``[W1, W2]`` and ``[W1;W2]`` are accepted.  Arbitrary Markdown
    link labels and prose inside brackets are intentionally ignored.
    """

    allowed_prefixes = (
        {str(prefix).casefold() for prefix in prefixes}
        if prefixes is not None
        else None
    )
    output: list[str] = []
    for match in _CITATION_GROUP.finditer(str(value or "")):
        for citation in _CITATION_ITEM.findall(match.group(1)):
            if allowed_prefixes is not None:
                prefix = citation[0].casefold()
                if prefix not in allowed_prefixes:
                    continue
            output.append(citation)
    return output


def strip_citations(value: str) -> str:
    """Remove supported citation groups before answer-text comparison."""

    return _CITATION_GROUP.sub("", str(value or ""))


def normalize_citation_groups(
    value: str,
    *,
    allowed_ids: Iterable[str] | None = None,
) -> str:
    """Render grouped citations as independent tokens and drop unknown IDs.

    This only rewrites syntactically recognized citation groups.  It cannot add
    evidence or convert arbitrary bracketed prose into a citation.
    """

    allowed = (
        {str(item).upper() for item in allowed_ids}
        if allowed_ids is not None
        else None
    )

    def replace(match: re.Match[str]) -> str:
        identifiers = [item.upper() for item in _CITATION_ITEM.findall(match.group(1))]
        if allowed is not None:
            identifiers = [item for item in identifiers if item in allowed]
        return "".join(f"[{item}]" for item in dict.fromkeys(identifiers))

    return _CITATION_GROUP.sub(replace, str(value or ""))
