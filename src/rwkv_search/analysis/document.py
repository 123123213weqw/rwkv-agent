from __future__ import annotations

import time
from typing import Iterable, Optional

from .core import AnalyzerCore
from .models import DocumentAnalysis


class DocumentAnalyzer:
    """Analyze index fields while retaining frequency and token order."""

    def __init__(self, core: Optional[AnalyzerCore] = None) -> None:
        self.core = core or AnalyzerCore()

    def analyze(
        self,
        *,
        title: str,
        body: str,
        headings: Iterable[str] = (),
        url: str = "",
    ) -> DocumentAnalysis:
        started = time.perf_counter()
        title_analysis = self.core.analyze(title, keep_duplicates=True, include_bigrams=True)
        body_analysis = self.core.analyze(body, keep_duplicates=True, include_bigrams=True)
        heading_analyses = tuple(
            self.core.analyze(item, keep_duplicates=True, include_bigrams=True)
            for item in headings
            if item.strip()
        )
        url_analysis = self.core.analyze(url, keep_duplicates=True, include_bigrams=False) if url else None
        return DocumentAnalysis(
            title=title_analysis,
            body=body_analysis,
            headings=heading_analyses,
            url=url_analysis,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
