from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .text import canonicalize_url, search_tokens


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "who",
    "with",
}


class PairScorer(Protocol):
    """Minimal interface shared by passage, evidence, and tool selection."""

    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class TransformersPairScorer:
    """Lazy local Cross-Encoder implementation of :class:`PairScorer`."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = True,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = str(model_name)
        self.device_name = str(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.fp16 = bool(fp16)
        self.local_files_only = bool(local_files_only)
        self._lock = threading.Lock()
        self._torch: Any = None
        self._device: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "semantic reranking requires torch and transformers"
            ) from exc
        device_name = self.device_name
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        ).to(device)
        if self.fp16 and device.type == "cuda":
            model = model.half()
        model.eval()
        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        with self._lock:
            self._load()
            pairs = [(str(query), str(document)) for document in documents]
            values: list[float] = []
            with self._torch.inference_mode():
                for start in range(0, len(pairs), self.batch_size):
                    inputs = self._tokenizer(
                        pairs[start : start + self.batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    inputs = {
                        key: value.to(self._device)
                        for key, value in inputs.items()
                    }
                    logits = self._model(**inputs).logits.float()
                    if logits.ndim == 2 and logits.shape[-1] > 1:
                        batch_scores = logits[:, -1]
                    else:
                        batch_scores = logits.reshape(-1)
                    values.extend(float(value) for value in batch_scores.cpu().tolist())
            return values


@dataclass(frozen=True)
class DiverseSelection:
    items: tuple[dict[str, Any], ...]
    query_views: tuple[str, ...]
    selected_scores: tuple[float, ...]
    strategy: str
    scorer_model: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query_views": list(self.query_views),
            "selected_scores": [round(value, 6) for value in self.selected_scores],
            "scorer_model": self.scorer_model,
        }


def unique_query_views(
    question: str,
    query_views: Sequence[str] = (),
    *,
    limit: int = 12,
) -> tuple[str, ...]:
    """Keep the original request and model-produced searches as semantic facets."""

    output: list[str] = []
    seen: set[str] = set()
    for raw in (question, *query_views):
        value = " ".join(str(raw or "").split()).strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        output.append(value[:500])
        if len(output) >= max(1, int(limit)):
            break
    return tuple(output)


def _rank_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0] * len(values)
    return [(float(value) - minimum) / (maximum - minimum) for value in values]


def _bm25_matrix(views: Sequence[str], documents: Sequence[str]) -> list[list[float]]:
    token_lists = [search_tokens(document) for document in documents]
    average_length = sum(len(tokens) for tokens in token_lists) / max(1, len(token_lists))
    matrix: list[list[float]] = []
    for view in views:
        terms = [
            term
            for term in dict.fromkeys(search_tokens(view))
            if term not in _QUERY_STOPWORDS
        ]
        document_frequency = {
            term: sum(term in set(tokens) for tokens in token_lists)
            for term in terms
        }
        raw_scores: list[float] = []
        for tokens in token_lists:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            length_ratio = len(tokens) / max(1.0, average_length)
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
                score += idf * (frequency * 2.2) / (
                    frequency + 1.2 * (0.25 + 0.75 * length_ratio)
                )
            raw_scores.append(score)
        maximum = max(raw_scores, default=0.0)
        matrix.append(
            [value / maximum for value in raw_scores]
            if maximum > 0
            else [0.0] * len(documents)
        )
    return matrix


def _semantic_matrix(
    scorer: PairScorer,
    views: Sequence[str],
    documents: Sequence[str],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for view in views:
        values = list(scorer.score(view, documents))
        if len(values) != len(documents):
            raise ValueError("semantic scorer returned an unexpected score count")
        matrix.append(_rank_normalize([float(value) for value in values]))
    return matrix


def _document(item: Mapping[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            str(item.get("title") or "").strip(),
            str(item.get("content") or item.get("snippet") or "").strip(),
            str(item.get("uri") or item.get("url") or "").strip(),
        )
        if value
    )[:6000]


def _canonical_key(item: Mapping[str, Any]) -> str:
    uri = str(item.get("uri") or item.get("url") or "").strip()
    canonical = canonicalize_url(uri) if uri else None
    if canonical:
        return canonical
    return " ".join(_document(item).casefold().split())[:2000]


def _merge_duplicate(current: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for field in ("title", "content", "snippet"):
        old = str(current.get(field) or "")
        new = str(incoming.get(field) or "")
        if len(new) > len(old):
            current[field] = new
    for field in ("source", "published_at", "published_hint"):
        if not current.get(field) and incoming.get(field):
            current[field] = incoming[field]
    current["_hits"] = int(current.get("_hits") or 1) + 1
    current["_best_position"] = min(
        int(current.get("_best_position") or 1),
        int(incoming.get("_best_position") or incoming.get("position") or 1),
    )
    current["_upstream_score"] = max(
        float(current.get("_upstream_score") or 0.0),
        float(
            incoming.get("_upstream_score")
            or incoming.get("score")
            or incoming.get("candidate_score")
            or incoming.get("rrf_score")
            or 0.0
        ),
    )


def _deduplicate(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    for order, source in enumerate(items):
        if not isinstance(source, Mapping):
            continue
        item = dict(source)
        key = _canonical_key(item)
        if not key:
            continue
        index = indexes.get(key)
        if index is not None:
            _merge_duplicate(output[index], item)
            continue
        item.setdefault("_order", order)
        item.setdefault("_hits", 1)
        item.setdefault("_best_position", int(item.get("position") or order + 1))
        item.setdefault(
            "_upstream_score",
            float(
                item.get("score")
                or item.get("candidate_score")
                or item.get("rrf_score")
                or 0.0
            ),
        )
        indexes[key] = len(output)
        output.append(item)
    return output


def _token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _host(item: Mapping[str, Any]) -> str:
    uri = str(item.get("uri") or item.get("url") or "")
    host = (urlsplit(uri).hostname or "").casefold()
    return host[4:] if host.startswith("www.") else host


def select_diverse_items(
    question: str,
    query_views: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    scorer: PairScorer | None = None,
    semantic_weight: float = 0.65,
    redundancy_weight: float = 0.30,
    domain_weight: float = 0.08,
    preference_weight: float = 0.0,
) -> DiverseSelection:
    """Select evidence by query-view coverage instead of topic-specific rules.

    Each model-produced search is treated as a separate information need. The
    greedy objective combines relevance, marginal coverage, upstream rank,
    content redundancy, and a soft domain repetition penalty. No entity,
    business domain, URL shape, or intent keyword is encoded here.
    """

    candidates = _deduplicate(items)
    views = unique_query_views(question, query_views) or ("",)
    cap = min(max(0, int(limit)), len(candidates))
    if not candidates or not views or cap == 0:
        return DiverseSelection((), views, (), "query_view_mmr_v1")

    documents = [_document(item) for item in candidates]
    lexical = _bm25_matrix(views, documents)
    if scorer is not None:
        semantic = _semantic_matrix(scorer, views, documents)
        weight = min(1.0, max(0.0, float(semantic_weight)))
        score_matrix = [
            [
                (1.0 - weight) * lexical[row][column]
                + weight * semantic[row][column]
                for column in range(len(candidates))
            ]
            for row in range(len(views))
        ]
    else:
        score_matrix = lexical

    upstream = _rank_normalize(
        [
            float(item.get("_upstream_score") or 0.0)
            + 1.0 / max(1, int(item.get("_best_position") or 1))
            + 0.05 * min(4, int(item.get("_hits") or 1) - 1)
            for item in candidates
        ]
    )
    preference = _rank_normalize(
        [float(item.get("_preference_score") or 0.0) for item in candidates]
    )
    token_sets = [set(search_tokens(document)) for document in documents]
    covered = [0.0] * len(views)
    selected: list[int] = []
    selected_scores: list[float] = []
    domain_counts: dict[str, int] = {}

    while len(selected) < cap:
        best_index = -1
        best_value = float("-inf")
        for index, item in enumerate(candidates):
            if index in selected:
                continue
            per_view = [row[index] for row in score_matrix]
            relevance = max(per_view, default=0.0)
            marginal = sum(
                max(0.0, value - covered[row])
                for row, value in enumerate(per_view)
            ) / max(1, len(views))
            redundancy = max(
                (_token_jaccard(token_sets[index], token_sets[other]) for other in selected),
                default=0.0,
            )
            host = _host(item)
            repetition = domain_counts.get(host, 0) if host else 0
            value = (
                0.58 * relevance
                + 0.32 * marginal
                + 0.10 * upstream[index]
                + float(preference_weight) * preference[index]
                - float(redundancy_weight) * redundancy
                - float(domain_weight) * repetition
            )
            tie_break = -int(item.get("_order") or 0) * 1e-9
            value += tie_break
            if value > best_value:
                best_index = index
                best_value = value
        if best_index < 0:
            break
        selected.append(best_index)
        selected_scores.append(best_value)
        for row in range(len(views)):
            covered[row] = max(covered[row], score_matrix[row][best_index])
        host = _host(candidates[best_index])
        if host:
            domain_counts[host] = domain_counts.get(host, 0) + 1

    cleaned: list[dict[str, Any]] = []
    for index in selected:
        item = {
            key: value
            for key, value in candidates[index].items()
            if not str(key).startswith("_")
        }
        item["selection_view_scores"] = {
            views[row]: round(score_matrix[row][index], 6)
            for row in range(len(views))
            if score_matrix[row][index] > 0
        }
        cleaned.append(item)
    return DiverseSelection(
        items=tuple(cleaned),
        query_views=views,
        selected_scores=tuple(selected_scores),
        strategy="query_view_mmr_v1" + ("+cross_encoder" if scorer else "+bm25"),
        scorer_model=str(getattr(scorer, "model_name", "")) if scorer else "",
    )


def rank_capabilities(
    query: str,
    capabilities: Mapping[str, str],
    *,
    scorer: PairScorer | None = None,
    limit: int = 2,
) -> tuple[str, ...]:
    """Rank declarative tool descriptions without query keyword branches."""

    names = list(capabilities)
    if not names or not str(query or "").strip() or limit <= 0:
        return ()
    documents = [str(capabilities[name]) for name in names]
    lexical = _bm25_matrix((query,), documents)[0]
    if scorer is not None:
        semantic = _semantic_matrix(scorer, (query,), documents)[0]
        scores = [0.35 * left + 0.65 * right for left, right in zip(lexical, semantic)]
    else:
        scores = lexical
    ranked = sorted(range(len(names)), key=lambda index: (-scores[index], index))
    # A zero lexical score means there is no evidence for activating a
    # specialized API. A learned scorer can still resolve paraphrases.
    selected = [
        names[index]
        for index in ranked
        if scorer is not None or scores[index] > 0
    ]
    return tuple(selected[: max(0, int(limit))])
