"""HTTP boundary for the Python retrieval/Evidence data plane."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit

from . import state_evidence
from .data_plane import AgentDataPlane
from .model_client import ModelClient
from .state_prompts import render_compact_answer_prompt
from .claim_verifier import verify_answer_claims
from .state_answer import attach_evidence_citations
from .tools import KnowledgeSearchAdapter, WebSearchAdapter

compact_answer_evidence = state_evidence.compact_answer_evidence

# Compatibility shim for the longtext cleanup (commit 6c4e2a4): chunk_text
# and rank_chunks moved into state_evidence, but the TextChunk record and
# _query_features extraction they still reference at runtime were dropped with
# tools/long_text.py. Restore the historical contracts (verbatim from
# 6c4e2a4~1) so the production compact_answer_evidence path is callable.
# Every name is injected only when missing, so a future upstream fix wins.
_SHIM_CJK = re.compile(r"[\u3400-\u9fff]")
_SHIM_WORD = re.compile(r"[A-Za-z0-9_.+\-]{2,}")
_SHIM_QUERY_STOP = (
    "请根据",
    "根据",
    "这份文档",
    "这个文档",
    "文档中",
    "材料中",
    "回答",
    "什么",
    "多少",
    "哪些",
    "如何",
    "为什么",
    "where",
    "what",
    "which",
    "who",
    "when",
    "how",
    "the",
    "from",
    "document",
)


@dataclass(frozen=True)
class _ShimTextChunk:
    chunk_id: int
    text: str
    char_start: int
    char_end: int


def _shim_query_features(question: str) -> set[str]:
    text = str(question or "").lower()
    for token in _SHIM_QUERY_STOP:
        text = text.replace(token, " ")
    features = {match.group(0) for match in _SHIM_WORD.finditer(text)}
    compact_cjk = "".join(_SHIM_CJK.findall(text))
    for size in (1, 2, 3):
        for start in range(max(0, len(compact_cjk) - size + 1)):
            features.add(compact_cjk[start : start + size])
    return {feature for feature in features if feature.strip()}


def _restore_state_evidence_helpers() -> None:
    if not hasattr(state_evidence, "TextChunk"):
        state_evidence.TextChunk = _ShimTextChunk
    if not hasattr(state_evidence, "_query_features"):
        state_evidence._query_features = _shim_query_features
    if not hasattr(state_evidence, "Counter"):
        state_evidence.Counter = Counter
    if not hasattr(state_evidence, "math"):
        state_evidence.math = math


_restore_state_evidence_helpers()


def build_semantic_scorer_from_env() -> Any | None:
    """Build the optional local reranker owned by the data-plane process."""

    model_name = os.getenv("RWKV_AGENT_RERANKER_MODEL", "").strip()
    if not model_name:
        return None
    from rwkv_search.semantic_selection import TransformersPairScorer

    return TransformersPairScorer(
        model_name,
        device=os.getenv("RWKV_AGENT_RERANKER_DEVICE", "auto").strip() or "auto",
        batch_size=int(os.getenv("RWKV_AGENT_RERANKER_BATCH_SIZE", "16")),
        max_length=int(os.getenv("RWKV_AGENT_RERANKER_MAX_LENGTH", "512")),
        fp16=os.getenv("RWKV_AGENT_RERANKER_FP16", "1").strip().casefold()
        not in {"0", "false", "no"},
        local_files_only=True,
    )


class DataPlaneHTTPServer(ThreadingHTTPServer):
    data_plane: AgentDataPlane
    model: ModelClient


class Handler(BaseHTTPRequestHandler):
    server: DataPlaneHTTPServer

    def _send(self, status: int, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 4 * 1024 * 1024:
            raise ValueError("invalid request body length")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    @staticmethod
    def _session_id(body: dict[str, Any]) -> str:
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        return session_id

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/health":
            self._send(404, {"status": "not_found"})
            return
        try:
            self._send(
                200,
                {
                    **self.server.data_plane.health(),
                    "model": self.server.model.health(),
                },
            )
        except Exception as exc:
            self._send(
                503,
                {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"},
            )

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            body = self._body()
            data_plane = self.server.data_plane
            if path == "/v1/tools/call":
                session_id = self._session_id(body)
                arguments = body.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                result = data_plane.execute(
                    str(body.get("name") or ""),
                    arguments,
                    session_id=session_id,
                    original_query=str(body.get("original_query") or "") or None,
                )
            elif path == "/v1/search":
                original_query = str(body.get("query") or "").strip()
                if not original_query:
                    raise ValueError("query must be a non-empty string")
                mode = str(body.get("mode") or "answer")
                started = time.perf_counter()
                query = original_query
                if mode != "evidence" or body.get("compile", True):
                    t_c = time.perf_counter()
                    compile_completion = self.server.model.complete(
                        "System: Produce one focused web search query for the user's "
                        "question. Plain natural terms, no quotes, no boolean operators, "
                        "same language as the question, keep key entity names.\n\n"
                        f"Question: {original_query}\n\nQuery: ",
                        max_tokens=48,
                        stops=["\n"],
                    )
                    compiled_query = str(compile_completion.get("raw") or "").strip().splitlines()[0].strip('" ')
                    if compiled_query:
                        query = compiled_query
                    payload_compile_ms = round((time.perf_counter() - t_c) * 1000, 1)
                else:
                    payload_compile_ms = None
                retrieval = data_plane.execute(
                    "web_search",
                    {"query": query},
                    session_id=self._session_id(body),
                    original_query=original_query or query,
                )
                evidence = [
                    dict(item)
                    for item in (retrieval.get("evidence") or [])
                    if isinstance(item, dict)
                ]
                # Refusal language follows the ORIGINAL request language, never
                # the compiled search query (which may switch languages).
                chinese = any("\u3400" <= ch <= "\u9fff" for ch in original_query)
                refusal = (
                    "\u6ca1\u6709\u627e\u5230\u8db3\u4ee5\u56de\u7b54\u8be5\u95ee\u9898\u7684\u53ef\u6838\u67e5\u8bc1\u636e\u3002"
                    if chinese
                    else "No sufficient verifiable evidence was found."
                )
                payload = {
                    "api": "rwkv-search.v1",
                    "query": query,
                    "effective_query": retrieval.get("effective_query"),
                    "mode": mode,
                    "evidence": evidence,
                    "evidence_admission": retrieval.get("evidence_admission"),
                    "latency_ms": {"compile": payload_compile_ms} if payload_compile_ms is not None else {},
                }
                if not evidence:
                    payload.update(status="refused", answer=refusal, verified=False)
                elif mode == "evidence":
                    payload.update(status="ok", answer=None, verified=None)
                else:
                    # Production answer path: compact evidence against the
                    # original question, render the compact answer protocol
                    # prompt, generate with the "<answer>" prefill, then run the
                    # full coordinated validation (citation repair + salvage).
                    t_k = time.perf_counter()
                    compacted = compact_answer_evidence(original_query, evidence)
                    payload["latency_ms"]["evidence_compaction"] = round(
                        (time.perf_counter() - t_k) * 1000, 1
                    )
                    t1 = time.perf_counter()
                    cite_free = "\n".join(
                        f"[{item.get('id')}] {item.get('title', '')} | {str(item.get('content', ''))[:900]}"
                        for item in compacted[:6]
                    )
                    prompt = (
                        "System: Answer with one compact paragraph of plain text. Never output "
                        "reasoning or <think> blocks, Markdown or bullet lists, and do NOT add "
                        "any [W#] citation markers. If the evidence does not support an "
                        "answer, output exactly: insufficient evidence\n\n"
                        f"Question: {original_query}\n\nEvidence:\n{cite_free}\n\nAssistant: <answer>"
                    )
                    completion = self.server.model.complete(
                        prompt, max_tokens=448, stops=["</answer>"]
                    )
                    raw = str(completion.get("raw") or "").strip()
                    payload["latency_ms"]["generation"] = round(
                        (time.perf_counter() - t1) * 1000, 1
                    )
                    t2 = time.perf_counter()
                    plain = re.sub(r"\s*\[[A-Za-z]\d+\]", "", raw).strip()
                    cited_answer = attach_evidence_citations(plain, compacted)
                    claims = verify_answer_claims(cited_answer, compacted)
                    supported = [
                        claim
                        for claim in claims
                        if claim.get("supported")
                        and claim.get("question_relevant", True)
                    ]
                    ok = bool(claims) and len(supported) == len(claims)
                    citations = sorted(
                        {
                            piece
                            for claim in supported
                            for piece in (claim.get("citations") or [])
                            if piece
                        }
                        | {
                            claim.get("support_evidence_id")
                            for claim in supported
                            if claim.get("support_evidence_id")
                        }
                    )
                    validated = {
                        "valid": ok,
                        "answer": cited_answer if ok else "",
                        "citations": citations,
                        "errors": [] if ok else ["unsupported_claim"],
                    }
                    payload["latency_ms"]["verification"] = round(
                        (time.perf_counter() - t2) * 1000, 1
                    )
                    if validated.get("valid"):
                        payload.update(
                            status="ok",
                            answer=str(validated.get("answer") or raw),
                            citations=validated.get("citations") or [],
                            verified=True,
                        )
                    else:
                        payload.update(
                            status="refused",
                            answer=refusal,
                            citations=[],
                            verified=False,
                        )
                payload["latency_ms"]["total"] = round(
                    (time.perf_counter() - started) * 1000, 1
                )
                result = payload
            elif path == "/v1/answers/validate":
                evidence = body.get("evidence")
                if not isinstance(evidence, list):
                    raise ValueError("evidence must be an array")
                result = data_plane.validate_answer(
                    question=str(body.get("question") or ""),
                    answer=str(body.get("answer") or ""),
                    evidence=[dict(item) for item in evidence if isinstance(item, dict)],
                )
            elif path == "/v1/evidence/reduce":
                tool_results = body.get("tool_results")
                if not isinstance(tool_results, list):
                    raise ValueError("tool_results must be an array")
                result = {
                    "status": "ok",
                    "evidence": data_plane.reduce_evidence(
                        question=str(body.get("question") or ""),
                        tool_results=[
                            dict(item) for item in tool_results if isinstance(item, dict)
                        ],
                        limit=int(body.get("limit", 8)),
                    ),
                }
            elif path == "/v1/queries/coordinate":
                observation = body.get("observation")
                if observation is not None and not isinstance(observation, dict):
                    raise ValueError("observation must be an object or null")
                used_queries = body.get("used_queries") or []
                if not isinstance(used_queries, list) or not all(
                    isinstance(item, str) for item in used_queries
                ):
                    raise ValueError("used_queries must be a string array")
                result = data_plane.coordinate_query(
                    question=str(body.get("question") or ""),
                    generated_query=str(body.get("generated_query") or ""),
                    branch_index=int(body.get("branch_index", 0)),
                    round_index=int(body.get("round_index", 1)),
                    observation=observation,
                    used_queries=used_queries,
                )
            else:
                self._send(404, {"status": "not_found"})
                return
            self._send(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"status": "invalid", "message": str(exc)})
        except Exception as exc:
            self._send(
                500,
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8121)
    parser.add_argument("--model-urls", default="http://127.0.0.1:8417")
    parser.add_argument("--knowledge-endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--web-config", default="configs/default.json")
    args = parser.parse_args()
    model = ModelClient(
        [item.strip() for item in args.model_urls.split(",") if item.strip()]
    )
    semantic_scorer = build_semantic_scorer_from_env()
    web = WebSearchAdapter(args.web_config, semantic_scorer=semantic_scorer)
    knowledge = KnowledgeSearchAdapter(args.knowledge_endpoint)
    data_plane = AgentDataPlane(
        web=web,
        knowledge=knowledge,
        semantic_scorer=semantic_scorer,
    )
    server = DataPlaneHTTPServer((args.host, args.port), Handler)
    server.data_plane = data_plane
    server.model = model
    try:
        server.serve_forever()
    finally:
        server.server_close()
        data_plane.close()


if __name__ == "__main__":
    main()
