"""HTTP client for stateless and recurrent G1I Sidecar operations."""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

from .chat_prompts import CHAT_STOPS


class ModelClient:
    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("at least one G1I sidecar URL is required")
        self.urls = [value.rstrip("/") for value in urls]
        self._index = 0
        self._lock = threading.Lock()

    def _next_url(self) -> str:
        with self._lock:
            url = self.urls[self._index % len(self.urls)]
            self._index += 1
            return url

    @staticmethod
    def _get(url: str) -> dict[str, Any]:
        with urlopen(url, timeout=10) as response:
            return json.load(response)

    @staticmethod
    def _post(
        url: str,
        payload: dict[str, Any],
        *,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode()
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError("sidecar returned a non-object response")
        return value

    def health(self) -> list[dict[str, Any]]:
        return [self._get(url + "/health") for url in self.urls]

    def state_prefill(
        self,
        *,
        owner_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        home_url = self._next_url()
        result = self._post(
            home_url + "/v1/states/prefill",
            {
                "owner_id": owner_id,
                "prompt": prompt,
                "branch": "root",
            },
        )
        state = dict(result["state"])
        state["home_url"] = home_url
        return state

    def state_fork(
        self,
        *,
        home_url: str,
        owner_id: str,
        parent_state_id: str,
        branches: list[str],
    ) -> list[dict[str, Any]]:
        result = self._post(
            home_url.rstrip("/") + f"/v1/states/{parent_state_id}/fork",
            {"owner_id": owner_id, "branches": branches},
        )
        return [dict(value) for value in result["states"]]

    def state_batch_continue(
        self,
        *,
        home_url: str,
        owner_id: str,
        items: list[dict[str, str]],
        stops: list[str],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        result = self._post(
            home_url.rstrip("/") + "/v1/states/batch_continue",
            {
                "owner_id": owner_id,
                "items": items,
                "stop": stops,
                "max_tokens": max_tokens,
            },
        )
        return [dict(value) for value in result["results"]]

    def state_chat_complete(
        self,
        *,
        home_url: str,
        owner_id: str,
        state_id: str,
        input_text: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        rows = self.state_batch_continue(
            home_url=home_url,
            owner_id=owner_id,
            items=[{"state_id": state_id, "input": input_text}],
            stops=list(CHAT_STOPS),
            max_tokens=max_tokens,
        )
        if len(rows) != 1:
            raise RuntimeError("chat state continuation returned an invalid row count")
        row = rows[0]
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return {
            "raw": str(row.get("text") or ""),
            "stop": str(row.get("stop_reason") or ""),
            "output_tokens": len(row.get("token_ids") or []),
            "model_elapsed_ms": float(row.get("elapsed_ms") or 0.0),
            "request_elapsed_ms": elapsed_ms,
            "model": None,
            "url": home_url.rstrip("/"),
            "state_id": str(row.get("state_id") or state_id),
            "seen_tokens": int(row.get("seen_tokens") or 0),
        }

    def state_batch_classify(
        self,
        *,
        home_url: str,
        owner_id: str,
        items: list[dict[str, str]],
        labels: dict[str, str],
    ) -> list[dict[str, Any]]:
        result = self._post(
            home_url.rstrip("/") + "/v1/states/batch_classify",
            {"owner_id": owner_id, "items": items, "labels": labels},
        )
        return [dict(value) for value in result["results"]]

    def state_release(
        self,
        *,
        home_url: str,
        owner_id: str,
        state_ids: list[str],
    ) -> dict[str, Any]:
        return self._post(
            home_url.rstrip("/") + "/v1/states/release",
            {"owner_id": owner_id, "state_ids": state_ids},
        )

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 192,
        stops: list[str] | None = None,
    ) -> dict[str, Any]:
        url = self._next_url()
        stop_values = stops or [
            "</tool_call>",
            "</tool_calls>",
            "</tool_code>",
            "\nUser:",
            "\nSystem:",
            "\n\nUser:",
            "</s>",
        ]
        started = time.perf_counter()
        data = self._post(
            url + "/v1/completions",
            {
                "prompt": prompt,
                "stop": stop_values,
                "max_tokens": max_tokens,
            },
        )
        g1i = data["g1i"]
        stop = str(g1i.get("stop_reason") or "")
        raw = str(g1i.get("text") or "") + (stop if stop.startswith("</tool") else "")
        return {
            "raw": raw,
            "stop": stop,
            "output_tokens": len(g1i.get("token_ids") or []),
            "model_elapsed_ms": float(g1i.get("elapsed_ms") or 0.0),
            "request_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "model": data.get("model"),
            "url": url,
        }

    def classify(
        self,
        prompt: str,
        *,
        labels: dict[str, str],
    ) -> dict[str, Any]:
        url = self._next_url()
        started = time.perf_counter()
        result = self._post(
            url + "/v1/classify",
            {"prompt": prompt, "labels": labels},
        )
        result["request_elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        result["url"] = url
        return result

    def gate_tool(
        self,
        message: str,
        *,
        threshold: float = 0.7,
        context: str = "",
        has_pasted_text: bool = False,
    ) -> dict[str, Any]:
        url = self._next_url()
        started = time.perf_counter()
        result = self._post(
            url + "/v1/gate/tool",
            {
                "message": message,
                "threshold": threshold,
                "context": context,
                "has_pasted_text": has_pasted_text,
            },
        )
        result["request_elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        result["url"] = url
        return result
