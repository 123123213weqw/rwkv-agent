#!/usr/bin/env python3
"""Isolated 4-bit PEFT quality-evaluation sidecar.

This is deliberately not the release runtime.  It exposes the same HTTP
surface as the native G1I sidecar while recomputing branch prompts, allowing a
new LoRA checkpoint to be screened on a smaller GPU before the compact raw
merge is moved to the V100 native runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Sequence
import uuid


@dataclass
class PromptState:
    state_id: str
    owner_id: str
    parent_state_id: str | None
    branch: str
    prompt: str
    seen_tokens: int


class PromptStateRuntime:
    """Owner-isolated prompt states for non-release checkpoint screening."""

    def __init__(
        self,
        *,
        encode: Callable[[str], Sequence[int]],
        generate: Callable[[str, Sequence[str], int], dict[str, Any]],
        capacity: int = 64,
    ) -> None:
        self.encode = encode
        self.generate = generate
        self.capacity = max(1, int(capacity))
        self.records: dict[str, PromptState] = {}
        self.lock = threading.RLock()
        self.metrics = {
            "created": 0,
            "forked": 0,
            "continued": 0,
            "released": 0,
            "expired": 0,
            "failed": 0,
        }

    @staticmethod
    def _owner(value: str) -> str:
        owner = str(value or "").strip()
        if not owner:
            raise ValueError("owner_id must not be empty")
        return owner

    @staticmethod
    def _id() -> str:
        return "state-" + uuid.uuid4().hex

    def _require(self, state_id: str, owner_id: str) -> PromptState:
        try:
            state = self.records[str(state_id)]
        except KeyError as exc:
            raise KeyError(f"unknown state_id: {state_id}") from exc
        if state.owner_id != owner_id:
            raise PermissionError("state owner mismatch")
        return state

    @staticmethod
    def _describe(state: PromptState) -> dict[str, Any]:
        return {
            "state_id": state.state_id,
            "owner_id": state.owner_id,
            "parent_state_id": state.parent_state_id,
            "branch": state.branch,
            "seen_tokens": state.seen_tokens,
        }

    def prefill(self, *, owner_id: str, prompt: str, branch: str) -> dict[str, Any]:
        owner = self._owner(owner_id)
        value = str(prompt or "")
        if not value:
            raise ValueError("state input must not be empty")
        with self.lock:
            if len(self.records) >= self.capacity:
                raise RuntimeError("persistent state capacity exceeded")
            state = PromptState(
                state_id=self._id(),
                owner_id=owner,
                parent_state_id=None,
                branch=str(branch or "root")[:80],
                prompt=value,
                seen_tokens=len(self.encode(value)),
            )
            self.records[state.state_id] = state
            self.metrics["created"] += 1
            return self._describe(state)

    def fork(
        self,
        *,
        owner_id: str,
        parent_state_id: str,
        branches: Sequence[str],
    ) -> list[dict[str, Any]]:
        owner = self._owner(owner_id)
        labels = [str(value or "").strip() for value in branches]
        if not labels or any(not value for value in labels):
            raise ValueError("branches must contain non-empty labels")
        if len(set(labels)) != len(labels):
            raise ValueError("branch labels must be unique")
        with self.lock:
            parent = self._require(parent_state_id, owner)
            if len(self.records) + len(labels) > self.capacity:
                raise RuntimeError("persistent state capacity exceeded")
            output = []
            for label in labels:
                child = PromptState(
                    state_id=self._id(),
                    owner_id=owner,
                    parent_state_id=parent.state_id,
                    branch=label[:80],
                    prompt=parent.prompt,
                    seen_tokens=parent.seen_tokens,
                )
                self.records[child.state_id] = child
                output.append(self._describe(child))
            self.metrics["forked"] += len(output)
            return output

    def continue_many(
        self,
        *,
        owner_id: str,
        items: Sequence[dict[str, Any]],
        stops: Sequence[str],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        owner = self._owner(owner_id)
        if not items:
            raise ValueError("items must not be empty")
        with self.lock:
            states = [
                self._require(str(item.get("state_id") or ""), owner)
                for item in items
            ]
            if len({state.state_id for state in states}) != len(states):
                raise ValueError("duplicate state_id in batch")
            output = []
            for state, item in zip(states, items):
                continuation = str(item.get("input") or "")
                if not continuation:
                    raise ValueError("state input must not be empty")
                prompt = state.prompt + continuation
                result = self.generate(prompt, stops, int(max_tokens))
                committed = str(result.get("committed_text") or result["text"])
                state.prompt = prompt + committed
                state.seen_tokens = len(self.encode(state.prompt))
                output.append(
                    {
                        "state_id": state.state_id,
                        "branch": state.branch,
                        "text": str(result["text"]),
                        "token_ids": list(result.get("token_ids") or []),
                        "stop_reason": str(result.get("stop_reason") or ""),
                        "seen_tokens": state.seen_tokens,
                    }
                )
            self.metrics["continued"] += len(output)
            return output

    def release(self, *, owner_id: str, state_ids: Sequence[str]) -> dict[str, Any]:
        owner = self._owner(owner_id)
        ids = [str(value or "").strip() for value in state_ids]
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("state_ids must be unique non-empty strings")
        with self.lock:
            for state_id in ids:
                self._require(state_id, owner)
            for state_id in ids:
                self.records.pop(state_id)
            self.metrics["released"] += len(ids)
        return {"released": len(ids), "state_ids": ids}

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "enabled": True,
                "capacity": self.capacity,
                "allocated": len(self.records),
                "free": self.capacity - len(self.records),
                "ttl_seconds": 0.0,
                "expired_on_health": 0,
                "oldest_idle_seconds": 0.0,
                "metrics": dict(self.metrics),
            }


class HFEvalService:
    def __init__(self, model_path: Path, adapter_path: Path | None, context: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        started = time.perf_counter()
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        if adapter_path is None:
            self.model = base
        else:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True
        self.torch = torch
        self.context = int(context)
        self.model_id = (
            f"fitgen-hf-eval:{adapter_path.name}"
            if adapter_path is not None
            else f"fitgen-hf-eval-base:{model_path.name}"
        )
        self.generate_lock = threading.Lock()
        self.calls = 0
        self.loaded_seconds = time.perf_counter() - started
        self.states = PromptStateRuntime(
            encode=self.tokenizer.encode,
            generate=self.generate,
            capacity=int(os.getenv("FITGEN_HF_STATE_CAPACITY", "64")),
        )

    def generate(
        self,
        prompt: str,
        stops: Sequence[str],
        max_tokens: int,
        prefix_token_ids: Sequence[int] = (),
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with self.generate_lock, self.torch.inference_mode():
            values = [*map(int, prefix_token_ids), *self.tokenizer.encode(prompt)]
            values = values[-max(1, self.context - int(max_tokens)) :]
            input_ids = self.torch.tensor([values], device="cuda", dtype=self.torch.long)
            result = self.model(input_ids=input_ids, use_cache=True)
            cache = result.past_key_values
            logits = result.logits[:, -1, :]
            output_ids: list[int] = []
            text = ""
            stop_reason = "max_tokens"
            committed_text = ""
            for _ in range(int(max_tokens)):
                token = int(self.torch.argmax(logits, dim=-1).item())
                if token == 0:
                    stop_reason = "</s>"
                    break
                output_ids.append(token)
                decoded = self.tokenizer.decode(output_ids)
                if "\ufffd" not in decoded:
                    committed_text = decoded
                    hits = [
                        (decoded.find(stop), stop)
                        for stop in stops
                        if stop and stop in decoded
                    ]
                    if hits:
                        index, stop_reason = min(hits)
                        text = decoded[:index]
                        break
                    text = decoded
                next_ids = self.torch.tensor([[token]], device="cuda")
                result = self.model(
                    input_ids=next_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = result.past_key_values
                logits = result.logits[:, -1, :]
            self.calls += 1
        return {
            "text": text,
            "committed_text": committed_text or text,
            "token_ids": output_ids,
            "stop_reason": stop_reason,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "queue_ms": 0.0,
            "batch_mode": "hf_eval_serial",
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": self.model_id,
            "context": self.context,
            "loaded_seconds": round(self.loaded_seconds, 3),
            "pipeline_devices": [0],
            "calls": self.calls,
            "gate_calls": 0,
            "memory_gate_calls": 0,
            "inference": {
                "mode": "hf_eval_serial",
                "worker_alive": True,
                "worker_error": "",
                "metrics": {"completed": self.calls},
                "scheduler": {"pool": {"capacity": self.states.capacity}},
            },
            "persistent_states": self.states.health(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }


def create_app(service: HFEvalService):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="FitGen HF checkpoint screening sidecar")

    def fail(exc: Exception) -> None:
        if isinstance(exc, PermissionError):
            raise HTTPException(403, str(exc)) from exc
        if isinstance(exc, KeyError):
            raise HTTPException(404, str(exc)) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        if isinstance(exc, RuntimeError):
            raise HTTPException(409, str(exc)) from exc
        raise exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": service.model_id}]}

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "")
        if not prompt:
            raise HTTPException(422, "prompt must be non-empty string")
        stops = payload.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        result = service.generate(
            prompt,
            stops,
            int(payload.get("max_tokens", 192)),
            payload.get("prefix_token_ids") or [],
        )
        return {
            "id": f"hf-eval-{int(time.time() * 1000)}",
            "object": "text_completion",
            "model": service.model_id,
            "choices": [{"index": 0, "text": result["text"], "finish_reason": result["stop_reason"]}],
            "g1i": result,
        }

    @app.post("/v1/states/prefill")
    def prefill(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            state = service.states.prefill(
                owner_id=str(payload.get("owner_id") or ""),
                prompt=str(payload.get("prompt") or ""),
                branch=str(payload.get("branch") or "root"),
            )
            return {"status": "ok", "state": state}
        except Exception as exc:
            fail(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/{state_id}/fork")
    def fork(state_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            states = service.states.fork(
                owner_id=str(payload.get("owner_id") or ""),
                parent_state_id=state_id,
                branches=payload.get("branches") or [],
            )
            return {"status": "ok", "states": states}
        except Exception as exc:
            fail(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/batch_continue")
    def batch_continue(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            results = service.states.continue_many(
                owner_id=str(payload.get("owner_id") or ""),
                items=payload.get("items") or [],
                stops=payload.get("stop") or [],
                max_tokens=int(payload.get("max_tokens", 192)),
            )
            return {"status": "ok", "results": results}
        except Exception as exc:
            fail(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/release")
    def release(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = service.states.release(
                owner_id=str(payload.get("owner_id") or ""),
                state_ids=payload.get("state_ids") or [],
            )
            return {"status": "ok", **result}
        except Exception as exc:
            fail(exc)
            raise AssertionError("unreachable")

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8317)
    parser.add_argument("--context", type=int, default=12288)
    args = parser.parse_args()
    service = HFEvalService(
        args.model.resolve(),
        None if args.adapter is None else args.adapter.resolve(),
        args.context,
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
