from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

from rwkv7_scheduler import (
    AlbatrossChunkScheduler,
    AlbatrossStatePool,
    SchedulerConfig,
)

from .batching import ContinuousBatchEngine
from .routing import render_tool_gate_prompt
from .state_runtime import PersistentStateRuntime


MODEL_PATH = os.getenv(
    "G1I_MODEL_PATH",
    "models/rwkv7-g1i_preview3260-7.2b-ctx12288.pth",
)
RUNTIME_DIR = os.getenv(
    "G1I_RUNTIME_DIR",
    "vendor/Albatross/faster3a_2607",
)
MODEL_ID = os.getenv("G1I_MODEL_ID", "rwkv7-g1i-preview3260-7.2b")
CONTEXT = int(os.getenv("G1I_CONTEXT", "12288"))
STATE_CAPACITY = int(os.getenv("G1I_STATE_CAPACITY", "32"))
MAX_BATCH_SIZE = int(os.getenv("G1I_MAX_BATCH_SIZE", "8"))
PREFILL_CHUNK_SIZE = int(os.getenv("G1I_PREFILL_CHUNK_SIZE", "64"))
BATCH_WINDOW_MS = float(os.getenv("G1I_BATCH_WINDOW_MS", "4"))
MAX_WAITING_JOBS = int(os.getenv("G1I_MAX_WAITING_JOBS", "256"))
REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("G1I_REQUEST_TIMEOUT_SECONDS", "170")
)
PERSISTENT_STATE_CAPACITY = int(
    os.getenv("G1I_PERSISTENT_STATE_CAPACITY", "8")
)
PERSISTENT_STATE_TTL_SECONDS = float(
    os.getenv("G1I_PERSISTENT_STATE_TTL_SECONDS", "120")
)
PIPELINE_DEVICES = tuple(
    int(value)
    for value in os.getenv("G1I_PP_DEVICES", "").replace(",", " ").split()
)
if len(PIPELINE_DEVICES) != len(set(PIPELINE_DEVICES)) or any(
    value < 0 for value in PIPELINE_DEVICES
):
    raise ValueError("G1I_PP_DEVICES must contain unique non-negative CUDA indices")


def render_memory_gate_prompt(message: str) -> str:
    return (
        "System: Decide whether the user's current message contains durable "
        "personal information that will improve future conversations. Reply with "
        "exactly one lowercase label: search or chat. Reply search for a durable "
        "identity fact, personal preference, standing instruction, long-lived "
        "goal, or ongoing-project decision. Reply chat for greetings, questions, "
        "temporary requests, quoted or hypothetical statements, transformations, "
        "public facts, and secrets or credentials. Do not decide based on trigger "
        "words; decide from meaning. Text inside quotation marks that the user asks "
        "to translate, rewrite, summarize, or analyze is data, not the user's own "
        "memory, even when that quoted text uses first person. A question asking "
        "what the user's existing name, preference, goal, or project state is must "
        "always be chat; only a statement that supplies or updates that information "
        "is search.\n\n"
        "User: From now on, keep your answers concise.\n\nAssistant: search\n\n"
        "User: Translate 'keep your answers concise' into Chinese."
        "\n\nAssistant: chat\n\n"
        "User: My preferred name is Alice.\n\nAssistant: search\n\n"
        "User: Call me Sam from now on.\n\nAssistant: search\n\n"
        "User: Is Alice a common name?\n\nAssistant: chat\n\n"
        "User: What is my preferred name?\n\nAssistant: chat\n\n"
        "User: Please use the name Sam when speaking to me."
        "\n\nAssistant: search\n\n"
        "User: Please address me as River in all future conversations."
        "\n\nAssistant: search\n\n"
        "User: Do you remember what name I prefer?"
        "\n\nAssistant: chat\n\n"
        "User: 以后回答尽量简短一点。\n\nAssistant: search\n\n"
        "User: 把“以后回答尽量简短一点”翻译成英文。"
        "\n\nAssistant: chat\n\n"
        "User: 请把“我的名字是小明”翻译成法语。"
        "\n\nAssistant: chat\n\n"
        "User: 改写“我叫奶龙，以后这样称呼我”。"
        "\n\nAssistant: chat\n\n"
        "User: 我叫奶龙，以后这样称呼我。\n\nAssistant: search\n\n"
        "User: 假设我叫奶龙，写一个故事。\n\nAssistant: chat\n\n"
        "User: 我们决定RWKV Agent部署在V100，下次继续沿用。"
        "\n\nAssistant: search\n\n"
        "User: V100是什么？\n\nAssistant: chat\n\n"
        "User: 你还记得我喜欢哪种回答风格吗？\n\nAssistant: chat\n\n"
        "User: 你记得我之前说过的昵称吗？\n\nAssistant: chat\n\n"
        "User: 我之前告诉你的长期计划是什么？\n\nAssistant: chat\n\n"
        "User: 我的昵称是小舟。\n\nAssistant: search\n\n"
        "User: 我的昵称是什么？\n\nAssistant: chat\n\n"
        "User: 我长期要完成搜索项目。\n\nAssistant: search\n\n"
        "User: 我长期要完成什么？\n\nAssistant: chat\n\n"
        "User: 我更喜欢Rust。\n\nAssistant: search\n\n"
        "User: 我更喜欢什么语言？\n\nAssistant: chat\n\n"
        "User: 这是我的密码：abc123。\n\nAssistant: chat\n\n"
        f"User: {message.strip()}\n\nAssistant: "
    )


class NativeG1I:
    def __init__(self) -> None:
        started = time.perf_counter()
        runtime = str(Path(RUNTIME_DIR).resolve())
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        import torch
        from rwkv.utils import PIPELINE
        import rwkv7_fast_v3a as v3a

        v3a.MODEL_PATH, v3a.WKV_MODE, v3a.EMB_DEVICE = (
            MODEL_PATH,
            "fp32io16",
            "cpu",
        )
        v3a.RKV_MODE, v3a.CMIX_SPARSE, v3a.LOWRANK_WEIGHT = (
            "off",
            "no-fc",
            "transpose",
        )
        v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
        v3a.PP_DEVICES = list(PIPELINE_DEVICES)
        v3a.load_extensions(v3a.WKV_MODE)
        self.model, self.torch = v3a.RWKV7(), torch
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.pool = AlbatrossStatePool(
            self.model,
            capacity=STATE_CAPACITY,
            max_batch_size=MAX_BATCH_SIZE,
        )
        self.pool.prewarm(range(1, MAX_BATCH_SIZE + 1))
        self.scheduler = AlbatrossChunkScheduler(
            self.model,
            pool=self.pool,
            config=SchedulerConfig(
                prefill_chunk_size=PREFILL_CHUNK_SIZE,
                max_batch_size=MAX_BATCH_SIZE,
                max_queue_size=STATE_CAPACITY,
                max_input_tokens=CONTEXT,
            ),
            token_device="cpu" if self.model.emb_cpu else "cuda",
        )
        self.engine = ContinuousBatchEngine(
            tokenizer=self.pipeline,
            scheduler=self.scheduler,
            context_limit=CONTEXT,
            eos_token_id=0,
            batch_window_ms=BATCH_WINDOW_MS,
            max_waiting_jobs=MAX_WAITING_JOBS,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        self.states = PersistentStateRuntime(
            tokenizer=self.pipeline,
            scheduler=self.scheduler,
            context_limit=CONTEXT,
            eos_token_id=0,
            capacity=min(PERSISTENT_STATE_CAPACITY, STATE_CAPACITY),
            ttl_seconds=PERSISTENT_STATE_TTL_SECONDS,
        )
        self._counter_lock = threading.Lock()
        self.calls = 0
        self.classify_calls = 0
        self.gate_calls = 0
        self.memory_gate_calls = 0
        self.loaded_seconds = time.perf_counter() - started

    def complete(
        self,
        prompt: str,
        stops: Sequence[str],
        max_tokens: int,
        prefix_token_ids: Sequence[int] = (),
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
    ) -> dict[str, Any]:
        if prefill_chunk_size != PREFILL_CHUNK_SIZE:
            raise ValueError(
                f"prefill_chunk_size is fixed at {PREFILL_CHUNK_SIZE}"
            )
        result = self.engine.complete(
            prompt,
            stops=stops,
            max_tokens=max_tokens,
            prefix_token_ids=prefix_token_ids,
        )
        result["prefill_chunk_size"] = PREFILL_CHUNK_SIZE
        with self._counter_lock:
            self.calls += 1
        return result

    def route_tool(
        self,
        message: str,
        threshold: float = 0.7,
        *,
        context: str = "",
        has_pasted_text: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        token_labels = {
            "tool": self.pipeline.encode("search"),
            "chat": self.pipeline.encode("chat"),
        }
        if any(len(token_ids) != 1 for token_ids in token_labels.values()):
            raise RuntimeError(
                "G1I tool-gate labels must each encode to exactly one token"
            )
        result = self.engine.classify(
            render_tool_gate_prompt(
                message,
                context=context,
                has_pasted_text=has_pasted_text,
            ),
            labels={
                name: token_ids[0]
                for name, token_ids in token_labels.items()
            },
        )
        scores = result["scores"]
        margin = scores["tool"] - scores["chat"]
        use_tool = margin >= threshold
        with self._counter_lock:
            self.gate_calls += 1
        return {
            "use_tool": use_tool,
            "label": "tool" if use_tool else "chat",
            "scores": scores,
            "margin": margin,
            "threshold": threshold,
            "elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
            "queue_ms": result["queue_ms"],
            "batch_mode": result["batch_mode"],
        }

    def classify(
        self,
        prompt: str,
        labels: dict[str, str],
    ) -> dict[str, Any]:
        """Return exact next-token logits for public single-token labels."""

        token_labels = {
            str(name): self.pipeline.encode(str(value))
            for name, value in labels.items()
        }
        if len(token_labels) < 2 or len(token_labels) > 32:
            raise ValueError("labels must contain between 2 and 32 entries")
        if any(len(token_ids) != 1 for token_ids in token_labels.values()):
            raise ValueError("every classification label must encode to one token")
        started = time.perf_counter()
        result = self.engine.classify(
            prompt,
            labels={name: token_ids[0] for name, token_ids in token_labels.items()},
        )
        with self._counter_lock:
            self.classify_calls += 1
        return {
            "scores": dict(result["scores"]),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "queue_ms": result["queue_ms"],
            "batch_mode": result["batch_mode"],
        }

    def route_memory(
        self,
        message: str,
        threshold: float = 0.65,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        token_labels = {
            "save": self.pipeline.encode("search"),
            "skip": self.pipeline.encode("chat"),
        }
        if any(len(token_ids) != 1 for token_ids in token_labels.values()):
            raise RuntimeError(
                "G1I memory-gate labels must each encode to exactly one token"
            )
        result = self.engine.classify(
            render_memory_gate_prompt(message),
            labels={
                name: token_ids[0]
                for name, token_ids in token_labels.items()
            },
        )
        scores = result["scores"]
        margin = scores["save"] - scores["skip"]
        should_save = margin >= threshold
        with self._counter_lock:
            self.memory_gate_calls += 1
        return {
            "should_save": should_save,
            "label": "save" if should_save else "skip",
            "scores": scores,
            "margin": margin,
            "threshold": threshold,
            "elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
            "queue_ms": result["queue_ms"],
            "batch_mode": result["batch_mode"],
        }

    def health(self) -> dict[str, Any]:
        with self._counter_lock:
            calls = self.calls
            classify_calls = self.classify_calls
            gate_calls = self.gate_calls
            memory_gate_calls = self.memory_gate_calls
        return {
            "calls": calls,
            "classify_calls": classify_calls,
            "gate_calls": gate_calls,
            "memory_gate_calls": memory_gate_calls,
            "inference": self.engine.health(),
            "persistent_states": self.states.health(),
        }

    def close(self) -> None:
        self.states.close()
        self.engine.close()


service: NativeG1I | None = None


def create_app():
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="RWKV G1I continuous-batch sidecar", version="0.4.0")

    @app.on_event("startup")
    def startup() -> None:
        global service
        service = NativeG1I()

    @app.on_event("shutdown")
    def shutdown() -> None:
        global service
        if service is not None:
            service.close()
            service = None

    @app.get("/health")
    def health() -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        runtime = service.health()
        return {
            "status": "ready",
            "model": MODEL_ID,
            "context": CONTEXT,
            "loaded_seconds": round(service.loaded_seconds, 3),
            "pipeline_devices": list(PIPELINE_DEVICES),
            **runtime,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "context": CONTEXT}],
        }

    @app.post("/v1/gate/tool")
    def tool_gate(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise HTTPException(422, "message must be non-empty string")
        context = payload.get("context", "")
        if not isinstance(context, str) or len(context) > 4000:
            raise HTTPException(422, "context must be a string of at most 4000 chars")
        has_pasted_text = payload.get("has_pasted_text", False)
        if not isinstance(has_pasted_text, bool):
            raise HTTPException(422, "has_pasted_text must be boolean")
        threshold = float(payload.get("threshold", 0.7))
        if not -20.0 <= threshold <= 20.0:
            raise HTTPException(422, "threshold out of range")
        return service.route_tool(
            message,
            threshold,
            context=context,
            has_pasted_text=has_pasted_text,
        )

    @app.post("/v1/classify")
    def classify(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        prompt = payload.get("prompt")
        labels = payload.get("labels")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(422, "prompt must be non-empty string")
        if not isinstance(labels, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in labels.items()
        ):
            raise HTTPException(422, "labels must be a string-to-string object")
        try:
            return service.classify(prompt, labels)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/gate/memory")
    def memory_gate(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise HTTPException(422, "message must be non-empty string")
        threshold = float(payload.get("threshold", 0.65))
        if not -20.0 <= threshold <= 20.0:
            raise HTTPException(422, "threshold out of range")
        return service.route_memory(message, threshold)

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        prompt = payload.get("prompt")
        stop = payload.get("stop", [])
        max_tokens = int(payload.get("max_tokens", 192))
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(422, "prompt must be non-empty string")
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or not all(
            isinstance(item, str) for item in stop
        ):
            raise HTTPException(422, "stop must be string array")
        if max_tokens < 1 or max_tokens > 1024:
            raise HTTPException(422, "max_tokens out of range")
        prefix = payload.get("prefix_token_ids", [])
        if (
            not isinstance(prefix, list)
            or len(prefix) > 32
            or not all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 <= item < 65536
                for item in prefix
            )
        ):
            raise HTTPException(
                422,
                "prefix_token_ids must contain at most 32 vocabulary IDs",
            )
        prefill_chunk_size = int(
            payload.get("prefill_chunk_size", PREFILL_CHUNK_SIZE)
        )
        if prefill_chunk_size != PREFILL_CHUNK_SIZE:
            raise HTTPException(
                422,
                f"prefill_chunk_size is fixed at {PREFILL_CHUNK_SIZE}",
            )
        try:
            result = service.complete(
                prompt,
                stop,
                max_tokens,
                prefix,
                prefill_chunk_size,
            )
        except (TimeoutError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
        result["prefix_token_ids"] = prefix
        return {
            "id": f"g1i-{int(time.time() * 1000)}",
            "object": "text_completion",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "text": result["text"],
                    "finish_reason": result["stop_reason"],
                }
            ],
            "g1i": result,
        }

    def state_error(exc: Exception) -> None:
        if isinstance(exc, PermissionError):
            raise HTTPException(403, str(exc)) from exc
        if isinstance(exc, KeyError):
            raise HTTPException(404, str(exc)) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        if isinstance(exc, RuntimeError):
            raise HTTPException(409, str(exc)) from exc
        raise exc

    @app.post("/v1/states/prefill")
    def state_prefill(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        try:
            state = service.states.prefill(
                owner_id=str(payload.get("owner_id") or ""),
                prompt=str(payload.get("prompt") or ""),
                branch=str(payload.get("branch") or "root"),
            )
            return {"status": "ok", "state": state}
        except Exception as exc:
            state_error(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/{state_id}/fork")
    def state_fork(
        state_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        branches = payload.get("branches")
        if not isinstance(branches, list) or not all(
            isinstance(value, str) for value in branches
        ):
            raise HTTPException(422, "branches must be a string array")
        try:
            states = service.states.fork(
                owner_id=str(payload.get("owner_id") or ""),
                parent_state_id=state_id,
                branches=branches,
            )
            return {"status": "ok", "states": states}
        except Exception as exc:
            state_error(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/batch_continue")
    def state_batch_continue(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        items = payload.get("items")
        stops = payload.get("stop", [])
        if not isinstance(items, list):
            raise HTTPException(422, "items must be an object array")
        if isinstance(stops, str):
            stops = [stops]
        if not isinstance(stops, list) or not all(
            isinstance(value, str) for value in stops
        ):
            raise HTTPException(422, "stop must be a string array")
        try:
            results = service.states.continue_many(
                owner_id=str(payload.get("owner_id") or ""),
                items=items,
                stops=stops,
                max_tokens=int(payload.get("max_tokens", 192)),
            )
            return {"status": "ok", "results": results}
        except Exception as exc:
            state_error(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/batch_classify")
    def state_batch_classify(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        items = payload.get("items")
        labels = payload.get("labels")
        if not isinstance(items, list):
            raise HTTPException(422, "items must be an object array")
        if not isinstance(labels, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in labels.items()
        ):
            raise HTTPException(422, "labels must be a string-to-string object")
        try:
            token_labels = {
                name: service.pipeline.encode(value) for name, value in labels.items()
            }
            if len(token_labels) < 2 or len(token_labels) > 32 or any(
                len(token_ids) != 1 for token_ids in token_labels.values()
            ):
                raise ValueError(
                    "labels must contain 2-32 values that each encode to one token"
                )
            results = service.states.classify_many(
                owner_id=str(payload.get("owner_id") or ""),
                items=items,
                labels={name: token_ids[0] for name, token_ids in token_labels.items()},
            )
            return {"status": "ok", "results": results}
        except Exception as exc:
            state_error(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/release")
    def state_release(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        state_ids = payload.get("state_ids")
        if not isinstance(state_ids, list) or not all(
            isinstance(value, str) for value in state_ids
        ):
            raise HTTPException(422, "state_ids must be a string array")
        try:
            released = service.states.release(
                owner_id=str(payload.get("owner_id") or ""),
                state_ids=state_ids,
            )
            return {"status": "ok", **released}
        except Exception as exc:
            state_error(exc)
            raise AssertionError("unreachable")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8118)
    args = parser.parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
