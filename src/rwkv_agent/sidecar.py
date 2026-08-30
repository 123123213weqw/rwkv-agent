from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from .batching import ContinuousBatchEngine
from .openai_compat import (
    completion_usage,
    normalize_stops,
    openai_finish_reason,
    render_chat_prompt,
    sse_data,
)
from .routing import (
    render_tool_gate_root,
    render_tool_gate_turn,
)
from .state_runtime import PersistentStateRuntime
from .statepool_worker import StatePoolWorkerAgent, WorkerSettings


MODEL_PATH = os.getenv(
    "G1I_MODEL_PATH",
    "models/rwkv7-g1i_preview3260-7.2b-ctx12288.pth",
)
RUNTIME_DIR = os.getenv(
    "G1I_RUNTIME_DIR",
    "vendor/Albatross/faster3a_2607",
)
MODEL_ID = os.getenv("G1I_MODEL_ID", "rwkv7-g1i-preview3260-7.2b")
MODEL_REVISION = os.getenv("G1I_MODEL_REVISION", "preview3260")
TOKENIZER_ID = os.getenv("G1I_TOKENIZER_ID", "rwkv_vocab_v20230424")
BACKEND = os.getenv("G1I_BACKEND", "albatross").strip().lower()
STATE_ABI = os.getenv(
    "G1I_STATE_ABI",
    f"rwkv7-{BACKEND}-recurrent-state-v1",
)
HF_MODEL_PATH = os.getenv("G1I_HF_MODEL_PATH", MODEL_PATH)
HF_DTYPE = os.getenv("G1I_HF_DTYPE", "fp16").strip().lower()
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
MAX_SNAPSHOT_BYTES = int(
    os.getenv("G1I_MAX_SNAPSHOT_BYTES", str(512 * 1024 * 1024))
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
        if BACKEND == "albatross":
            self._load_albatross()
        elif BACKEND == "hf_recurrent":
            self._load_hf_recurrent()
        else:
            raise ValueError(
                "G1I_BACKEND must be either albatross or hf_recurrent"
            )
        self.engine = ContinuousBatchEngine(
            tokenizer=self.pipeline,
            scheduler=self.scheduler,
            context_limit=CONTEXT,
            eos_token_id=0,
            batch_window_ms=BATCH_WINDOW_MS,
            max_waiting_jobs=MAX_WAITING_JOBS,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_state_rows=MAX_BATCH_SIZE,
        )
        self.states = PersistentStateRuntime(
            tokenizer=self.pipeline,
            scheduler=self.scheduler,
            context_limit=CONTEXT,
            eos_token_id=0,
            capacity=min(PERSISTENT_STATE_CAPACITY, STATE_CAPACITY),
            ttl_seconds=PERSISTENT_STATE_TTL_SECONDS,
            decode_engine=self.engine,
            max_snapshot_bytes=MAX_SNAPSHOT_BYTES,
        )
        self._counter_lock = threading.Lock()
        self.calls = 0
        self.classify_calls = 0
        self.gate_calls = 0
        self.memory_gate_calls = 0
        self._tool_gate_owner = "system-tool-gate-v1"
        self._tool_gate_root: dict[str, Any] | None = None
        self._tool_gate_lock = threading.RLock()
        self._tool_gate_sequence = 0
        self._tool_gate_root_builds = 0
        self._tool_gate_root_reuses = 0
        self._tool_gate_forks = 0
        self._tool_gate_failures = 0
        # Pay the immutable semantic-gate prefill cost once at startup. Every
        # request then forks this recurrent State and appends only local input.
        with self._tool_gate_lock:
            self._ensure_tool_gate_root_locked()
        self.loaded_seconds = time.perf_counter() - started

    def _ensure_tool_gate_root_locked(self) -> tuple[dict[str, Any], bool]:
        root = self._tool_gate_root
        if root is not None and self.states.has_state(
            owner_id=self._tool_gate_owner,
            state_id=str(root["state_id"]),
            touch=True,
        ):
            self._tool_gate_root_reuses += 1
            return root, True
        root = self.states.prefill(
            owner_id=self._tool_gate_owner,
            prompt=render_tool_gate_root(),
            branch="tool-gate-root",
        )
        self._tool_gate_root = root
        self._tool_gate_root_builds += 1
        return root, False

    def _load_albatross(self) -> None:
        from rwkv7_scheduler import (
            AlbatrossChunkScheduler,
            AlbatrossStatePool,
            SchedulerConfig,
        )

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

    def _load_hf_recurrent(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rwkv7_scheduler import HFRecurrentScheduler, SchedulerConfig

        if HF_DTYPE not in {"fp16", "bf16"}:
            raise ValueError("G1I_HF_DTYPE must be fp16 or bf16")
        # Keep a deployment-provided symlink name intact.  Transformers 4.x
        # derives a Python package name from the final directory component;
        # resolving a safe ``rwkv7_01b_hf`` symlink back to a hyphenated model
        # directory produces an invalid dynamic-module import path.
        model_path = Path(HF_MODEL_PATH).expanduser().absolute()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"G1I_HF_MODEL_PATH is not a directory: {model_path}"
            )
        dtype = torch.float16 if HF_DTYPE == "fp16" else torch.bfloat16
        self.pipeline = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            # ``torch_dtype`` works on the pinned Transformers 4.x runtime as
            # well as newer releases.  Passing the newer ``dtype`` alias into
            # 4.x is treated as a config field and fails JSON serialization.
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        self.torch = torch
        self.pool = None
        self.scheduler = HFRecurrentScheduler(
            self.model,
            config=SchedulerConfig(
                prefill_chunk_size=PREFILL_CHUNK_SIZE,
                max_batch_size=MAX_BATCH_SIZE,
                max_queue_size=STATE_CAPACITY,
                max_input_tokens=CONTEXT,
                eos_token_id=0,
            ),
            device="cuda",
            capacity=STATE_CAPACITY,
        )
        self.pool = self.scheduler.pool

    def complete(
        self,
        prompt: str,
        stops: Sequence[str],
        max_tokens: int,
        prefix_token_ids: Sequence[int] = (),
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
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
            event_sink=event_sink,
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
        child: dict[str, Any] | None = None
        root_reused = False
        try:
            with self._tool_gate_lock:
                root, root_reused = self._ensure_tool_gate_root_locked()
                self._tool_gate_sequence += 1
                child = self.states.fork(
                    owner_id=self._tool_gate_owner,
                    parent_state_id=str(root["state_id"]),
                    branches=[f"tool-gate-{self._tool_gate_sequence}"],
                )[0]
                self._tool_gate_forks += 1
            result = self.states.classify_many(
                owner_id=self._tool_gate_owner,
                items=[
                    {
                        "state_id": child["state_id"],
                        "input": render_tool_gate_turn(
                            message,
                            context=context,
                            has_pasted_text=has_pasted_text,
                        ),
                    }
                ],
                labels={
                    name: token_ids[0]
                    for name, token_ids in token_labels.items()
                },
            )[0]
        except Exception:
            with self._tool_gate_lock:
                self._tool_gate_failures += 1
            raise
        finally:
            if child is not None:
                try:
                    self.states.release(
                        owner_id=self._tool_gate_owner,
                        state_ids=[str(child["state_id"])],
                    )
                except Exception:
                    with self._tool_gate_lock:
                        self._tool_gate_failures += 1
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
            "queue_ms": 0.0,
            "batch_mode": "persistent_root_fork",
            "root_reused": root_reused,
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
        with self._tool_gate_lock:
            tool_gate_root_id = (
                str(self._tool_gate_root["state_id"])
                if self._tool_gate_root is not None
                else ""
            )
            tool_gate_root_available = bool(
                tool_gate_root_id
                and self.states.has_state(
                    owner_id=self._tool_gate_owner,
                    state_id=tool_gate_root_id,
                )
            )
            tool_gate_metrics = {
                "root_available": tool_gate_root_available,
                "root_builds": self._tool_gate_root_builds,
                "root_reuses": self._tool_gate_root_reuses,
                "forks": self._tool_gate_forks,
                "failures": self._tool_gate_failures,
                "mode": "persistent_root_fork",
            }
        persistent_states = self.states.health()
        # The immutable tool-gate root has no user/session data and is rebuilt
        # from render_tool_gate_root() when a Worker starts. Expose it
        # explicitly so StatePool drain does not mistake this reproducible
        # system cache for a dirty user State that must be snapshotted.
        persistent_states["reconstructible"] = int(tool_gate_root_available)
        return {
            "backend": BACKEND,
            "calls": calls,
            "classify_calls": classify_calls,
            "gate_calls": gate_calls,
            "memory_gate_calls": memory_gate_calls,
            "tool_gate_state": tool_gate_metrics,
            "inference": self.engine.health(),
            "persistent_states": persistent_states,
        }

    def close(self) -> None:
        self.engine.close()
        self.states.close()


service: NativeG1I | None = None
worker_agent: StatePoolWorkerAgent | None = None


def _requires_worker_admission(path: str) -> bool:
    """Return whether a POST can begin or mutate inference work.

    Snapshot and release remain available while draining so the Controller can
    make every resident State durable and free its slot. Control-plane routes
    likewise remain reachable for polling.
    """

    if not path.startswith("/v1/"):
        return False
    if path.startswith("/v1/statepool/") or path == "/v1/states/release":
        return False
    if path.startswith("/v1/states/") and path.endswith("/snapshot"):
        return False
    return True


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="RWKV StateServe", version="0.4.0")

    @app.middleware("http")
    async def statepool_admission(request, call_next):
        agent = worker_agent
        admitted = False
        if (
            agent is not None
            and request.method == "POST"
            and _requires_worker_admission(request.url.path)
        ):
            admitted = agent.enter_request()
            if not admitted:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "statepool_worker_draining",
                        "message": "Worker is draining and rejects new inference",
                    },
                    headers={"Retry-After": "1"},
                )
        try:
            response = await call_next(request)
            if admitted and agent is not None and hasattr(response, "body_iterator"):
                original_body = response.body_iterator

                async def release_after_body():
                    nonlocal admitted
                    try:
                        async for chunk in original_body:
                            yield chunk
                    finally:
                        if admitted:
                            admitted = False
                            agent.exit_request()

                response.body_iterator = release_after_body()
            return response
        finally:
            if admitted and agent is not None:
                admitted = False
                agent.exit_request()

    def model_ref() -> dict[str, str]:
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer": TOKENIZER_ID,
            "state_abi": STATE_ABI,
        }

    def require_model_ref(payload: dict[str, Any]) -> None:
        supplied = payload.get("model_ref")
        if supplied != model_ref():
            raise HTTPException(409, "exact State model_ref mismatch")

    def generation_options(
        payload: Mapping[str, Any],
        *,
        chat: bool,
    ) -> tuple[list[str], int, bool, bool]:
        requested_model = payload.get("model")
        if requested_model is not None and requested_model != MODEL_ID:
            raise HTTPException(404, f"model {requested_model!r} is not available")
        for name, expected in (("n", 1), ("best_of", 1)):
            value = payload.get(name, expected)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise HTTPException(422, f"{name} must be {expected}")
        temperature = payload.get("temperature", 0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise HTTPException(422, "temperature must be numeric")
        if float(temperature) != 0.0:
            raise HTTPException(
                422,
                "this RWKV serving profile currently supports greedy temperature=0 only",
            )
        top_p = payload.get("top_p", 1)
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise HTTPException(422, "top_p must be numeric")
        if float(top_p) != 1.0:
            raise HTTPException(
                422,
                "this RWKV serving profile currently supports top_p=1 only",
            )
        if payload.get("logprobs") not in (None, False):
            raise HTTPException(422, "logprobs are not supported")
        if payload.get("echo") not in (None, False):
            raise HTTPException(422, "echo is not supported")
        for field in ("tools", "tool_choice", "functions", "function_call"):
            if payload.get(field) is not None:
                raise HTTPException(
                    422,
                    f"{field} is not supported by the base inference endpoint",
                )
        max_tokens_value = payload.get("max_completion_tokens")
        if max_tokens_value is None:
            max_tokens_value = payload.get("max_tokens", 192)
        if isinstance(max_tokens_value, bool) or not isinstance(max_tokens_value, int):
            raise HTTPException(422, "max_tokens must be an integer")
        if max_tokens_value < 1 or max_tokens_value > 1024:
            raise HTTPException(422, "max_tokens out of range")
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise HTTPException(422, "stream must be boolean")
        stream_options = payload.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, Mapping):
            raise HTTPException(422, "stream_options must be an object")
        include_usage = False
        if stream_options is not None:
            include_usage = stream_options.get("include_usage", False)
            if not isinstance(include_usage, bool):
                raise HTTPException(
                    422,
                    "stream_options.include_usage must be boolean",
                )
        try:
            stops = normalize_stops(payload.get("stop"), chat=chat)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return stops, max_tokens_value, stream, include_usage

    def stream_completion(
        *,
        prompt: str,
        stops: Sequence[str],
        max_tokens: int,
        request_id: str,
        created: int,
        chat: bool,
        include_usage: bool,
        prefix_token_ids: Sequence[int] = (),
    ) -> StreamingResponse:
        def chunk(delta: str, finish_reason: str | None = None) -> dict[str, Any]:
            choice: dict[str, Any] = {
                "index": 0,
                "finish_reason": finish_reason,
            }
            if chat:
                choice["delta"] = {"content": delta} if delta else {}
            else:
                choice["text"] = delta
            return {
                "id": request_id,
                "object": (
                    "chat.completion.chunk" if chat else "text_completion"
                ),
                "created": created,
                "model": MODEL_ID,
                "choices": [choice],
            }

        def body():
            events: queue.Queue[dict[str, Any] | None] = queue.Queue()

            def emit(event: dict[str, Any]) -> None:
                events.put(event)

            def run() -> None:
                try:
                    assert service is not None
                    result = service.complete(
                        prompt,
                        stops,
                        max_tokens,
                        prefix_token_ids,
                        PREFILL_CHUNK_SIZE,
                        event_sink=emit,
                    )
                    events.put({"type": "done", "result": result})
                except Exception as exc:
                    events.put(
                        {
                            "type": "error",
                            "error": {
                                "message": str(exc),
                                "type": type(exc).__name__,
                            },
                        }
                    )
                finally:
                    events.put(None)

            if chat:
                role = chunk("")
                role["choices"][0]["delta"] = {"role": "assistant"}
                yield sse_data(role)
            threading.Thread(
                target=run,
                name="rwkv-openai-stream",
                daemon=True,
            ).start()
            emitted = ""
            while True:
                event = events.get()
                if event is None:
                    return
                if event.get("type") == "delta":
                    text = str(event.get("text") or "")
                    if text.startswith(emitted):
                        delta = text[len(emitted) :]
                        emitted = text
                        if delta:
                            yield sse_data(chunk(delta))
                    continue
                if event.get("type") == "error":
                    yield "event: error\n" + sse_data({"error": event["error"]})
                    yield sse_data("[DONE]")
                    return
                if event.get("type") != "done":
                    continue
                result = event["result"]
                final_text = str(result.get("text") or "")
                if final_text.startswith(emitted):
                    delta = final_text[len(emitted) :]
                    if delta:
                        yield sse_data(chunk(delta))
                yield sse_data(
                    chunk("", openai_finish_reason(result.get("stop_reason")))
                )
                if include_usage:
                    yield sse_data(
                        {
                            "id": request_id,
                            "object": (
                                "chat.completion.chunk"
                                if chat
                                else "text_completion"
                            ),
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [],
                            "usage": completion_usage(result),
                        }
                    )
                yield sse_data("[DONE]")
                return

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.on_event("startup")
    def startup() -> None:
        global service, worker_agent
        service = NativeG1I()
        settings = WorkerSettings.from_environment()
        if settings is not None:
            worker_agent = StatePoolWorkerAgent(settings, service.health)
            worker_agent.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        global service, worker_agent
        if worker_agent is not None:
            worker_agent.stop()
            worker_agent = None
        if service is not None:
            service.close()
            service = None

    @app.get("/live")
    def live() -> dict[str, Any]:
        return {"status": "live"}

    @app.get("/ready")
    def ready():
        if service is None:
            raise HTTPException(503, "loading")
        if worker_agent is not None and not worker_agent.ready():
            raise HTTPException(503, "StatePool Worker is not registered or is draining")
        return {"status": "ready"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        runtime = service.health()
        return {
            "status": "ready",
            "model": MODEL_ID,
            "model_ref": model_ref(),
            "backend": BACKEND,
            "context": CONTEXT,
            "loaded_seconds": round(service.loaded_seconds, 3),
            "pipeline_devices": list(PIPELINE_DEVICES),
            **runtime,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "statepool_worker": (
                worker_agent.status()
                if worker_agent is not None
                else {"enabled": False, "ready": True}
            ),
        }

    @app.get("/v1/statepool/worker")
    def statepool_worker() -> dict[str, Any]:
        if worker_agent is None:
            raise HTTPException(404, "StatePool Worker adapter is disabled")
        return {
            "status": worker_agent.status(),
            "capability": worker_agent.capability(),
        }

    @app.post("/v1/statepool/drain")
    def statepool_drain(payload: dict[str, Any]) -> dict[str, Any]:
        if worker_agent is None:
            raise HTTPException(404, "StatePool Worker adapter is disabled")
        timeout_seconds = float(payload.get("timeout_seconds", 120))
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            raise HTTPException(422, "timeout_seconds must be in (0, 3600]")
        return worker_agent.begin_draining(timeout_seconds=timeout_seconds)

    @app.get("/v1/statepool/drain")
    def statepool_drain_status() -> dict[str, Any]:
        if worker_agent is None:
            raise HTTPException(404, "StatePool Worker adapter is disabled")
        return worker_agent.drain_status()

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
    def completions(payload: dict[str, Any]):
        if service is None:
            raise HTTPException(503, "loading")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(422, "prompt must be non-empty string")
        stop, max_tokens, stream, include_usage = generation_options(
            payload,
            chat=False,
        )
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
        created = int(time.time())
        request_id = "cmpl-rwkv-" + uuid.uuid4().hex
        if stream:
            return stream_completion(
                prompt=prompt,
                stops=stop,
                max_tokens=max_tokens,
                request_id=request_id,
                created=created,
                chat=False,
                include_usage=include_usage,
                prefix_token_ids=prefix,
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
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "text": result["text"],
                    "finish_reason": openai_finish_reason(result["stop_reason"]),
                }
            ],
            "usage": completion_usage(result),
            "g1i": result,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]):
        if service is None:
            raise HTTPException(503, "loading")
        try:
            prompt = render_chat_prompt(payload.get("messages"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        stop, max_tokens, stream, include_usage = generation_options(
            payload,
            chat=True,
        )
        created = int(time.time())
        request_id = "chatcmpl-rwkv-" + uuid.uuid4().hex
        if stream:
            return stream_completion(
                prompt=prompt,
                stops=stop,
                max_tokens=max_tokens,
                request_id=request_id,
                created=created,
                chat=True,
                include_usage=include_usage,
            )
        try:
            result = service.complete(
                prompt,
                stop,
                max_tokens,
                (),
                PREFILL_CHUNK_SIZE,
            )
        except (TimeoutError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": openai_finish_reason(result["stop_reason"]),
                }
            ],
            "usage": completion_usage(result),
            "rwkv": {
                "batch_mode": result.get("batch_mode"),
                "elapsed_ms": result.get("elapsed_ms"),
                "queue_ms": result.get("queue_ms"),
            },
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

    @app.post("/v1/states/batch_prefill")
    def state_batch_prefill(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        items = payload.get("items")
        if not isinstance(items, list):
            raise HTTPException(422, "items must be an object array")
        try:
            states = service.states.prefill_many(items=items)
            return {"status": "ok", "states": states}
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

    @app.post("/v1/states/stream_continue")
    def state_stream_continue(payload: dict[str, Any]) -> StreamingResponse:
        if service is None:
            raise HTTPException(503, "loading")
        items = payload.get("items")
        stops = payload.get("stop", [])
        if not isinstance(items, list) or len(items) != 1:
            raise HTTPException(422, "streaming requires exactly one state item")
        if isinstance(stops, str):
            stops = [stops]
        if not isinstance(stops, list) or not all(
            isinstance(value, str) for value in stops
        ):
            raise HTTPException(422, "stop must be a string array")
        owner_id = str(payload.get("owner_id") or "")
        max_tokens = int(payload.get("max_tokens", 192))

        def body():
            events: queue.Queue[dict[str, Any] | None] = queue.Queue()

            def emit(event: dict[str, Any]) -> None:
                events.put(event)

            def run() -> None:
                try:
                    results = service.states.continue_many(
                        owner_id=owner_id,
                        items=items,
                        stops=stops,
                        max_tokens=max_tokens,
                        event_sink=emit,
                    )
                    events.put({"type": "done", "results": results})
                except Exception as exc:
                    events.put(
                        {
                            "type": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    events.put(None)

            threading.Thread(
                target=run,
                name="rwkv-state-stream",
                daemon=True,
            ).start()
            while True:
                event = events.get()
                if event is None:
                    return
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(
            body(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

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

    @app.post("/v1/states/{state_id}/snapshot")
    def state_snapshot(state_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        require_model_ref(payload)
        if payload.get("target_tier", "cpu") != "cpu":
            raise HTTPException(422, "Sidecar snapshot target_tier must be cpu")
        try:
            snapshot = service.states.snapshot(
                owner_id=str(payload.get("owner_id") or ""),
                state_id=state_id,
            )
            checkpoint_id = "checkpoint-" + snapshot["checksum"].removeprefix(
                "sha256:"
            )[:32]
            return {
                "status": "ok",
                "checkpoint": {
                    "checkpoint_id": checkpoint_id,
                    "model_ref": model_ref(),
                    "provider_mode": "rwkv_recurrent",
                    "placement": "cpu",
                    "checksum": snapshot["checksum"],
                    "size_bytes": snapshot["size_bytes"],
                    "atomic": True,
                    "seen_tokens": snapshot["seen_tokens"],
                },
                "payload_base64": base64.b64encode(snapshot["payload"]).decode("ascii"),
            }
        except Exception as exc:
            state_error(exc)
            raise AssertionError("unreachable")

    @app.post("/v1/states/restore")
    def state_restore(payload: dict[str, Any]) -> dict[str, Any]:
        if service is None:
            raise HTTPException(503, "loading")
        require_model_ref(payload)
        encoded = payload.get("payload_base64")
        if not isinstance(encoded, str) or not encoded:
            raise HTTPException(422, "payload_base64 must be a non-empty string")
        if len(encoded) > ((MAX_SNAPSHOT_BYTES + 2) // 3) * 4 + 4:
            raise HTTPException(413, "snapshot exceeds configured byte limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(422, "payload_base64 is invalid") from exc
        expected_checksum = str(payload.get("checksum") or "")
        actual_checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
        if expected_checksum != actual_checksum:
            raise HTTPException(422, "snapshot checksum mismatch")
        try:
            state = service.states.restore(
                owner_id=str(payload.get("owner_id") or ""),
                payload=raw,
                branch=(
                    str(payload["branch"])
                    if payload.get("branch") is not None
                    else None
                ),
            )
            return {"status": "ok", "state": state}
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
