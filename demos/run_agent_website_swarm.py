#!/usr/bin/env python3
"""Run the Gate 4 website swarm through the real Rust Agent Controller.

Every logical job has a unique prompt, Controller session/owner, RWKV State and
sandbox workspace.  RWKV autonomously calls ``run_command`` to create and
validate a compact content artifact.  This runner owns only task scheduling,
strict validation, deterministic HTML rendering, telemetry and evidence.
It never calls the model or the Sidecar generation API directly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from demos.generate_website_swarm import (
    Generation,
    WebsiteSpec,
    build_specs,
    render_gallery,
    render_html,
    validate_html,
)


SCHEMA = "rwkv-agent-gate4-website-swarm.v1"
CONTENT_KEYS = {"title", "headline", "summary", "cta", "features"}
PROTOCOL_MARKERS = ("<tool_call>", "<tool_result>", "<think>", "System:", "Assistant:")
ACTIVE_STATUSES = {
    "prefilling",
    "model_action",
    "tool_execution",
    "observation",
    "validating",
    "rendering",
    "retrying",
}
PALETTES = (
    ("#090D18", "#121B2D", "#77E4C8", "#F4F8FF", "#8B7CFF"),
    ("#120D0A", "#211713", "#FFB86B", "#FFF7EF", "#E96E7C"),
    ("#071513", "#102521", "#95E06C", "#EDFFF7", "#31C4A4"),
    ("#111014", "#1D1A22", "#D7C7FF", "#FAF7FF", "#FF84C1"),
    ("#081218", "#10232D", "#65D8E6", "#F0FBFF", "#F7C86B"),
    ("#151008", "#281D0F", "#F8D477", "#FFF9EA", "#D78C4A"),
    ("#0E1012", "#1A1E21", "#E9EEF2", "#F8FAFC", "#6EE7A8"),
    ("#110B18", "#22152F", "#CDA1FF", "#FFF4FF", "#6AF0DC"),
)


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    spec: WebsiteSpec
    owner_id: str
    session_id: str
    workspace: str
    prompt: str
    prompt_sha256: str


def percentile(values: Sequence[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    return rows[max(0, math.ceil(quantile * len(rows)) - 1)]


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_task_prompt(spec: WebsiteSpec, *, repair_errors: Sequence[str] = ()) -> str:
    repair = ""
    if repair_errors:
        repair = (
            " content.json already exists but failed the runtime checks: "
            + ", ".join(repair_errors[:8])
            + ". Inspect it, replace it with a corrected version, and validate again."
        )
    return (
        "Create only content.json for one original landing page in the current isolated workspace. "
        "This is an artifact task, not a chat answer. Use one run_command containing both actions in order: "
        "first write content.json with a quoted heredoc; then run python3 -c to parse that same file, assert "
        "that it has five keys and exactly three features, and print the exact line VALID. The combined command "
        "must really execute both actions. Do not create index.html or any other file. "
        "The JSON must have exactly these keys: title, headline, summary, cta, features. "
        f"title must contain the exact brand {spec.brand}. headline must be 2 to 20 words. "
        "summary must be 2 to 100 words. cta must be 1 to 8 words. features must be an array of exactly "
        "3 short phrases. All values must be original English plain text on one line, with no "
        "Markdown, HTML, protocol tags, apostrophes, or double quote characters inside values. "
        f"Site identity: {spec.site_id}. Category: {spec.category}. Audience: {spec.audience}. "
        f"Visual mood: {spec.mood}. Invent copy specific to this identity; never use another site brand."
        + repair
    )


def build_identities(specs: Sequence[WebsiteSpec], run_id: str) -> list[TaskIdentity]:
    identities = []
    for spec in specs:
        prompt = build_task_prompt(spec)
        session_id = f"{run_id}-{spec.site_id}"
        workspace = f"gate4/{run_id}/{spec.site_id}"
        identities.append(
            TaskIdentity(
                spec=spec,
                owner_id="gate4-" + sha256(f"{run_id}:{spec.site_id}".encode()).hexdigest()[:24],
                session_id=session_id,
                workspace=workspace,
                prompt=prompt,
                prompt_sha256=sha256(prompt.encode()).hexdigest(),
            )
        )
    return identities


def _plain(value: Any, path: str, errors: list[str], *, minimum: int, maximum: int) -> str:
    if not isinstance(value, str):
        errors.append(f"{path}:not_string")
        return ""
    text = " ".join(value.split())
    if not minimum <= len(text) <= maximum:
        errors.append(f"{path}:length")
    if any(marker.casefold() in text.casefold() for marker in PROTOCOL_MARKERS) or "<" in text or ">" in text:
        errors.append(f"{path}:protocol_or_markup")
    return text


def validate_content(value: Any, spec: WebsiteSpec, all_brands: Sequence[str]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["root:not_object"]
    if set(value) != CONTENT_KEYS:
        errors.append("root:keys")
    title = _plain(value.get("title"), "title", errors, minimum=3, maximum=80)
    headline = _plain(value.get("headline"), "headline", errors, minimum=2, maximum=160)
    summary = _plain(value.get("summary"), "summary", errors, minimum=2, maximum=600)
    cta = _plain(value.get("cta"), "cta", errors, minimum=1, maximum=80)
    if spec.brand.casefold() not in title.casefold():
        errors.append("title:missing_brand")
    for other in all_brands:
        if other != spec.brand and other.casefold() in title.casefold():
            errors.append("title:foreign_brand")
            break
    word_ranges = (("headline", headline, 2, 20), ("summary", summary, 2, 100), ("cta", cta, 1, 8))
    for path, text, minimum, maximum in word_ranges:
        count = len(text.split())
        if not minimum <= count <= maximum:
            errors.append(f"{path}:words")
    raw_features = value.get("features")
    features: list[str] = []
    if not isinstance(raw_features, list) or len(raw_features) != 3:
        errors.append("features:count")
    else:
        for index, item in enumerate(raw_features):
            features.append(_plain(item, f"features.{index}", errors, minimum=1, maximum=160))
    if errors:
        return None, errors
    return {"title": title, "headline": headline, "summary": summary, "cta": cta, "features": features}, []


def content_to_dsl(spec: WebsiteSpec, content: dict[str, Any]) -> dict[str, Any]:
    background, surface, primary, text, accent = PALETTES[(spec.index - 1) % len(PALETTES)]
    feature_counts: dict[str, int] = {}
    feature_names = []
    for name in content["features"]:
        key = name.casefold()
        feature_counts[key] = feature_counts.get(key, 0) + 1
        suffix = f" {feature_counts[key]}" if list(map(str.casefold, content["features"])).count(key) > 1 else ""
        feature_names.append(name + suffix)
    features = [
        {
            "title": name,
            "description": f"{name} shaped for {spec.audience} in a {spec.mood} experience.",
        }
        for name in feature_names
    ]
    branded_summary = f"{spec.brand} — {content['summary']}"
    return {
        "title": content["title"],
        "tagline": branded_summary,
        "theme": {
            "background": background,
            "surface": surface,
            "primary": primary,
            "text": text,
            "accent": accent,
        },
        "hero": {
            "eyebrow": spec.category,
            "headline": content["headline"],
            "summary": branded_summary,
            "cta": content["cta"],
        },
        "features": features,
        "stats": [
            {"value": "LOCAL", "label": "private inference"},
            {"value": "1", "label": "independent RWKV State"},
            {"value": f"{spec.index:03d}", "label": "swarm artifact"},
        ],
        "footer": f"{spec.brand} — generated locally by RWKV.",
    }


class HttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get(self, path: str) -> dict[str, Any]:
        with urlopen(self.base_url + path, timeout=self.timeout_seconds) as response:
            return json.load(response)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {body[:1000]}") from error


class TaskWall:
    def __init__(self, output: Path, identities: Sequence[TaskIdentity], *, run_id: str, concurrency: int) -> None:
        self.output = output
        self.lock = threading.Lock()
        self.started = time.time()
        self.run_id = run_id
        self.concurrency = concurrency
        self.run_status = "initializing"
        self.runtime: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.tasks = {
            item.spec.site_id: {
                "site_id": item.spec.site_id,
                "brand": item.spec.brand,
                "category": item.spec.category,
                "mood": item.spec.mood,
                "session_id": item.session_id,
                "workspace": item.workspace,
                "prompt_sha256": item.prompt_sha256,
                "status": "queued",
                "attempt": 0,
                "elapsed_ms": 0.0,
                "tool_steps": 0,
                "error": "",
                "gallery": "",
            }
            for item in identities
        }
        self.flush()

    def set_run_status(self, status: str) -> None:
        with self.lock:
            self.run_status = status
            self._flush_locked()

    def update_task(self, site_id: str, **values: Any) -> None:
        with self.lock:
            self.tasks[site_id].update(values)
            event = {"time": round(time.time(), 3), "site_id": site_id, "status": values.get("status", "")}
            if values.get("error"):
                event["error"] = str(values["error"])[:180]
            self.events.append(event)
            self.events = self.events[-80:]
            self._flush_locked()

    def update_runtime(self, values: dict[str, Any]) -> None:
        with self.lock:
            self.runtime.update(values)
            self._flush_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()

    def flush(self) -> None:
        with self.lock:
            self._flush_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        rows = list(self.tasks.values())
        counts = {
            "total": len(rows),
            "completed": sum(row["status"] == "done" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "active": sum(row["status"] in ACTIVE_STATUSES for row in rows),
            "queued": sum(row["status"] == "queued" for row in rows),
        }
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "run_status": self.run_status,
            "updated_at_unix": round(time.time(), 3),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "physical_concurrency": self.concurrency,
            "counts": counts,
            "runtime": self.runtime,
            "tasks": rows,
            "events": list(reversed(self.events[-20:])),
        }

    def _flush_locked(self) -> None:
        atomic_write(self.output / "tasks.json", json.dumps(self._snapshot_locked(), ensure_ascii=False, indent=2) + "\n")


def rocm_sample() -> dict[str, Any]:
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = list(csv.DictReader(io.StringIO(completed.stdout.strip())))
    if not rows:
        raise RuntimeError("rocm-smi returned no GPU rows")
    row = rows[0]
    return {
        "gpu_busy_pct": float(row["GPU use (%)"]),
        "vram_total_bytes": int(row["VRAM Total Memory (B)"]),
        "vram_used_bytes": int(row["VRAM Total Used Memory (B)"]),
    }


class TelemetrySampler:
    def __init__(self, sidecar: HttpClient, wall: TaskWall, output: Path, interval: float) -> None:
        self.sidecar = sidecar
        self.wall = wall
        self.output = output
        self.interval = max(0.1, interval)
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gate4-telemetry", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=20)
        with (self.output / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _run(self) -> None:
        while not self.stop_event.is_set():
            started = time.time()
            try:
                health = self.sidecar.get("/health")
                persistent = dict(health.get("persistent_states") or {})
                batching = dict(persistent.get("batching") or {})
                inference = dict(health.get("inference") or {})
                scheduler = dict(inference.get("scheduler") or {})
                sample = {
                    "timestamp_unix": round(started, 3),
                    **rocm_sample(),
                    "resident_states": int(persistent.get("allocated") or 0),
                    "active_state_rows": int(batching.get("active_rows") or 0),
                    "waiting_jobs": int(inference.get("waiting") or 0),
                    "prefilling_rows": int(inference.get("prefilling") or 0),
                    "decoding_rows": int(inference.get("decoding") or 0),
                    "decode_tokens": int(scheduler.get("decode_tokens") or 0),
                    "max_batch_observed": int(scheduler.get("max_batch_observed") or 0),
                }
                self.samples.append(sample)
                self.wall.update_runtime(sample)
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}"[:300])
            self.stop_event.wait(max(0.0, self.interval - (time.time() - started)))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


def start_dashboard(output: Path, host: str, port: int) -> ThreadingHTTPServer | None:
    if port <= 0:
        return None
    server = ThreadingHTTPServer((host, port), partial(QuietHandler, directory=str(output)))
    threading.Thread(target=server.serve_forever, name="gate4-dashboard", daemon=True).start()
    return server


def _trace_summary(response: dict[str, Any]) -> dict[str, Any]:
    agent = dict((response.get("trace") or {}).get("agent") or {})
    events = list(agent.get("events") or [])
    opened = [str(row.get("state_id")) for row in events if row.get("type") == "state_opened"]
    released = [
        str(row.get("state_id"))
        for row in events
        if row.get("type") == "state_released" and row.get("success") is True
    ]
    owners = [str(row.get("owner_id")) for row in events if row.get("type") == "run_started"]
    tool_steps = list(agent.get("tool_steps") or [])
    validated = any(
        "VALID" in str((step.get("result") or {}).get("stdout") or "")
        for step in tool_steps
    )
    answer = str(response.get("answer") or "")
    leak = any(marker.casefold() in answer.casefold() for marker in PROTOCOL_MARKERS)
    return {
        "owners": owners,
        "opened_states": opened,
        "released_states": released,
        "all_states_released": bool(opened) and sorted(opened) == sorted(released),
        "tool_steps": len(tool_steps),
        "runtime_validator_passed": validated,
        "protocol_leak": leak,
        "model_turns": int(agent.get("model_turns") or 0),
    }


def _read_content(path: Path, spec: WebsiteSpec, all_brands: Sequence[str]) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, ["content:missing"]
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"content:{type(exc).__name__}:{exc}"[:180]]
    return validate_content(value, spec, all_brands)


def run_task(
    identity: TaskIdentity,
    *,
    controller: HttpClient,
    workspace_root: Path,
    output: Path,
    all_brands: Sequence[str],
    repair_attempts: int,
    wall: TaskWall,
) -> dict[str, Any]:
    spec = identity.spec
    workspace_path = workspace_root / identity.workspace
    attempts: list[dict[str, Any]] = []
    errors: list[str] = []
    content: dict[str, Any] | None = None
    final_response: dict[str, Any] = {}
    trace: dict[str, Any] = {}
    started = time.perf_counter()
    for attempt in range(1, repair_attempts + 2):
        prompt = identity.prompt if attempt == 1 else build_task_prompt(spec, repair_errors=errors)
        wall.update_task(
            spec.site_id,
            status="model_action" if attempt == 1 else "retrying",
            attempt=attempt,
            error="" if attempt == 1 else ", ".join(errors[:4]),
        )
        attempt_started = time.perf_counter()
        response_error = ""
        try:
            final_response = controller.post(
                "/v1/agent/run",
                {"message": prompt, "session_id": f"{identity.session_id}-a{attempt}", "working_directory": identity.workspace},
            )
        except Exception as exc:
            final_response = {}
            response_error = f"controller:{type(exc).__name__}:{exc}"[:500]
        wall.update_task(spec.site_id, status="validating")
        content, content_errors = _read_content(workspace_path / "content.json", spec, all_brands)
        trace = _trace_summary(final_response)
        errors = list(content_errors)
        if response_error:
            errors.append(response_error)
        if final_response.get("status") != "ok":
            errors.append(f"controller_status:{final_response.get('status', 'missing')}")
        if trace.get("tool_steps", 0) < 1:
            errors.append("trace:missing_artifact_tool_step")
        if not trace.get("runtime_validator_passed"):
            errors.append("trace:validator_missing_VALID")
        if not trace.get("all_states_released"):
            errors.append("trace:state_release")
        if trace.get("protocol_leak"):
            errors.append("trace:protocol_leak")
        attempts.append(
            {
                "attempt": attempt,
                "prompt": prompt,
                "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                "elapsed_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                "errors": errors,
                "trace": trace,
                "response": final_response,
            }
        )
        if not errors and content is not None:
            break

    if errors or content is None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        wall.update_task(
            spec.site_id,
            status="failed",
            elapsed_ms=round(elapsed_ms, 3),
            tool_steps=sum(int(row["trace"].get("tool_steps") or 0) for row in attempts),
            error=", ".join(errors[:6]),
        )
        return {
            "spec": asdict(spec),
            "identity": asdict(identity),
            "status": "failed",
            "errors": errors,
            "attempts": attempts,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    wall.update_task(spec.site_id, status="rendering")
    dsl = content_to_dsl(spec, content)
    html = render_html(spec, dsl)
    html_errors = validate_html(html)
    if html_errors:
        errors.extend(f"html:{error}" for error in html_errors)
    if not errors:
        artifact_dir = workspace_path
        atomic_write(artifact_dir / "index.html", html)
        atomic_write(artifact_dir / "design.json", json.dumps(dsl, ensure_ascii=False, indent=2) + "\n")
        gallery_dir = output / "gallery" / spec.site_id
        gallery_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_dir / "index.html", gallery_dir / "index.html")
        shutil.copy2(artifact_dir / "content.json", gallery_dir / "content.json")
        shutil.copy2(artifact_dir / "design.json", gallery_dir / "design.json")
    elapsed_ms = (time.perf_counter() - started) * 1000
    status = "done" if not errors else "failed"
    wall.update_task(
        spec.site_id,
        status=status,
        elapsed_ms=round(elapsed_ms, 3),
        tool_steps=sum(int(row["trace"].get("tool_steps") or 0) for row in attempts),
        error=", ".join(errors[:6]),
        gallery=f"gallery/{spec.site_id}/index.html" if status == "done" else "",
    )
    return {
        "spec": asdict(spec),
        "identity": asdict(identity),
        "status": status,
        "content": content,
        "dsl": dsl,
        "errors": errors,
        "attempts": attempts,
        "elapsed_ms": round(elapsed_ms, 3),
    }


STATE_AGENT_ROOT = (
    "System: You are a bounded RWKV workspace agent. Complete the supplied artifact task with run_command. "
    "Output exactly one strict envelope per turn: <tool_call>{\"name\":\"run_command\",\"arguments\":"
    "{\"command\":\"...\"}}</tool_call> or <answer>concise result</answer>. Output no reasoning, roles, "
    "Markdown fences, empty commands, or text outside the envelope. The selected workspace is /workspace; "
    "only its files persist. There is no network or package installation. Python is available as python3. "
    "There are no read_file, write_file, edit_file, list_files, or shell functions. Use run_command for all "
    "workspace actions and inspect the real Tool Result before finalizing. One tool call per turn."
)
RUN_COMMAND_JSON_PREFIX = '{"name":"run_command","arguments":{"command":"'


def state_root_prompt(
    identity: TaskIdentity,
    *,
    recovery_attempt: int = 0,
    recovery_errors: Sequence[str] = (),
) -> str:
    recovery = ""
    if recovery_attempt:
        recovery = (
            f"\nSystem: This is fresh recovery State {recovery_attempt}. A previous independent State failed: "
            f"{', '.join(recovery_errors[:8])}. Do not repeat its invalid output. Because the Assistant prefix "
            "already contains <tool_call>, begin with a JSON object whose first field is name and whose name "
            "is run_command. Replace the existing content.json and validate it."
        )
    return (
        STATE_AGENT_ROOT
        + f"\nSystem: This is an independent State owned only by {identity.spec.site_id} / {identity.spec.brand}. "
        "Never use content, names, or observations from another State."
        + recovery
    )


def parse_run_command(raw: str, stop_reason: str) -> tuple[dict[str, Any] | None, list[str]]:
    value = raw.strip()
    if value.startswith("<tool_call>"):
        value = value[len("<tool_call>") :]
    if value.endswith("</tool_call>"):
        value = value[: -len("</tool_call>")]
    errors: list[str] = []
    if stop_reason != "</tool_call>":
        errors.append(f"tool_stop:{stop_reason or 'missing'}")
    try:
        parsed, end = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError as exc:
        return None, errors + [f"tool_json:{exc.msg}@{exc.pos}"]
    if value[end:].strip():
        errors.append("tool_json:trailing")
    if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
        errors.append("tool_json:envelope")
        return None, errors
    arguments = parsed.get("arguments")
    if parsed.get("name") != "run_command" or not isinstance(arguments, dict) or set(arguments) != {"command"}:
        errors.append("tool_json:run_command_schema")
        return None, errors
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        errors.append("tool_json:empty_command")
        return None, errors
    return {"name": "run_command", "arguments": {"command": command}}, errors


def _continue_one(
    sidecar: HttpClient,
    identity: TaskIdentity,
    state_id: str,
    input_text: str,
    *,
    stop: str,
    max_tokens: int,
    barrier: threading.Barrier | None,
) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait(timeout=60)
    response = sidecar.post(
        "/v1/states/batch_continue",
        {
            "owner_id": identity.owner_id,
            "items": [{"state_id": state_id, "input": input_text}],
            "stop": [stop],
            "max_tokens": max_tokens,
        },
    )
    return dict(response["results"][0])


def continue_wave(
    sidecar: HttpClient,
    rows: Sequence[tuple[TaskIdentity, str, str]],
    *,
    stop: str,
    max_tokens: int,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for start in range(0, len(rows), concurrency):
        chunk = list(rows[start : start + concurrency])
        barrier = threading.Barrier(len(chunk)) if len(chunk) > 1 else None
        with ThreadPoolExecutor(max_workers=len(chunk), thread_name_prefix="gate4-state") as executor:
            futures = {
                executor.submit(
                    _continue_one,
                    sidecar,
                    identity,
                    state_id,
                    input_text,
                    stop=stop,
                    max_tokens=max_tokens,
                    barrier=barrier,
                ): identity
                for identity, state_id, input_text in chunk
            }
            for future in as_completed(futures):
                identity = futures[future]
                try:
                    results[identity.spec.site_id] = future.result()
                except Exception as exc:
                    results[identity.spec.site_id] = {
                        "state_id": next(state for item, state, _input in chunk if item == identity),
                        "text": "",
                        "stop_reason": "error",
                        "token_ids": [],
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
    return results


def execute_tools(
    controller: HttpClient,
    rows: Sequence[tuple[TaskIdentity, dict[str, Any]]],
    *,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    def execute(identity: TaskIdentity, call: dict[str, Any]) -> dict[str, Any]:
        return controller.post(
            "/v1/tools/call",
            {
                "name": call["name"],
                "arguments": call["arguments"],
                "session_id": identity.session_id,
                "working_directory": identity.workspace,
            },
        )

    output: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, len(rows)), thread_name_prefix="gate4-tool") as executor:
        futures = {executor.submit(execute, identity, call): identity for identity, call in rows}
        for future in as_completed(futures):
            identity = futures[future]
            try:
                output[identity.spec.site_id] = future.result()
            except Exception as exc:
                output[identity.spec.site_id] = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}"[:500],
                }
    return output


def release_state_rows(
    sidecar: HttpClient,
    identities: Sequence[TaskIdentity],
    states: dict[str, str],
    *,
    concurrency: int,
) -> dict[str, bool]:
    def release(identity: TaskIdentity) -> bool:
        response = sidecar.post(
            "/v1/states/release",
            {"owner_id": identity.owner_id, "state_ids": [states[identity.spec.site_id]]},
        )
        return int(response.get("released") or 0) == 1

    output: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, len(identities)), thread_name_prefix="gate4-release") as executor:
        futures = {executor.submit(release, identity): identity for identity in identities}
        for future in as_completed(futures):
            identity = futures[future]
            try:
                output[identity.spec.site_id] = bool(future.result())
            except Exception:
                output[identity.spec.site_id] = False
    return output


def run_state_swarm(
    identities: Sequence[TaskIdentity],
    *,
    controller: HttpClient,
    sidecar: HttpClient,
    workspace_root: Path,
    output: Path,
    wall: TaskWall,
    concurrency: int,
    repair_attempts: int,
) -> list[dict[str, Any]]:
    all_brands = [identity.spec.brand for identity in identities]
    by_site = {identity.spec.site_id: identity for identity in identities}
    for identity in identities:
        wall.update_task(identity.spec.site_id, status="prefilling")
    prefill = sidecar.post(
        "/v1/states/batch_prefill",
        {
            "items": [
                {
                    "owner_id": identity.owner_id,
                    "prompt": state_root_prompt(identity),
                    "branch": identity.spec.site_id,
                }
                for identity in identities
            ]
        },
    )
    state_rows = list(prefill.get("states") or [])
    if len(state_rows) != len(identities):
        raise RuntimeError(f"batch prefill returned {len(state_rows)} states for {len(identities)} tasks")
    states = {
        identity.spec.site_id: str(state["state_id"])
        for identity, state in zip(identities, state_rows, strict=True)
    }
    results = {
        identity.spec.site_id: {
            "spec": asdict(identity.spec),
            "identity": asdict(identity),
            "status": "running",
            "content": None,
            "dsl": None,
            "errors": [],
            "attempts": [],
            "elapsed_ms": 0.0,
        }
        for identity in identities
    }
    started = {identity.spec.site_id: time.perf_counter() for identity in identities}
    active = list(identities)
    try:
        for attempt_index in range(1, repair_attempts + 2):
            if not active:
                break
            continuation_rows = []
            prompt_by_site: dict[str, str] = {}
            forced_prefix_by_site: dict[str, str] = {}
            for identity in active:
                site_id = identity.spec.site_id
                wall.update_task(
                    site_id,
                    status="model_action" if attempt_index == 1 else "retrying",
                    attempt=attempt_index,
                    error=", ".join(results[site_id]["errors"][:4]),
                )
                prompt = identity.prompt if attempt_index == 1 else build_task_prompt(
                    identity.spec,
                    repair_errors=results[site_id]["errors"],
                )
                prompt_by_site[site_id] = prompt
                forced_prefix = RUN_COMMAND_JSON_PREFIX if attempt_index > 1 else ""
                forced_prefix_by_site[site_id] = forced_prefix
                input_text = (
                    f"\n\nUser: Workspace task:\n{prompt}\n\nAssistant: <tool_call>{forced_prefix}"
                )
                continuation_rows.append((identity, states[site_id], input_text))
            generated = continue_wave(
                sidecar,
                continuation_rows,
                stop="</tool_call>",
                max_tokens=384,
                concurrency=concurrency,
            )
            calls: list[tuple[TaskIdentity, dict[str, Any]]] = []
            attempt_by_site: dict[str, dict[str, Any]] = {}
            for identity in active:
                site_id = identity.spec.site_id
                prompt = prompt_by_site[site_id]
                row = generated[site_id]
                raw_generation = forced_prefix_by_site[site_id] + str(row.get("text") or "")
                call, errors = parse_run_command(raw_generation, str(row.get("stop_reason") or ""))
                if row.get("error"):
                    errors.append(f"model:{row['error']}")
                record = {
                    "attempt": attempt_index,
                    "prompt": prompt,
                    "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                    "elapsed_ms": float(row.get("elapsed_ms") or 0.0),
                    "errors": list(errors),
                    "trace": {
                        "owners": [identity.owner_id] if attempt_index == 1 else [],
                        "opened_states": [states[site_id]],
                        "released_states": [],
                        "all_states_released": False,
                        "tool_steps": 1 if call is not None else 0,
                        "runtime_validator_passed": False,
                        "protocol_leak": any(marker.casefold() in str(row.get("text") or "").casefold() for marker in ("<think>", "System:", "Assistant:")),
                        "model_turns": 1,
                    },
                    "response": {
                        "generation": row,
                        "forced_prefix": forced_prefix_by_site[site_id],
                    },
                    "tool_result": None,
                }
                attempt_by_site[site_id] = record
                results[site_id]["attempts"].append(record)
                if call is not None and not errors:
                    calls.append((identity, call))
                else:
                    results[site_id]["errors"] = list(errors)
            for identity, _call in calls:
                wall.update_task(identity.spec.site_id, status="tool_execution")
            tool_results = execute_tools(controller, calls, concurrency=concurrency) if calls else {}
            next_active = []
            for identity in active:
                site_id = identity.spec.site_id
                record = attempt_by_site[site_id]
                tool_result = tool_results.get(site_id)
                record["tool_result"] = tool_result
                record["response"]["tool_result"] = tool_result
                errors = list(record["errors"])
                if tool_result is None:
                    errors.append("tool:not_executed")
                else:
                    stdout = str(tool_result.get("stdout") or "")
                    valid_signal = tool_result.get("status") == "ok" and any(
                        line.strip() == "VALID" for line in stdout.splitlines()
                    )
                    record["trace"]["runtime_validator_passed"] = valid_signal
                    if not valid_signal:
                        errors.append("tool:missing_VALID")
                wall.update_task(site_id, status="validating")
                content, content_errors = _read_content(
                    workspace_root / identity.workspace / "content.json",
                    identity.spec,
                    all_brands,
                )
                errors.extend(content_errors)
                record["errors"] = errors
                results[site_id]["errors"] = errors
                if errors or content is None:
                    next_active.append(identity)
                    continue
                dsl = content_to_dsl(identity.spec, content)
                html = render_html(identity.spec, dsl)
                html_errors = validate_html(html)
                if html_errors:
                    results[site_id]["errors"] = [f"html:{value}" for value in html_errors]
                    record["errors"] = results[site_id]["errors"]
                    next_active.append(identity)
                    continue
                wall.update_task(site_id, status="rendering")
                artifact_dir = workspace_root / identity.workspace
                atomic_write(artifact_dir / "index.html", html)
                atomic_write(artifact_dir / "design.json", json.dumps(dsl, ensure_ascii=False, indent=2) + "\n")
                gallery_dir = output / "gallery" / site_id
                gallery_dir.mkdir(parents=True, exist_ok=True)
                for name in ("index.html", "content.json", "design.json"):
                    shutil.copy2(artifact_dir / name, gallery_dir / name)
                results[site_id]["content"] = content
                results[site_id]["dsl"] = dsl
                results[site_id]["errors"] = []
            active = next_active
            if active and attempt_index <= repair_attempts:
                released_for_recovery = release_state_rows(sidecar, active, states, concurrency=concurrency)
                for identity in active:
                    site_id = identity.spec.site_id
                    record = attempt_by_site[site_id]
                    if released_for_recovery.get(site_id):
                        record["trace"]["released_states"] = [states[site_id]]
                        record["trace"]["all_states_released"] = True
                    else:
                        record["errors"].append("state:recovery_release_failed")
                        results[site_id]["errors"] = list(record["errors"])
                release_failures = [
                    identity.spec.site_id
                    for identity in active
                    if not released_for_recovery.get(identity.spec.site_id)
                ]
                if release_failures:
                    raise RuntimeError(f"failed to release recovery states: {release_failures}")
                recovery_prefill = sidecar.post(
                    "/v1/states/batch_prefill",
                    {
                        "items": [
                            {
                                "owner_id": identity.owner_id,
                                "prompt": state_root_prompt(
                                    identity,
                                    recovery_attempt=attempt_index + 1,
                                    recovery_errors=results[identity.spec.site_id]["errors"],
                                ),
                                "branch": f"{identity.spec.site_id}-recovery-{attempt_index + 1}",
                            }
                            for identity in active
                        ]
                    },
                )
                recovery_rows = list(recovery_prefill.get("states") or [])
                if len(recovery_rows) != len(active):
                    raise RuntimeError(
                        f"recovery prefill returned {len(recovery_rows)} states for {len(active)} tasks"
                    )
                for identity, state in zip(active, recovery_rows, strict=True):
                    states[identity.spec.site_id] = str(state["state_id"])

        successful = [
            identity for identity in identities if results[identity.spec.site_id]["content"] is not None
        ]
        for identity in successful:
            wall.update_task(identity.spec.site_id, status="observation")
        final_rows = continue_wave(
            sidecar,
            [
                (
                    identity,
                    states[identity.spec.site_id],
                    "\n\nTool: <tool_result>{\"status\":\"ok\",\"stdout\":\"VALID\"}</tool_result>"
                    "\n\nUser: The real artifact validator passed. Return the final status word DONE and nothing else."
                    "\n\nAssistant: <answer>",
                )
                for identity in successful
            ],
            stop="</answer>",
            max_tokens=32,
            concurrency=concurrency,
        ) if successful else {}
        for identity in identities:
            site_id = identity.spec.site_id
            if results[site_id]["content"] is None:
                results[site_id]["status"] = "failed"
                continue
            final = final_rows[site_id]
            final_text = str(final.get("text") or "").strip()
            final_ok = str(final.get("stop_reason") or "") == "</answer>" and final_text == "DONE"
            results[site_id]["attempts"][-1]["response"]["final"] = final
            if not final_ok:
                results[site_id]["status"] = "failed"
                results[site_id]["errors"] = [f"final_protocol:{final_text[:80]}:{final.get('stop_reason')}"]
            else:
                results[site_id]["status"] = "done"
    finally:
        released = release_state_rows(sidecar, identities, states, concurrency=concurrency)
        for identity in identities:
            site_id = identity.spec.site_id
            attempts = results[site_id]["attempts"]
            if attempts:
                attempts[-1]["trace"]["released_states"] = [states[site_id]] if released.get(site_id) else []
                attempts[-1]["trace"]["all_states_released"] = bool(released.get(site_id))
            if not released.get(site_id):
                results[site_id]["status"] = "failed"
                results[site_id]["errors"].append("state:release_failed")

    output_rows = []
    for identity in identities:
        site_id = identity.spec.site_id
        row = results[site_id]
        row["elapsed_ms"] = round((time.perf_counter() - started[site_id]) * 1000, 3)
        wall.update_task(
            site_id,
            status=row["status"],
            elapsed_ms=row["elapsed_ms"],
            tool_steps=sum(int(attempt["trace"].get("tool_steps") or 0) for attempt in row["attempts"]),
            error=", ".join(row["errors"][:6]),
            gallery=f"gallery/{site_id}/index.html" if row["status"] == "done" else "",
        )
        output_rows.append(row)
    return output_rows


def telemetry_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    busy = [float(row["gpu_busy_pct"]) for row in samples]
    vram = [int(row["vram_used_bytes"]) for row in samples]
    return {
        "samples": len(samples),
        "gpu_busy_pct": {
            "mean": round(statistics.fmean(busy), 3) if busy else 0.0,
            "p50": round(percentile(busy, 0.5), 3),
            "p95": round(percentile(busy, 0.95), 3),
            "peak": round(max(busy), 3) if busy else 0.0,
        },
        "vram_used_bytes": {
            "mean": round(statistics.fmean(vram)) if vram else 0,
            "peak": max(vram) if vram else 0,
        },
        "resident_states_peak": max((int(row.get("resident_states") or 0) for row in samples), default=0),
        "active_state_rows_peak": max((int(row.get("active_state_rows") or 0) for row in samples), default=0),
        "decoding_rows_peak": max((int(row.get("decoding_rows") or 0) for row in samples), default=0),
        "max_batch_observed": max((int(row.get("max_batch_observed") or 0) for row in samples), default=0),
    }


def scheduler_metrics(health: dict[str, Any]) -> dict[str, Any]:
    return dict(((health.get("inference") or {}).get("scheduler") or {}))


def metric_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after.get(key) or 0) - int(before.get(key) or 0)


def write_dashboard_assets(output: Path) -> None:
    source = Path(__file__).with_name("website_swarm_dashboard")
    for name in ("index.html", "app.css", "app.js"):
        shutil.copy2(source / name, output / name)


def write_evidence(
    output: Path,
    results: Sequence[dict[str, Any]],
    identities: Sequence[TaskIdentity],
    *,
    args: argparse.Namespace,
    controller_health: dict[str, Any],
    sidecar_before: dict[str, Any],
    sidecar_after: dict[str, Any],
    sampler: TelemetrySampler,
    wall: TaskWall,
    elapsed: float,
) -> dict[str, Any]:
    valid = [row for row in results if row["status"] == "done"]
    generations = [
        Generation(
            spec=WebsiteSpec(**row["spec"]),
            prompt=str(row["identity"]["prompt"]),
            raw=json.dumps(row["content"], ensure_ascii=False),
            dsl=row["dsl"],
            errors=[],
            attempts=len(row["attempts"]),
            output_tokens=0,
        )
        for row in valid
    ]
    atomic_write(output / "gallery" / "index.html", render_gallery(generations))
    with (output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    titles = [str(row["content"]["title"]).casefold() for row in valid]
    main_copy = [
        (
            str(row["dsl"]["hero"]["headline"])
            + " "
            + str(row["dsl"]["hero"]["summary"])
        ).casefold()
        for row in valid
    ]
    owners = [
        owner
        for row in results
        for attempt in row["attempts"]
        for owner in attempt["trace"].get("owners", [])
    ]
    states = [
        state
        for row in results
        for attempt in row["attempts"]
        for state in attempt["trace"].get("opened_states", [])
    ]
    scheduler_before = scheduler_metrics(sidecar_before)
    scheduler_after = scheduler_metrics(sidecar_after)
    persistent_after = dict(sidecar_after.get("persistent_states") or {})
    pool_after = dict(scheduler_after.get("pool") or {})
    wall_times = [float(row["elapsed_ms"]) for row in results]
    protocol_leaks = sum(
        bool(attempt["trace"].get("protocol_leak"))
        for row in results
        for attempt in row["attempts"]
    )
    metrics = {
        "schema": SCHEMA,
        "status": "pass" if len(valid) == args.count else "fail",
        "run_id": args.run_id,
        "logical_tasks": args.count,
        "physical_concurrency": args.concurrency,
        "claim": f"{args.count} independent tasks · physical concurrency {args.concurrency}",
        "completed": len(valid),
        "failed": args.count - len(valid),
        "wall_seconds": round(elapsed, 3),
        "task_latency_ms": {
            "p50": round(percentile(wall_times, 0.5), 3),
            "p95": round(percentile(wall_times, 0.95), 3),
            "max": round(max(wall_times), 3) if wall_times else 0.0,
        },
        "unique_prompts": len({item.prompt_sha256 for item in identities}),
        "unique_sessions": len({item.session_id for item in identities}),
        "unique_workspaces": len({item.workspace for item in identities}),
        "unique_owner_ids": len(set(owners)),
        "owner_ids_observed": len(owners),
        "unique_state_ids": len(set(states)),
        "state_ids_observed": len(states),
        "unique_title_rate": round(len(set(titles)) / args.count, 6),
        "unique_main_copy_rate": round(len(set(main_copy)) / args.count, 6),
        "artifact_valid_rate": round(len(valid) / args.count, 6),
        "runtime_validation_rate": round(
            sum(any(attempt["trace"].get("runtime_validator_passed") for attempt in row["attempts"]) for row in results)
            / args.count,
            6,
        ),
        "protocol_leaks": protocol_leaks,
        "repair_count": sum(len(row["attempts"]) > 1 for row in results),
        "tool_steps": sum(
            int(attempt["trace"].get("tool_steps") or 0)
            for row in results
            for attempt in row["attempts"]
        ),
        "scheduler_delta": {
            key: metric_delta(scheduler_before, scheduler_after, key)
            for key in ("admitted", "forward_calls", "forward_rows", "forward_tokens", "decode_calls", "decode_tokens", "released")
        },
        "telemetry": telemetry_summary(sampler.samples),
        "telemetry_errors": sampler.errors,
        "final_state_counters": {
            "persistent_allocated": int(persistent_after.get("allocated") or 0),
            "persistent_busy": int(persistent_after.get("busy") or 0),
            "pool_allocated": int(pool_after.get("allocated") or 0),
            "waiting": int((sidecar_after.get("inference") or {}).get("waiting") or 0),
        },
        "model": sidecar_after.get("model"),
        "backend": sidecar_after.get("backend"),
        "context": sidecar_after.get("context"),
        "controller": {
            "control_plane": controller_health.get("control_plane"),
            "command": controller_health.get("command"),
            "tools": controller_health.get("tools"),
        },
        "failed_sites": {row["spec"]["site_id"]: row["errors"] for row in results if row["status"] != "done"},
    }
    if (
        metrics["unique_prompts"] != args.count
        or metrics["unique_sessions"] != args.count
        or metrics["unique_workspaces"] != args.count
        or metrics["unique_owner_ids"] != metrics["owner_ids_observed"]
        or metrics["unique_owner_ids"] < args.count
        or metrics["unique_state_ids"] != metrics["state_ids_observed"]
        or metrics["unique_state_ids"] < args.count
        or metrics["unique_title_rate"] != 1.0
        or metrics["unique_main_copy_rate"] != 1.0
        or metrics["runtime_validation_rate"] != 1.0
        or protocol_leaks != 0
        or any(metrics["final_state_counters"].values())
    ):
        metrics["status"] = "fail"
    atomic_write(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    wall.update_runtime(
        {
            "resident_states": metrics["final_state_counters"]["persistent_allocated"],
            "active_state_rows": metrics["final_state_counters"]["persistent_busy"],
            "waiting_jobs": metrics["final_state_counters"]["waiting"],
            "prefilling_rows": 0,
            "decoding_rows": 0,
        }
    )
    wall.set_run_status("complete" if metrics["status"] == "pass" else "failed")
    manifest = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "command": vars(args),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "controller_health": controller_health,
        "sidecar_health_before": sidecar_before,
        "sidecar_health_after": sidecar_after,
        "task_wall_final": wall.snapshot(),
    }
    manifest["command"] = {key: str(value) if isinstance(value, Path) else value for key, value in manifest["command"].items()}
    atomic_write(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    checksum_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        checksum_rows.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}")
    atomic_write(output / "SHA256SUMS", "\n".join(checksum_rows) + "\n")
    return metrics


def prepare_directories(output: Path, workspace_root: Path, identities: Sequence[TaskIdentity]) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_workspace = workspace_root / Path(identities[0].workspace).parent
    if run_workspace.exists() and any(run_workspace.iterdir()):
        raise ValueError(f"workspace run directory must be empty: {run_workspace}")
    for item in identities:
        path = workspace_root / item.workspace
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    workspace_root = args.workspace_root.resolve()
    specs = build_specs(args.count, args.seed)
    identities = build_identities(specs, args.run_id)
    if len({item.prompt_sha256 for item in identities}) != args.count:
        raise RuntimeError("task prompts are not unique")
    prepare_directories(output, workspace_root, identities)
    write_dashboard_assets(output)
    controller = HttpClient(args.controller_url, args.request_timeout_seconds)
    sidecar = HttpClient(args.sidecar_url, args.request_timeout_seconds)
    controller_health = controller.get("/health")
    sidecar_before = sidecar.get("/health")
    if controller_health.get("control_plane") != "rust":
        raise RuntimeError("Gate 4 requires the Rust Controller")
    model_name = str(sidecar_before.get("model") or "")
    if "13.3b" not in model_name.casefold():
        raise RuntimeError(f"Gate 4 requires the frozen 13.3B model, got {model_name}")
    wall = TaskWall(output, identities, run_id=args.run_id, concurrency=args.concurrency)
    wall.update_runtime(
        {
            "model": sidecar_before.get("model"),
            "backend": sidecar_before.get("backend"),
            "context": sidecar_before.get("context"),
            "gpu": "AMD Radeon gfx1100",
            "rocm": "7.2.1",
        }
    )
    dashboard = start_dashboard(output, args.serve_host, args.serve_port)
    sampler = TelemetrySampler(
        HttpClient(args.sidecar_url, 5.0),
        wall,
        output,
        args.telemetry_interval_seconds,
    )
    sampler.start()
    wall.set_run_status("running")
    started = time.perf_counter()
    results: list[dict[str, Any]]
    try:
        results = run_state_swarm(
            identities,
            controller=controller,
            sidecar=sidecar,
            workspace_root=workspace_root,
            output=output,
            wall=wall,
            concurrency=args.concurrency,
            repair_attempts=args.repair_attempts,
        )
    finally:
        sampler.stop()
    results.sort(key=lambda row: int(row["spec"]["index"]))
    sidecar_after = sidecar.get("/health")
    elapsed = time.perf_counter() - started
    wall.set_run_status("finalizing")
    metrics = write_evidence(
        output,
        results,
        identities,
        args=args,
        controller_health=controller_health,
        sidecar_before=sidecar_before,
        sidecar_after=sidecar_after,
        sampler=sampler,
        wall=wall,
        elapsed=elapsed,
    )
    if dashboard is not None:
        dashboard.shutdown()
        dashboard.server_close()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-url", default="http://127.0.0.1:18120")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:18118")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=0.5)
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 100:
        parser.error("--count must be 1..100")
    if not 1 <= args.concurrency <= 32:
        parser.error("--concurrency must be 1..32 for the frozen Gate 3 Scheduler")
    if not 0 <= args.repair_attempts <= 4:
        parser.error("--repair-attempts must be 0..4")
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, hyphen, or underscore")
    return args


def main() -> None:
    metrics = run(parse_args())
    if metrics["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
