"""Optional StatePool registration, heartbeat and conservative drain control.

The Worker adapter is dormant unless ``RWKV_STATEPOOL_URL`` is configured.
It deliberately treats every resident recurrent State as unpersisted: a Pod is
never declared safe to stop merely because inference is idle.  The Controller
must first snapshot and release those States through the fenced StatePool
lifecycle, after which the next heartbeat can prove a zero dirty-State count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import socket
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WORKER_CONTRACT_VERSION = "statepool-worker-capability.v1"
DRAIN_REQUEST_VERSION = "statepool-drain-request.v1"


class WorkerConfigurationError(ValueError):
    """Raised when the explicitly enabled Worker adapter is incomplete."""


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    statepool_url: str
    worker_id: str
    endpoint: str
    zone: str
    model_ref: dict[str, str]
    state_slots: int
    max_batch: int
    heartbeat_seconds: float = 5.0
    request_timeout_seconds: float = 3.0
    device_vendor: str = "unknown"
    device_model: str = "unknown"
    device_runtime: str = ""
    device_memory_bytes: int = 0
    price: dict[str, Any] | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> WorkerSettings | None:
        statepool_url = os.getenv("RWKV_STATEPOOL_URL", "").strip().rstrip("/")
        if not statepool_url:
            return None
        worker_id = (
            os.getenv("RWKV_WORKER_ID")
            or os.getenv("POD_UID")
            or os.getenv("POD_NAME")
            or socket.gethostname()
        ).strip()
        port = _positive_int(os.getenv("RWKV_WORKER_PORT", "8118"), "port")
        advertised_host = os.getenv("POD_IP", "127.0.0.1").strip()
        endpoint = os.getenv(
            "RWKV_WORKER_ENDPOINT", f"http://{advertised_host}:{port}"
        ).strip().rstrip("/")
        zone = os.getenv("RWKV_WORKER_ZONE", "cloud").strip().lower()
        if zone not in {"local", "edge", "cloud"}:
            raise WorkerConfigurationError(
                "RWKV_WORKER_ZONE must be local, edge or cloud"
            )
        model_ref = {
            "model_id": _required_env("RWKV_WORKER_MODEL_ID", "G1I_MODEL_ID"),
            "revision": _required_env(
                "RWKV_WORKER_MODEL_REVISION", "G1I_MODEL_REVISION"
            ),
            "tokenizer": _required_env(
                "RWKV_WORKER_TOKENIZER", "G1I_TOKENIZER_ID"
            ),
            "state_abi": _required_env(
                "RWKV_WORKER_STATE_ABI", "G1I_STATE_ABI"
            ),
        }
        labels = _labels_from_environment()
        currency = os.getenv("RWKV_WORKER_PRICE_CURRENCY", "").strip().upper()
        price_value = os.getenv("RWKV_WORKER_PRICE_PER_GPU_HOUR", "").strip()
        price = None
        if currency or price_value:
            if len(currency) != 3 or not currency.isalpha() or not price_value:
                raise WorkerConfigurationError(
                    "Worker price requires a three-letter currency and per-GPU-hour value"
                )
            amount = _non_negative_float(price_value, "per GPU hour")
            price = {"currency": currency, "per_gpu_hour": amount}
        settings = cls(
            statepool_url=statepool_url,
            worker_id=worker_id,
            endpoint=endpoint,
            zone=zone,
            model_ref=model_ref,
            state_slots=_positive_int(
                os.getenv("RWKV_WORKER_STATE_SLOTS", "8"), "state slots"
            ),
            max_batch=_positive_int(
                os.getenv("RWKV_WORKER_MAX_BATCH", "8"), "max batch"
            ),
            heartbeat_seconds=_positive_float(
                os.getenv("RWKV_WORKER_HEARTBEAT_SECONDS", "5"),
                "heartbeat seconds",
            ),
            request_timeout_seconds=_positive_float(
                os.getenv("RWKV_WORKER_REQUEST_TIMEOUT_SECONDS", "3"),
                "request timeout seconds",
            ),
            device_vendor=os.getenv(
                "RWKV_WORKER_DEVICE_VENDOR", "unknown"
            ).strip()
            or "unknown",
            device_model=os.getenv("RWKV_WORKER_DEVICE_MODEL", "unknown").strip()
            or "unknown",
            device_runtime=os.getenv("RWKV_WORKER_DEVICE_RUNTIME", "").strip(),
            device_memory_bytes=_non_negative_int(
                os.getenv("RWKV_WORKER_DEVICE_MEMORY_BYTES", "0"),
                "device memory bytes",
            ),
            price=price,
            labels=labels,
        )
        if not settings.worker_id or not settings.endpoint:
            raise WorkerConfigurationError(
                "Worker ID and advertised endpoint must not be empty"
            )
        return settings


class StatePoolWorkerAgent:
    """Own one Worker's registration and liveness lease with StatePool."""

    def __init__(
        self,
        settings: WorkerSettings,
        health_provider: Callable[[], dict[str, Any]],
        *,
        clock: Callable[[], float] = time.time,
        transport: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._health_provider = health_provider
        self._clock = clock
        self._transport = transport or _post_json
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = "starting"
        self._registered = False
        self._active_ingress = 0
        self._drain_deadline_ms: int | None = None
        self._last_success_ms = 0
        self._last_error = ""

    @property
    def enabled(self) -> bool:
        return True

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._lifecycle = "ready"
            self._thread = threading.Thread(
                target=self._run,
                name="statepool-worker-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def heartbeat_once(self) -> dict[str, Any]:
        """Synchronously publish one capability update (tests/probes/preStop)."""

        self._send_once(force_register=False)
        return self.status()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            self._lifecycle = "offline"
            registered = self._registered
        self._stop.set()
        self._wake.set()
        if registered:
            self._send_once(force_register=False)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))

    def enter_request(self) -> bool:
        with self._lock:
            if self._lifecycle != "ready":
                return False
            self._active_ingress += 1
            return True

    def exit_request(self) -> None:
        with self._lock:
            if self._active_ingress > 0:
                self._active_ingress -= 1
            if self._lifecycle == "draining":
                self._wake.set()

    def begin_draining(self, *, timeout_seconds: float) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("drain timeout must be positive")
        deadline_ms = int((self._clock() + timeout_seconds) * 1000)
        with self._lock:
            self._lifecycle = "draining"
            self._drain_deadline_ms = deadline_ms
        # Publish the admission change before querying the control plane.
        self._send_once(force_register=False)
        try:
            remote = self._transport(
                f"{self.settings.statepool_url}/plugin/v1/workers/"
                f"{self.settings.worker_id}/drain",
                {
                    "contract_version": DRAIN_REQUEST_VERSION,
                    "deadline_ms": deadline_ms,
                },
                self.settings.request_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in status, never ignored
            with self._lock:
                self._last_error = f"drain request failed: {exc}"
            remote = {"status": "control_plane_unavailable"}
        return {**self.drain_status(deadline_ms=deadline_ms), "control_plane": remote}

    def drain_status(self, *, deadline_ms: int | None = None) -> dict[str, Any]:
        if deadline_ms is None:
            with self._lock:
                deadline_ms = self._drain_deadline_ms
        capability = self.capability()
        capacity = capability["capacity"]
        active = int(capacity["running_requests"]) + int(capacity["queue_depth"])
        unpersisted = int(capacity["unpersisted_state_slots"])
        safe = active == 0 and unpersisted == 0
        expired = deadline_ms is not None and int(self._clock() * 1000) >= deadline_ms
        return {
            "contract_version": "statepool-drain-status.v1",
            "worker_id": self.settings.worker_id,
            "status": (
                "safe_to_stop"
                if safe
                else "deadline_exceeded"
                if expired
                else "draining"
            ),
            "active_requests": active,
            "unpersisted_states": unpersisted,
        }

    def ready(self) -> bool:
        with self._lock:
            return self._lifecycle == "ready" and self._registered

    def status(self) -> dict[str, Any]:
        with self._lock:
            lifecycle = self._lifecycle
            registered = self._registered
            last_success_ms = self._last_success_ms
            last_error = self._last_error
            active_ingress = self._active_ingress
        return {
            "enabled": True,
            "worker_id": self.settings.worker_id,
            "lifecycle": lifecycle,
            "registered": registered,
            "ready": lifecycle == "ready" and registered,
            "active_ingress": active_ingress,
            "last_success_ms": last_success_ms,
            "last_error": last_error,
        }

    def capability(self) -> dict[str, Any]:
        health = self._health_provider()
        persistent = health.get("persistent_states") or {}
        inference = health.get("inference") or {}
        batching = persistent.get("batching") or {}
        allocated = _as_non_negative_int(persistent.get("allocated"))
        free = _as_non_negative_int(persistent.get("free"))
        queue_depth = max(
            _as_non_negative_int(inference.get("waiting")),
            _as_non_negative_int(batching.get("waiting")),
        )
        runtime_active = max(
            _as_non_negative_int(inference.get("prefilling"))
            + _as_non_negative_int(inference.get("decoding")),
            _as_non_negative_int(batching.get("active_rows")),
            _as_non_negative_int(persistent.get("busy")),
        )
        with self._lock:
            lifecycle = self._lifecycle
            active_ingress = self._active_ingress
        reported_capacity = _as_non_negative_int(persistent.get("capacity"))
        capacity = max(
            reported_capacity or self.settings.state_slots,
            allocated + free,
        )
        max_batch = (
            _as_non_negative_int(batching.get("max_batch_size"))
            or self.settings.max_batch
        )
        payload: dict[str, Any] = {
            "contract_version": WORKER_CONTRACT_VERSION,
            "worker_id": self.settings.worker_id,
            "zone": self.settings.zone,
            "endpoint": self.settings.endpoint,
            "lifecycle": lifecycle,
            "models": [dict(self.settings.model_ref)],
            "device": {
                "vendor": self.settings.device_vendor,
                "model": self.settings.device_model,
                "runtime": self.settings.device_runtime,
                "memory_bytes": self.settings.device_memory_bytes,
            },
            "capacity": {
                "state_slots": capacity,
                "free_state_slots": min(free, capacity),
                "max_batch": max_batch,
                "queue_depth": queue_depth,
                "running_requests": max(runtime_active, active_ingress),
                # Conservative by construction: a resident State is dirty until
                # Controller snapshot + Worker release makes it disappear.
                "unpersisted_state_slots": allocated,
            },
            "price": self.settings.price,
            "labels": dict(self.settings.labels),
            "reported_at_ms": int(self._clock() * 1000),
        }
        return payload

    def _run(self) -> None:
        while not self._stop.is_set():
            self._send_once(force_register=False)
            self._wake.wait(self.settings.heartbeat_seconds)
            self._wake.clear()

    def _send_once(self, *, force_register: bool) -> None:
        with self._lock:
            registered = self._registered and not force_register
        path = (
            f"/plugin/v1/workers/{self.settings.worker_id}/heartbeat"
            if registered
            else "/plugin/v1/workers/register"
        )
        try:
            response = self._transport(
                self.settings.statepool_url + path,
                self.capability(),
                self.settings.request_timeout_seconds,
            )
            with self._lock:
                self._registered = True
                self._last_success_ms = int(self._clock() * 1000)
                self._last_error = ""
                desired = response.get("lifecycle")
                if desired == "draining":
                    self._lifecycle = "draining"
        except WorkerNotRegistered:
            with self._lock:
                self._registered = False
            if registered:
                self._send_once(force_register=True)
        except Exception as exc:  # noqa: BLE001 - retained for readiness/diagnostics
            with self._lock:
                self._last_error = str(exc)


class WorkerNotRegistered(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        if exc.code == 404 and url.endswith("/heartbeat"):
            raise WorkerNotRegistered("StatePool forgot Worker registration") from exc
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"StatePool HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"StatePool unavailable: {exc}") from exc
    if not raw:
        return {"status": "ok"}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("StatePool returned non-object JSON")
    return value


def _required_env(primary: str, fallback: str) -> str:
    value = os.getenv(primary, os.getenv(fallback, "")).strip()
    if not value:
        raise WorkerConfigurationError(f"{primary} is required when Worker mode is enabled")
    return value


def _labels_from_environment() -> dict[str, str]:
    encoded = os.getenv("RWKV_WORKER_LABELS_JSON", "").strip()
    labels: dict[str, str] = {}
    if encoded:
        value = json.loads(encoded)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise WorkerConfigurationError(
                "RWKV_WORKER_LABELS_JSON must be a string-to-string object"
            )
        labels.update(value)
    for key, env_name in (
        ("pod", "POD_NAME"),
        ("namespace", "POD_NAMESPACE"),
        ("node", "NODE_NAME"),
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            labels[key] = value
    return labels


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise WorkerConfigurationError(f"Worker {name} must be positive")
    return parsed


def _non_negative_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise WorkerConfigurationError(f"Worker {name} must not be negative")
    return parsed


def _positive_float(value: str, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise WorkerConfigurationError(f"Worker {name} must be positive")
    return parsed


def _non_negative_float(value: str, name: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise WorkerConfigurationError(f"Worker {name} must not be negative")
    return parsed


def _as_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
