from __future__ import annotations

import pytest

from rwkv_agent.sidecar import _requires_worker_admission
from rwkv_agent import statepool_drain
from rwkv_agent.statepool_worker import (
    StatePoolWorkerAgent,
    WorkerConfigurationError,
    WorkerNotRegistered,
    WorkerSettings,
)


MODEL_REF = {
    "model_id": "rwkv-test",
    "revision": "revision-test",
    "tokenizer": "tokenizer-test",
    "state_abi": "state-abi-test",
}


def settings() -> WorkerSettings:
    return WorkerSettings(
        statepool_url="http://statepool.test",
        worker_id="worker-a",
        endpoint="http://worker-a.test:8118",
        zone="cloud",
        model_ref=MODEL_REF,
        state_slots=8,
        max_batch=4,
        heartbeat_seconds=60,
        device_vendor="NVIDIA",
        device_model="V100",
        device_runtime="CUDA 12",
        device_memory_bytes=32 * 1024**3,
    )


def health(*, allocated: int = 2, busy: int = 0) -> dict:
    return {
        "inference": {"waiting": 3, "prefilling": 1, "decoding": 1},
        "persistent_states": {
            "capacity": 8,
            "allocated": allocated,
            "free": 8 - allocated,
            "busy": busy,
            "batching": {"waiting": 2, "active_rows": 1},
        },
    }


def test_worker_mode_is_absent_without_statepool_url(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_STATEPOOL_URL", raising=False)
    assert WorkerSettings.from_environment() is None


def test_enabled_worker_requires_exact_model_identity(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_STATEPOOL_URL", "http://statepool.test")
    for name in (
        "RWKV_WORKER_MODEL_ID",
        "RWKV_WORKER_MODEL_REVISION",
        "RWKV_WORKER_TOKENIZER",
        "RWKV_WORKER_STATE_ABI",
        "G1I_MODEL_ID",
        "G1I_MODEL_REVISION",
        "G1I_TOKENIZER_ID",
        "G1I_STATE_ABI",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(WorkerConfigurationError, match="MODEL_ID"):
        WorkerSettings.from_environment()


def test_capability_reports_real_load_and_conservative_dirty_states() -> None:
    agent = StatePoolWorkerAgent(settings(), lambda: health(allocated=2, busy=1))
    # Before start, the Worker cannot be selected.
    assert agent.capability()["lifecycle"] == "starting"
    agent._lifecycle = "ready"
    assert agent.enter_request()
    capability = agent.capability()
    assert capability["models"] == [MODEL_REF]
    assert capability["device"]["model"] == "V100"
    assert capability["capacity"] == {
        "state_slots": 8,
        "free_state_slots": 6,
        "max_batch": 4,
        "queue_depth": 3,
        "running_requests": 2,
        "unpersisted_state_slots": 2,
    }
    agent.exit_request()


def test_drain_rejects_new_admission_and_never_claims_dirty_state_safe() -> None:
    current = {"allocated": 1, "busy": 0}
    calls: list[tuple[str, dict]] = []

    def transport(url: str, payload: dict, _timeout: float) -> dict:
        calls.append((url, payload))
        if url.endswith("/drain"):
            return {"status": "draining"}
        return {"status": "ok", "lifecycle": payload["lifecycle"]}

    agent = StatePoolWorkerAgent(
        settings(),
        lambda: health(**current),
        clock=lambda: 100.0,
        transport=transport,
    )
    agent._lifecycle = "ready"
    agent._registered = True
    result = agent.begin_draining(timeout_seconds=10)
    assert result["status"] == "draining"
    assert result["unpersisted_states"] == 1
    assert not agent.enter_request()
    assert any(url.endswith("/worker-a/drain") for url, _ in calls)

    current["allocated"] = 0
    agent._health_provider = lambda: {
        "inference": {"waiting": 0, "prefilling": 0, "decoding": 0},
        "persistent_states": {
            "capacity": 8,
            "allocated": 0,
            "free": 8,
            "busy": 0,
            "batching": {"waiting": 0, "active_rows": 0},
        },
    }
    assert agent.drain_status(deadline_ms=110_000)["status"] == "safe_to_stop"


def test_missing_registration_is_recreated_before_next_heartbeat() -> None:
    calls: list[str] = []

    def transport(url: str, _payload: dict, _timeout: float) -> dict:
        calls.append(url)
        if url.endswith("/heartbeat"):
            raise WorkerNotRegistered("forgotten")
        return {"status": "ok"}

    agent = StatePoolWorkerAgent(settings(), lambda: health(allocated=0), transport=transport)
    agent._lifecycle = "ready"
    agent._registered = True
    agent.heartbeat_once()
    assert calls[-2].endswith("/heartbeat")
    assert calls[-1].endswith("/workers/register")
    assert agent.status()["registered"] is True


def test_drain_route_keeps_snapshot_and_release_available() -> None:
    assert _requires_worker_admission("/v1/states/prefill")
    assert _requires_worker_admission("/v1/states/batch_continue")
    assert _requires_worker_admission("/v1/states/restore")
    assert not _requires_worker_admission("/v1/states/state-1/snapshot")
    assert not _requires_worker_admission("/v1/states/release")
    assert not _requires_worker_admission("/v1/statepool/drain")


def test_prestop_client_polls_until_controller_releases_dirty_states(monkeypatch) -> None:
    statuses = iter(
        [
            {
                "status": "draining",
                "active_requests": 0,
                "unpersisted_states": 1,
            },
            {
                "status": "safe_to_stop",
                "active_requests": 0,
                "unpersisted_states": 0,
            },
        ]
    )
    requests: list[tuple[str, dict | None]] = []

    def request(url: str, payload=None):
        requests.append((url, payload))
        return next(statuses)

    monkeypatch.setattr(statepool_drain, "_request", request)
    monkeypatch.setattr(statepool_drain.time, "sleep", lambda _seconds: None)
    result = statepool_drain.drain("http://worker", 10, 0.01)
    assert result["status"] == "safe_to_stop"
    assert requests[0][1] == {"timeout_seconds": 10}
    assert requests[1] == ("http://worker/v1/statepool/drain", None)
