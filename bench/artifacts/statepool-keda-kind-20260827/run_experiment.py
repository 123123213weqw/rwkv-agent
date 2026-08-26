#!/usr/bin/env python3
from __future__ import annotations
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

NS = "statepool-keda"
RUN = Path("/home/wzu/codex-run/statepool-keda-kind-20260827/evidence-run-04")
KUBECTL = "/home/wzu/codex-tools/statepool-keda/bin/kubectl"
ENV = dict(os.environ)
ENV["KUBECONFIG"] = "/home/wzu/codex-tools/statepool-keda/kubeconfig"
LABEL = "app.kubernetes.io%2Fcomponent%3Dinference-worker%2Capp.kubernetes.io%2Finstance%3Dstatepool"
PROXY = "http://127.0.0.1:18001"
PLUGIN = "http://127.0.0.1:18130"
PROM = "http://127.0.0.1:19090"

def iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def start(cmd, stdout=None, stderr=None):
    return subprocess.Popen(cmd, env=ENV, stdout=stdout or subprocess.DEVNULL,
                            stderr=stderr or subprocess.STDOUT, text=True,
                            start_new_session=True)

def wait_http(url, timeout=30):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.read()
        except Exception as exc:
            last = exc
            time.sleep(0.25)
    raise RuntimeError(f"timeout waiting for {url}: {last}")

def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)

def get_text(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return response.read().decode()

def post_json(url, value):
    data = json.dumps(value, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, json.load(response)

def metric(text, name):
    prefix = name + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):])
    return None

def condition(obj, kind):
    for cond in obj.get("status", {}).get("conditions", []):
        if cond.get("type") == kind:
            return cond.get("status")
    return None

def pod_phase_summary(items):
    values = {}
    for pod in items:
        phase = pod.get("status", {}).get("phase", "Unknown")
        values[phase] = values.get(phase, 0) + 1
    return ",".join(f"{key}:{values[key]}" for key in sorted(values))

forward_log = (RUN / "port-forwards.log").open("w")
proxy_log = (RUN / "kubectl-proxy.log").open("w")
processes = [
    start([KUBECTL, "proxy", "--port=18001"], proxy_log),
    start([KUBECTL, "-n", NS, "port-forward", "service/statepool", "18130:8130"], forward_log),
    start([KUBECTL, "-n", NS, "port-forward", "service/prometheus", "19090:9090"], forward_log),
]
log_procs = {}
log_files = {}
try:
    wait_http(PROXY + "/version")
    wait_http(PLUGIN + "/plugin/v1/health")
    wait_http(PROM + "/-/ready")

    dep_url = PROXY + f"/apis/apps/v1/namespaces/{NS}/deployments/statepool-worker"
    hpa_url = PROXY + f"/apis/autoscaling/v2/namespaces/{NS}/horizontalpodautoscalers/keda-hpa-statepool-worker"
    so_url = PROXY + f"/apis/keda.sh/v1alpha1/namespaces/{NS}/scaledobjects/statepool-worker"
    pods_url = PROXY + f"/api/v1/namespaces/{NS}/pods?labelSelector={LABEL}"

    baseline_dep = get_json(dep_url)
    if int(baseline_dep.get("spec", {}).get("replicas", 0)) != 0:
        raise RuntimeError("baseline Worker replicas are not zero")
    if metric(get_text(PLUGIN + "/metrics"), "statepool_pending_requests") != 0:
        raise RuntimeError("baseline pending_requests is not zero")

    requests = []
    model_ref = {
        "model_id": "rwkv7-keda-sim",
        "revision": "keda-sim-20260827",
        "tokenizer": "rwkv_vocab_v20230424",
        "state_abi": "rwkv7-keda-sim-state-v1",
    }
    trigger_at = iso()
    for index in range(1, 4):
        payload = {
            "contract_version": "statepool-plan-request.v1",
            "request_id": f"keda-scale-{index:02d}",
            "session_id": f"keda-session-{index:02d}",
            "owner_id": f"session:keda-session-{index:02d}",
            "model_ref": model_ref,
            "privacy": "cloud_allowed",
            "latency_slo_ms": 5000,
            "max_cost": None,
            "preferred_zone": "cloud",
            "state_ref": None,
            "estimated_input_tokens": 128,
            "estimated_output_tokens": 100,
        }
        status, response = post_json(PLUGIN + "/plugin/v1/plan", payload)
        requests.append({"at": iso(), "http_status": status, "request": payload, "response": response})
    (RUN / "trigger-responses.json").write_text(json.dumps({
        "trigger_at": trigger_at, "requests": requests
    }, indent=2) + "\n")

    fields = [
        "timestamp_utc", "elapsed_seconds", "plugin_pending", "plugin_decode_seconds",
        "prom_pending", "prom_decode_seconds", "deployment_desired", "deployment_current",
        "deployment_ready", "deployment_available", "pod_count", "pod_phases",
        "hpa_current", "hpa_desired", "scaledobject_ready", "scaledobject_active",
        "registered_worker_count"
    ]
    observed_desired = []
    observed_ready = []
    start_at = time.monotonic()
    reached_n = False
    reached_ready_n = False
    back_to_zero = False
    samples = []
    deadline = start_at + 240

    with (RUN / "transitions.csv").open("w", newline="") as csv_file,          (RUN / "transitions.jsonl").open("w") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        while time.monotonic() < deadline:
            now = time.monotonic()
            dep = get_json(dep_url)
            hpa = get_json(hpa_url)
            so = get_json(so_url)
            pods = get_json(pods_url).get("items", [])
            plugin_metrics = get_text(PLUGIN + "/metrics")
            workers = get_json(PLUGIN + "/plugin/v1/workers")
            prom_pending_data = get_json(PROM + "/api/v1/query?" + urllib.parse.urlencode({"query": "sum(statepool_pending_requests)"}))
            prom_decode_data = get_json(PROM + "/api/v1/query?" + urllib.parse.urlencode({"query": "sum(statepool_estimated_decode_seconds)"}))

            for pod in pods:
                name = pod["metadata"]["name"]
                if name not in log_procs:
                    handle = (RUN / f"worker-{name}.log").open("w")
                    log_files[name] = handle
                    command = (
                        f'while {KUBECTL} -n {NS} get pod {name} >/dev/null 2>&1; do '
                        f'{KUBECTL} -n {NS} logs -f pod/{name} '
                        '--all-containers=true --prefix=true --timestamps=true; '
                        'sleep 0.25; done'
                    )
                    log_procs[name] = start(["/bin/bash", "-lc", command], handle)

            def prom_value(data):
                result = data.get("data", {}).get("result", [])
                return float(result[0]["value"][1]) if result else None

            desired = int(dep.get("spec", {}).get("replicas") or 0)
            current = int(dep.get("status", {}).get("replicas") or 0)
            ready = int(dep.get("status", {}).get("readyReplicas") or 0)
            available = int(dep.get("status", {}).get("availableReplicas") or 0)
            hpa_current = int(hpa.get("status", {}).get("currentReplicas") or 0)
            hpa_desired = int(hpa.get("status", {}).get("desiredReplicas") or 0)
            observed_desired.append(desired)
            observed_ready.append(ready)
            reached_n = reached_n or desired >= 3 or current >= 3
            reached_ready_n = reached_ready_n or ready >= 3
            if reached_n and desired == 0 and len(pods) == 0:
                back_to_zero = True

            row = {
                "timestamp_utc": iso(),
                "elapsed_seconds": round(now - start_at, 3),
                "plugin_pending": metric(plugin_metrics, "statepool_pending_requests"),
                "plugin_decode_seconds": metric(plugin_metrics, "statepool_estimated_decode_seconds"),
                "prom_pending": prom_value(prom_pending_data),
                "prom_decode_seconds": prom_value(prom_decode_data),
                "deployment_desired": desired,
                "deployment_current": current,
                "deployment_ready": ready,
                "deployment_available": available,
                "pod_count": len(pods),
                "pod_phases": pod_phase_summary(pods),
                "hpa_current": hpa_current,
                "hpa_desired": hpa_desired,
                "scaledobject_ready": condition(so, "Ready"),
                "scaledobject_active": condition(so, "Active"),
                "registered_worker_count": int(workers.get("count", 0)),
            }
            writer.writerow(row)
            csv_file.flush()
            jsonl_file.write(json.dumps({
                "sample": row,
                "pod_names": [pod["metadata"]["name"] for pod in pods],
                "workers": workers.get("workers", []),
                "scaledobject_triggers_activity": so.get("status", {}).get("triggersActivity", {}),
                "hpa_current_metrics": hpa.get("status", {}).get("currentMetrics", []),
            }, separators=(",", ":")) + "\n")
            jsonl_file.flush()
            samples.append(row)

            if back_to_zero:
                # Leave time for log-follow processes to flush final preStop output.
                time.sleep(3)
                break
            time.sleep(0.25)

    sequence = []
    for value in observed_desired:
        if not sequence or sequence[-1] != value:
            sequence.append(value)
    summary = {
        "started_at": trigger_at,
        "finished_at": iso(),
        "success": bool(reached_n and back_to_zero),
        "reached_desired_n": reached_n,
        "reached_ready_n": reached_ready_n,
        "returned_to_zero": back_to_zero,
        "desired_replica_transition_sequence": sequence,
        "max_desired_replicas": max(observed_desired, default=0),
        "max_ready_replicas": max(observed_ready, default=0),
        "sample_count": len(samples),
        "duration_seconds": round(time.monotonic() - start_at, 3),
        "worker_log_files": sorted(f"worker-{name}.log" for name in log_procs),
    }
    (RUN / "experiment-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["success"]:
        raise SystemExit(2)
finally:
    for proc in log_procs.values():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    time.sleep(1)
    for handle in log_files.values():
        handle.close()
    for proc in processes:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    forward_log.close()
    proxy_log.close()
