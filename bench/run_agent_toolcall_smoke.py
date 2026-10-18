#!/usr/bin/env python3
"""Smoke batteries for agent tool calling against a running rwkv-agent server.

Targets the canonical Rust controller (default http://127.0.0.1:8122, override
with RWKV_AGENT_ENDPOINT). Each battery prints one compact block per turn:
route, gate margin, compiled query, evidence admission, answer, and timing.

Batteries:
  gate     chat-vs-tool discrimination (control prompts vs search intent)
  tools    sandboxed command/file tools (needs --enable-command + a workspace
           subdir passed as --workspace)
  long     one 15-turn session mixing tools and recall (state persistence)
  staged   two-stage task_spec through the durable ledger
  all      everything above

This is a live smoke tool: it talks to real web search and a real sandbox.
It stores nothing and is safe to rerun.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "rwkv-agent.service.v1"


def base_url():
    return os.environ.get("RWKV_AGENT_ENDPOINT", "http://127.0.0.1:8122").rstrip("/")


def stream_turn(request_id, session_id, message, working_directory=None, timeout=240):
    body = {
        "api_version": API,
        "request_id": request_id,
        "owner_id": "smoke",
        "session_id": session_id,
        "message": message,
    }
    if working_directory:
        body["working_directory"] = working_directory
    req = urllib.request.Request(
        base_url() + "/v1/tasks/stream",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
    )
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            raw = raw.strip()
            if raw:
                events.append(json.loads(raw))
    return events


def report(label, events):
    final = next((e for e in events if e["type"] == "final"), None)
    if final is None:
        error = next((e for e in events if e["type"] == "error"), {})
        print(f"[{label}] ERROR {str(error.get('error', 'no terminal event'))[:300]}")
        return
    response = final["response"]
    trace = response.get("trace", {})
    gate = trace.get("gate", {})
    route = response.get("route", {})
    completion = trace.get("answer_completion", {}) or {}
    state = trace.get("context", {}).get("session_state", {})
    tool_result = response.get("tool_result") or {}
    admission = tool_result.get("evidence_admission", {})
    print(f"[{label}] {time.strftime('%H:%M:%S')}")
    print(f"  route    : {route.get('mode')} tool={route.get('tool')} steps={route.get('steps')}")
    if gate:
        scores = gate.get("scores", {})
        print(
            f"  gate     : chat={scores.get('chat')} tool={scores.get('tool')} "
            f"margin={gate.get('margin')} use_tool={gate.get('use_tool')}"
        )
    if tool_result:
        print(f"  query    : {tool_result.get('effective_query')!r}")
        print(
            f"  evidence : admitted={admission.get('admitted')} "
            f"rejected={admission.get('rejected')}"
        )
    if state:
        print(f"  state    : reused={state.get('reused')} seen_tokens={state.get('seen_tokens')}")
    answer = (response.get("answer") or "")[:300].replace("\n", " ")
    print(f"  answer   : {answer}")
    print(
        f"  timing   : model={completion.get('model_elapsed_ms')}ms "
        f"total={trace.get('elapsed_ms')}ms"
    )
    print()


GATE_BATTERY = [
    ("control-math", "What is 2+2? Reply with just the number."),
    ("control-greet", "Reply with exactly: hello"),
    ("explicit-search", "Search the web: what is the latest stable version of the Rust programming language?"),
    ("implicit-search", "What is new in the most recent RWKV-7 release?"),
    ("chinese-search", "搜索一下：RWKV-7 最新进展"),
]

TOOL_BATTERY = [
    ("cmd-echo", "Use run_command to run: echo smoke-ok. Then tell me the exact output."),
    ("file-write-read", "Create a file named smoke.txt containing one line: smoke value 7. Then read it back and tell me its exact contents."),
]

LONG_BATTERY = [
    ("01-create", "Create a file named mission.txt containing exactly one line: codeword BANANA-42"),
    ("02-recall", "From our conversation memory alone, without using any tool: what codeword did you write to mission.txt?"),
    ("03-search", "Search the web: who created the Linux kernel?"),
    ("04-recall-search", "In one short sentence: what did the search you just ran conclude?"),
    ("05-verify", "Read mission.txt and tell me its exact current contents"),
    ("06-summary", "Without using any tool, summarize in 3 short bullets what we have done together in this session."),
]


def run_battery(name, cases, workspace):
    stamp = time.strftime("%H%M%S")
    for index, (label, message) in enumerate(cases, start=1):
        body_extra = workspace if name in ("tools", "long") else None
        events = stream_turn(
            f"smoke-{name}-{stamp}-{index:02d}",
            f"smoke-{name}-{stamp}",
            message,
            body_extra,
        )
        report(label, events)


def staged_battery(workspace):
    body = {
        "api_version": API,
        "request_id": f"smoke-staged-{time.strftime('%H%M%S')}",
        "owner_id": "smoke",
        "session_id": f"smoke-staged-{time.strftime('%H%M%S')}",
        "task_spec": {
            "objective": "Find what RWKV stands for via web search, then save the answer to rwkv-meaning.txt in the workspace.",
            "acceptance_criteria": ["rwkv-meaning.txt contains the expansion of RWKV"],
            "working_directory": workspace,
            "stages": [
                {"id": "research", "objective": "Search the web for what the acronym RWKV stands for and note the exact expansion."},
                {"id": "persist", "objective": "Write the expansion found in the research stage into rwkv-meaning.txt (one line).", "depends_on": ["research"]},
            ],
        },
    }
    req = urllib.request.Request(
        base_url() + "/v1/tasks", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        print(f"[staged] HTTP {error.code} {error.read().decode()[:300]}")
        return
    trace = result.get("trace", {})
    print("[staged] status:", result.get("status"))
    print("  ledger:", json.dumps(trace.get("task_ledger", {}).get("stages")))
    print("  answer:", (result.get("answer") or "")[:200])
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("battery", choices=["gate", "tools", "long", "staged", "all"], nargs="?", default="gate")
    parser.add_argument("--workspace", default=None, help="existing workspace subdir for command tools")
    args = parser.parse_args()
    if not args.workspace:
        args.workspace = os.environ.get("RWKV_AGENT_WORKSPACE")
    if args.battery in ("tools", "all") and not args.workspace:
        print("tools battery needs --workspace (an existing subdirectory of the command workspace root)", file=sys.stderr)
        return 2
    selections = {
        "gate": [("gate", GATE_BATTERY)],
        "tools": [("tools", TOOL_BATTERY)],
        "long": [("long", LONG_BATTERY)],
        "staged": [],
        "all": [("gate", GATE_BATTERY), ("tools", TOOL_BATTERY), ("long", LONG_BATTERY)],
    }
    for name, cases in selections[args.battery]:
        run_battery(name, cases, args.workspace)
    if args.battery in ("staged", "all"):
        staged_battery(args.workspace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
