import { createReadStream, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.argv[2] || 19091);
const tasks = new Map();
let taskSequence = 3;

const now = Date.now();
for (const sample of [
  ["task-running", "Inspect the workspace and repair the failing inventory tests.", "running", now - 42_000],
  ["task-complete", "Summarize the API contract and verify every canonical route.", "succeeded", now - 180_000],
  ["task-failed", "Apply the migration and run the exact verifier.", "failed", now - 320_000],
]) {
  const [id, objective, status, created] = sample;
  tasks.set(id, taskRecord(id, "owner-demo", "session-demo", objective, status, created));
}

function taskRecord(id, ownerId, sessionId, objective, status = "running", created = Date.now()) {
  const terminal = ["succeeded", "failed", "cancelled", "interrupted"].includes(status);
  return {
    ledger_schema_version: "rwkv-agent.task-ledger.v1",
    task_id: id,
    request_id: `request-${id}`,
    owner_id: ownerId,
    session_id: sessionId,
    status,
    current_stage: terminal ? null : "verify",
    created_unix_ms: created,
    updated_unix_ms: terminal ? created + 12_500 : Date.now(),
    revision: terminal ? 6 : 3,
    recovery_count: status === "interrupted" ? 1 : 0,
    task_spec_summary: {
      objective_preview: objective,
      objective_chars: objective.length,
      acceptance_criteria_count: 2,
      constraint_count: 1,
      verification_command_count: 1,
      stage_count: 3,
    },
    stages: [
      { id: "inspect", status: "succeeded", attempts: 1, started_unix_ms: created, completed_unix_ms: created + 2_000, error: "" },
      { id: "repair", status: status === "failed" ? "failed" : "succeeded", attempts: 1, started_unix_ms: created + 2_100, completed_unix_ms: created + 8_000, error: status === "failed" ? "exact verifier failed" : "" },
      { id: "verify", status: status === "running" ? "running" : status === "succeeded" ? "succeeded" : "pending", attempts: status === "running" ? 1 : 0, started_unix_ms: created + 8_100, completed_unix_ms: status === "succeeded" ? created + 12_000 : null, error: "" },
    ],
    events: [
      { sequence: 1, unix_ms: created, kind: "task_created", stage_id: null, detail: "durable task created" },
      { sequence: 2, unix_ms: created + 100, kind: "stage_started", stage_id: "inspect", detail: "bounded stage started" },
      { sequence: 3, unix_ms: created + 2_000, kind: "stage_completed", stage_id: "inspect", detail: "checkpoint committed" },
    ],
    final_summary: terminal ? { status: status === "succeeded" ? "ok" : "error", route: "tool", answer_chars: 72, tool_steps: 3 } : null,
    error: status === "failed" ? "exact verifier failed" : null,
  };
}

function summary(task) {
  const status = task.status === "succeeded" ? "complete" : ["failed", "cancelled", "interrupted"].includes(task.status) ? "error" : task.status;
  return {
    id: task.task_id,
    request_id: task.request_id,
    owner_id: task.owner_id,
    session_id: task.session_id,
    message: task.task_spec_summary.objective_preview,
    kind: "agent",
    status,
    route: status === "running" ? "tool" : "direct",
    state: status === "running" ? task.current_stage : "released",
    tool_count: status === "running" ? 2 : 3,
    created_unix_ms: task.created_unix_ms,
    elapsed_ms: task.updated_unix_ms - task.created_unix_ms,
    error: task.error,
    revision: task.revision,
    recovery_count: task.recovery_count,
  };
}

function json(response, status = 200) {
  return { status, body: Buffer.from(JSON.stringify(response)), type: "application/json" };
}

function collect(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {}); }
      catch (error) { reject(error); }
    });
    request.on("error", reject);
  });
}

function sendFile(response, relative, type) {
  response.writeHead(200, {
    "content-type": type,
    "content-security-policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'",
    "x-content-type-options": "nosniff",
  });
  createReadStream(join(root, relative)).pipe(response);
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  if (request.method === "GET" && ["/", "/tasks", "/status"].includes(url.pathname)) return sendFile(response, "index.html", "text/html; charset=utf-8");
  if (request.method === "GET" && url.pathname === "/assets/app.css") return sendFile(response, "app.css", "text/css; charset=utf-8");
  if (request.method === "GET" && url.pathname === "/assets/app.js") return sendFile(response, "dist/app.js", "text/javascript; charset=utf-8");
  if (request.method === "GET" && url.pathname === "/assets/api-client.js") return sendFile(response, "dist/api-client.js", "text/javascript; charset=utf-8");
  if (request.method === "GET" && url.pathname === "/live") return reply(response, json({ status: "alive", api_version: "rwkv-agent.service.v1", control_plane: "rust", runtime_revision: "frontend-e2e" }));
  if (request.method === "GET" && url.pathname === "/ready") return reply(response, json({
    status: "ready",
    api_version: "rwkv-agent.service.v1",
    control_plane: "rust",
    runtime_revision: "frontend-e2e",
    tools: ["run_command", "read_file", "edit_file", "knowledge_search", "web_search"],
    model: [{ status: "ready", model: "rwkv7-g1j-13.3b", backend: "hf_recurrent", context: 16384, device: "cuda:1", dtype: "float16" }],
    components: {
      model_sidecar: { status: "ready" },
      data_plane: { status: "ready" },
      sandbox: { status: "ready", mode: "bubblewrap_no_network_no_unsafe_fallback" },
      state_capacity: { status: "ready", available: 6, capacity: 8 },
      task_ledger: { status: "ready", active_tasks: 1 },
    },
    agent_limits: { max_tool_steps: 18, max_model_tokens_per_turn: 1024, max_run_seconds: 1200 },
  }));
  if (request.method === "GET" && url.pathname === "/v1/tasks") {
    const ownerId = url.searchParams.get("owner_id");
    const visible = [...tasks.values()].filter((task) => ownerId === "owner-demo" || task.owner_id === ownerId).map(summary);
    const counts = { total: visible.length, running: visible.filter((task) => task.status === "running").length, complete: visible.filter((task) => task.status === "complete").length, failed: visible.filter((task) => task.status === "error").length };
    return reply(response, json({ status: "ok", counts, tasks: visible }));
  }
  const detail = url.pathname.match(/^\/v1\/tasks\/([^/]+)$/);
  if (request.method === "GET" && detail) {
    const task = tasks.get(decodeURIComponent(detail[1]));
    const ownerId = url.searchParams.get("owner_id");
    if (!task || (ownerId !== "owner-demo" && task.owner_id !== ownerId)) return reply(response, json({ status: "error", error: "not found" }, 404));
    return reply(response, json({ status: "ok", task }));
  }
  const control = url.pathname.match(/^\/v1\/tasks\/([^/]+)\/(cancel|resume)$/);
  if (request.method === "POST" && control) {
    const task = tasks.get(decodeURIComponent(control[1]));
    const body = await collect(request);
    if (!task || (body.owner_id !== "owner-demo" && body.owner_id !== task.owner_id)) return reply(response, json({ status: "error", error: "not found" }, 404));
    task.status = control[2] === "cancel" ? "cancelled" : "running";
    task.revision += 1;
    task.updated_unix_ms = Date.now();
    return reply(response, json({ status: "ok", task }));
  }
  if (request.method === "POST" && url.pathname === "/v1/research") {
    await collect(request);
    return reply(response, json({ status: "ok", answer: "The bounded research branches completed with three official sources.", trace: { elapsed_ms: 840, context: { session_state: { used: true, cached: true } }, agent: { tool_steps: [{ step: 1, name: "web_search", arguments: { query: "official source" }, result: { status: "ok", evidence: [{ id: "W1" }, { id: "W2" }, { id: "W3" }] } }] } } }));
  }
  if (request.method === "POST" && url.pathname === "/v1/tasks/stream") {
    const body = await collect(request);
    const taskId = `task-e2e-${++taskSequence}`;
    const task = taskRecord(taskId, body.owner_id, body.session_id, body.message);
    tasks.set(taskId, task);
    let completed = false;
    response.on("close", () => {
      if (!completed) {
        task.status = "cancelled";
        task.updated_unix_ms = Date.now();
        task.revision += 1;
      }
    });
    response.writeHead(200, { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store" });
    const fail = String(body.message).toLowerCase().includes("mock failure");
    const events = [
      { type: "task_created", sequence: 1, task_id: taskId, status: "running" },
      { type: "stage_started", sequence: 2, task_id: taskId, stage_id: "inspect", stage_index: 1, stage_count: 3 },
      { type: "future_event", sequence: 3, task_id: taskId, payload: { forward_compatible: true } },
      { type: "phase", sequence: 4, task_id: taskId, phase: "tool" },
      { type: "delta", sequence: 5, task_id: taskId, text: "<img src=x onerror=alert(1)> was rendered as plain text. Verified result: ready." },
      fail
        ? { type: "error", sequence: 6, task_id: taskId, error: "mock verifier failed" }
        : { type: "final", sequence: 6, task_id: taskId, response: { status: "ok", answer: "<img src=x onerror=alert(1)> was rendered as plain text. Verified result: ready.", trace: { elapsed_ms: 980, context: { session_state: { used: true, reused: true, cached: true } }, agent: { tool_steps: [{ step: 1, name: "run_command", arguments: { command: "printf '<script>unsafe</script>'" }, result: { status: "ok", stdout: "<script>unsafe</script>" } }] } } } },
    ];
    for (const [index, event] of events.entries()) {
      await new Promise((resolve) => setTimeout(resolve, index === 0 ? 30 : 70));
      const line = `${JSON.stringify(event)}\n`;
      if (index === 2) { response.write(line.slice(0, 23)); response.write(line.slice(23)); }
      else response.write(line);
    }
    task.status = fail ? "failed" : "succeeded";
    task.error = fail ? "mock verifier failed" : null;
    task.updated_unix_ms = Date.now();
    task.final_summary = { status: fail ? "error" : "ok", route: "tool", answer_chars: fail ? 0 : 72, tool_steps: 1 };
    completed = true;
    response.end();
    return;
  }
  reply(response, json({ status: "error", error: "not found" }, 404));
});

function reply(response, result) {
  response.writeHead(result.status, { "content-type": result.type, "content-length": result.body.length });
  response.end(result.body);
}

server.listen(port, "127.0.0.1", () => console.log(`frontend mock controller listening on http://127.0.0.1:${port}`));

process.on("SIGINT", () => server.close());

// Keep the fixture deterministic: fail immediately when checked-in assets are missing.
for (const path of ["index.html", "app.css", "dist/app.js", "dist/api-client.js"]) readFileSync(join(root, path));
