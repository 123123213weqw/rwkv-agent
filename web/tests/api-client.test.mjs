import assert from "node:assert/strict";
import test from "node:test";

import { API_VERSION, AgentApi } from "../dist/api-client.js";

const identity = {
  api_version: API_VERSION,
  request_id: "request-test",
  owner_id: "owner-test",
};

function chunkedResponse(chunks, status = 200) {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  }), {
    status,
    headers: { "content-type": "application/x-ndjson" },
  });
}

function streamClient(chunks) {
  return new AgentApi("", async () => chunkedResponse(chunks));
}

test("stream parser accepts split NDJSON lines and preserves unknown events", async () => {
  const received = [];
  const client = streamClient([
    '{"type":"task_created","sequence":1,"task_id":"task-1"}\n{"type":"future_',
    'event","sequence":2,"payload":{"safe":true}}\n',
    '{"type":"final","sequence":3,"response":{"status":"ok","answer":"done"}}\n',
  ]);
  const terminal = await client.streamTask(
    { ...identity, session_id: "session-test", message: "hello" },
    (event) => received.push(event),
  );
  assert.deepEqual(received.map((event) => event.type), ["task_created", "future_event", "final"]);
  assert.equal(terminal.type, "final");
  assert.equal(terminal.response.answer, "done");
});

test("stream parser rejects a sequence gap", async () => {
  const client = streamClient([
    '{"type":"task_created","sequence":1}\n',
    '{"type":"final","sequence":3,"response":{"status":"ok"}}\n',
  ]);
  await assert.rejects(
    client.streamTask({ ...identity, session_id: "session-test", message: "hello" }, () => {}),
    /expected 2, got 3/,
  );
});

test("stream parser rejects data after the unique terminal event", async () => {
  const client = streamClient([
    '{"type":"final","sequence":1,"response":{"status":"ok"}}\n',
    '{"type":"runtime_event","sequence":2}\n',
  ]);
  await assert.rejects(
    client.streamTask({ ...identity, session_id: "session-test", message: "hello" }, () => {}),
    /after a terminal event/,
  );
});

test("stream error is surfaced as a terminal failure", async () => {
  const received = [];
  const client = streamClient(['{"type":"error","sequence":1,"error":"provider unavailable"}\n']);
  await assert.rejects(
    client.streamTask(
      { ...identity, session_id: "session-test", message: "hello" },
      (event) => received.push(event.type),
    ),
    /provider unavailable/,
  );
  assert.deepEqual(received, ["error"]);
});

test("owner identity is included in task list and control requests", async () => {
  const calls = [];
  const client = new AgentApi("", async (input, init = {}) => {
    calls.push({ url: String(input), init });
    return Response.json({ status: "ok", tasks: [], counts: {} });
  });
  await client.listTasks(identity);
  await client.cancelTask("task/unsafe", identity);
  await client.resumeTask("task-resume", identity);
  assert.match(calls[0].url, /api_version=rwkv-agent\.service\.v1/);
  assert.match(calls[0].url, /owner_id=owner-test/);
  assert.equal(calls[1].url, "/v1/tasks/task%2Funsafe/cancel");
  assert.deepEqual(JSON.parse(calls[1].init.body), identity);
  assert.equal(calls[2].url, "/v1/tasks/task-resume/resume");
  assert.deepEqual(JSON.parse(calls[2].init.body), identity);
});

test("readiness keeps a structured 503 body for component diagnostics", async () => {
  const client = new AgentApi("", async () => Response.json({
    status: "unavailable",
    components: { model_sidecar: { status: "unavailable" } },
  }, { status: 503 }));
  const result = await client.ready();
  assert.equal(result.ok, false);
  assert.equal(result.status, 503);
  assert.equal(result.body.components.model_sidecar.status, "unavailable");
});
