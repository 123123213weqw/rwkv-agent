const $ = (selector) => document.querySelector(selector);
const API_VERSION = "rwkv-agent.service.v1";

const elements = {
  composer: $("#composer"),
  input: $("#message-input"),
  send: $("#send"),
  messages: $("#messages"),
  charCount: $("#char-count"),
  connection: $("#connection"),
  connectionLabel: $("#connection-label"),
  sessionLabel: $("#session-label"),
  sessionState: $("#session-state"),
  stateBadge: $("#state-badge span:last-child"),
  topModel: $("#top-model"),
  topBackend: $("#top-backend"),
  runtimeModel: $("#runtime-model"),
  runtimeContext: $("#runtime-context"),
  metricController: $("#metric-controller"),
  metricBackend: $("#metric-backend"),
  metricState: $("#metric-state"),
  metricSandbox: $("#metric-sandbox"),
  toolCount: $("#tool-count"),
  toolStack: $("#tool-stack"),
  traceSummary: $("#trace-summary"),
  lastTurnStatus: $("#last-turn-status"),
  endpointLabel: $("#endpoint-label"),
  toast: $("#toast"),
  conversationHead: $("#conversation-head"),
  composerWrap: $("#composer-wrap"),
  taskView: $("#task-view"),
  taskList: $("#task-list"),
  tasksTotal: $("#tasks-total"),
  tasksRunning: $("#tasks-running"),
  tasksComplete: $("#tasks-complete"),
  tasksFailed: $("#tasks-failed"),
};

let pending = false;
let healthTimer = null;
let taskTimer = null;
let toastTimer = null;
let sessionId = loadSession();
let ownerId = loadOwner();
const taskViewActive = window.location.pathname === "/tasks";

function loadSession() {
  const stored = localStorage.getItem("rwkv-agent-session");
  if (stored) return stored;
  const created = `web-${crypto.randomUUID()}`;
  localStorage.setItem("rwkv-agent-session", created);
  return created;
}

function loadOwner() {
  const stored = localStorage.getItem("rwkv-agent-owner");
  if (stored) return stored;
  const created = `web-owner-${crypto.randomUUID()}`;
  localStorage.setItem("rwkv-agent-owner", created);
  return created;
}

function requestId() {
  return `web-request-${crypto.randomUUID()}`;
}

function requestIdentity() {
  return {
    api_version: API_VERSION,
    request_id: requestId(),
    owner_id: ownerId,
  };
}

function short(value, limit = 40) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function formatModel(value) {
  return String(value || "RWKV G1I 13.3B")
    .replace(/^rwkv7-/i, "RWKV-7 ")
    .replace(/preview4922/i, "Preview4922");
}

function sanitizeAnswer(value) {
  return String(value ?? "")
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<tool_(?:call|result)>[\s\S]*?<\/tool_(?:call|result)>/gi, "")
    .replace(/^\s*(?:System|User|Assistant|Tool):\s*/gim, "")
    .trim();
}

function setText(element, value) {
  if (element) element.textContent = String(value ?? "");
}

function toast(message) {
  setText(elements.toast, message);
  elements.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 2200);
}

function updateSessionLabels() {
  setText(elements.sessionLabel, short(sessionId.replace(/^web-/, ""), 22));
  setText(elements.endpointLabel, window.location.origin);
}

function autoSizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 190)}px`;
  const count = [...elements.input.value].length;
  setText(elements.charCount, count.toLocaleString());
  elements.send.disabled = pending || !elements.input.value.trim();
}

function clearWelcome() {
  const welcome = elements.messages.querySelector(".welcome-card");
  if (welcome) welcome.remove();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  });
}

function messageRow(role, text, meta = {}) {
  const article = document.createElement("article");
  article.className = `message-row ${role}`;

  const header = document.createElement("div");
  header.className = "message-meta";
  const author = document.createElement("strong");
  author.textContent = role === "user" ? "YOU" : "RWKV";
  const time = document.createElement("span");
  time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  header.append(author, time);

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  article.append(header, body);

  if (role === "assistant") attachMessageFooter(article, meta);
  elements.messages.append(article);
  scrollToBottom();
  return article;
}

function attachMessageFooter(article, meta = {}) {
  if ((meta.elapsed || meta.state) && !article.querySelector(".message-foot")) {
    const footer = document.createElement("div");
    footer.className = "message-foot";
    if (meta.state) {
      const state = document.createElement("span");
      state.className = "ok";
      state.textContent = meta.state;
      footer.append(state);
    }
    if (meta.elapsed) {
      const elapsed = document.createElement("span");
      elapsed.textContent = meta.elapsed;
      footer.append(elapsed);
    }
    article.append(footer);
  }
}

function thinkingRow() {
  const row = document.createElement("div");
  row.className = "thinking-card";
  row.id = "thinking";
  const bars = document.createElement("span");
  bars.className = "thinking-bars";
  bars.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
  const label = document.createElement("span");
  label.textContent = "Opening recurrent state…";
  row.append(bars, label);
  elements.messages.append(row);
  scrollToBottom();

  const phases = ["Opening recurrent state…", "Greedy decoding…", "Waiting for verified completion…"];
  let index = 0;
  const timer = setInterval(() => {
    index = Math.min(index + 1, phases.length - 1);
    label.textContent = phases[index];
  }, 5000);
  return {
    setPhase(phase) {
      clearInterval(timer);
      const names = {
        routing: "Routing with cached gate state…",
        decoding: "Greedy decoding · streaming tokens…",
        tool: "Executing verified tool loop…",
      };
      label.textContent = names[phase] || String(phase || "Working…");
    },
    stop() {
      clearInterval(timer);
      row.remove();
    },
  };
}

function evidenceCount(result) {
  return Array.isArray(result?.evidence) ? result.evidence.length : 0;
}

function resultSummary(step) {
  const args = step?.arguments || {};
  const result = step?.result || {};
  if (step.name === "run_command") {
    const output = String(result.stdout || result.stderr || "").trim();
    return `${short(args.command, 76)}${output ? ` → ${short(output, 70)}` : ""}`;
  }
  if (step.name === "long_text_qa") {
    const hint = result.answer_hint ? ` · ${result.answer_hint}` : "";
    return `${evidenceCount(result)} grounded evidence${hint}`;
  }
  const query = args.query || args.question || "";
  return `${short(query, 74)} · ${evidenceCount(result)} evidence`;
}

function renderToolTimeline(response) {
  const steps = response?.trace?.agent?.tool_steps;
  if (!Array.isArray(steps) || !steps.length) return;
  const timeline = document.createElement("section");
  timeline.className = "tool-timeline";
  for (const step of steps) {
    const card = document.createElement("div");
    card.className = "tool-card";
    const head = document.createElement("div");
    head.className = "tool-card-head";
    const name = document.createElement("strong");
    name.textContent = `${String(step.step || "·").padStart(2, "0")}  ${step.name || "tool"}`;
    const status = document.createElement("span");
    status.textContent = step?.result?.status || "complete";
    const summary = document.createElement("p");
    summary.textContent = resultSummary(step);
    head.append(name, status);
    card.append(head, summary);
    timeline.append(card);
  }
  elements.messages.append(timeline);
  scrollToBottom();
}

function elapsedLabel(response, clientElapsed) {
  const milliseconds = Number(response?.trace?.elapsed_ms || clientElapsed);
  return milliseconds >= 1000 ? `${(milliseconds / 1000).toFixed(1)}s` : `${Math.round(milliseconds)}ms`;
}

function stateLabel(response) {
  const state = response?.trace?.context?.session_state;
  if (!state?.used) return "state released";
  return state.reused ? "state reused" : state.cached ? "state cached" : "state complete";
}

function renderTrace(response) {
  const events = response?.trace?.agent?.events || [];
  const directState = response?.trace?.context?.session_state;
  elements.traceSummary.className = "trace-list";
  elements.traceSummary.replaceChildren();

  const rows = [];
  if (directState) {
    rows.push([directState.reused ? "Reused recurrent state" : "Opened recurrent state", true]);
    if (directState.cached) rows.push([`${directState.seen_tokens || 0} tokens retained`, true]);
  } else if (Array.isArray(events)) {
    for (const event of events) {
      if (event.type === "state_opened") rows.push([`State ${short(event.state_id, 16)} opened`, false]);
      if (event.type === "tool_completed") rows.push([`${event.name} · ${event.status}`, event.status === "ok"]);
      if (event.type === "state_released") rows.push([`State released · ${event.success ? "clean" : "error"}`, Boolean(event.success)]);
    }
  }
  if (!rows.length) rows.push(["Turn completed without retained tool state", true]);
  for (const [label, ok] of rows.slice(-6)) {
    const item = document.createElement("div");
    item.className = `trace-item${ok ? " ok" : ""}`;
    item.textContent = label;
    elements.traceSummary.append(item);
  }
}

function errorCard(message) {
  const card = document.createElement("div");
  card.className = "error-card";
  card.textContent = message;
  elements.messages.append(card);
  scrollToBottom();
}

function formatElapsed(milliseconds) {
  const value = Number(milliseconds || 0);
  if (value >= 60_000) return `${(value / 60_000).toFixed(1)}m`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

function renderTasks(payload) {
  const counts = payload?.counts || {};
  const tasks = Array.isArray(payload?.tasks) ? payload.tasks : [];
  setText(elements.tasksTotal, counts.total || 0);
  setText(elements.tasksRunning, counts.running || 0);
  setText(elements.tasksComplete, counts.complete || 0);
  setText(elements.tasksFailed, counts.failed || 0);
  elements.taskList.replaceChildren();
  if (!tasks.length) {
    const empty = document.createElement("div");
    empty.className = "task-empty";
    empty.textContent = "No tasks have reached this Controller yet.";
    elements.taskList.append(empty);
    return;
  }
  for (const task of tasks) {
    const row = document.createElement("article");
    row.className = `task-row status-${task.status || "unknown"}`;

    const indicator = document.createElement("span");
    indicator.className = "task-indicator";

    const identity = document.createElement("div");
    identity.className = "task-identity";
    const id = document.createElement("strong");
    id.textContent = task.id || "task";
    const kind = document.createElement("small");
    kind.textContent = `${String(task.kind || "agent").toUpperCase()} · ${short(task.session_id, 20)}`;
    identity.append(id, kind);

    const copy = document.createElement("div");
    copy.className = "task-copy";
    const prompt = document.createElement("p");
    prompt.textContent = task.message || "—";
    const trace = document.createElement("small");
    const route = task.route ? `route ${task.route}` : "routing";
    trace.textContent = `${route} · state ${task.state || "active"} · ${task.tool_count || 0} tools`;
    copy.append(prompt, trace);

    const result = document.createElement("div");
    result.className = "task-result";
    const status = document.createElement("strong");
    status.textContent = String(task.status || "unknown").toUpperCase();
    const elapsed = document.createElement("small");
    elapsed.textContent = formatElapsed(task.elapsed_ms);
    result.append(status, elapsed);

    row.append(indicator, identity, copy, result);
    elements.taskList.append(row);
  }
}

async function refreshTasks() {
  try {
    const query = new URLSearchParams(requestIdentity());
    renderTasks(await request(`/v1/tasks?${query}`, { headers: {} }));
  } catch (error) {
    elements.taskList.replaceChildren();
    const card = document.createElement("div");
    card.className = "task-empty error";
    card.textContent = `Task wall unavailable: ${error.message}`;
    elements.taskList.append(card);
  }
}

function configureView() {
  document.body.dataset.view = taskViewActive ? "tasks" : "chat";
  for (const link of document.querySelectorAll("[data-view]")) {
    link.classList.toggle("active", link.dataset.view === (taskViewActive ? "tasks" : "chat"));
  }
  if (!taskViewActive) return;
  elements.conversationHead.hidden = true;
  elements.messages.hidden = true;
  elements.composerWrap.hidden = true;
  elements.taskView.hidden = false;
  refreshTasks();
  taskTimer = setInterval(refreshTasks, 1_000);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers || {}) },
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Controller returned HTTP ${response.status} without JSON`);
  }
  if (!response.ok) throw new Error(body.error || body.message || `HTTP ${response.status}`);
  return body;
}

async function requestStream(path, payload, onEvent) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(text || `Controller returned HTTP ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = done ? "" : lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "error") throw new Error(event.error || "Stream failed");
      onEvent(event);
    }
    if (done) break;
  }
}

async function sendMessage(message) {
  if (pending || !message.trim()) return;
  clearWelcome();
  pending = true;
  autoSizeInput();
  messageRow("user", message);
  const thinking = thinkingRow();
  setText(elements.lastTurnStatus, "RUNNING");
  setText(elements.sessionState, "State active");
  setText(elements.stateBadge, "STATE ACTIVE");
  const started = performance.now();
  let assistantRow = null;
  let response = null;

  try {
    await requestStream(
      "/v1/tasks/stream",
      { ...requestIdentity(), session_id: sessionId, message },
      (event) => {
      if (event.type === "phase") {
        thinking.setPhase(event.phase);
        return;
      }
      if (event.type === "delta") {
        if (!assistantRow) {
          thinking.stop();
          assistantRow = messageRow("assistant", "");
        }
        const visible = sanitizeAnswer(event.text);
        setText(assistantRow.querySelector(".message-body"), visible);
        scrollToBottom();
        return;
      }
      if (event.type === "final") response = event.response;
      },
    );
    if (!response) throw new Error("Controller stream ended without a final event");
    thinking.stop();
    renderToolTimeline(response);
    const answer = sanitizeAnswer(response.answer) || "The turn completed without a displayable answer.";
    const meta = {
      elapsed: elapsedLabel(response, performance.now() - started),
      state: stateLabel(response),
    };
    if (assistantRow) {
      setText(assistantRow.querySelector(".message-body"), answer);
      attachMessageFooter(assistantRow, meta);
    } else {
      messageRow("assistant", answer, meta);
    }
    renderTrace(response);
    setText(elements.lastTurnStatus, response.status === "ok" ? "COMPLETE" : String(response.status).toUpperCase());
    setText(elements.sessionState, response?.trace?.context?.session_state?.reused ? "State reused" : "State ready");
    setText(elements.stateBadge, response?.trace?.context?.session_state?.reused ? "STATE REUSED" : "STATE READY");
  } catch (error) {
    thinking.stop();
    errorCard(`Request failed: ${error.message}`);
    setText(elements.lastTurnStatus, "ERROR");
    setText(elements.sessionState, "Turn failed");
    setText(elements.stateBadge, "STATE READY");
  } finally {
    pending = false;
    elements.input.value = "";
    autoSizeInput();
    elements.input.focus();
  }
}

function renderTools(tools) {
  elements.toolStack.replaceChildren();
  const icons = { run_command: "⌘", knowledge_search: "◫", long_text_qa: "≋", web_search: "◎" };
  for (const tool of tools) {
    const chip = document.createElement("div");
    chip.className = "tool-chip";
    const icon = document.createElement("span");
    icon.textContent = icons[tool] || "◇";
    const name = document.createElement("strong");
    name.textContent = tool;
    const status = document.createElement("small");
    chip.append(icon, name, status);
    elements.toolStack.append(chip);
  }
  setText(elements.toolCount, `${tools.length} READY`);
}

async function refreshHealth(showToast = false) {
  try {
    const health = await request("/ready", { headers: {} });
    const model = Array.isArray(health.model) ? health.model[0] || {} : {};
    const modelName = formatModel(model.model);
    const backend = model.backend || "unknown";
    const state = health?.context?.session_state || {};
    const sandbox = health?.command?.sandbox || "disabled";
    const tools = Array.isArray(health.tools) ? health.tools : [];

    elements.connection.dataset.status = health.status === "ready" ? "ready" : "error";
    setText(elements.connectionLabel, health.status === "ready" ? "RUNTIME READY" : "DEGRADED");
    setText(elements.topModel, short(modelName, 28));
    setText(elements.topBackend, backend === "hf_recurrent" ? "ROCm · recurrent" : backend);
    setText(elements.runtimeModel, short(modelName, 34));
    setText(elements.runtimeContext, `Context ${Number(model.context || 0).toLocaleString()} · greedy`);
    setText(elements.metricController, health.status || "unknown");
    setText(elements.metricBackend, backend);
    setText(elements.metricState, `${state.allocated || 0}/${state.capacity || 0} resident`);
    setText(elements.metricSandbox, sandbox.replace(/_no_unsafe_fallback$/, ""));
    renderTools(tools);
    if (showToast) toast("Runtime status refreshed");
  } catch (error) {
    elements.connection.dataset.status = "error";
    setText(elements.connectionLabel, "OFFLINE");
    setText(elements.metricController, "unreachable");
    if (showToast) toast(`Runtime unavailable: ${error.message}`);
  }
}

function newSession() {
  sessionId = `web-${crypto.randomUUID()}`;
  localStorage.setItem("rwkv-agent-session", sessionId);
  if (taskViewActive) {
    window.location.assign("/");
    return;
  }
  updateSessionLabels();
  elements.messages.replaceChildren();
  const note = document.createElement("article");
  note.className = "message-row assistant";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = "New isolated session created. No previous transcript or recurrent state is attached.";
  note.append(body);
  elements.messages.append(note);
  elements.traceSummary.className = "trace-empty";
  elements.traceSummary.replaceChildren();
  const line = document.createElement("span");
  line.className = "trace-line";
  const text = document.createElement("p");
  text.textContent = "Waiting for the first turn in this session.";
  elements.traceSummary.append(line, text);
  setText(elements.lastTurnStatus, "IDLE");
  setText(elements.sessionState, "State ready");
  setText(elements.stateBadge, "STATE READY");
  toast("Created an isolated session");
  elements.input.focus();
}

elements.input.addEventListener("input", autoSizeInput);
elements.input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(elements.input.value);
});
$("#new-session").addEventListener("click", newSession);
$("#refresh-health").addEventListener("click", () => refreshHealth(true));
$("#runtime-refresh").addEventListener("click", () => refreshHealth(true));
document.addEventListener("keydown", (event) => {
  if (event.metaKey && event.key.toLowerCase() === "k") {
    event.preventDefault();
    newSession();
  }
});
for (const starter of document.querySelectorAll(".starter")) {
  starter.addEventListener("click", () => {
    elements.input.value = starter.dataset.prompt || "";
    autoSizeInput();
    elements.input.focus();
  });
}

updateSessionLabels();
autoSizeInput();
configureView();
refreshHealth();
healthTimer = setInterval(() => refreshHealth(false), 15_000);
window.addEventListener("beforeunload", () => {
  clearInterval(healthTimer);
  clearInterval(taskTimer);
});
