import { API_VERSION, AgentApi, newRequestId, } from "./api-client.js";
function element(selector) {
    const found = document.querySelector(selector);
    if (!found)
        throw new Error(`Frontend element is missing: ${selector}`);
    return found;
}
const elements = {
    layout: element("#app-layout"),
    sidebar: element("#sidebar"),
    messages: element("#messages"),
    conversationView: element("#conversation-view"),
    tasksView: element("#tasks-view"),
    statusView: element("#status-view"),
    composer: element("#composer"),
    input: element("#message-input"),
    send: element("#send"),
    sendLabel: element("#send-label"),
    charCount: element("#char-count"),
    sessionList: element("#session-list"),
    ownerLabel: element("#owner-label"),
    pageTitle: element("#page-title"),
    pageSubtitle: element("#page-subtitle"),
    modelPill: element("#model-pill"),
    connectionButton: element("#connection-button"),
    connectionLabel: element("#connection-label"),
    sidebarStatusPin: element("#sidebar-status-pin"),
    runningBadge: element("#task-running-badge"),
    toast: element("#toast"),
    inspector: element("#inspector"),
    inspectorTitle: element("#inspector-title"),
    contextOwner: element("#context-owner"),
    contextSession: element("#context-session"),
    lastTurnStatus: element("#last-turn-status"),
    turnTimeline: element("#turn-timeline"),
    toolCount: element("#tool-count"),
    toolList: element("#tool-list"),
    runtimeModel: element("#runtime-model"),
    runtimeContext: element("#runtime-context"),
    runtimeDetails: element("#runtime-detail-list"),
    taskList: element("#task-list"),
    taskTotals: {
        total: element("#tasks-total"),
        running: element("#tasks-running"),
        complete: element("#tasks-complete"),
        failed: element("#tasks-failed"),
    },
    componentGrid: element("#component-grid"),
    statusHero: element("#status-hero"),
    statusTitle: element("#status-title"),
    statusMessage: element("#status-message"),
    runtimeRevision: element("#runtime-revision"),
    modelStatus: element("#model-status"),
    modelDetails: element("#model-details"),
    limitDetails: element("#limit-details"),
    identityDetails: element("#identity-details"),
    taskDetailEmpty: element("#task-detail-empty"),
    taskDetail: element("#task-detail"),
    taskDetailDot: element("#task-detail-dot"),
    taskDetailTitle: element("#task-detail-title"),
    taskDetailMeta: element("#task-detail-meta"),
    taskDetailObjective: element("#task-detail-objective"),
    cancelTask: element("#cancel-task"),
    resumeTask: element("#resume-task"),
    stageCount: element("#stage-count"),
    stageList: element("#stage-list"),
    eventCount: element("#event-count"),
    eventList: element("#event-list"),
    finalSummary: element("#final-summary"),
};
const api = new AgentApi();
const OWNER_KEY = "rwkv-agent-owner";
const SESSION_KEY = "rwkv-agent-session";
const SESSIONS_KEY = "rwkv-agent-recent-sessions";
const TASK_KEY = "rwkv-agent-active-task";
const THEME_KEY = "rwkv-agent-theme";
let ownerId = storedIdentity(OWNER_KEY, "web-owner");
let sessionId = storedIdentity(SESSION_KEY, "web");
let runMode = "agent";
let pending = false;
let activeAbort = null;
let activeTaskId = localStorage.getItem(TASK_KEY) || "";
let selectedTaskId = "";
let toastTimer = 0;
let taskTimer = 0;
let healthTimer = 0;
let lastTasks = [];
const currentView = viewFromPath(window.location.pathname);
function isObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function asObject(value) {
    return isObject(value) ? value : {};
}
function asArray(value) {
    return Array.isArray(value) ? value.filter(isObject) : [];
}
function storedIdentity(key, prefix) {
    const stored = localStorage.getItem(key);
    if (stored)
        return stored;
    const created = `${prefix}-${crypto.randomUUID()}`;
    localStorage.setItem(key, created);
    return created;
}
function viewFromPath(path) {
    if (path === "/tasks")
        return "tasks";
    if (path === "/status")
        return "status";
    return "chat";
}
function identity() {
    return { api_version: API_VERSION, request_id: newRequestId(), owner_id: ownerId };
}
function short(value, limit = 48) {
    const text = String(value ?? "").replace(/\s+/g, " ").trim();
    return text.length > limit ? `${text.slice(0, Math.max(0, limit - 1))}…` : text;
}
function statusText(value) {
    return String(value || "unknown").toLowerCase();
}
function formatElapsed(milliseconds) {
    const value = Number(milliseconds || 0);
    if (value >= 60_000)
        return `${(value / 60_000).toFixed(1)}m`;
    if (value >= 1_000)
        return `${(value / 1_000).toFixed(1)}s`;
    return `${Math.round(value)}ms`;
}
function formatTime(unixMs) {
    const value = Number(unixMs || 0);
    if (!value)
        return "—";
    return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function modelName(value) {
    return String(value || "RWKV")
        .replace(/^rwkv7-/i, "RWKV-7 ")
        .replace(/preview4922/i, "Preview4922");
}
function cleanAnswer(value) {
    return String(value ?? "")
        .replace(/<think>[\s\S]*?<\/think>/gi, "")
        .replace(/<tool_(?:call|result)>[\s\S]*?<\/tool_(?:call|result)>/gi, "")
        .replace(/^\s*(?:System|User|Assistant|Tool):\s*/gim, "")
        .trim();
}
function setText(target, value) {
    target.textContent = String(value ?? "");
}
function toast(message) {
    setText(elements.toast, message);
    elements.toast.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 2400);
}
function clearChildren(target) {
    target.replaceChildren();
}
function autoSizeInput() {
    elements.input.style.height = "auto";
    elements.input.style.height = `${Math.min(elements.input.scrollHeight, 210)}px`;
    setText(elements.charCount, [...elements.input.value].length.toLocaleString());
    elements.send.disabled = !pending && !elements.input.value.trim();
}
function sessions() {
    try {
        const parsed = JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]");
        if (!Array.isArray(parsed))
            return [];
        return parsed.filter((item) => isObject(item) && typeof item.id === "string" && typeof item.title === "string" && typeof item.updated === "number");
    }
    catch {
        return [];
    }
}
function rememberSession(title) {
    const current = sessions();
    const old = current.find((item) => item.id === sessionId);
    const next = {
        id: sessionId,
        title: title ? short(title, 42) : old?.title || "New session",
        updated: Date.now(),
    };
    const merged = [next, ...current.filter((item) => item.id !== sessionId)]
        .sort((a, b) => b.updated - a.updated)
        .slice(0, 8);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(merged));
    renderSessions();
}
function renderSessions() {
    clearChildren(elements.sessionList);
    for (const session of sessions()) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `session-item${session.id === sessionId ? " active" : ""}`;
        button.dataset.sessionId = session.id;
        const glyph = document.createElement("span");
        glyph.textContent = "›_";
        const copy = document.createElement("span");
        const title = document.createElement("strong");
        title.textContent = session.title;
        const meta = document.createElement("small");
        meta.textContent = `${short(session.id.replace(/^web-/, ""), 14)} · ${new Date(session.updated).toLocaleDateString()}`;
        copy.append(title, meta);
        const active = document.createElement("i");
        button.append(glyph, copy, active);
        button.addEventListener("click", () => {
            if (session.id === sessionId && currentView === "chat")
                return;
            localStorage.setItem(SESSION_KEY, session.id);
            window.location.assign("/");
        });
        elements.sessionList.append(button);
    }
}
function createSession() {
    sessionId = `web-${crypto.randomUUID()}`;
    localStorage.setItem(SESSION_KEY, sessionId);
    localStorage.removeItem(TASK_KEY);
    rememberSession("New session");
    if (currentView !== "chat") {
        window.location.assign("/");
        return;
    }
    window.location.reload();
}
function configureView() {
    elements.conversationView.hidden = currentView !== "chat";
    elements.tasksView.hidden = currentView !== "tasks";
    elements.statusView.hidden = currentView !== "status";
    for (const link of document.querySelectorAll("[data-view]")) {
        link.classList.toggle("active", link.dataset.view === currentView);
    }
    const titles = {
        chat: [sessions().find((item) => item.id === sessionId)?.title || "New session", "State-native agent workspace"],
        tasks: ["Task wall", "Durable Task Ledger"],
        status: ["System status", "Rust control plane"],
    };
    const [title, subtitle] = titles[currentView];
    setText(elements.pageTitle, title);
    setText(elements.pageSubtitle, subtitle);
}
function setInspector(open, tab) {
    elements.layout.classList.toggle("inspector-open", open);
    elements.inspector.setAttribute("aria-hidden", String(!open));
    element("#inspector-toggle").setAttribute("aria-pressed", String(open));
    if (tab)
        selectInspectorTab(tab);
}
function selectInspectorTab(tab) {
    for (const button of document.querySelectorAll("[data-inspector-tab]")) {
        const selected = button.dataset.inspectorTab === tab;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-selected", String(selected));
    }
    for (const panel of document.querySelectorAll("[data-inspector-panel]")) {
        panel.hidden = panel.dataset.inspectorPanel !== tab;
    }
    const titles = { context: "Session context", runtime: "Runtime status", task: "Task detail" };
    setText(elements.inspectorTitle, titles[tab]);
}
function messageRow(role, text) {
    const article = document.createElement("article");
    article.className = `message-row ${role}`;
    const header = document.createElement("div");
    header.className = "message-header";
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "Y" : "R";
    const author = document.createElement("strong");
    author.textContent = role === "user" ? "You" : "RWKV Agent";
    const time = document.createElement("time");
    time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    header.append(avatar, author, time);
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;
    article.append(header, body);
    elements.messages.append(article);
    scrollMessages();
    return article;
}
function addMessageFooter(article, state, elapsed) {
    const footer = document.createElement("div");
    footer.className = "message-footer";
    const stateText = document.createElement("span");
    stateText.className = "ok";
    stateText.textContent = state;
    const elapsedText = document.createElement("span");
    elapsedText.textContent = elapsed;
    footer.append(stateText, elapsedText);
    article.append(footer);
}
function scrollMessages() {
    requestAnimationFrame(() => { elements.messages.scrollTop = elements.messages.scrollHeight; });
}
function thinkingCard(label = "Opening recurrent state…") {
    const card = document.createElement("div");
    card.className = "thinking-card";
    const spinner = document.createElement("span");
    spinner.className = "thinking-spinner";
    const text = document.createElement("span");
    text.textContent = label;
    card.append(spinner, text);
    elements.messages.append(card);
    scrollMessages();
    return { update: (value) => { text.textContent = value; }, stop: () => card.remove() };
}
function showError(message) {
    const card = document.createElement("div");
    card.className = "inline-error";
    card.textContent = message;
    elements.messages.append(card);
    scrollMessages();
}
function timelineRows(rows) {
    clearChildren(elements.turnTimeline);
    elements.turnTimeline.className = rows.length ? "timeline-list" : "timeline-empty";
    if (!rows.length) {
        elements.turnTimeline.textContent = "No runtime events yet.";
        return;
    }
    for (const row of rows.slice(-12)) {
        const item = document.createElement("div");
        const state = statusText(row.status);
        item.className = `timeline-item${["ok", "complete", "succeeded", "final"].includes(state) ? " ok" : state === "error" ? " error" : ""}`;
        const dot = document.createElement("i");
        const copy = document.createElement("span");
        copy.textContent = row.label;
        if (row.meta) {
            const meta = document.createElement("small");
            meta.textContent = row.meta;
            copy.append(meta);
        }
        item.append(dot, copy);
        elements.turnTimeline.append(item);
    }
}
function phaseLabel(event) {
    if (event.type === "task_created")
        return "Durable task created";
    if (event.type === "stage_started")
        return `Stage ${event.stage_index || "·"}/${event.stage_count || "·"} started`;
    if (event.type === "stage_completed")
        return `Stage ${event.stage_id || "·"} checkpointed`;
    if (event.type === "phase") {
        const names = { routing: "Selecting next action", tool: "Executing verified tool", decoding: "Streaming model output" };
        return names[String(event.phase)] || `Runtime phase · ${event.phase || "unknown"}`;
    }
    if (event.type === "final")
        return "Terminal response committed";
    if (event.type === "error")
        return "Task ended with an error";
    return `Runtime event · ${event.type}`;
}
function toolSummary(step) {
    const args = asObject(step.arguments);
    const result = asObject(step.result);
    if (step.name === "run_command") {
        return `${short(args.command, 70)}${result.stdout || result.stderr ? ` → ${short(result.stdout || result.stderr, 58)}` : ""}`;
    }
    const evidence = Array.isArray(result.evidence) ? result.evidence.length : 0;
    const query = args.query || args.question || args.path || "";
    return `${short(query, 70)}${evidence ? ` · ${evidence} evidence` : ""}` || "Completed with a validated result";
}
function renderToolsFromResponse(response) {
    const trace = asObject(response.trace);
    const agent = asObject(trace.agent);
    const steps = asArray(agent.tool_steps);
    if (!steps.length)
        return;
    const timeline = document.createElement("section");
    timeline.className = "tool-timeline";
    for (const step of steps) {
        const details = document.createElement("details");
        details.className = "tool-card";
        const summary = document.createElement("summary");
        const number = document.createElement("span");
        number.textContent = String(step.step || "·").padStart(2, "0");
        const name = document.createElement("strong");
        name.textContent = String(step.name || "tool");
        const status = document.createElement("b");
        status.textContent = String(asObject(step.result).status || "complete");
        summary.append(number, name, status);
        const text = document.createElement("p");
        text.textContent = toolSummary(step);
        const raw = document.createElement("pre");
        raw.className = "tool-json";
        raw.textContent = JSON.stringify({ arguments: step.arguments, result: step.result }, null, 2).slice(0, 12_000);
        details.append(summary, text, raw);
        timeline.append(details);
    }
    elements.messages.append(timeline);
    scrollMessages();
}
function responseState(response) {
    const state = asObject(asObject(asObject(response.trace).context).session_state);
    if (!state.used)
        return "State released";
    if (state.reused)
        return "State reused";
    if (state.cached)
        return "State cached";
    return "State complete";
}
function setPending(value) {
    pending = value;
    elements.send.disabled = !value && !elements.input.value.trim();
    elements.send.classList.toggle("stop", value);
    setText(elements.sendLabel, value ? "Stop" : "Send");
    setText(elements.send.querySelector("b") || elements.send, value ? "■" : "↑");
}
async function runAgent(message) {
    const requestId = newRequestId();
    const started = performance.now();
    const thinking = thinkingCard();
    const events = [];
    const assistant = { element: null };
    let finalResponse = {};
    let terminalSeen = false;
    try {
        await api.streamTask({ api_version: API_VERSION, request_id: requestId, owner_id: ownerId, session_id: sessionId, message }, (event) => {
            events.push({ label: phaseLabel(event), meta: event.stage_id || event.phase || `sequence ${event.sequence}`, status: event.type });
            timelineRows(events);
            if (event.task_id && !activeTaskId) {
                activeTaskId = event.task_id;
                localStorage.setItem(TASK_KEY, activeTaskId);
            }
            if (event.type === "task_created" && event.task_id) {
                activeTaskId = event.task_id;
                localStorage.setItem(TASK_KEY, activeTaskId);
                void refreshTasks();
            }
            if (event.type === "phase")
                thinking.update(phaseLabel(event));
            if (event.type === "stage_started")
                thinking.update(phaseLabel(event));
            if (event.type === "delta") {
                if (!assistant.element) {
                    thinking.stop();
                    assistant.element = messageRow("assistant", "");
                }
                const body = assistant.element.querySelector(".message-body");
                if (body)
                    body.textContent = cleanAnswer(event.text);
                scrollMessages();
            }
            if (event.type === "final") {
                terminalSeen = true;
                finalResponse = asObject(event.response);
            }
        }, activeAbort?.signal);
        thinking.stop();
        if (!terminalSeen)
            throw new Error("Controller stream ended without a final response");
        renderToolsFromResponse(finalResponse);
        const answer = cleanAnswer(finalResponse.answer) || "The task completed without a displayable answer.";
        if (!assistant.element)
            assistant.element = messageRow("assistant", answer);
        else {
            const body = assistant.element.querySelector(".message-body");
            if (body)
                body.textContent = answer;
        }
        addMessageFooter(assistant.element, responseState(finalResponse), formatElapsed(Number(asObject(finalResponse.trace).elapsed_ms) || performance.now() - started));
        setText(elements.lastTurnStatus, statusText(finalResponse.status) === "ok" ? "COMPLETE" : String(finalResponse.status || "complete").toUpperCase());
    }
    catch (error) {
        thinking.stop();
        const aborted = error instanceof DOMException && error.name === "AbortError";
        showError(aborted ? "Run stopped. Reconciling the durable task and State release…" : `Request failed: ${error instanceof Error ? error.message : String(error)}`);
        setText(elements.lastTurnStatus, aborted ? "CANCELLED" : "ERROR");
    }
    finally {
        if (activeTaskId)
            await reconcileTask(activeTaskId);
    }
}
async function runResearch(message) {
    const started = performance.now();
    const thinking = thinkingCard("Opening parallel recurrent branches…");
    timelineRows([{ label: "Bounded research started", meta: "4 branches · 2 rounds", status: "running" }]);
    try {
        const response = await api.research({ api_version: API_VERSION, request_id: newRequestId(), owner_id: ownerId, session_id: sessionId, message, branch_width: 4, max_rounds: 2 }, activeAbort?.signal);
        thinking.stop();
        const answer = cleanAnswer(response.answer) || "Research completed without a displayable answer.";
        const article = messageRow("assistant", answer);
        addMessageFooter(article, responseState(response), formatElapsed(Number(asObject(response.trace).elapsed_ms) || performance.now() - started));
        renderToolsFromResponse(response);
        timelineRows([{ label: "Bounded research started", meta: "4 branches · 2 rounds", status: "running" }, { label: "Research response completed", meta: formatElapsed(performance.now() - started), status: "complete" }]);
        setText(elements.lastTurnStatus, "COMPLETE");
    }
    catch (error) {
        thinking.stop();
        const aborted = error instanceof DOMException && error.name === "AbortError";
        showError(aborted ? "Research request stopped." : `Research failed: ${error instanceof Error ? error.message : String(error)}`);
        setText(elements.lastTurnStatus, aborted ? "CANCELLED" : "ERROR");
    }
}
async function sendMessage() {
    const message = elements.input.value.trim();
    if (pending) {
        activeAbort?.abort();
        return;
    }
    if (!message)
        return;
    document.querySelector("#welcome-card")?.remove();
    rememberSession(message);
    setText(elements.pageTitle, short(message, 54));
    messageRow("user", message);
    elements.input.value = "";
    autoSizeInput();
    activeAbort = new AbortController();
    setPending(true);
    setText(elements.lastTurnStatus, "RUNNING");
    try {
        if (runMode === "research")
            await runResearch(message);
        else
            await runAgent(message);
    }
    finally {
        activeAbort = null;
        setPending(false);
        elements.input.focus();
    }
}
function renderTaskWall(payload) {
    const counts = payload.counts || {};
    for (const key of Object.keys(elements.taskTotals))
        setText(elements.taskTotals[key], counts[key] || 0);
    const running = Number(counts.running || 0);
    setText(elements.runningBadge, running);
    elements.runningBadge.hidden = running === 0;
    lastTasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    clearChildren(elements.taskList);
    if (!lastTasks.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No tasks have reached this Controller.";
        elements.taskList.append(empty);
        return;
    }
    for (const task of lastTasks) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = `task-row status-${statusText(task.status)}${task.id === selectedTaskId ? " selected" : ""}`;
        const taskId = document.createElement("span");
        taskId.className = "task-id";
        const id = document.createElement("strong");
        id.textContent = short(task.id, 22);
        const session = document.createElement("small");
        session.textContent = short(task.session_id, 20);
        taskId.append(id, session);
        const objective = document.createElement("span");
        objective.className = "task-objective-cell";
        const objectiveText = document.createElement("strong");
        objectiveText.textContent = task.message || "Untitled task";
        const route = document.createElement("small");
        route.textContent = `${task.route || "routing"} · ${task.tool_count || 0} tools · rev ${task.revision || 0}`;
        objective.append(objectiveText, route);
        const runtime = document.createElement("span");
        runtime.className = "task-runtime-cell";
        const elapsed = document.createElement("strong");
        elapsed.textContent = formatElapsed(task.elapsed_ms);
        const state = document.createElement("small");
        state.textContent = String(task.state || "checkpointed");
        runtime.append(elapsed, state);
        const status = document.createElement("span");
        status.className = "task-status-cell";
        const dot = document.createElement("i");
        const statusLabel = document.createElement("span");
        statusLabel.textContent = task.status || "unknown";
        status.append(dot, statusLabel);
        row.append(taskId, objective, runtime, status);
        row.addEventListener("click", () => void openTaskDetail(task.id));
        elements.taskList.append(row);
    }
}
async function refreshTasks(showToast = false) {
    try {
        const payload = await api.listTasks(identity());
        renderTaskWall(payload);
        if (showToast)
            toast("Task wall refreshed");
    }
    catch (error) {
        if (currentView === "tasks") {
            clearChildren(elements.taskList);
            const empty = document.createElement("div");
            empty.className = "empty-state error";
            empty.textContent = `Task wall unavailable: ${error instanceof Error ? error.message : String(error)}`;
            elements.taskList.append(empty);
        }
    }
}
async function reconcileTask(taskId) {
    try {
        const response = await api.task(taskId, identity());
        if (selectedTaskId === taskId || elements.layout.classList.contains("inspector-open"))
            renderTaskDetail(asObject(response.task));
        await refreshTasks();
    }
    catch (error) {
        toast(`Task reconciliation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
}
async function openTaskDetail(taskId) {
    selectedTaskId = taskId;
    renderTaskWall({ counts: {
            total: lastTasks.length,
            running: lastTasks.filter((task) => task.status === "running").length,
            complete: lastTasks.filter((task) => task.status === "complete").length,
            failed: lastTasks.filter((task) => task.status === "error").length,
        }, tasks: lastTasks });
    elements.taskDetail.hidden = true;
    elements.taskDetailEmpty.hidden = false;
    elements.taskDetailEmpty.textContent = "Loading owner-scoped task detail…";
    setInspector(true, "task");
    try {
        const response = await api.task(taskId, identity());
        renderTaskDetail(asObject(response.task));
    }
    catch (error) {
        elements.taskDetailEmpty.textContent = `Task detail unavailable: ${error instanceof Error ? error.message : String(error)}`;
    }
}
function renderTaskDetail(task) {
    elements.taskDetailEmpty.hidden = true;
    elements.taskDetail.hidden = false;
    const status = statusText(task.status);
    elements.taskDetailDot.className = `task-status-dot ${status}`;
    setText(elements.taskDetailTitle, short(task.task_id, 32));
    setText(elements.taskDetailMeta, `${status.toUpperCase()} · revision ${task.revision || 0} · ${formatElapsed(Number(task.updated_unix_ms) - Number(task.created_unix_ms))}`);
    const summary = asObject(task.task_spec_summary);
    setText(elements.taskDetailObjective, summary.objective_preview || "No safe objective preview available.");
    elements.cancelTask.disabled = ["succeeded", "cancelled"].includes(status);
    elements.resumeTask.disabled = !["failed", "interrupted"].includes(status);
    elements.cancelTask.dataset.taskId = String(task.task_id || "");
    elements.resumeTask.dataset.taskId = String(task.task_id || "");
    const stages = asArray(task.stages);
    setText(elements.stageCount, stages.length);
    clearChildren(elements.stageList);
    for (const stage of stages) {
        const item = document.createElement("div");
        item.className = `stage-item ${statusText(stage.status)}`;
        const dot = document.createElement("i");
        const name = document.createElement("strong");
        name.textContent = String(stage.id || "stage");
        const attempts = document.createElement("small");
        attempts.textContent = `${stage.status || "pending"} · ${stage.attempts || 0} attempt`;
        item.append(dot, name, attempts);
        elements.stageList.append(item);
    }
    if (!stages.length)
        elements.stageList.textContent = "No stages recorded.";
    const events = asArray(task.events);
    setText(elements.eventCount, events.length);
    clearChildren(elements.eventList);
    for (const event of events.slice(-20)) {
        const item = document.createElement("div");
        const kind = statusText(event.kind);
        item.className = `event-item${kind.includes("complete") || kind.includes("succeed") ? " ok" : kind.includes("fail") || kind.includes("cancel") ? " error" : ""}`;
        const dot = document.createElement("i");
        const copy = document.createElement("span");
        copy.textContent = String(event.kind || "event");
        const meta = document.createElement("small");
        meta.textContent = `${formatTime(event.unix_ms)}${event.stage_id ? ` · ${event.stage_id}` : ""}${event.detail ? ` · ${short(event.detail, 74)}` : ""}`;
        copy.append(meta);
        item.append(dot, copy);
        elements.eventList.append(item);
    }
    if (!events.length)
        elements.eventList.textContent = "No events recorded.";
    renderDefinitionList(elements.finalSummary, asObject(task.final_summary), ["status", "route", "answer_chars", "tool_steps"]);
}
async function controlSelectedTask(action) {
    const button = action === "cancel" ? elements.cancelTask : elements.resumeTask;
    const taskId = button.dataset.taskId || selectedTaskId;
    if (!taskId || button.disabled)
        return;
    button.disabled = true;
    try {
        if (action === "cancel")
            await api.cancelTask(taskId, identity());
        else
            await api.resumeTask(taskId, identity());
        toast(action === "cancel" ? "Cancellation persisted" : "Resume accepted");
        await openTaskDetail(taskId);
        await refreshTasks();
    }
    catch (error) {
        toast(`${action === "cancel" ? "Cancel" : "Resume"} failed: ${error instanceof Error ? error.message : String(error)}`);
        button.disabled = false;
    }
}
function renderDefinitionList(target, values, keys) {
    clearChildren(target);
    const entries = keys ? keys.map((key) => [key, values[key]]) : Object.entries(values);
    for (const [key, value] of entries) {
        if (value === undefined || value === null || typeof value === "object")
            continue;
        const row = document.createElement("div");
        const term = document.createElement("dt");
        term.textContent = key.replaceAll("_", " ");
        const detail = document.createElement("dd");
        detail.textContent = String(value);
        detail.title = String(value);
        row.append(term, detail);
        target.append(row);
    }
    if (!target.childElementCount) {
        const row = document.createElement("div");
        const term = document.createElement("dt");
        term.textContent = "status";
        const detail = document.createElement("dd");
        detail.textContent = "not reported";
        row.append(term, detail);
        target.append(row);
    }
}
function componentMessage(value) {
    if (typeof value.error === "string" && value.error)
        return short(value.error, 66);
    if (typeof value.mode === "string")
        return value.mode;
    if (typeof value.active_tasks === "number")
        return `${value.active_tasks} active task`;
    if (typeof value.available === "number" || typeof value.capacity === "number")
        return `${value.available || 0}/${value.capacity || 0} available`;
    return "No blocking error reported";
}
function renderStatus(liveResult, readyResult) {
    const live = liveResult.body;
    const ready = readyResult.body;
    const liveOk = liveResult.ok && live.status === "alive";
    const readyOk = readyResult.ok && ready.status === "ready";
    const overall = !liveOk ? "error" : readyOk ? "ready" : "error";
    elements.statusHero.dataset.status = overall;
    elements.connectionButton.dataset.status = overall;
    elements.sidebarStatusPin.className = `status-pin ${overall}`;
    setText(elements.connectionLabel, readyOk ? "Ready" : liveOk ? "Degraded" : "Offline");
    setText(elements.statusTitle, readyOk ? "Runtime ready" : liveOk ? "Process alive, dependencies unavailable" : "Controller offline");
    setText(elements.statusMessage, readyOk ? "All required model, data, State and sandbox checks passed." : "Inspect the component status below before starting a task.");
    setText(elements.runtimeRevision, ready.runtime_revision || live.runtime_revision || "—");
    const components = asObject(ready.components);
    clearChildren(elements.componentGrid);
    const names = { model_sidecar: "Model", data_plane: "Data plane", sandbox: "Sandbox", state_capacity: "State capacity", task_ledger: "Task ledger", statepool_cloud_plugin: "StatePool" };
    for (const key of ["model_sidecar", "data_plane", "sandbox", "state_capacity", "task_ledger"]) {
        const value = asObject(components[key]);
        const status = statusText(value.status);
        const card = document.createElement("article");
        card.className = `component-card ${["ready", "ok", "available"].includes(status) ? "ready" : "error"}`;
        const header = document.createElement("header");
        const label = document.createElement("span");
        label.textContent = names[key] || key;
        const dot = document.createElement("i");
        header.append(label, dot);
        const state = document.createElement("strong");
        state.textContent = status;
        const message = document.createElement("small");
        message.textContent = componentMessage(value);
        card.append(header, state, message);
        elements.componentGrid.append(card);
    }
    const models = Array.isArray(ready.model) ? ready.model.filter(isObject) : [];
    const model = models[0] || {};
    const displayedModel = modelName(model.model);
    setText(elements.modelPill, `${short(displayedModel, 30)} · ${model.backend || "runtime"}`);
    setText(elements.runtimeModel, displayedModel);
    setText(elements.runtimeContext, `Context ${Number(model.context || 0).toLocaleString()} · greedy`);
    setText(elements.modelStatus, model.status || (models.length ? "reported" : "unavailable"));
    renderDefinitionList(elements.modelDetails, model, ["model", "backend", "context", "device", "dtype"]);
    renderDefinitionList(elements.limitDetails, asObject(ready.agent_limits));
    renderDefinitionList(elements.identityDetails, {
        api_version: ready.api_version,
        control_plane: ready.control_plane,
        runtime_revision: ready.runtime_revision,
        liveness_http: liveResult.status,
        readiness_http: readyResult.status,
    });
    renderDefinitionList(elements.runtimeDetails, {
        controller: ready.status,
        backend: model.backend || "—",
        runtime_revision: ready.runtime_revision || "—",
        data_plane: asObject(components.data_plane).status || "—",
        sandbox: asObject(components.sandbox).mode || "—",
        state_capacity: componentMessage(asObject(components.state_capacity)),
        task_ledger: asObject(components.task_ledger).status || "—",
    });
    renderToolList(Array.isArray(ready.tools) ? ready.tools.map(String) : []);
}
function renderToolList(tools) {
    clearChildren(elements.toolList);
    setText(elements.toolCount, `${tools.length} READY`);
    const glyphs = { run_command: "⌘", knowledge_search: "◫", long_text_qa: "≋", web_search: "◎", read_file: "R", edit_file: "E", write_file: "W" };
    for (const tool of tools) {
        const chip = document.createElement("div");
        chip.className = "tool-chip";
        const icon = document.createElement("span");
        icon.textContent = glyphs[tool] || "◇";
        const name = document.createElement("strong");
        name.textContent = tool;
        const dot = document.createElement("i");
        chip.append(icon, name, dot);
        elements.toolList.append(chip);
    }
}
async function refreshStatus(showToast = false) {
    try {
        const [live, ready] = await Promise.all([api.live(), api.ready()]);
        renderStatus(live, ready);
        if (showToast)
            toast("Runtime status refreshed");
    }
    catch (error) {
        elements.connectionButton.dataset.status = "error";
        elements.sidebarStatusPin.className = "status-pin error";
        elements.statusHero.dataset.status = "error";
        setText(elements.connectionLabel, "Offline");
        setText(elements.statusTitle, "Controller unreachable");
        setText(elements.statusMessage, error instanceof Error ? error.message : String(error));
        if (showToast)
            toast("Runtime is unavailable");
    }
}
function setMode(mode) {
    runMode = mode;
    for (const button of document.querySelectorAll("[data-mode]")) {
        const active = button.dataset.mode === mode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
    }
    setText(element("#composer-context"), mode === "research" ? "4 branches · 2 rounds · synchronous" : "Auto tools · durable task");
}
function setTheme(theme) {
    if (theme === "system")
        document.documentElement.removeAttribute("data-theme");
    else
        document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
    setText(element("#theme-label"), `${theme[0]?.toUpperCase() || "S"}${theme.slice(1)}`);
}
function cycleTheme() {
    const current = (localStorage.getItem(THEME_KEY) || "system");
    const next = current === "system" ? "light" : current === "light" ? "dark" : "system";
    setTheme(next);
}
function bindEvents() {
    elements.input.addEventListener("input", autoSizeInput);
    elements.input.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            void sendMessage();
        }
    });
    elements.composer.addEventListener("submit", (event) => { event.preventDefault(); void sendMessage(); });
    element("#new-session").addEventListener("click", createSession);
    element("#theme-toggle").addEventListener("click", cycleTheme);
    element("#refresh-tasks").addEventListener("click", () => void refreshTasks(true));
    element("#tasks-refresh-button").addEventListener("click", () => void refreshTasks(true));
    element("#status-refresh-button").addEventListener("click", () => void refreshStatus(true));
    elements.connectionButton.addEventListener("click", () => setInspector(true, "runtime"));
    element("#inspector-toggle").addEventListener("click", () => setInspector(!elements.layout.classList.contains("inspector-open")));
    element("#inspector-close").addEventListener("click", () => setInspector(false));
    element("#sidebar-toggle").addEventListener("click", () => {
        const collapsed = elements.layout.classList.toggle("sidebar-collapsed");
        element("#sidebar-toggle").setAttribute("aria-pressed", String(collapsed));
    });
    element("#mobile-menu").addEventListener("click", () => document.body.classList.add("mobile-sidebar-open"));
    element("#sidebar-scrim").addEventListener("click", () => document.body.classList.remove("mobile-sidebar-open"));
    elements.cancelTask.addEventListener("click", () => void controlSelectedTask("cancel"));
    elements.resumeTask.addEventListener("click", () => void controlSelectedTask("resume"));
    for (const button of document.querySelectorAll("[data-mode]"))
        button.addEventListener("click", () => setMode(button.dataset.mode));
    for (const button of document.querySelectorAll("[data-inspector-tab]"))
        button.addEventListener("click", () => selectInspectorTab(button.dataset.inspectorTab));
    for (const starter of document.querySelectorAll(".starter-card")) {
        starter.addEventListener("click", () => {
            elements.input.value = starter.dataset.prompt || "";
            if (starter.textContent?.includes("Research"))
                setMode("research");
            autoSizeInput();
            elements.input.focus();
        });
    }
    document.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
            event.preventDefault();
            createSession();
        }
        if (event.key === "Escape") {
            setInspector(false);
            document.body.classList.remove("mobile-sidebar-open");
        }
    });
    window.addEventListener("beforeunload", () => {
        window.clearInterval(taskTimer);
        window.clearInterval(healthTimer);
        activeAbort?.abort();
    });
}
function initialize() {
    const selectedTheme = (localStorage.getItem(THEME_KEY) || "system");
    setTheme(["system", "light", "dark"].includes(selectedTheme) ? selectedTheme : "system");
    rememberSession();
    configureView();
    setText(elements.ownerLabel, short(ownerId.replace(/^web-owner-/, ""), 20));
    setText(elements.contextOwner, ownerId);
    setText(elements.contextSession, sessionId);
    bindEvents();
    autoSizeInput();
    void refreshStatus();
    void refreshTasks();
    if (currentView === "tasks")
        taskTimer = window.setInterval(() => void refreshTasks(), 2_000);
    healthTimer = window.setInterval(() => void refreshStatus(), 15_000);
    if (activeTaskId)
        void reconcileTask(activeTaskId);
}
initialize();
