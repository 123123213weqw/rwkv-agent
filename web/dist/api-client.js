export const API_VERSION = "rwkv-agent.service.v1";
function isObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function errorMessage(body, status) {
    const detail = body.error_detail;
    if (isObject(detail) && typeof detail.message === "string")
        return detail.message;
    if (typeof body.error === "string")
        return body.error;
    if (typeof body.message === "string")
        return body.message;
    return `Controller returned HTTP ${status}`;
}
export function newRequestId(prefix = "web-request") {
    return `${prefix}-${crypto.randomUUID()}`;
}
export class AgentApi {
    baseUrl;
    fetchImpl;
    constructor(baseUrl = "", fetchImpl = fetch.bind(globalThis)) {
        this.baseUrl = baseUrl.replace(/\/$/, "");
        this.fetchImpl = fetchImpl;
    }
    url(path) {
        return `${this.baseUrl}${path}`;
    }
    async json(path, init = {}) {
        const response = await this.fetchImpl(this.url(path), {
            ...init,
            headers: {
                accept: "application/json",
                ...(init.body ? { "content-type": "application/json" } : {}),
                ...(init.headers || {}),
            },
        });
        let body;
        try {
            body = await response.json();
        }
        catch {
            throw new Error(`Controller returned HTTP ${response.status} without JSON`);
        }
        if (!isObject(body))
            throw new Error("Controller returned a non-object JSON response");
        return { ok: response.ok, status: response.status, body: body };
    }
    async live() {
        return this.json("/live");
    }
    async ready() {
        return this.json("/ready");
    }
    async listTasks(identity) {
        const query = new URLSearchParams(identity);
        const result = await this.json(`/v1/tasks?${query}`);
        if (!result.ok)
            throw new Error(errorMessage(result.body, result.status));
        return result.body;
    }
    async task(taskId, identity) {
        const query = new URLSearchParams(identity);
        const result = await this.json(`/v1/tasks/${encodeURIComponent(taskId)}?${query}`);
        if (!result.ok)
            throw new Error(errorMessage(result.body, result.status));
        return result.body;
    }
    async cancelTask(taskId, identity) {
        return this.control(taskId, "cancel", identity);
    }
    async resumeTask(taskId, identity) {
        return this.control(taskId, "resume", identity);
    }
    async control(taskId, action, identity) {
        const result = await this.json(`/v1/tasks/${encodeURIComponent(taskId)}/${action}`, {
            method: "POST",
            body: JSON.stringify(identity),
        });
        if (!result.ok)
            throw new Error(errorMessage(result.body, result.status));
        return result.body;
    }
    async research(request, signal) {
        const result = await this.json("/v1/research", {
            method: "POST",
            body: JSON.stringify(request),
            signal,
        });
        if (!result.ok)
            throw new Error(errorMessage(result.body, result.status));
        return result.body;
    }
    async streamTask(request, onEvent, signal) {
        const response = await this.fetchImpl(this.url("/v1/tasks/stream"), {
            method: "POST",
            headers: {
                accept: "application/x-ndjson",
                "content-type": "application/json",
            },
            body: JSON.stringify(request),
            signal,
        });
        if (!response.ok || !response.body) {
            const raw = await response.text();
            let message = raw || `Controller returned HTTP ${response.status}`;
            try {
                const parsed = JSON.parse(raw);
                if (isObject(parsed))
                    message = errorMessage(parsed, response.status);
            }
            catch {
                // Preserve the bounded text body when it is not JSON.
            }
            throw new Error(message);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let expectedSequence = 1;
        const terminal = {};
        const consume = (line) => {
            if (!line.trim())
                return;
            let parsed;
            try {
                parsed = JSON.parse(line);
            }
            catch {
                throw new Error("Controller returned malformed NDJSON");
            }
            if (!isObject(parsed) || typeof parsed.type !== "string" || !Number.isInteger(parsed.sequence)) {
                throw new Error("Controller returned an invalid stream event");
            }
            const event = parsed;
            if (event.sequence !== expectedSequence) {
                throw new Error(`Stream sequence mismatch: expected ${expectedSequence}, got ${event.sequence}`);
            }
            expectedSequence += 1;
            if (terminal.event)
                throw new Error("Controller returned data after a terminal event");
            onEvent(event);
            if (event.type === "final" || event.type === "error")
                terminal.event = event;
        };
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            const lines = buffer.split("\n");
            buffer = done ? "" : (lines.pop() || "");
            for (const line of lines)
                consume(line);
            if (done)
                break;
        }
        if (buffer.trim())
            consume(buffer);
        const terminalEvent = terminal.event;
        if (!terminalEvent)
            throw new Error("Task stream ended without a terminal event");
        if (terminalEvent.type === "error")
            throw new Error(terminalEvent.error || "Task stream failed");
        return terminalEvent;
    }
}
