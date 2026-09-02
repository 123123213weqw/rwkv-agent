export const API_VERSION = "rwkv-agent.service.v1" as const;

export type JsonObject = Record<string, unknown>;

export interface RequestIdentity {
  api_version: typeof API_VERSION;
  request_id: string;
  owner_id: string;
}

export interface TaskRequest extends RequestIdentity {
  session_id: string;
  message: string;
  task_id?: string;
}

export interface ResearchRequest extends RequestIdentity {
  session_id: string;
  message: string;
  branch_width?: number;
  max_rounds?: number;
}

export interface TaskSummary {
  id: string;
  request_id?: string;
  session_id?: string;
  message?: string;
  kind?: string;
  status?: string;
  route?: string;
  state?: string;
  tool_count?: number;
  created_unix_ms?: number;
  elapsed_ms?: number;
  error?: string | null;
  revision?: number;
  recovery_count?: number;
}

export interface TaskListResponse extends JsonObject {
  counts?: Record<string, number>;
  tasks?: TaskSummary[];
}

export interface StreamEvent extends JsonObject {
  type: string;
  sequence: number;
  task_id?: string;
  phase?: string;
  stage_id?: string;
  stage_index?: number;
  stage_count?: number;
  text?: string;
  error?: string;
  response?: JsonObject;
}

export interface HttpResult<T extends JsonObject = JsonObject> {
  ok: boolean;
  status: number;
  body: T;
}

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(body: JsonObject, status: number): string {
  const detail = body.error_detail;
  if (isObject(detail) && typeof detail.message === "string") return detail.message;
  if (typeof body.error === "string") return body.error;
  if (typeof body.message === "string") return body.message;
  return `Controller returned HTTP ${status}`;
}

export function newRequestId(prefix = "web-request"): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

export class AgentApi {
  readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(baseUrl = "", fetchImpl: FetchLike = fetch.bind(globalThis)) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.fetchImpl = fetchImpl;
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private async json<T extends JsonObject>(path: string, init: RequestInit = {}): Promise<HttpResult<T>> {
    const response = await this.fetchImpl(this.url(path), {
      ...init,
      headers: {
        accept: "application/json",
        ...(init.body ? { "content-type": "application/json" } : {}),
        ...(init.headers || {}),
      },
    });
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new Error(`Controller returned HTTP ${response.status} without JSON`);
    }
    if (!isObject(body)) throw new Error("Controller returned a non-object JSON response");
    return { ok: response.ok, status: response.status, body: body as T };
  }

  async live(): Promise<HttpResult> {
    return this.json("/live");
  }

  async ready(): Promise<HttpResult> {
    return this.json("/ready");
  }

  async listTasks(identity: RequestIdentity): Promise<TaskListResponse> {
    const query = new URLSearchParams(identity as unknown as Record<string, string>);
    const result = await this.json<TaskListResponse>(`/v1/tasks?${query}`);
    if (!result.ok) throw new Error(errorMessage(result.body, result.status));
    return result.body;
  }

  async task(taskId: string, identity: RequestIdentity): Promise<JsonObject> {
    const query = new URLSearchParams(identity as unknown as Record<string, string>);
    const result = await this.json(`/v1/tasks/${encodeURIComponent(taskId)}?${query}`);
    if (!result.ok) throw new Error(errorMessage(result.body, result.status));
    return result.body;
  }

  async cancelTask(taskId: string, identity: RequestIdentity): Promise<JsonObject> {
    return this.control(taskId, "cancel", identity);
  }

  async resumeTask(taskId: string, identity: RequestIdentity): Promise<JsonObject> {
    return this.control(taskId, "resume", identity);
  }

  private async control(taskId: string, action: "cancel" | "resume", identity: RequestIdentity): Promise<JsonObject> {
    const result = await this.json(`/v1/tasks/${encodeURIComponent(taskId)}/${action}`, {
      method: "POST",
      body: JSON.stringify(identity),
    });
    if (!result.ok) throw new Error(errorMessage(result.body, result.status));
    return result.body;
  }

  async research(request: ResearchRequest, signal?: AbortSignal): Promise<JsonObject> {
    const result = await this.json("/v1/research", {
      method: "POST",
      body: JSON.stringify(request),
      signal,
    });
    if (!result.ok) throw new Error(errorMessage(result.body, result.status));
    return result.body;
  }

  async streamTask(
    request: TaskRequest,
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<StreamEvent> {
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
        const parsed: unknown = JSON.parse(raw);
        if (isObject(parsed)) message = errorMessage(parsed, response.status);
      } catch {
        // Preserve the bounded text body when it is not JSON.
      }
      throw new Error(message);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let expectedSequence = 1;
    const terminal: { event?: StreamEvent } = {};

    const consume = (line: string): void => {
      if (!line.trim()) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new Error("Controller returned malformed NDJSON");
      }
      if (!isObject(parsed) || typeof parsed.type !== "string" || !Number.isInteger(parsed.sequence)) {
        throw new Error("Controller returned an invalid stream event");
      }
      const event = parsed as StreamEvent;
      if (event.sequence !== expectedSequence) {
        throw new Error(`Stream sequence mismatch: expected ${expectedSequence}, got ${event.sequence}`);
      }
      expectedSequence += 1;
      if (terminal.event) throw new Error("Controller returned data after a terminal event");
      onEvent(event);
      if (event.type === "final" || event.type === "error") terminal.event = event;
    };

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = done ? "" : (lines.pop() || "");
      for (const line of lines) consume(line);
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
    const terminalEvent = terminal.event;
    if (!terminalEvent) throw new Error("Task stream ended without a terminal event");
    if (terminalEvent.type === "error") throw new Error(terminalEvent.error || "Task stream failed");
    return terminalEvent;
  }
}
