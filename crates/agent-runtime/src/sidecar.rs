use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use reqwest::Client;
use rwkv_agent_core::{
    ModelOutput, OpenStateRequest, RunContext, StateContinueRequest, StateHandle, StateModel,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::cloud_plugin::CloudModelRef;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SidecarState {
    #[serde(default)]
    pub owner_id: String,
    pub state_id: String,
    #[serde(default)]
    pub home_url: String,
    #[serde(default)]
    pub branch: String,
    #[serde(default)]
    pub seen_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SidecarSnapshot {
    pub checksum: String,
    pub size_bytes: u64,
    pub payload_base64: String,
    pub seen_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct SidecarCheckpoint {
    model_ref: CloudModelRef,
    provider_mode: String,
    placement: String,
    checksum: String,
    size_bytes: u64,
    atomic: bool,
    #[serde(default)]
    seen_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct SidecarSnapshotResponse {
    status: String,
    checkpoint: SidecarCheckpoint,
    payload_base64: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchContinuation {
    pub state_id: String,
    #[serde(default)]
    pub branch: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub stop_reason: String,
    #[serde(default)]
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub seen_tokens: u64,
    #[serde(default)]
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateDecision {
    #[serde(default)]
    pub use_tool: bool,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(flatten)]
    pub trace: serde_json::Map<String, Value>,
}

#[derive(Clone)]
pub struct SidecarClient {
    endpoints: Arc<Vec<String>>,
    next: Arc<AtomicUsize>,
    http: Client,
}

impl SidecarClient {
    pub fn new(endpoints: Vec<String>) -> Result<Self, String> {
        let endpoints = endpoints
            .into_iter()
            .map(|value| value.trim_end_matches('/').trim().to_string())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if endpoints.is_empty() {
            return Err("at least one sidecar endpoint is required".into());
        }
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(190))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            endpoints: Arc::new(endpoints),
            next: Arc::new(AtomicUsize::new(0)),
            http,
        })
    }

    fn choose(&self) -> String {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        self.endpoints[index % self.endpoints.len()].clone()
    }

    async fn get_json(&self, url: String) -> Result<Value, String> {
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|error| format!("sidecar unavailable: {error}"))?;
        let status = response.status();
        let value = response
            .json::<Value>()
            .await
            .map_err(|error| format!("sidecar invalid JSON response: {error}"))?;
        if !status.is_success() {
            return Err(format!("sidecar HTTP {status}: {value}"));
        }
        Ok(value)
    }

    async fn post_json(&self, url: String, body: Value) -> Result<Value, String> {
        let response = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("sidecar unavailable: {error}"))?;
        let status = response.status();
        let value = response
            .json::<Value>()
            .await
            .map_err(|error| format!("sidecar invalid JSON response: {error}"))?;
        if !status.is_success() {
            return Err(format!("sidecar HTTP {status}: {value}"));
        }
        Ok(value)
    }

    pub async fn health(&self) -> Result<Vec<Value>, String> {
        let mut values = Vec::with_capacity(self.endpoints.len());
        for endpoint in self.endpoints.iter() {
            values.push(self.get_json(format!("{endpoint}/health")).await?);
        }
        Ok(values)
    }

    pub async fn gate(
        &self,
        message: &str,
        threshold: f64,
        context: &str,
        has_pasted_text: bool,
    ) -> Result<GateDecision, String> {
        let endpoint = self.choose();
        let value = self
            .post_json(
                format!("{endpoint}/v1/gate/tool"),
                json!({
                    "message": message,
                    "threshold": threshold,
                    "context": context,
                    "has_pasted_text": has_pasted_text,
                }),
            )
            .await?;
        serde_json::from_value(value)
            .map_err(|error| format!("sidecar invalid Gate response: {error}"))
    }

    pub async fn classify(&self, prompt: &str, labels: Value) -> Result<Value, String> {
        let endpoint = self.choose();
        self.post_json(
            format!("{endpoint}/v1/classify"),
            json!({"prompt":prompt,"labels":labels}),
        )
        .await
    }

    pub async fn prefill(&self, owner_id: &str, prompt: &str) -> Result<SidecarState, String> {
        let endpoint = self.choose();
        self.prefill_at(&endpoint, owner_id, prompt).await
    }

    /// Opens a recurrent State on an explicitly selected Worker.
    ///
    /// The default local path continues to use [`Self::prefill`] and its
    /// existing round-robin endpoint selection. StatePool placement uses this
    /// method only after a versioned plugin plan selects a remote Worker.
    pub async fn prefill_at(
        &self,
        endpoint: &str,
        owner_id: &str,
        prompt: &str,
    ) -> Result<SidecarState, String> {
        let endpoint = endpoint.trim().trim_end_matches('/').to_string();
        if endpoint.is_empty() {
            return Err("sidecar endpoint must not be empty".into());
        }
        let value = self
            .post_json(
                format!("{endpoint}/v1/states/prefill"),
                json!({"owner_id": owner_id, "prompt": prompt, "branch": "root"}),
            )
            .await?;
        let state = value
            .get("state")
            .cloned()
            .ok_or_else(|| "sidecar prefill response has no state".to_string())?;
        let mut parsed: SidecarState = serde_json::from_value(state)
            .map_err(|error| format!("sidecar invalid prefill response: {error}"))?;
        parsed.owner_id = owner_id.to_string();
        parsed.home_url = endpoint;
        Ok(parsed)
    }

    pub async fn fork(
        &self,
        root: &SidecarState,
        branches: &[String],
    ) -> Result<Vec<SidecarState>, String> {
        let value = self
            .post_json(
                format!("{}/v1/states/{}/fork", root.home_url, root.state_id),
                json!({"owner_id": root.owner_id, "branches": branches}),
            )
            .await?;
        value
            .get("states")
            .and_then(Value::as_array)
            .ok_or_else(|| "sidecar fork response has no states".to_string())?
            .iter()
            .map(|item| {
                let mut state: SidecarState = serde_json::from_value(item.clone())
                    .map_err(|error| format!("sidecar invalid fork response: {error}"))?;
                state.owner_id = root.owner_id.clone();
                state.home_url = root.home_url.clone();
                Ok(state)
            })
            .collect()
    }

    pub async fn batch_continue(
        &self,
        home_url: &str,
        owner_id: &str,
        items: Vec<Value>,
        stops: &[String],
        max_tokens: u32,
    ) -> Result<Vec<BatchContinuation>, String> {
        let value = self
            .post_json(
                format!(
                    "{}/v1/states/batch_continue",
                    home_url.trim_end_matches('/')
                ),
                json!({
                    "owner_id": owner_id,
                    "items": items,
                    "stop": stops,
                    "max_tokens": max_tokens,
                }),
            )
            .await?;
        serde_json::from_value(
            value
                .get("results")
                .cloned()
                .ok_or_else(|| "sidecar continuation response has no results".to_string())?,
        )
        .map_err(|error| format!("sidecar invalid continuation response: {error}"))
    }

    pub async fn continue_one(
        &self,
        state: &SidecarState,
        input: &str,
        stops: &[String],
        max_tokens: u32,
    ) -> Result<BatchContinuation, String> {
        let mut rows = self
            .batch_continue(
                &state.home_url,
                &state.owner_id,
                vec![json!({"state_id": state.state_id, "input": input})],
                stops,
                max_tokens,
            )
            .await?;
        if rows.len() != 1 {
            return Err(format!("expected one continuation row, got {}", rows.len()));
        }
        Ok(rows.remove(0))
    }

    pub async fn continue_one_stream(
        &self,
        state: &SidecarState,
        input: &str,
        stops: &[String],
        max_tokens: u32,
        events: mpsc::Sender<Value>,
    ) -> Result<BatchContinuation, String> {
        self.continue_one_stream_inner(state, input, stops, max_tokens, events, false)
            .await
            .map(|(row, _)| row)
    }

    pub async fn continue_one_stream_captured(
        &self,
        state: &SidecarState,
        input: &str,
        stops: &[String],
        max_tokens: u32,
        events: mpsc::Sender<Value>,
    ) -> Result<(BatchContinuation, Vec<Value>), String> {
        self.continue_one_stream_inner(state, input, stops, max_tokens, events, true)
            .await
    }

    async fn continue_one_stream_inner(
        &self,
        state: &SidecarState,
        input: &str,
        stops: &[String],
        max_tokens: u32,
        events: mpsc::Sender<Value>,
        capture_events: bool,
    ) -> Result<(BatchContinuation, Vec<Value>), String> {
        let mut response = self
            .http
            .post(format!(
                "{}/v1/states/stream_continue",
                state.home_url.trim_end_matches('/')
            ))
            .json(&json!({
                "owner_id": state.owner_id,
                "items": [{"state_id": state.state_id, "input": input}],
                "stop": stops,
                "max_tokens": max_tokens,
            }))
            .send()
            .await
            .map_err(|error| format!("sidecar unavailable: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .map_err(|error| format!("sidecar response read failed: {error}"))?;
            return Err(format!("sidecar HTTP {status}: {body}"));
        }

        let mut pending = Vec::new();
        let mut rows: Option<Vec<BatchContinuation>> = None;
        let mut captured = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("sidecar stream read failed: {error}"))?
        {
            pending.extend_from_slice(&chunk);
            while let Some(index) = pending.iter().position(|byte| *byte == b'\n') {
                let line = pending.drain(..=index).collect::<Vec<_>>();
                Self::consume_stream_line(
                    &line[..line.len().saturating_sub(1)],
                    &events,
                    &mut rows,
                    capture_events.then_some(&mut captured),
                )
                .await?;
            }
        }
        if !pending.is_empty() {
            Self::consume_stream_line(
                &pending,
                &events,
                &mut rows,
                capture_events.then_some(&mut captured),
            )
            .await?;
        }
        let mut rows = rows.ok_or_else(|| "sidecar stream ended without done event".to_string())?;
        if rows.len() != 1 {
            return Err(format!(
                "expected one streamed continuation row, got {}",
                rows.len()
            ));
        }
        Ok((rows.remove(0), captured))
    }

    async fn consume_stream_line(
        line: &[u8],
        events: &mpsc::Sender<Value>,
        rows: &mut Option<Vec<BatchContinuation>>,
        captured: Option<&mut Vec<Value>>,
    ) -> Result<(), String> {
        if line.iter().all(u8::is_ascii_whitespace) {
            return Ok(());
        }
        let value: Value = serde_json::from_slice(line)
            .map_err(|error| format!("sidecar invalid stream event: {error}"))?;
        if let Some(captured) = captured {
            captured.push(value.clone());
        }
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "delta" => {
                // Provider continuation is an atomic boundary. The server
                // cancellation token is observed immediately after this
                // boundary and releases the State before another action.
                let _ = events.send(value).await;
            }
            "done" => {
                *rows = Some(
                    serde_json::from_value(
                        value
                            .get("results")
                            .cloned()
                            .ok_or_else(|| "stream done event has no results".to_string())?,
                    )
                    .map_err(|error| format!("sidecar invalid done event: {error}"))?,
                );
            }
            "error" => {
                return Err(value
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("sidecar stream failed")
                    .to_string());
            }
            kind => return Err(format!("unknown sidecar stream event: {kind}")),
        }
        Ok(())
    }

    pub async fn release_many(
        &self,
        home_url: &str,
        owner_id: &str,
        state_ids: &[String],
    ) -> Result<Value, String> {
        self.post_json(
            format!("{}/v1/states/release", home_url.trim_end_matches('/')),
            json!({"owner_id": owner_id, "state_ids": state_ids}),
        )
        .await
    }

    /// Exports an exact recurrent State into a portable, integrity-checked
    /// CPU snapshot while leaving the hot source State resident.
    pub async fn snapshot(
        &self,
        state: &SidecarState,
        model_ref: &CloudModelRef,
    ) -> Result<SidecarSnapshot, String> {
        let value = self
            .post_json(
                format!(
                    "{}/v1/states/{}/snapshot",
                    state.home_url.trim_end_matches('/'),
                    state.state_id
                ),
                json!({
                    "owner_id": state.owner_id,
                    "model_ref": model_ref,
                    "target_tier": "cpu",
                }),
            )
            .await?;
        let response: SidecarSnapshotResponse = serde_json::from_value(value)
            .map_err(|error| format!("sidecar invalid snapshot response: {error}"))?;
        if response.status != "ok"
            || response.checkpoint.model_ref != *model_ref
            || response.checkpoint.provider_mode != "rwkv_recurrent"
            || response.checkpoint.placement != "cpu"
            || !response.checkpoint.atomic
        {
            return Err("sidecar snapshot identity or atomicity mismatch".into());
        }
        validate_snapshot_payload(
            &response.checkpoint.checksum,
            response.checkpoint.size_bytes,
            &response.payload_base64,
        )?;
        Ok(SidecarSnapshot {
            checksum: response.checkpoint.checksum,
            size_bytes: response.checkpoint.size_bytes,
            payload_base64: response.payload_base64,
            seen_tokens: response.checkpoint.seen_tokens,
        })
    }

    /// Imports a committed snapshot on the explicitly selected Worker. The
    /// payload is verified locally before any Worker State is allocated.
    pub async fn restore_at(
        &self,
        endpoint: &str,
        owner_id: &str,
        model_ref: &CloudModelRef,
        checksum: &str,
        payload_base64: &str,
    ) -> Result<SidecarState, String> {
        let endpoint = endpoint.trim().trim_end_matches('/').to_string();
        if endpoint.is_empty() || owner_id.trim().is_empty() {
            return Err("sidecar restore endpoint and owner must not be empty".into());
        }
        validate_snapshot_payload(checksum, 0, payload_base64)?;
        let value = self
            .post_json(
                format!("{endpoint}/v1/states/restore"),
                json!({
                    "owner_id": owner_id,
                    "model_ref": model_ref,
                    "checksum": checksum,
                    "payload_base64": payload_base64,
                }),
            )
            .await?;
        let state = value
            .get("state")
            .cloned()
            .ok_or_else(|| "sidecar restore response has no state".to_string())?;
        let mut parsed: SidecarState = serde_json::from_value(state)
            .map_err(|error| format!("sidecar invalid restore response: {error}"))?;
        if parsed.state_id.trim().is_empty()
            || (!parsed.owner_id.is_empty() && parsed.owner_id != owner_id)
        {
            return Err("sidecar restored a mismatched State".into());
        }
        parsed.owner_id = owner_id.to_string();
        parsed.home_url = endpoint;
        Ok(parsed)
    }

    pub async fn complete(
        &self,
        prompt: &str,
        stops: &[String],
        max_tokens: u32,
    ) -> Result<BatchContinuation, String> {
        let endpoint = self.choose();
        let value = self
            .post_json(
                format!("{endpoint}/v1/completions"),
                json!({"prompt": prompt, "stop": stops, "max_tokens": max_tokens}),
            )
            .await?;
        let g1i = value
            .get("g1i")
            .cloned()
            .ok_or_else(|| "completion response has no g1i".to_string())?;
        serde_json::from_value(g1i)
            .map_err(|error| format!("sidecar invalid completion response: {error}"))
    }

    fn reconstruct_envelope(input: &str, row: &BatchContinuation) -> String {
        let generated = row.text.trim_start();
        let tool_prefill = input
            .trim_end()
            .rsplit_once("Assistant: ")
            .map(|(_, suffix)| suffix)
            .filter(|suffix| suffix.starts_with("<tool_call>") && !suffix.contains("</tool_call>"));
        if let Some(prefill) = tool_prefill
            && let Some(body) = first_complete_tool_json(prefill, generated)
        {
            // NF4/recurrent checkpoints occasionally close the JSON object but
            // omit </tool_call>, then imitate a User tool response. Commit
            // only the first structurally complete call and synthesize the
            // transport delimiter. The strict protocol parser and registry
            // still validate the extracted object and its exact schema.
            return format!("<tool_call>{body}</tool_call>");
        }
        match row.stop_reason.as_str() {
            "</tool_call>" => {
                let prefix = if generated.starts_with("<tool_call>") {
                    ""
                } else if generated.starts_with("{\"name\":\"") {
                    // Some recurrent checkpoints echo a complete JSON call
                    // after a deeper controller prefix instead of continuing
                    // only the missing suffix. Treat that as a framing echo,
                    // not as part of the arguments value. The strict parser
                    // and registry still validate the model's complete call.
                    "<tool_call>"
                } else {
                    tool_prefill.unwrap_or("<tool_call>")
                };
                let suffix = if generated.ends_with("</tool_call>") {
                    ""
                } else {
                    "</tool_call>"
                };
                format!("{prefix}{generated}{suffix}")
            }
            "</answer>" => {
                let prefix = if generated.starts_with("<answer>") {
                    ""
                } else {
                    "<answer>"
                };
                let suffix = if generated.ends_with("</answer>") {
                    ""
                } else {
                    "</answer>"
                };
                format!("{prefix}{generated}{suffix}")
            }
            _ if tool_prefill.is_some() => format!("{}{generated}", tool_prefill.unwrap()),
            _ if input.trim_end().ends_with("<answer>") => format!("<answer>{generated}"),
            _ => row.text.clone(),
        }
    }
}

fn validate_snapshot_payload(
    checksum: &str,
    expected_size_bytes: u64,
    payload_base64: &str,
) -> Result<(), String> {
    if !checksum.starts_with("sha256:") || checksum.len() != 71 {
        return Err("sidecar snapshot checksum must use sha256:<64 lowercase hex>".into());
    }
    let payload = BASE64_STANDARD
        .decode(payload_base64)
        .map_err(|error| format!("sidecar snapshot payload is not valid base64: {error}"))?;
    if payload.is_empty() {
        return Err("sidecar snapshot payload must not be empty".into());
    }
    if expected_size_bytes != 0 && payload.len() as u64 != expected_size_bytes {
        return Err("sidecar snapshot payload size mismatch".into());
    }
    let digest = ring::digest::digest(&ring::digest::SHA256, &payload);
    let actual = format!(
        "sha256:{}",
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    if actual != checksum {
        return Err("sidecar snapshot payload checksum mismatch".into());
    }
    Ok(())
}

fn first_complete_tool_json(prefill: &str, generated: &str) -> Option<String> {
    let generated = generated.trim_start();
    let candidate = if let Some(value) = generated.strip_prefix("<tool_call>") {
        value.to_string()
    } else if generated.starts_with("{\"name\"") {
        generated.to_string()
    } else {
        format!("{}{}", prefill.strip_prefix("<tool_call>")?, generated)
    };
    let candidate = candidate.trim_start();
    let mut values = serde_json::Deserializer::from_str(candidate).into_iter::<Value>();
    let value = values.next()?.ok()?;
    let consumed = values.byte_offset();
    let object = value.as_object()?;
    if !object.contains_key("name") || !object.contains_key("arguments") {
        return None;
    }
    Some(candidate.get(..consumed)?.to_string())
}

impl StateModel for SidecarClient {
    async fn open(
        &mut self,
        request: OpenStateRequest,
        context: RunContext,
    ) -> Result<StateHandle, String> {
        context.check().map_err(|e| e.to_string())?;
        let state = tokio::time::timeout(
            context.remaining(),
            self.prefill(&request.owner_id, &request.root_prompt),
        )
        .await
        .map_err(|_| "run deadline exceeded during State prefill".to_string())??;
        Ok(StateHandle {
            endpoint: state.home_url,
            owner_id: state.owner_id,
            state_id: state.state_id,
        })
    }

    async fn continue_state(
        &mut self,
        request: StateContinueRequest,
        context: RunContext,
    ) -> Result<ModelOutput, String> {
        context.check().map_err(|e| e.to_string())?;
        let state = SidecarState {
            owner_id: request.state.owner_id,
            state_id: request.state.state_id.clone(),
            home_url: request.state.endpoint,
            branch: String::new(),
            seen_tokens: 0,
        };
        let row = tokio::time::timeout(
            context.remaining(),
            self.continue_one(&state, &request.input, &request.stops, request.max_tokens),
        )
        .await
        .map_err(|_| "run deadline exceeded during State continuation".to_string())??;
        let text = Self::reconstruct_envelope(&request.input, &row);
        Ok(ModelOutput {
            state_id: row.state_id,
            text,
            stop_reason: (!row.stop_reason.is_empty()).then_some(row.stop_reason),
        })
    }

    async fn fork_state(
        &mut self,
        root: StateHandle,
        context: RunContext,
    ) -> Result<StateHandle, String> {
        context.check().map_err(|error| error.to_string())?;
        let source = SidecarState {
            owner_id: root.owner_id,
            state_id: root.state_id,
            home_url: root.endpoint,
            branch: "root".into(),
            seen_tokens: 0,
        };
        let mut children = tokio::time::timeout(
            context.remaining(),
            self.fork(&source, &["agent-worker".into()]),
        )
        .await
        .map_err(|_| "run deadline exceeded during State fork".to_string())??;
        if children.len() != 1 {
            return Err(format!("expected one forked state, got {}", children.len()));
        }
        let child = children.remove(0);
        Ok(StateHandle {
            endpoint: child.home_url,
            owner_id: child.owner_id,
            state_id: child.state_id,
        })
    }

    async fn release(&mut self, state: StateHandle) -> Result<(), String> {
        self.release_many(&state.endpoint, &state.owner_id, &[state.state_id])
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::State as AxumState, routing::post};
    use rwkv_agent_core::TOOL_CALL_JSON_PREFIX;

    fn row(text: &str, stop_reason: &str) -> BatchContinuation {
        BatchContinuation {
            state_id: "state-1".into(),
            branch: String::new(),
            text: text.into(),
            stop_reason: stop_reason.into(),
            token_ids: Vec::new(),
            seen_tokens: 0,
            elapsed_ms: 0.0,
        }
    }

    fn cloud_model_ref() -> CloudModelRef {
        CloudModelRef {
            model_id: "rwkv7".into(),
            revision: "revision".into(),
            tokenizer: "tokenizer".into(),
            state_abi: "rwkv7-state-v1".into(),
        }
    }

    #[derive(Clone)]
    struct SnapshotMock {
        model_ref: CloudModelRef,
        checksum: String,
        payload_base64: String,
    }

    async fn mock_snapshot_sidecar() -> (String, SnapshotMock) {
        async fn snapshot(
            AxumState(mock): AxumState<SnapshotMock>,
            Json(request): Json<Value>,
        ) -> Json<Value> {
            assert_eq!(request["owner_id"], "owner");
            assert_eq!(request["model_ref"], json!(mock.model_ref));
            Json(json!({
                "status":"ok",
                "checkpoint":{
                    "checkpoint_id":"checkpoint-test",
                    "model_ref":mock.model_ref,
                    "provider_mode":"rwkv_recurrent",
                    "placement":"cpu",
                    "checksum":mock.checksum,
                    "size_bytes":16,
                    "atomic":true,
                    "seen_tokens":42
                },
                "payload_base64":mock.payload_base64,
            }))
        }

        async fn restore(
            AxumState(mock): AxumState<SnapshotMock>,
            Json(request): Json<Value>,
        ) -> Json<Value> {
            assert_eq!(request["owner_id"], "owner");
            assert_eq!(request["model_ref"], json!(mock.model_ref));
            assert_eq!(request["checksum"], mock.checksum);
            assert_eq!(request["payload_base64"], mock.payload_base64);
            Json(json!({
                "status":"ok",
                "state":{
                    "state_id":"restored-state",
                    "owner_id":"owner",
                    "branch":"restored",
                    "seen_tokens":42
                }
            }))
        }

        let payload = b"snapshot-payload";
        let digest = ring::digest::digest(&ring::digest::SHA256, payload);
        let mock = SnapshotMock {
            model_ref: cloud_model_ref(),
            checksum: format!(
                "sha256:{}",
                digest
                    .as_ref()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            ),
            payload_base64: BASE64_STANDARD.encode(payload),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/states/{state_id}/snapshot", post(snapshot))
            .route("/v1/states/restore", post(restore))
            .with_state(mock.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}"), mock)
    }

    #[tokio::test]
    async fn snapshot_and_restore_validate_payload_and_model_identity() {
        let (endpoint, mock) = mock_snapshot_sidecar().await;
        let client = SidecarClient::new(vec![endpoint.clone()]).unwrap();
        let state = SidecarState {
            owner_id: "owner".into(),
            state_id: "source-state".into(),
            home_url: endpoint.clone(),
            branch: "root".into(),
            seen_tokens: 0,
        };
        let snapshot = client.snapshot(&state, &mock.model_ref).await.unwrap();
        assert_eq!(snapshot.checksum, mock.checksum);
        assert_eq!(snapshot.size_bytes, 16);
        assert_eq!(snapshot.seen_tokens, 42);

        let restored = client
            .restore_at(
                &endpoint,
                "owner",
                &mock.model_ref,
                &snapshot.checksum,
                &snapshot.payload_base64,
            )
            .await
            .unwrap();
        assert_eq!(restored.state_id, "restored-state");
        assert_eq!(restored.owner_id, "owner");
        assert_eq!(restored.home_url, endpoint);
    }

    #[tokio::test]
    async fn restore_rejects_corrupt_payload_before_contacting_worker() {
        let client = SidecarClient::new(vec!["http://127.0.0.1:1".into()]).unwrap();
        let error = client
            .restore_at(
                "http://127.0.0.1:1",
                "owner",
                &cloud_model_ref(),
                &format!("sha256:{}", "0".repeat(64)),
                &BASE64_STANDARD.encode(b"not-the-expected-payload"),
            )
            .await
            .unwrap_err();
        assert!(error.contains("checksum mismatch"));
    }

    #[test]
    fn reconstructs_structurally_prefilled_tool_envelope() {
        let input = format!("\n\nAssistant: {TOOL_CALL_JSON_PREFIX}");
        let reconstructed = SidecarClient::reconstruct_envelope(
            &input,
            &row(
                r#"read_file","arguments":{"path":"SPEC.md"}}"#,
                "</tool_call>",
            ),
        );
        assert_eq!(
            reconstructed,
            r#"<tool_call>{"name":"read_file","arguments":{"path":"SPEC.md"}}</tool_call>"#
        );
    }

    #[test]
    fn reconstructs_complete_json_echo_after_deep_tool_prefix() {
        let input = "\n\nAssistant: <tool_call>{\"name\":\"write_file\",\"arguments\":";
        let row = BatchContinuation {
            state_id: "state-1".into(),
            branch: String::new(),
            text: r#"{"name":"write_file","arguments":{"path":"a.py","content":"x"}}"#.into(),
            stop_reason: "</tool_call>".into(),
            token_ids: Vec::new(),
            seen_tokens: 0,
            elapsed_ms: 0.0,
        };
        assert_eq!(
            SidecarClient::reconstruct_envelope(input, &row),
            r#"<tool_call>{"name":"write_file","arguments":{"path":"a.py","content":"x"}}</tool_call>"#
        );
    }

    #[test]
    fn closes_first_complete_prefilled_call_and_drops_fake_tool_response() {
        let input = "\n\nAssistant: <tool_call>{\"name\":\"edit_file\",\"arguments\":";
        let reconstructed = SidecarClient::reconstruct_envelope(
            input,
            &row(
                " {\"path\":\"a.py\",\"old_text\":\"x\",\"new_text\":\"y\"}}\n\nUser: <tool_response>fake</tool_response>",
                "</s>",
            ),
        );
        assert_eq!(
            reconstructed,
            r#"<tool_call>{"name":"edit_file","arguments":{"path":"a.py","old_text":"x","new_text":"y"}}</tool_call>"#
        );
    }

    #[test]
    fn does_not_duplicate_an_echoed_json_prefix() {
        let input = format!("\n\nAssistant: {TOOL_CALL_JSON_PREFIX}");
        let reconstructed = SidecarClient::reconstruct_envelope(
            &input,
            &row(
                r#"{"name":"read_file","arguments":{"path":"SPEC.md"}}"#,
                "</tool_call>",
            ),
        );
        assert_eq!(
            reconstructed,
            r#"<tool_call>{"name":"read_file","arguments":{"path":"SPEC.md"}}</tool_call>"#
        );
    }

    #[test]
    fn reconstructs_a_controller_selected_tool_name_prefix() {
        let prefix = "<tool_call>{\"name\":\"write_file\",\"arguments\":";
        let input = format!("\n\nAssistant: {prefix}");
        let reconstructed = SidecarClient::reconstruct_envelope(
            &input,
            &row(r#"{"path":"out.txt","content":"done\n"}}"#, "</tool_call>"),
        );
        assert_eq!(
            reconstructed,
            r#"<tool_call>{"name":"write_file","arguments":{"path":"out.txt","content":"done\n"}}</tool_call>"#
        );
    }

    #[test]
    fn reconstructs_controller_grounded_path_and_first_required_key() {
        let prefix = "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"migrate.py\",\"content\":";
        let input = format!("\n\nAssistant: {prefix}");
        let reconstructed = SidecarClient::reconstruct_envelope(
            &input,
            &row(r#""print('ok')\n"}}"#, "</tool_call>"),
        );
        assert_eq!(
            reconstructed,
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"migrate.py\",\"content\":\"print('ok')\\n\"}}</tool_call>"
        );
    }
}
