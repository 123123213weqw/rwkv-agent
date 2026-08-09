use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use reqwest::Client;
use rwkv_agent_core::{
    ModelOutput, OpenStateRequest, RunContext, StateContinueRequest, StateHandle, StateModel,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;

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
        let response = self.http.get(url).send().await.map_err(|e| e.to_string())?;
        let status = response.status();
        let value = response.json::<Value>().await.map_err(|e| e.to_string())?;
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
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let value = response.json::<Value>().await.map_err(|e| e.to_string())?;
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
        serde_json::from_value(value).map_err(|e| e.to_string())
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
        let mut parsed: SidecarState = serde_json::from_value(state).map_err(|e| e.to_string())?;
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
                let mut state: SidecarState =
                    serde_json::from_value(item.clone()).map_err(|e| e.to_string())?;
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
        .map_err(|e| e.to_string())
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
            .map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.map_err(|error| error.to_string())?;
            return Err(format!("sidecar HTTP {status}: {body}"));
        }

        let mut pending = Vec::new();
        let mut rows: Option<Vec<BatchContinuation>> = None;
        while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
            pending.extend_from_slice(&chunk);
            while let Some(index) = pending.iter().position(|byte| *byte == b'\n') {
                let line = pending.drain(..=index).collect::<Vec<_>>();
                Self::consume_stream_line(
                    &line[..line.len().saturating_sub(1)],
                    &events,
                    &mut rows,
                )
                .await?;
            }
        }
        if !pending.is_empty() {
            Self::consume_stream_line(&pending, &events, &mut rows).await?;
        }
        let mut rows = rows.ok_or_else(|| "sidecar stream ended without done event".to_string())?;
        if rows.len() != 1 {
            return Err(format!(
                "expected one streamed continuation row, got {}",
                rows.len()
            ));
        }
        Ok(rows.remove(0))
    }

    async fn consume_stream_line(
        line: &[u8],
        events: &mpsc::Sender<Value>,
        rows: &mut Option<Vec<BatchContinuation>>,
    ) -> Result<(), String> {
        if line.iter().all(u8::is_ascii_whitespace) {
            return Ok(());
        }
        let value: Value = serde_json::from_slice(line).map_err(|error| error.to_string())?;
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "delta" => {
                // A disconnected browser must not interrupt the bounded State
                // continuation. Keep consuming until the Sidecar commits it.
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
                    .map_err(|error| error.to_string())?,
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
        serde_json::from_value(g1i).map_err(|e| e.to_string())
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
