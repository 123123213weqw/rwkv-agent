use std::time::Duration;

use reqwest::Client;
use rwkv_agent_core::{RunContext, ToolCall, ToolExecutor};
use serde_json::{Value, json};

#[derive(Clone)]
pub struct DataPlaneClient {
    endpoint: String,
    http: Client,
    session_id: String,
    original_query: String,
}

impl DataPlaneClient {
    pub fn new(endpoint: impl Into<String>) -> Result<Self, String> {
        let endpoint = endpoint.into().trim().trim_end_matches('/').to_string();
        if endpoint.is_empty() {
            return Err("data-plane endpoint must not be empty".into());
        }
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            endpoint,
            http,
            session_id: "default".into(),
            original_query: String::new(),
        })
    }

    pub fn for_turn(&self, session_id: &str, original_query: &str) -> Self {
        let mut value = self.clone();
        value.session_id = session_id.to_string();
        value.original_query = original_query.to_string();
        value
    }

    async fn get(&self, path: &str) -> Result<Value, String> {
        let response = self
            .http
            .get(format!("{}{}", self.endpoint, path))
            .send()
            .await
            .map_err(|error| format!("data plane unavailable: {error}"))?;
        Self::decode(response).await
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value, String> {
        let response = self
            .http
            .post(format!("{}{}", self.endpoint, path))
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("data plane unavailable: {error}"))?;
        Self::decode(response).await
    }

    async fn decode(response: reqwest::Response) -> Result<Value, String> {
        let status = response.status();
        let value = response
            .json::<Value>()
            .await
            .map_err(|error| format!("data plane invalid JSON response: {error}"))?;
        if !status.is_success() {
            return Err(format!("data plane HTTP {status}: {value}"));
        }
        Ok(value)
    }

    pub async fn health(&self) -> Result<Value, String> {
        self.get("/health").await
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
        session_id: &str,
        original_query: &str,
    ) -> Result<Value, String> {
        self.post(
            "/v1/tools/call",
            json!({
                "name": name,
                "arguments": arguments,
                "session_id": session_id,
                "original_query": original_query,
            }),
        )
        .await
    }

    pub async fn capture_text(&self, session_id: &str, text: &str) -> Result<Value, String> {
        self.post(
            "/v1/session/text",
            json!({"session_id": session_id, "text": text}),
        )
        .await
    }

    pub async fn text_status(&self, session_id: &str) -> Result<Value, String> {
        self.post("/v1/session/text/status", json!({"session_id": session_id}))
            .await
    }

    pub async fn validate_answer(
        &self,
        question: &str,
        answer: &str,
        evidence: &[Value],
    ) -> Result<Value, String> {
        self.post(
            "/v1/answers/validate",
            json!({"question": question, "answer": answer, "evidence": evidence}),
        )
        .await
    }

    pub async fn reduce_evidence(
        &self,
        question: &str,
        tool_results: &[Value],
        limit: usize,
    ) -> Result<Vec<Value>, String> {
        let value = self
            .post(
                "/v1/evidence/reduce",
                json!({"question": question, "tool_results": tool_results, "limit": limit}),
            )
            .await?;
        value
            .get("evidence")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| "data plane returned no evidence array".into())
    }

    pub async fn coordinate_query(
        &self,
        question: &str,
        generated_query: &str,
        branch_index: usize,
        round_index: usize,
        observation: Option<&Value>,
        used_queries: &[String],
    ) -> Result<Value, String> {
        self.post(
            "/v1/queries/coordinate",
            json!({
                "question": question,
                "generated_query": generated_query,
                "branch_index": branch_index,
                "round_index": round_index,
                "observation": observation,
                "used_queries": used_queries,
            }),
        )
        .await
    }
}

impl ToolExecutor for DataPlaneClient {
    async fn execute(&mut self, call: ToolCall, context: RunContext) -> Result<Value, String> {
        context.check().map_err(|e| e.to_string())?;
        tokio::time::timeout(
            context.remaining(),
            self.call_tool(
                &call.name,
                Value::Object(call.arguments),
                &self.session_id,
                &self.original_query,
            ),
        )
        .await
        .map_err(|_| "run deadline exceeded during tool execution".to_string())?
    }
}
