use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::TaskSpec;

pub const SERVICE_API_VERSION: &str = "rwkv-agent.service.v1";

fn default_api_version() -> String {
    SERVICE_API_VERSION.to_string()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRunRequest {
    #[serde(default = "default_api_version")]
    pub api_version: String,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub owner_id: Option<String>,
    pub session_id: String,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub working_directory: Option<String>,
    #[serde(default)]
    pub task_spec: Option<TaskSpec>,
}

impl TaskRunRequest {
    pub fn resolve_task_spec(&self) -> Result<TaskSpec, String> {
        self.validate_version()?;
        let legacy_message = self.message.trim();
        let legacy_directory = self
            .working_directory
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let task = match &self.task_spec {
            Some(task) => {
                if !legacy_message.is_empty() && legacy_message != task.objective.trim() {
                    return Err(
                        "message and task_spec.objective must match when both are provided".into(),
                    );
                }
                let mut task = task.clone();
                match (task.working_directory.as_deref(), legacy_directory) {
                    (Some(spec), Some(legacy)) if spec.trim() != legacy => {
                        return Err("working_directory and task_spec.working_directory must match when both are provided".into());
                    }
                    (None, Some(legacy)) => task.working_directory = Some(legacy.to_string()),
                    _ => {}
                }
                task
            }
            None => TaskSpec::legacy(legacy_message, legacy_directory.map(str::to_string)),
        };
        task.normalize().map_err(|error| error.to_string())
    }

    pub fn validate_version(&self) -> Result<(), String> {
        validate_api_version(&self.api_version)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCallRequest {
    #[serde(default = "default_api_version")]
    pub api_version: String,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub owner_id: Option<String>,
    pub session_id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
    #[serde(default)]
    pub working_directory: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchRequest {
    #[serde(default = "default_api_version")]
    pub api_version: String,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub owner_id: Option<String>,
    pub session_id: String,
    pub message: String,
    #[serde(default = "default_branch_width")]
    pub branch_width: usize,
    #[serde(default = "default_max_rounds")]
    pub max_rounds: usize,
}

fn default_branch_width() -> usize {
    4
}

fn default_max_rounds() -> usize {
    2
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskControlRequest {
    #[serde(default = "default_api_version")]
    pub api_version: String,
    pub request_id: String,
    pub owner_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestIdentity {
    pub api_version: String,
    pub request_id: String,
    pub owner_id: String,
    pub session_id: String,
}

impl RequestIdentity {
    pub fn resolve(
        api_version: &str,
        request_id: Option<&str>,
        owner_id: Option<&str>,
        session_id: &str,
        generated_request_id: impl FnOnce() -> String,
    ) -> Result<Self, String> {
        validate_api_version(api_version)?;
        let session_id = validate_session_id(session_id)?;
        let request_id = match request_id {
            Some(value) => validate_identifier("request_id", value)?,
            None => validate_identifier("request_id", &generated_request_id())?,
        };
        let default_owner = format!("session:{session_id}");
        let owner_id = validate_identifier("owner_id", owner_id.unwrap_or(&default_owner))?;
        Ok(Self {
            api_version: SERVICE_API_VERSION.into(),
            request_id,
            owner_id,
            session_id,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceErrorCode {
    InvalidRequest,
    NotFound,
    Conflict,
    Unavailable,
    Unsupported,
    Cancelled,
    DeadlineExceeded,
    Internal,
}

impl ServiceErrorCode {
    pub fn retryable(self) -> bool {
        matches!(self, Self::Unavailable | Self::DeadlineExceeded)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceErrorDetail {
    pub code: ServiceErrorCode,
    pub message: String,
    pub retryable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServiceStreamEvent {
    pub api_version: String,
    pub request_id: String,
    pub owner_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    pub sequence: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub data: Map<String, Value>,
}

impl ServiceStreamEvent {
    pub fn new(
        identity: &RequestIdentity,
        task_id: Option<String>,
        sequence: u64,
        kind: impl Into<String>,
        mut data: Map<String, Value>,
    ) -> Self {
        for reserved in [
            "api_version",
            "request_id",
            "owner_id",
            "session_id",
            "task_id",
            "sequence",
            "type",
        ] {
            data.remove(reserved);
        }
        Self {
            api_version: SERVICE_API_VERSION.into(),
            request_id: identity.request_id.clone(),
            owner_id: identity.owner_id.clone(),
            session_id: identity.session_id.clone(),
            task_id,
            sequence,
            kind: kind.into(),
            data,
        }
    }
}

pub fn validate_api_version(value: &str) -> Result<(), String> {
    if value.trim() == SERVICE_API_VERSION {
        Ok(())
    } else {
        Err(format!(
            "unsupported api_version `{}`; expected `{SERVICE_API_VERSION}`",
            value.trim()
        ))
    }
}

pub fn validate_identifier(field: &str, value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty()
        || value.chars().count() > 128
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._:-/".contains(character))
    {
        return Err(format!(
            "{field} must be 1-128 ASCII letters, digits, `.`, `_`, `:`, `-`, or `/`"
        ));
    }
    Ok(value.to_string())
}

pub fn validate_session_id(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() || value.chars().count() > 128 || value.chars().any(char::is_control) {
        return Err("session_id must be 1-128 non-control characters".into());
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_message_is_an_explicit_task_spec_conversion() {
        let request = TaskRunRequest {
            api_version: SERVICE_API_VERSION.into(),
            request_id: None,
            owner_id: None,
            session_id: "s1".into(),
            task_id: None,
            message: " fix calc.py ".into(),
            working_directory: Some(" repo ".into()),
            task_spec: None,
        };
        let task = request.resolve_task_spec().unwrap();
        assert_eq!(task.objective, "fix calc.py");
        assert_eq!(task.working_directory.as_deref(), Some("repo"));
    }

    #[test]
    fn identity_defaults_owner_to_the_session_boundary() {
        let identity =
            RequestIdentity::resolve(SERVICE_API_VERSION, None, None, "session-1", || {
                "request-1".into()
            })
            .unwrap();
        assert_eq!(identity.owner_id, "session:session-1");
        assert_eq!(identity.request_id, "request-1");
    }

    #[test]
    fn unknown_versions_and_mismatched_legacy_fields_are_rejected() {
        assert!(validate_api_version("v2").is_err());
        let request = TaskRunRequest {
            api_version: SERVICE_API_VERSION.into(),
            request_id: None,
            owner_id: None,
            session_id: "s1".into(),
            task_id: None,
            message: "legacy".into(),
            working_directory: None,
            task_spec: Some(TaskSpec::new("structured")),
        };
        assert!(request.resolve_task_spec().is_err());
    }

    #[test]
    fn stream_event_identity_cannot_be_overridden_by_payload() {
        let identity = RequestIdentity::resolve(
            SERVICE_API_VERSION,
            Some("r1"),
            Some("owner-1"),
            "s1",
            String::new,
        )
        .unwrap();
        let event = ServiceStreamEvent::new(
            &identity,
            Some("task-1".into()),
            1,
            "phase",
            Map::from_iter([("type".into(), Value::String("forged".into()))]),
        );
        assert_eq!(event.kind, "phase");
        assert!(!event.data.contains_key("type"));
    }

    #[test]
    fn request_deserialization_rejects_unknown_wire_fields() {
        let value = serde_json::json!({
            "api_version":SERVICE_API_VERSION,
            "request_id":"request-1",
            "owner_id":"owner-1",
            "session_id":"session-1",
            "message":"hello",
            "unexpected":true
        });
        assert!(serde_json::from_value::<TaskRunRequest>(value).is_err());
    }

    #[test]
    fn checked_in_schema_tracks_wire_version_requests_and_error_codes() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../contracts/agent-service-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema
                .pointer("/$defs/apiVersion/const")
                .and_then(Value::as_str),
            Some(SERVICE_API_VERSION)
        );
        for definition in [
            "taskSpec",
            "taskRunRequest",
            "toolCallRequest",
            "researchRequest",
            "taskControlRequest",
            "streamEvent",
            "errorResponse",
        ] {
            assert!(schema.pointer(&format!("/$defs/{definition}")).is_some());
        }
        let codes = schema
            .pointer("/$defs/errorDetail/properties/code/enum")
            .and_then(Value::as_array)
            .unwrap();
        for code in [
            ServiceErrorCode::InvalidRequest,
            ServiceErrorCode::NotFound,
            ServiceErrorCode::Conflict,
            ServiceErrorCode::Unavailable,
            ServiceErrorCode::Unsupported,
            ServiceErrorCode::Cancelled,
            ServiceErrorCode::DeadlineExceeded,
            ServiceErrorCode::Internal,
        ] {
            let encoded = serde_json::to_value(code).unwrap();
            assert!(codes.contains(&encoded));
        }
    }
}
