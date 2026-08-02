use serde::Deserialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::{Action, ToolCall};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const ANSWER_OPEN: &str = "<answer>";
const ANSWER_CLOSE: &str = "</answer>";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("output must contain exactly one strict tool-call or answer envelope")]
    Envelope,
    #[error("invalid tool-call JSON: {0}")]
    Json(String),
    #[error("tool name must not be empty")]
    ToolName,
    #[error("answer must not be empty")]
    EmptyAnswer,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireToolCall {
    name: String,
    arguments: Map<String, Value>,
}

pub fn parse_action(raw: &str) -> Result<Action, ProtocolError> {
    let value = raw.trim();
    if let Some(body) = strict_envelope(value, TOOL_OPEN, TOOL_CLOSE) {
        let call: WireToolCall =
            serde_json::from_str(body).map_err(|error| ProtocolError::Json(error.to_string()))?;
        if call.name.trim().is_empty() {
            return Err(ProtocolError::ToolName);
        }
        return Ok(Action::Tool(ToolCall {
            name: call.name,
            arguments: call.arguments,
        }));
    }
    if let Some(body) = strict_envelope(value, ANSWER_OPEN, ANSWER_CLOSE) {
        let answer = body.trim();
        if answer.is_empty() {
            return Err(ProtocolError::EmptyAnswer);
        }
        return Ok(Action::Answer(answer.to_string()));
    }
    Err(ProtocolError::Envelope)
}

fn strict_envelope<'a>(value: &'a str, open: &str, close: &str) -> Option<&'a str> {
    value.strip_prefix(open)?.strip_suffix(close)
}

pub fn normalize_tool_result(result: Value) -> Value {
    match result {
        Value::Object(mut object) => {
            object
                .entry("status".to_string())
                .or_insert_with(|| Value::String("ok".to_string()));
            Value::Object(object)
        }
        other => json!({
            "status": "error",
            "message": "tool returned a non-object result",
            "value": other,
        }),
    }
}

pub fn render_observation(result: &Value) -> String {
    render_observation_with_suffix(result, "Assistant:")
}

pub fn render_answer_observation(result: &Value) -> String {
    render_observation_with_suffix(result, "Assistant: <answer>")
}

fn render_observation_with_suffix(result: &Value, suffix: &str) -> String {
    let compact_value = compact_observation_value(normalize_tool_result(result.clone()));
    let compact =
        serde_json::to_string(&compact_value).expect("serde_json::Value is always serializable");
    format!(
        "\n\nTool: <tool_result>{compact}</tool_result>\n\n\
         User: Continue the original task. Call a tool again only if more work or \
         verification is required; otherwise return the final answer. Output exactly \
         one protocol envelope and no reasoning.\n\n{suffix}"
    )
}

fn compact_observation_value(value: Value) -> Value {
    let Some(object) = value.as_object() else {
        return value;
    };
    if let Some(evidence) = object.get("evidence").and_then(Value::as_array) {
        let mut compact = Map::new();
        for key in [
            "status",
            "tool",
            "query",
            "effective_query",
            "message",
            "document",
        ] {
            if let Some(value) = object.get(key) {
                compact.insert(key.to_string(), bounded_value(value, &mut 1200usize));
            }
        }
        let rows = evidence
            .iter()
            .take(8)
            .map(|row| {
                let Some(source) = row.as_object() else {
                    return bounded_value(row, &mut 700usize);
                };
                let mut item = Map::new();
                for (key, limit) in [
                    ("id", 64usize),
                    ("title", 240),
                    ("content", 700),
                    ("uri", 500),
                    ("source", 120),
                    ("score", 40),
                    ("limited_evidence", 40),
                ] {
                    if let Some(value) = source.get(key) {
                        let mut field_budget = limit;
                        item.insert(key.to_string(), bounded_value(value, &mut field_budget));
                    }
                }
                Value::Object(item)
            })
            .collect();
        compact.insert("evidence".into(), Value::Array(rows));
        return Value::Object(compact);
    }
    bounded_value(&Value::Object(object.clone()), &mut 6000usize)
}

fn bounded_value(value: &Value, remaining: &mut usize) -> Value {
    match value {
        Value::String(text) => {
            let take = text.chars().count().min(*remaining).min(2000);
            *remaining = (*remaining).saturating_sub(take);
            let mut compact = text.chars().take(take).collect::<String>();
            if take < text.chars().count() {
                compact.push_str("…[truncated]");
            }
            Value::String(compact)
        }
        Value::Array(values) => Value::Array(
            values
                .iter()
                .take(16)
                .map(|value| bounded_value(value, remaining))
                .collect(),
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .take(32)
                .map(|(key, value)| (key.clone(), bounded_value(value, remaining)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_parser_accepts_one_tool_or_answer() {
        let action = parse_action(
            r#"<tool_call>{"name":"run_command","arguments":{"command":"pwd"}}</tool_call>"#,
        )
        .unwrap();
        let Action::Tool(call) = action else {
            panic!("expected tool call");
        };
        assert_eq!(call.name, "run_command");
        assert_eq!(call.arguments["command"], "pwd");
        assert_eq!(
            parse_action(" <answer>done</answer> ").unwrap(),
            Action::Answer("done".into())
        );
    }

    #[test]
    fn strict_parser_rejects_leakage_and_extra_wire_keys() {
        assert_eq!(
            parse_action("reasoning <answer>done</answer>"),
            Err(ProtocolError::Envelope)
        );
        assert!(matches!(
            parse_action(
                r#"<tool_call>{"name":"run_command","arguments":{},"extra":1}</tool_call>"#
            ),
            Err(ProtocolError::Json(_))
        ));
    }

    #[test]
    fn observation_is_compact_and_status_normalized() {
        let value = render_observation(&json!({"stdout": "ok\n"}));
        assert!(value.contains(r#"<tool_result>{"status":"ok","stdout":"ok\n"}</tool_result>"#));
        assert!(value.ends_with("Assistant:"));
    }

    #[test]
    fn observation_bounds_large_evidence_without_mutating_tool_result() {
        let result = json!({
            "status":"ok",
            "evidence":(1..=12).map(|index| json!({
                "id":format!("W{index}"),
                "title":"t".repeat(500),
                "content":"x".repeat(5000),
                "uri":"https://example.test/fact",
                "debug":"must not enter model context",
            })).collect::<Vec<_>>(),
            "trace":"z".repeat(10000),
        });
        let rendered = render_observation(&result);
        assert!(rendered.len() < 10_000);
        assert!(rendered.contains("W8"));
        assert!(!rendered.contains("W9"));
        assert!(!rendered.contains("must not enter model context"));
        assert_eq!(result["evidence"].as_array().unwrap().len(), 12);
    }

    #[test]
    fn answer_observation_commits_the_greedy_answer_prefix() {
        let rendered = render_answer_observation(&json!({"status":"ok","stdout":"done"}));
        assert!(rendered.ends_with("Assistant: <answer>"));
    }
}
