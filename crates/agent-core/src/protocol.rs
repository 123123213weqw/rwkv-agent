use serde::Deserialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::{Action, ToolCall};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const ANSWER_OPEN: &str = "<answer>";
const ANSWER_CLOSE: &str = "</answer>";

/// Structural prefill used whenever the controller has already decided that
/// the next model action must be a tool call. Keeping the opening JSON shape
/// in the prompt prevents small or quantized models from drifting to an
/// incompatible `function`/`args` wire format; the strict parser remains the
/// sole authority for the completed envelope.
pub const TOOL_CALL_JSON_PREFIX: &str = "<tool_call>{\"name\":\"";

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

pub fn render_observation_with_progress(
    result: &Value,
    completed_steps: usize,
    max_steps: usize,
) -> String {
    render_observation_with_progress_and_reminder(result, completed_steps, max_steps, "")
}

pub fn render_observation_with_progress_and_reminder(
    result: &Value,
    completed_steps: usize,
    max_steps: usize,
    reminder: &str,
) -> String {
    let instruction = if completed_steps >= max_steps {
        format!(
            "Tool step {completed_steps}/{max_steps} is complete. No tool budget remains. Return the best truthful final answer now and do not call another tool."
        )
    } else {
        format!(
            "Tool step {completed_steps}/{max_steps} is complete. Continue the original task. Do not repeat an identical successful command. Call one tool only if a distinct required action or verification remains; otherwise return the final answer now."
        )
    };
    let instruction = if reminder.trim().is_empty() {
        instruction
    } else {
        format!(
            "Original task (authoritative): {}\n{instruction}\nBefore answering, compare every requested literal, filename, prefix, suffix, and verification result against this task.",
            reminder.trim()
        )
    };
    render_observation_with_instruction(result, &instruction, "Assistant:")
}

pub fn render_tool_observation_with_progress_and_reminder(
    result: &Value,
    completed_steps: usize,
    max_steps: usize,
    reminder: &str,
) -> String {
    render_tool_observation_with_progress_reminder_and_prefix(
        result,
        completed_steps,
        max_steps,
        reminder,
        TOOL_CALL_JSON_PREFIX,
    )
}

pub fn render_tool_observation_with_progress_reminder_and_prefix(
    result: &Value,
    completed_steps: usize,
    max_steps: usize,
    reminder: &str,
    tool_prefix: &str,
) -> String {
    let mut value =
        render_observation_with_progress_and_reminder(result, completed_steps, max_steps, reminder);
    if value.ends_with("Assistant:") {
        value.push(' ');
        value.push_str(tool_prefix);
    }
    value
}

pub fn render_answer_observation(result: &Value) -> String {
    render_observation_with_suffix(result, "Assistant: <answer>")
}

pub fn render_answer_observation_with_reminder(result: &Value, reminder: &str) -> String {
    render_observation_with_instruction(
        result,
        &format!(
            "All mandatory execution phases are complete. Answer the original task now. Include every explicitly requested value, count, relation, filename/source, and exact literal; do not omit labels or provenance. Original task (authoritative): {}",
            reminder.trim()
        ),
        "Assistant: <answer>",
    )
}

fn render_observation_with_suffix(result: &Value, suffix: &str) -> String {
    render_observation_with_instruction(
        result,
        "Continue the original task. Call a tool again only if more work or verification is required; otherwise return the final answer.",
        suffix,
    )
}

fn render_observation_with_instruction(result: &Value, instruction: &str, suffix: &str) -> String {
    let compact_value = compact_observation_value(normalize_tool_result(result.clone()));
    let compact =
        serde_json::to_string(&compact_value).expect("serde_json::Value is always serializable");
    format!(
        "\n\nTool: <tool_result>{compact}</tool_result>\n\n\
         User: {instruction} Output exactly one protocol envelope and no reasoning.\n\n{suffix}"
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

    #[test]
    fn progress_observation_exposes_budget_and_forces_final_boundary() {
        let middle = render_observation_with_progress(&json!({"status":"ok"}), 2, 4);
        assert!(middle.contains("Tool step 2/4"));
        assert!(middle.contains("Do not repeat an identical successful command"));
        let final_step = render_observation_with_progress(&json!({"status":"ok"}), 4, 4);
        assert!(final_step.contains("No tool budget remains"));
        assert!(final_step.ends_with("Assistant:"));
    }

    #[test]
    fn tool_observation_commits_the_greedy_tool_prefix() {
        let value = render_tool_observation_with_progress_and_reminder(
            &json!({"stdout":"input\n"}),
            1,
            4,
            "write out.txt next",
        );
        assert!(value.contains("write out.txt next"));
        assert!(value.ends_with("Assistant: <tool_call>{\"name\":\""));
    }

    #[test]
    fn tool_observation_can_commit_a_controller_selected_tool_name() {
        let value = render_tool_observation_with_progress_reminder_and_prefix(
            &json!({"stdout":"inputs inspected"}),
            3,
            8,
            "create output.py",
            "<tool_call>{\"name\":\"write_file\",\"arguments\":",
        );
        assert!(value.ends_with("Assistant: <tool_call>{\"name\":\"write_file\",\"arguments\":"));
    }

    #[test]
    fn progress_observation_repeats_authoritative_task_contract() {
        let rendered = render_observation_with_progress_and_reminder(
            &json!({"status":"ok","stdout":"4\n"}),
            2,
            6,
            "write exactly nonblank=4 to count.txt",
        );
        assert!(rendered.contains("Original task (authoritative)"));
        assert!(rendered.contains("write exactly nonblank=4 to count.txt"));
        assert!(rendered.contains("requested literal"));
    }
}
