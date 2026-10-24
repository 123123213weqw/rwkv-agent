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
    let value: &str = &strip_think_blocks(raw.trim());
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
    if let Some(action) = parse_function_call(value) {
        return Ok(action);
    }
    Err(ProtocolError::Envelope)
}

/// Lenient parse used by the agent loop: accepts the checkpoint's most
/// frequent near-miss shapes — spontaneous `<think>` blocks (stripped
/// first), and one sentence of explanation followed by one complete
/// native call whose name is in `allowed`. The largest measured single
/// gain (+42%) in the G1L prompt-format study came from this tolerance.
pub fn parse_action_lenient(raw: &str, allowed: &[String]) -> Result<Action, ProtocolError> {
    let value: String = strip_think_blocks(raw.trim());
    if let Ok(action) = parse_action(&value) {
        return Ok(action);
    }
    for name in allowed {
        if name.is_empty() {
            continue;
        }
        let mut from = 0usize;
        while let Some(offset) = value[from..].find(name.as_str()) {
            let start = from + offset;
            let rest = &value[start..];
            let ends_ident = rest[name.len()..]
                .chars()
                .next()
                .is_some_and(|c| c == '(');
            let before_ok = start == 0
                || !value[..start]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            if ends_ident && before_ok
                && let Some(action @ Action::Tool(_)) = parse_function_call(rest)
            {
                return Ok(action);
            }
            from = start + name.len();
        }
    }
    Err(ProtocolError::Envelope)
}

/// Plain-answer acceptance for checkpoints that cannot close an envelope:
/// everything before a self-played role label is the answer.
pub fn parse_plain_answer(raw: &str) -> Option<String> {
    let value = strip_think_blocks(raw.trim());
    if value.is_empty() {
        return None;
    }
    let cut = value
        .find("\n\nUser:")
        .or_else(|| value.find("\nUser:"))
        .or_else(|| value.find("\n\nTool:"))
        .unwrap_or(value.len());
    let answer = value[..cut].trim();
    (!answer.is_empty()).then(|| answer.to_string())
}

fn strip_think_blocks(value: &str) -> String {
    let Some(open) = value.find("<think>") else {
        return value.to_string();
    };
    let Some(close_rel) = value[open..].find("</think>") else {
        return value[..open].trim_end().to_string();
    };
    let head = value[..open].trim_end();
    let tail = value[open + close_rel + "</think>".len()..].trim_start();
    match (head.is_empty(), tail.is_empty()) {
        (true, true) => String::new(),
        (true, false) => strip_think_blocks(tail),
        (false, true) => head.to_string(),
        (false, false) => strip_think_blocks(&format!("{head}\n{tail}")),
    }
}

/// Native-format fallback for checkpoints trained on function-call syntax:
/// `write_file(path="out.html", content="<html>…")`. Accepted only when the
/// whole output is a single call, so prose that merely starts like a call
/// still falls through to the protocol-retry path. Values follow JSON
/// escaping; the final value may be unterminated when generation was bounded
/// by max_tokens instead of the closing quote.
fn parse_function_call(value: &str) -> Option<Action> {
    let open = value.find('(')?;
    let name = &value[..open];
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let mut body = &value[open + 1..];
    if let Some(stripped) = body.strip_suffix(')') {
        body = stripped;
    }
    let is_arg = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut arguments = Map::new();
    let mut pos = 0usize;
    while pos < body.len() {
        let rest = &body[pos..];
        let first = rest.chars().next()?;
        if first == ',' || first.is_whitespace() {
            pos += first.len_utf8();
            continue;
        }
        if !is_arg(first) {
            return None;
        }
        let key_end = rest
            .char_indices()
            .find(|&(index, c)| index > 0 && !is_arg(c))
            .map_or(rest.len(), |(index, _)| index);
        let key = rest[..key_end].to_string();
        let after_key = rest[key_end..].trim_start();
        let value_start_in_rest = key_end + (rest.len() - key_end - after_key.len());
        if !after_key.starts_with('=') {
            return None;
        }
        let after_eq = after_key[1..].trim_start();
        let value_start = pos + value_start_in_rest + 1 + (after_key.len() - 1 - after_eq.len());
        let (raw_value, next_pos, terminated) = if after_eq.starts_with('"') {
            let inner = value_start + 1;
            match find_unescaped_quote(body, inner) {
                Some(close) => (&body[inner..close], close + 1, true),
                None => (body[inner..].trim_end(), body.len(), false),
            }
        } else {
            let rest_from = &body[value_start..];
            let end = rest_from.find(',').unwrap_or(rest_from.len());
            (rest_from[..end].trim(), value_start + end, true)
        };
        let parsed = if after_eq.starts_with('"') {
            unescape_value(raw_value)
        } else {
            serde_json::from_str::<Value>(raw_value)
                .unwrap_or_else(|_| Value::String(raw_value.to_string()))
        };
        arguments.insert(key, parsed);
        if !terminated {
            break;
        }
        pos = next_pos;
        let following = body[pos..].trim_start();
        if !following.starts_with(',') {
            // Tolerate role labels or prose the model appends after the
            // closing quote; the registry still validates the call itself.
            break;
        }
    }
    if arguments.is_empty() {
        return None;
    }
    Some(Action::Tool(ToolCall { name: name.to_string(), arguments }))
}

fn find_unescaped_quote(body: &str, from: usize) -> Option<usize> {
    let mut escaped = false;
    for (index, c) in body[from..].char_indices() {
        match c {
            '\\' => escaped = !escaped,
            '"' if !escaped => return Some(from + index),
            _ => escaped = false,
        }
    }
    None
}

fn unescape_value(raw: &str) -> Value {
    let quoted = format!("\"{}\"", raw.replace('\n', "\\n"));
    serde_json::from_str::<String>(&quoted)
        .map(Value::String)
        .unwrap_or_else(|_| Value::String(raw.to_string()))
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
    if value.ends_with("Assistant:") && !tool_prefix.is_empty() {
        value.push(' ');
        value.push_str(tool_prefix);
    }
    value
}

pub fn render_answer_observation(result: &Value) -> String {
    render_observation_with_suffix(result, "Assistant: <answer>")
}

/// Plain-answer mode for checkpoints that cannot close an `<answer>`
/// envelope: the final turn is unprefilled and any non-tool text is
/// accepted as the answer, trimmed at the first self-played role label.
pub fn render_plain_answer_observation_with_reminder(result: &Value, reminder: &str) -> String {
    render_observation_with_instruction(
        result,
        &format!(
            "All mandatory execution phases are complete. Answer the original task now in one short plain sentence of plain text. No tags, no role labels, no further tool calls, no reasoning. Original task (authoritative): {}",
            reminder.trim()
        ),
        "Assistant:",
    )
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
         User: {instruction} Output exactly one native tool call like name(argument=\"value\") or one <answer> envelope, and no reasoning.\n\n{suffix}"
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
    fn function_call_fallback_parses_native_syntax() {
        let action = parse_action(
            "write_file(path=\"card.html\", content=\"<div class=\\\"box\\\">hi</div>\")",
        )
        .unwrap();
        let Action::Tool(call) = action else {
            panic!("expected tool call");
        };
        assert_eq!(call.name, "write_file");
        assert_eq!(call.arguments["path"], "card.html");
        assert_eq!(call.arguments["content"], "<div class=\"box\">hi</div>");
    }

    #[test]
    fn function_call_fallback_accepts_unterminated_content() {
        let action = parse_action("write_file(path=\"g.html\", content=\"<html>unterminated").unwrap();
        let Action::Tool(call) = action else {
            panic!("expected tool call");
        };
        assert_eq!(call.arguments["path"], "g.html");
        assert_eq!(call.arguments["content"], "<html>unterminated");
    }

    #[test]
    fn function_call_fallback_supports_multiline_and_scalars() {
        let action =
            parse_action("run_command(command=\"echo hi\", retries=2, verbose=true)").unwrap();
        let Action::Tool(call) = action else {
            panic!("expected tool call");
        };
        assert_eq!(call.arguments["command"], "echo hi");
        assert_eq!(call.arguments["retries"], 2);
        assert_eq!(call.arguments["verbose"], true);
    }

    #[test]
    fn function_call_fallback_rejects_prose_starting_like_a_call() {
        assert_eq!(
            parse_action("Note (2024): the file above is fine"),
            Err(ProtocolError::Envelope)
        );
        assert_eq!(parse_action("write_file (no kwargs here)"), Err(ProtocolError::Envelope));
    }

    #[test]
    fn lenient_parser_accepts_sentence_then_complete_call() {
        let allowed = vec!["read_file".to_string()];
        let action = parse_action_lenient(
            "I'll inspect the workspace first to see the layout.\nread_file(path=\"SPEC.md\")",
            &allowed,
        )
        .unwrap();
        let Action::Tool(call) = action else {
            panic!("expected tool call");
        };
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments["path"], "SPEC.md");
    }

    #[test]
    fn think_blocks_are_stripped_from_every_path() {
        let action = parse_action(
            "<think>The user wants a file. Let me plan.</think>write_file(path=\"a.txt\", content=\"x\")",
        )
        .unwrap();
        assert!(matches!(action, Action::Tool(_)));
        let answer = parse_plain_answer("<think>hm</think>The file snake.html is complete with canvas and score.");
        assert_eq!(answer.as_deref(), Some("The file snake.html is complete with canvas and score."));
    }

    #[test]
    fn plain_answer_cuts_at_self_played_roles() {
        let text = "Task complete: pong.html written.\n\nUser: <tool_result>{\"ok\":1}";
        assert_eq!(
            parse_plain_answer(text).as_deref(),
            Some("Task complete: pong.html written.")
        );
        assert_eq!(parse_plain_answer("   "), None);
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
