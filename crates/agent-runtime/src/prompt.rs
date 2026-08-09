use crate::Exchange;
use rwkv_agent_core::{TOOL_CALL_JSON_PREFIX, TaskSpec};

pub const CHAT_STOPS: &[&str] = &["</answer>", "\n\nUser:", "\nUser:", "\nSystem:", "</s>"];

pub fn bounded_context(history: &[Exchange], max_chars: usize) -> String {
    let mut rows = Vec::new();
    let mut used = 0usize;
    for exchange in history.iter().rev() {
        for (label, text) in [
            ("Assistant", exchange.assistant.as_str()),
            ("User", exchange.user.as_str()),
        ] {
            let line = format!("{label}: {}", text.trim());
            let remaining = max_chars.saturating_sub(used + usize::from(!rows.is_empty()));
            if remaining == 0 {
                break;
            }
            let line = if line.chars().count() > remaining {
                line.chars()
                    .rev()
                    .take(remaining)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect()
            } else {
                line
            };
            used += line.chars().count() + usize::from(!rows.is_empty());
            rows.push(line);
        }
    }
    rows.reverse();
    rows.join("\n")
}

pub fn direct_prefix(context: &str) -> String {
    let mut value = String::from(
        "System: Answer directly in the user's language. Do not use tools, sources, <think>, or hidden reasoning. \
Continue `Assistant: <answer>` with only the visible answer, then close `</answer>`.\n\n",
    );
    if !context.is_empty() {
        value.push_str(context);
        value.push_str("\n\n");
    }
    value
}

pub fn direct_turn(message: &str, continuation: bool, previous_stop: &str) -> String {
    if continuation && matches!(previous_stop, "\n\nUser:" | "\nUser:") {
        return format!(" {}\n\nAssistant: <answer>", message.trim());
    }
    format!(
        "{}User: {}\n\nAssistant: <answer>",
        if continuation { "\n\n" } else { "" },
        message.trim()
    )
}

pub fn agent_prompt(message: &str, has_text: bool, context: &str, command: bool) -> String {
    let command_schema = if command {
        "- run_command(command): run one bounded command in the configured sandbox workspace\n"
    } else {
        ""
    };
    let selection_contract = if has_text {
        "Tool-selection contract: An active pasted text exists. Questions about the supplied, active, current, or just-pasted material MUST use long_text_qa. Do not substitute web_search for the active text. Use web_search only when the user explicitly asks for live Internet information; use knowledge_search only when the user explicitly asks for the local index.\n\n\
User: What is the final approval code in the supplied material?\n\nAssistant: <tool_call>{\"name\":\"long_text_qa\",\"arguments\":{\"question\":\"What is the final approval code?\"}}</tool_call>\n\n\
Tool: <tool_result>{\"status\":\"ok\",\"answer_hint\":\"AX-17\",\"evidence\":[{\"id\":\"L1\",\"content\":\"The final approval code is AX-17.\"}]}</tool_result>\n\n\
User: Continue the original task. Call a tool again only if more work is required; otherwise return the final answer. Output exactly one protocol envelope and no reasoning.\n\n\
Assistant: <answer>AX-17 [L1]</answer>"
    } else {
        "Tool-selection contract: No active pasted text exists. Use web_search for live public facts and knowledge_search for the local index. Do not call long_text_qa until text is active.\n\n\
User: Who maintains ExampleDB?\n\nAssistant: <tool_call>{\"name\":\"web_search\",\"arguments\":{\"query\":\"ExampleDB maintainer official\"}}</tool_call>\n\n\
Tool: <tool_result>{\"status\":\"ok\",\"evidence\":[{\"id\":\"W1\",\"title\":\"ExampleDB\",\"content\":\"ExampleDB is maintained by Example Foundation.\",\"uri\":\"https://example.invalid/db\"}]}</tool_result>\n\n\
User: Continue the original task. Call a tool again only if more work is required; otherwise return the final answer. Output exactly one protocol envelope and no reasoning.\n\n\
Assistant: <answer>Example Foundation maintains ExampleDB [W1].</answer>"
    };
    format!(
        "System: You are a bounded RWKV tool agent. At every turn output exactly one strict envelope: \
<tool_call>{{\"name\":...,\"arguments\":{{...}}}}</tool_call> or <answer>concise user-visible answer</answer>. \
Never output reasoning, role labels, Markdown fences, or protocol text outside the envelope. Use tools when evidence or execution is needed, \
inspect each Tool Result, and stop when the task is complete. One tool call per turn.\nFunctions:\n\
- web_search(query): live public Internet\n\
- knowledge_search(query): local indexed knowledge\n\
- long_text_qa(question): active pasted session text only\n\
{command_schema}Active pasted long text: {}.\n\n\
{}\n\n\
Recent conversation:\n{}\n\nUser: {}",
        if has_text { "yes" } else { "no" },
        selection_contract,
        context,
        message.trim(),
    )
}

pub fn task_spec_message(task: &TaskSpec) -> String {
    if task.acceptance_criteria.is_empty()
        && task.constraints.is_empty()
        && task.verification_commands.is_empty()
        && task.requires_mutation.is_none()
    {
        return task.objective.clone();
    }
    format!(
        "TaskSpec:\n{}",
        serde_json::to_string_pretty(task).expect("TaskSpec is always JSON serializable")
    )
}

#[cfg(test)]
pub fn workspace_agent_prompt(
    message: &str,
    context: &str,
    display_directory: &str,
    max_tool_steps: usize,
) -> String {
    let task = TaskSpec::legacy(message, Some(display_directory.to_string()));
    workspace_agent_prompt_for_task(&task, context, display_directory, max_tool_steps)
}

pub fn workspace_agent_prompt_for_task(
    task: &TaskSpec,
    context: &str,
    display_directory: &str,
    max_tool_steps: usize,
) -> String {
    let rendered_task = render_workspace_task_spec(task, display_directory);
    format!(
        "System: You are a bounded workspace agent. Complete the user's task autonomously with the workspace tools and inspect every Tool Result before choosing the next action. \
At each turn output exactly one strict envelope: <tool_call>{{\"name\":\"tool_name\",\"arguments\":{{...}}}}</tool_call> \
or <answer>concise user-visible answer</answer>. Output no reasoning, role labels, Markdown fences, empty commands, or text outside the envelope. \
Tool environment contract: the selected workspace is `/workspace`, every command starts there, and only files inside it persist between calls. \
Each call is a new isolated process: `/tmp`, shell variables, and process state do not persist. Never use a temporary path to communicate between calls. \
There is no network or package installation. Python is available only as `python3`; `python`, Pytest, and dependency installation are unavailable. \
A Python test file supplied by the user is a standalone script and can be executed directly with `python3` and its filename. \
	The Controller supplies a bounded authoritative inventory of existing workspace files with the task. Use it to distinguish inputs from not-yet-created targets; do not guess a missing target is an input. List directories only if the inventory is truncated or a deeper detail is genuinely needed. \
	Do not create or modify artifacts the user did not request. Use read_file for named text, edit_file for one exact replacement, write_file for complete file content, and run_command for listing, tests, or multi-file verification. \
	The Controller may run the exact declared verifier after inspection or mutation and may read one grounded repair target after a failure. These controller actions are authoritative observations, not requests to repeat the same command. \
	When the Controller enters MUTATE it may pre-fill `<answer>` and name one target path. In that phase, return only the complete replacement file text inside the strict answer envelope; the Controller will package it into the required edit_file/write_file call, validate the strict Tool schema, and execute it in the same Sandbox. Never include reasoning, Markdown fences, tests, specifications, role labels, or fake Tool results in replacement content. \
	You have at most {max_tool_steps} tool calls. Use the fewest necessary actions while following the user's requested order. Treat exact content as a byte contract: add no spaces, labels, or files that were not requested. \
If a command fails, use its actual result to choose a different corrective action instead of repeating it unchanged. \
After the requested verification succeeds, answer on the next turn instead of calling another command merely to restate the same result. \
Answer only when the task is complete; otherwise call the tool once and continue in the same task state. \
Functions:\n\
- read_file(path): read one UTF-8 text file; path is relative to `/workspace`\n\
- write_file(path, content): atomically create or replace one UTF-8 text file; include exact final content\n\
- edit_file(path, old_text, new_text): replace old_text exactly once; use text copied from read_file\n\
- run_command(command): run one bounded shell command for listing or verification in the isolated workspace\n\
Never invent a tool result. If edit_file reports zero or multiple matches, read the file again and choose a unique exact old_text.\n\n\
Recent conversation:\n{context}\n\nUser: TaskSpec:\n{rendered_task}",
    )
}

#[cfg(test)]
pub fn workspace_agent_state_prompts(
    message: &str,
    context: &str,
    display_directory: &str,
    max_tool_steps: usize,
) -> (String, String) {
    let task = TaskSpec::legacy(message, Some(display_directory.to_string()));
    workspace_agent_state_prompts_for_task(&task, context, display_directory, max_tool_steps, &[])
}

pub fn workspace_agent_state_prompts_for_task(
    task: &TaskSpec,
    context: &str,
    display_directory: &str,
    max_tool_steps: usize,
    workspace_inventory: &[String],
) -> (String, String) {
    let rendered =
        workspace_agent_prompt_for_task(task, context, display_directory, max_tool_steps);
    let marker = "\n\nRecent conversation:\n";
    let root_end = rendered
        .rfind(marker)
        .expect("workspace prompt always contains the conversation marker");
    let root = format!(
        "{}\n\nSystem: The next User message supplies the current workspace task. Keep its complete execution trajectory in this task state.",
        &rendered[..root_end]
    );
    let task = render_workspace_task_spec(task, display_directory);
    let inventory = if workspace_inventory.is_empty() {
        "- (empty workspace)".to_string()
    } else {
        workspace_inventory
            .iter()
            .map(|path| format!("- {path}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let initial = format!(
        "\n\nUser: Recent conversation:\n{context}\n\nWorkspace TaskSpec:\n{task}\n\nWorkspace inventory at task start (authoritative; a path not listed does not exist yet):\n{inventory}\n\nAssistant: {TOOL_CALL_JSON_PREFIX}"
    );
    (root, initial)
}

fn render_workspace_task_spec(task: &TaskSpec, display_directory: &str) -> String {
    let mut normalized = task.clone();
    normalized.objective = normalize_workspace_task(&normalized.objective, display_directory);
    for value in &mut normalized.acceptance_criteria {
        *value = normalize_workspace_task(value, display_directory);
    }
    for value in &mut normalized.constraints {
        *value = normalize_workspace_task(value, display_directory);
    }
    for value in &mut normalized.verification_commands {
        *value = normalize_workspace_task(value, display_directory);
    }
    normalized.working_directory = Some("/workspace".into());
    serde_json::to_string_pretty(&normalized).expect("TaskSpec is always JSON serializable")
}

pub fn normalize_workspace_task(message: &str, display_directory: &str) -> String {
    if display_directory.is_empty() {
        message.trim().to_string()
    } else {
        message
            .replace(display_directory, "/workspace")
            .trim()
            .to_string()
    }
}

pub fn research_root(question: &str) -> String {
    format!(
        "System: You are a bounded state-native research agent. The Controller will fork this recurrent state. \
Branches emit exactly one web_search Tool Call when instructed. The retained root answers only in the final-answer stage, \
using supplied Evidence and citing every factual claim with [W#]. Never expose reasoning or invent evidence.\n\nUser: {}",
        question.trim()
    )
}

pub fn branch_step(
    question: &str,
    mission: &str,
    round_index: usize,
    observation: Option<&serde_json::Value>,
) -> String {
    if round_index == 1 {
        return format!(
            "\n\nUser: Branch mission: {mission}\nOriginal question: {}\nProduce one focused web_search call now. \
Output only {{\"name\":\"web_search\",\"arguments\":{{\"query\":\"...\"}}}}.\n\nAssistant: {TOOL_CALL_JSON_PREFIX}",
            question.trim()
        );
    }
    format!(
        "\n\nTool: <tool_result>{}</tool_result>\n\nUser: Continue the same mission. Do not repeat the prior query. \
Search one missing entity, relation, value, date, contradiction, or primary source while retaining the original subject. \
Output only {{\"name\":\"web_search\",\"arguments\":{{\"query\":\"...\"}}}}.\n\nAssistant: {TOOL_CALL_JSON_PREFIX}",
        observation
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"status":"empty","evidence":[]}))
    )
}

pub fn research_final(question: &str, evidence: &[serde_json::Value]) -> String {
    format!(
        "\n\nTool: <tool_result>{}</tool_result>\n\nUser: Final answer stage. Answer the original question directly in its language \
using only Evidence. Cite every factual claim with [W#]. Preserve exact relation labels. If insufficient, say so. \
Output only concise answer text followed by </answer>. Original question: {}\n\nAssistant: <answer>",
        serde_json::json!({"status": if evidence.is_empty() {"empty"} else {"ok"}, "evidence": evidence}),
        question.trim()
    )
}

pub const MISSIONS: &[&str] = &[
    "Find the primary answer and strongest directly relevant source.",
    "Prefer official, primary, or first-party sources for key claims.",
    "Find an independent source that corroborates the likely answer.",
    "Look for missing facts, ambiguity, date issues, or contradictions.",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_chat_forces_visible_answer_envelope() {
        assert_eq!(CHAT_STOPS.first(), Some(&"</answer>"));
        assert!(direct_prefix("").contains("close `</answer>`"));
        assert_eq!(
            direct_turn("hello", false, ""),
            "User: hello\n\nAssistant: <answer>"
        );
        assert_eq!(
            direct_turn("again", true, "</answer>"),
            "\n\nUser: again\n\nAssistant: <answer>"
        );
    }

    #[test]
    fn workspace_prompt_maps_requested_directory_to_relative_commands() {
        let value = workspace_agent_prompt(
            "Work in runs/c1/case and read input.txt.",
            "",
            "runs/c1/case",
            8,
        );
        assert!(value.contains("workspace is `/workspace`"));
        assert!(value.contains("only files inside it persist"));
        assert!(value.contains("at most 8 tool calls"));
        assert!(value.contains("different corrective action"));
        assert!(value.contains("- read_file(path)"));
        assert!(value.contains("- write_file(path, content)"));
        assert!(value.contains("- edit_file(path, old_text, new_text)"));
        assert!(value.contains("Work in /workspace and read input.txt."));
        assert!(value.ends_with('}'));
    }

    #[test]
    fn active_text_prompt_prioritizes_long_text_without_hiding_other_tools() {
        let value = agent_prompt("What does the material say?", true, "", true);
        assert!(value.contains("MUST use long_text_qa"));
        assert!(value.contains("Do not substitute web_search"));
        assert!(value.contains("\"name\":\"long_text_qa\""));
        assert!(value.contains("- web_search(query)"));
        assert!(value.contains("- run_command(command)"));
    }

    #[test]
    fn structured_non_workspace_task_keeps_acceptance_contract() {
        let mut task = TaskSpec::new("Find the current release");
        task.acceptance_criteria = vec!["Use an official source".into()];
        let rendered = task_spec_message(&task);
        assert!(rendered.starts_with("TaskSpec:\n{"));
        assert!(rendered.contains("Use an official source"));
    }

    #[test]
    fn workspace_state_root_is_pristine_and_task_starts_the_worker() {
        let (root, initial) = workspace_agent_state_prompts(
            "Work in runs/c1/a and read input.txt.",
            "",
            "runs/c1/a",
            8,
        );
        assert!(!root.contains("Work in /workspace and read input.txt."));
        assert!(root.contains("next User message supplies the current workspace task"));
        assert!(root.contains("Keep its complete execution trajectory"));
        assert!(initial.contains("Workspace TaskSpec:"));
        assert!(initial.contains("Work in /workspace and read input.txt."));
        assert!(initial.contains("\"schema_version\": 1"));
        assert!(initial.ends_with("Assistant: <tool_call>{\"name\":\""));
    }

    #[test]
    fn structured_task_spec_is_rendered_with_workspace_alias() {
        let mut task = TaskSpec::legacy("Fix /repo/calc.py", Some("/repo".into()));
        task.acceptance_criteria = vec!["/repo/calc.py passes its test".into()];
        task.constraints = vec!["Do not change /repo/api.py".into()];
        task.verification_commands = vec!["python3 /repo/test_calc.py".into()];
        task.requires_mutation = Some(true);
        let (_, initial) = workspace_agent_state_prompts_for_task(
            &task,
            "",
            "/repo",
            6,
            &[
                "calc.py (24 bytes)".into(),
                "test_calc.py (41 bytes)".into(),
            ],
        );
        assert!(initial.contains("Fix /workspace/calc.py"));
        assert!(initial.contains("/workspace/calc.py passes its test"));
        assert!(initial.contains("Do not change /workspace/api.py"));
        assert!(initial.contains("python3 /workspace/test_calc.py"));
        assert!(initial.contains("calc.py (24 bytes)"));
        assert!(initial.contains("a path not listed does not exist yet"));
        assert!(!initial.contains("/repo"));
    }
}
