use crate::Exchange;

pub const CHAT_STOPS: &[&str] = &["\n\nUser:", "\nUser:", "\nSystem:", "</s>"];

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
        "System: You are a helpful conversational assistant. Answer directly in the user's language. \
Do not claim to have searched, invent sources or citation IDs, call a tool, output <think>, or expose hidden reasoning. \
The supplied conversation is the only memory; there is no extracted long-term profile.\n\n",
    );
    if !context.is_empty() {
        value.push_str(context);
        value.push_str("\n\n");
    }
    value
}

pub fn direct_turn(message: &str, continuation: bool, previous_stop: &str) -> String {
    if continuation && matches!(previous_stop, "\n\nUser:" | "\nUser:") {
        return format!(" {}\n\nAssistant:", message.trim());
    }
    format!(
        "{}User: {}\n\nAssistant:",
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
    format!(
        "System: You are a bounded RWKV tool agent. At every turn output exactly one strict envelope: \
<tool_call>{{\"name\":...,\"arguments\":{{...}}}}</tool_call> or <answer>concise user-visible answer</answer>. \
Never output reasoning, role labels, Markdown fences, or protocol text outside the envelope. Use tools when evidence or execution is needed, \
inspect each Tool Result, and stop when the task is complete. One tool call per turn.\nFunctions:\n\
- web_search(query): live public Internet\n\
- knowledge_search(query): local indexed knowledge\n\
- long_text_qa(question): active pasted session text only\n\
{command_schema}Active pasted long text: {}.\n\n\
User: Who maintains ExampleDB?\n\nAssistant: <tool_call>{{\"name\":\"web_search\",\"arguments\":{{\"query\":\"ExampleDB maintainer official\"}}}}</tool_call>\n\n\
Tool: <tool_result>{{\"status\":\"ok\",\"evidence\":[{{\"id\":\"W1\",\"title\":\"ExampleDB\",\"content\":\"ExampleDB is maintained by Example Foundation.\",\"uri\":\"https://example.invalid/db\"}}]}}</tool_result>\n\n\
User: Continue the original task. Call a tool again only if more work is required; otherwise return the final answer. Output exactly one protocol envelope and no reasoning.\n\n\
Assistant: <answer>Example Foundation maintains ExampleDB [W1].</answer>\n\n\
Recent conversation:\n{}\n\nUser: {}",
        if has_text { "yes" } else { "no" },
        context,
        message.trim(),
    )
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
Output only {{\"name\":\"web_search\",\"arguments\":{{\"query\":\"...\"}}}}.\n\nAssistant: <tool_call>",
            question.trim()
        );
    }
    format!(
        "\n\nTool: <tool_result>{}</tool_result>\n\nUser: Continue the same mission. Do not repeat the prior query. \
Search one missing entity, relation, value, date, contradiction, or primary source while retaining the original subject. \
Output only {{\"name\":\"web_search\",\"arguments\":{{\"query\":\"...\"}}}}.\n\nAssistant: <tool_call>",
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
