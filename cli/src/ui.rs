use std::env;
use std::io::{self, IsTerminal};

use console::{Term, measure_text_width};
use serde_json::Value;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const PURPLE: &str = "\x1b[38;5;141m";
const CYAN: &str = "\x1b[38;5;81m";
const GREEN: &str = "\x1b[38;5;78m";
const YELLOW: &str = "\x1b[38;5;214m";
const RED: &str = "\x1b[38;5;203m";

#[derive(Clone, Debug)]
pub struct Ui {
    color: bool,
    width: usize,
}

impl Ui {
    pub fn for_stdout() -> Self {
        let tty = io::stdout().is_terminal();
        Self::new(tty, terminal_width(&Term::stdout()))
    }

    pub fn for_stderr() -> Self {
        let tty = io::stderr().is_terminal();
        Self::new(tty, terminal_width(&Term::stderr()))
    }

    #[cfg(test)]
    pub fn plain(width: usize) -> Self {
        Self {
            color: false,
            width: width.clamp(48, 88),
        }
    }

    fn new(tty: bool, width: usize) -> Self {
        let color = tty
            && env::var_os("NO_COLOR").is_none()
            && env::var("TERM").map_or(true, |value| value != "dumb");
        Self {
            color,
            width: width.clamp(48, 88),
        }
    }

    pub fn prompt(&self) -> String {
        self.paint(PURPLE, "❯ ")
    }

    pub fn continuation_prompt(&self) -> String {
        self.paint(DIM, "… ")
    }

    pub fn banner(&self, session: &str, endpoint: &str) -> String {
        let mut lines = Vec::new();
        lines.push(self.paint(PURPLE, &self.panel_top("RWKV Agent")));
        lines.push(self.panel_line("Local-first agent · tools · parallel states"));
        lines.push(self.panel_line(""));
        lines.push(self.panel_line(&format!("Session   {}", shorten(session, self.width - 14))));
        lines.push(self.panel_line(&format!("Endpoint  {}", shorten(endpoint, self.width - 15))));
        lines.push(self.paint(PURPLE, &self.panel_bottom()));
        lines.push(String::new());
        lines.push(format!(
            "  {}",
            self.paint(DIM, "/help 查看命令 · Ctrl-D 退出")
        ));
        lines.join("\n")
    }

    pub fn response(&self, value: &Value, raw_json: bool) -> String {
        if raw_json {
            return serde_json::to_string_pretty(value).unwrap_or_else(|_| compact_json(value));
        }

        if let Some(answer) = value.get("answer").and_then(Value::as_str) {
            let mut sections = Vec::new();
            if value.pointer("/route/mode").and_then(Value::as_str) == Some("document_capture") {
                sections.push(self.document_capture_summary(value));
            } else if value
                .pointer("/route/tool")
                .and_then(Value::as_str)
                .is_some()
            {
                sections.push(self.tool_summary(value));
            }
            sections.push(answer.trim().to_string());
            let sources = self.cited_sources(answer, value);
            if !sources.is_empty() {
                sections.push(sources);
            }
            return sections
                .into_iter()
                .filter(|section| !section.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
        }

        if value.get("tools").is_some() || value.get("model").is_some() {
            return self.health(value);
        }

        if value.get("evidence").and_then(Value::as_array).is_some() {
            return self.evidence_result(value);
        }

        let mut lines = vec![format!("status: {}", status_of(value))];
        if let Some(memory_id) = value.get("memory_id").and_then(Value::as_str) {
            lines.push(format!("memory: {memory_id}"));
        }
        if let Some(message) = value.get("message").and_then(Value::as_str) {
            lines.push(message.to_string());
        }
        if let Some(tool_result) = value.get("tool_result") {
            lines.push(format!("evidence: {}", evidence_count(tool_result)));
        }
        lines.join("\n")
    }

    pub fn doctor(&self, endpoint: &str, checks: &[(impl AsRef<str>, bool, String)]) -> String {
        let ready = checks.iter().all(|(_, passed, _)| *passed);
        let title = if ready {
            "Doctor · ready"
        } else {
            "Doctor · attention required"
        };
        let mut lines =
            vec![self.paint(if ready { GREEN } else { YELLOW }, &self.panel_top(title))];
        lines.push(self.panel_line(&format!("Endpoint  {}", shorten(endpoint, self.width - 15))));
        lines.push(self.panel_line(""));
        for (name, passed, detail) in checks {
            let mark = if *passed { "✓" } else { "!" };
            let row = format!(
                "{mark} {:<22} {}",
                name.as_ref(),
                shorten(detail, self.width.saturating_sub(30))
            );
            lines.push(self.panel_line(&row));
        }
        lines.push(self.paint(if ready { GREEN } else { YELLOW }, &self.panel_bottom()));
        lines.join("\n")
    }

    pub fn help(&self) -> String {
        let rows = [
            ("/help", "显示命令"),
            ("/status", "检查后端与模型"),
            ("/tools", "列出可用Function Call"),
            ("/session [name]", "显示或切换Session"),
            ("/knowledge <query>", "搜索本地知识库"),
            ("/web <query>", "搜索实时网页"),
            ("/research <question>", "B4 × 2并行State研究"),
            ("/longtext <question>", "询问当前粘贴长文本"),
            ("/json on|off", "切换完整JSON响应"),
            ("/clear", "清空终端"),
            ("/exit, /quit", "退出"),
        ];
        self.command_panel("Commands", &rows, Some("输入末尾加 \\ 可继续下一行"))
    }

    pub fn tools(&self) -> String {
        let rows = [
            ("web_search(query)", "实时发现与网页提取"),
            ("knowledge_search(query)", "本地索引知识"),
            ("long_text_qa(question)", "并行处理Session长文本"),
            ("/research <question>", "B4 × 2 State-native Web研究"),
        ];
        self.command_panel(
            "Available function calls",
            &rows,
            Some("直接粘贴长文本，等待接收后即可提问"),
        )
    }

    pub fn error(&self, error: &str) -> String {
        let connection = error.contains("request to ") && error.contains("failed");
        let title = if connection {
            "Connection failed"
        } else {
            "Request failed"
        };
        let mut lines = vec![self.paint(RED, &self.panel_top(title))];
        for line in wrap_words(error, self.width.saturating_sub(4)) {
            lines.push(self.panel_line(&line));
        }
        if connection {
            lines.push(self.panel_line(""));
            lines.push(self.panel_line("Try: rwkv-agent doctor"));
        }
        lines.push(self.paint(RED, &self.panel_bottom()));
        lines.join("\n")
    }

    fn tool_summary(&self, value: &Value) -> String {
        let state_research =
            value.pointer("/route/mode").and_then(Value::as_str) == Some("state_parallel_search");
        if state_research {
            return self.research_summary(value);
        }

        let tool = value
            .pointer("/route/tool")
            .and_then(Value::as_str)
            .unwrap_or("tool");
        let result = value.get("tool_result").unwrap_or(&Value::Null);
        let mut lines = vec![self.paint(CYAN, &format!("  ◇ {tool}"))];
        if let Some(arguments) = value.pointer("/route/arguments")
            && let Some(query) = arguments
                .get("query")
                .or_else(|| arguments.get("question"))
                .and_then(Value::as_str)
        {
            lines.push(format!("  ├─ Query    {}", shorten(query, self.width - 15)));
        }
        lines.push(format!("  ├─ Sources  {}", evidence_count(result)));
        if let Some(workers) = result.get("workers") {
            let completed = workers
                .get("completed")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let concurrency = workers
                .get("concurrency")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let candidates = workers
                .get("candidates")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            lines.push(format!(
                "  ├─ Chunks   {completed} · concurrency {concurrency} · {candidates} candidates"
            ));
        }
        let status = status_of(result);
        let elapsed = elapsed_label(value);
        lines.push(format!(
            "  └─ {}     {}{}",
            if status == "ok" { "Done" } else { "Status" },
            status,
            elapsed.map_or_else(String::new, |item| format!(" · {item}"))
        ));
        lines.join("\n")
    }

    fn document_capture_summary(&self, value: &Value) -> String {
        let document = value
            .pointer("/tool_result/document")
            .unwrap_or(&Value::Null);
        let characters = document.get("chars").and_then(Value::as_u64).unwrap_or(0);
        let name = document
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("session text");
        let mut lines = vec![self.paint(GREEN, "  ✓ Long text captured")];
        lines.push(format!(
            "    {} characters · {} · session only",
            characters,
            shorten(name, self.width.saturating_sub(34))
        ));
        lines.join("\n")
    }

    fn research_summary(&self, value: &Value) -> String {
        let branches = value
            .pointer("/route/branch_width")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let rounds = value
            .pointer("/route/rounds")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mut lines = vec![self.paint(
            CYAN,
            &format!("  ◇ Parallel state research · B{branches} × {rounds}"),
        )];
        let trace_rounds = value
            .pointer("/trace/rounds")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if trace_rounds.is_empty() {
            lines.push(format!(
                "  ├─ Plan      {branches} branches · {rounds} rounds"
            ));
        } else {
            for row in trace_rounds {
                let round = row.get("round").and_then(Value::as_u64).unwrap_or(0);
                let branch_rows = row
                    .get("branches")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let queries = branch_rows
                    .iter()
                    .filter(|item| {
                        item.pointer("/route/strict").and_then(Value::as_bool) == Some(true)
                    })
                    .count();
                let evidence: u64 = branch_rows
                    .iter()
                    .filter_map(|item| item.get("evidence_count").and_then(Value::as_u64))
                    .sum();
                lines.push(format!(
                    "  ├─ Round {round:<2}  {queries} queries · {evidence} evidence"
                ));
            }
        }
        let result = value.get("tool_result").unwrap_or(&Value::Null);
        lines.push(format!(
            "  ├─ Reduced   {} evidence",
            evidence_count(result)
        ));
        let elapsed = elapsed_label(value).unwrap_or_else(|| status_of(result).to_string());
        lines.push(format!("  └─ Done      {elapsed}"));
        lines.join("\n")
    }

    fn cited_sources(&self, answer: &str, value: &Value) -> String {
        let evidence = value
            .pointer("/tool_result/evidence")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let cited = citation_ids(answer);
        if cited.is_empty() || evidence.is_empty() {
            return String::new();
        }
        let selected = evidence
            .iter()
            .filter(|item| {
                item.get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| cited.iter().any(|wanted| wanted == id))
            })
            .take(6)
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return String::new();
        }
        let mut lines = vec![self.paint(DIM, "  Sources")];
        for item in selected {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("?");
            let title = item
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("Untitled");
            lines.push(format!("  [{id}] {}", shorten(title, self.width - 9)));
            if let Some(uri) = item.get("uri").and_then(Value::as_str) {
                lines.push(format!("       {}", self.paint(DIM, &short_host(uri))));
            }
        }
        lines.join("\n")
    }

    fn evidence_result(&self, value: &Value) -> String {
        let status = status_of(value);
        let evidence = value
            .get("evidence")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut lines = vec![self.paint(
            if status == "ok" { GREEN } else { YELLOW },
            &format!("  ◇ Evidence · {status} · {} sources", evidence.len()),
        )];
        for item in evidence.iter().take(8) {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("?");
            let title = item
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("Untitled");
            lines.push(format!("  [{id}] {}", shorten(title, self.width - 9)));
            if let Some(content) = item.get("content").and_then(Value::as_str) {
                lines.push(format!("       {}", shorten(content, self.width - 9)));
            }
            if let Some(uri) = item.get("uri").and_then(Value::as_str) {
                lines.push(format!(
                    "       {}",
                    self.paint(DIM, &shorten(uri, self.width - 9))
                ));
            }
        }
        lines.join("\n")
    }

    fn health(&self, value: &Value) -> String {
        let status = status_of(value);
        let ready = status == "ready" || status == "ok";
        let models = value
            .get("model")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let model_names = models
            .iter()
            .filter_map(|item| item.get("model").and_then(Value::as_str))
            .collect::<Vec<_>>();
        let tools = value
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let tool_names = tools.iter().filter_map(Value::as_str).collect::<Vec<_>>();
        let mut lines = vec![self.paint(
            if ready { GREEN } else { YELLOW },
            &self.panel_top("Agent status"),
        )];
        lines.push(self.panel_line(&format!("status: {status}")));
        if !model_names.is_empty() {
            lines.push(self.panel_line(&format!(
                "model:  {}",
                shorten(&model_names.join(", "), self.width - 12)
            )));
        } else if !models.is_empty() {
            lines.push(self.panel_line(&format!("models: {} ready", models.len())));
        }
        if !tool_names.is_empty() {
            lines.push(self.panel_line(&format!(
                "tools:  {}",
                shorten(&tool_names.join(", "), self.width - 12)
            )));
        }
        lines.push(self.paint(if ready { GREEN } else { YELLOW }, &self.panel_bottom()));
        lines.join("\n")
    }

    fn command_panel(&self, title: &str, rows: &[(&str, &str)], footer: Option<&str>) -> String {
        let mut lines = vec![self.paint(PURPLE, &self.panel_top(title))];
        for (command, description) in rows {
            lines.push(self.panel_line(&format!("{command:<23} {description}")));
        }
        if let Some(footer) = footer {
            lines.push(self.panel_line(""));
            lines.push(self.panel_line(footer));
        }
        lines.push(self.paint(PURPLE, &self.panel_bottom()));
        lines.join("\n")
    }

    fn panel_top(&self, title: &str) -> String {
        let prefix = format!("╭─ {title} ");
        let fill = self
            .width
            .saturating_sub(measure_text_width(&prefix) + measure_text_width("╮"));
        format!("{prefix}{}╮", "─".repeat(fill))
    }

    fn panel_bottom(&self) -> String {
        format!("╰{}╯", "─".repeat(self.width.saturating_sub(2)))
    }

    fn panel_line(&self, value: &str) -> String {
        let content = shorten(value, self.width.saturating_sub(4));
        let padding = self.width.saturating_sub(measure_text_width(&content) + 4);
        format!("│ {content}{} │", " ".repeat(padding))
    }

    fn paint(&self, code: &str, value: &str) -> String {
        if self.color {
            format!("{code}{value}{RESET}")
        } else {
            value.to_string()
        }
    }
}

fn terminal_width(term: &Term) -> usize {
    let (_, columns) = term.size();
    usize::from(columns).clamp(48, 88)
}

fn elapsed_label(value: &Value) -> Option<String> {
    let milliseconds = value
        .pointer("/trace/elapsed_ms")
        .or_else(|| value.pointer("/tool_result/retrieval/latency_ms"))
        .and_then(Value::as_f64)?;
    if milliseconds >= 1_000.0 {
        Some(format!("{:.1}s", milliseconds / 1_000.0))
    } else {
        Some(format!("{milliseconds:.0}ms"))
    }
}

fn citation_ids(answer: &str) -> Vec<String> {
    let mut output = Vec::new();
    let bytes = answer.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'[' {
            index += 1;
            continue;
        }
        let Some(relative_end) = answer[index + 1..].find(']') else {
            break;
        };
        let end = index + 1 + relative_end;
        let candidate = &answer[index + 1..end];
        let mut chars = candidate.chars();
        let prefix = chars.next();
        if prefix.is_some_and(|value| matches!(value, 'W' | 'K' | 'L' | 'M'))
            && chars.clone().next().is_some()
            && chars.all(|value| value.is_ascii_digit())
            && !output.iter().any(|value| value == candidate)
        {
            output.push(candidate.to_string());
        }
        index = end + 1;
    }
    output
}

fn short_host(uri: &str) -> String {
    uri.split_once("://")
        .map(|(_, tail)| tail.split('/').next().unwrap_or(tail))
        .unwrap_or(uri)
        .trim_start_matches("www.")
        .to_string()
}

fn wrap_words(value: &str, width: usize) -> Vec<String> {
    if value.trim().is_empty() {
        return vec![String::new()];
    }
    let mut output = Vec::new();
    let mut current = String::new();
    for word in value.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        if measure_text_width(&candidate) > width && !current.is_empty() {
            output.push(current);
            current = word.to_string();
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        output.push(shorten(&current, width));
    }
    output
}

pub fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid JSON>".to_string())
}

pub fn evidence_count(value: &Value) -> usize {
    value
        .get("evidence")
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

pub fn status_of(value: &Value) -> &str {
    value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

pub fn shorten(value: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if measure_text_width(&normalized) <= limit {
        return normalized;
    }
    let target = limit.saturating_sub(1);
    let mut output = String::new();
    for character in normalized.chars() {
        let candidate = format!("{output}{character}");
        if measure_text_width(&candidate) > target {
            break;
        }
        output.push(character);
    }
    output.push('…');
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_banner_has_stable_width_without_ansi() {
        let ui = Ui::plain(58);
        let banner = ui.banner("demo", "http://127.0.0.1:8120");
        assert!(banner.contains("RWKV Agent"));
        assert!(!banner.contains("\x1b["));
        for line in banner
            .lines()
            .filter(|line| line.starts_with(['╭', '│', '╰']))
        {
            assert_eq!(measure_text_width(line), 58);
        }
    }

    #[test]
    fn answer_hides_raw_tool_payload_and_lists_only_cited_sources() {
        let ui = Ui::plain(64);
        let value = json!({
            "status": "ok",
            "route": {
                "tool": "web_search",
                "arguments": {"query": "RWKV author official"}
            },
            "tool_result": {
                "status": "ok",
                "evidence": [
                    {"id": "W1", "title": "Official", "uri": "https://rwkv.com/a"},
                    {"id": "W2", "title": "Unused", "uri": "https://example.com/b"}
                ]
            },
            "answer": "The author is documented here [W1]."
        });
        let output = ui.response(&value, false);
        assert!(output.contains("◇ web_search"));
        assert!(output.contains("RWKV author official"));
        assert!(output.contains("[W1] Official"));
        assert!(!output.contains("[W2] Unused"));
        assert!(!output.contains("\"arguments\""));
    }

    #[test]
    fn research_summary_uses_real_round_trace() {
        let ui = Ui::plain(64);
        let value = json!({
            "status": "ok",
            "route": {"mode": "state_parallel_search", "tool": "web_search", "branch_width": 4, "rounds": 2},
            "tool_result": {"status": "ok", "evidence": [{"id": "W1"}]},
            "answer": "answer",
            "trace": {
                "elapsed_ms": 18700.0,
                "rounds": [
                    {"round": 1, "branches": [
                        {"route": {"strict": true}, "evidence_count": 3},
                        {"route": {"strict": true}, "evidence_count": 2}
                    ]}
                ]
            }
        });
        let output = ui.response(&value, false);
        assert!(output.contains("Parallel state research · B4 × 2"));
        assert!(output.contains("Round 1"));
        assert!(output.contains("2 queries · 5 evidence"));
        assert!(output.contains("18.7s"));
    }

    #[test]
    fn raw_json_mode_is_unstyled_and_complete() {
        let ui = Ui::plain(64);
        let value = json!({"status": "ok", "answer": "hello"});
        let output = ui.response(&value, true);
        assert!(output.contains("\"answer\": \"hello\""));
        assert!(!output.contains("\x1b["));
    }

    #[test]
    fn utf8_shortening_uses_terminal_width() {
        assert_eq!(shorten("这是一个很长的中文句子", 12), "这是一个很…");
        assert_eq!(shorten("short text", 20), "short text");
    }
}
