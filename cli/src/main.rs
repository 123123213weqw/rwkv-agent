use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde_json::{Value, json};

const PASTED_TEXT_CAPTURE_CHARS: usize = 4_000;

#[derive(Debug, Parser)]
#[command(
    name = "rwkv-agent",
    version,
    about = "Claude-style terminal client for RWKV tools and state-native research"
)]
struct Cli {
    /// Agent HTTP endpoint.
    #[arg(
        long,
        env = "RWKV_AGENT_ENDPOINT",
        default_value = "http://127.0.0.1:8120",
        global = true
    )]
    endpoint: String,

    /// Resume a named session instead of creating a fresh conversation.
    #[arg(long, short = 's', env = "RWKV_AGENT_SESSION", global = true)]
    session: Option<String>,

    /// Print the complete API response as JSON.
    #[arg(long, global = true)]
    json: bool,

    /// HTTP timeout in seconds.
    #[arg(long, default_value_t = 180, global = true)]
    timeout: u64,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Verify the controller, model Sidecar and required tools.
    Doctor,

    /// Check Agent and model health.
    Health,

    /// Send one message through model routing and tool execution.
    Ask {
        #[arg(required = true, trailing_var_arg = true)]
        message: Vec<String>,
    },

    /// Run bounded parallel-state Web research (four branches, two rounds).
    Research {
        #[arg(required = true, trailing_var_arg = true)]
        question: Vec<String>,

        /// Number of independent recurrent-state branches (1-4).
        #[arg(long, default_value_t = 4)]
        branches: u8,

        /// Number of search-and-observe rounds per branch (1-3).
        #[arg(long, default_value_t = 2)]
        rounds: u8,
    },

    /// Start an interactive terminal conversation.
    Chat,

    /// Call a function directly without model routing.
    Tool {
        #[command(subcommand)]
        command: ToolCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ToolCommand {
    /// Call live web discovery and extraction.
    WebSearch {
        #[arg(required = true, trailing_var_arg = true)]
        query: Vec<String>,
    },

    /// Search local indexed knowledge.
    KnowledgeSearch {
        #[arg(required = true, trailing_var_arg = true)]
        query: Vec<String>,
    },

    /// Ask the long text currently pasted into this session.
    LongTextQa {
        #[arg(required = true, trailing_var_arg = true)]
        question: Vec<String>,
    },
}

struct AgentClient {
    endpoint: String,
    session: String,
    http: Client,
}

impl AgentClient {
    fn new(endpoint: &str, session: Option<&str>, timeout: u64) -> Result<Self, String> {
        let normalized = endpoint.trim().trim_end_matches('/');
        if normalized.is_empty() {
            return Err("endpoint must not be empty".to_string());
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(timeout.max(1)))
            .build()
            .map_err(|error| format!("cannot construct HTTP client: {error}"))?;
        Ok(Self {
            endpoint: normalized.to_string(),
            session: resolve_session(session),
            http,
        })
    }

    fn health(&self) -> Result<Value, String> {
        self.request("GET", "/health", None)
    }

    fn ask(&self, message: &str) -> Result<Value, String> {
        self.request(
            "POST",
            "/v1/agent/run",
            Some(json!({
                "session_id": self.session,
                "message": require_text(message, "message")?,
            })),
        )
    }

    fn research(&self, question: &str, branch_width: u8, max_rounds: u8) -> Result<Value, String> {
        if !(1..=4).contains(&branch_width) {
            return Err("branches must be between 1 and 4".to_string());
        }
        if !(1..=3).contains(&max_rounds) {
            return Err("rounds must be between 1 and 3".to_string());
        }
        self.request(
            "POST",
            "/v1/agent/run_stateful",
            Some(json!({
                "session_id": self.session,
                "message": require_text(question, "question")?,
                "branch_width": branch_width,
                "max_rounds": max_rounds,
            })),
        )
    }

    fn tool(&self, name: &str, arguments: Value) -> Result<Value, String> {
        self.request(
            "POST",
            "/v1/tools/call",
            Some(json!({
                "session_id": self.session,
                "name": name,
                "arguments": arguments,
            })),
        )
    }

    fn set_session(&mut self, session: &str) {
        self.session = normalize_session(session);
    }

    fn request(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
        let url = format!("{}{}", self.endpoint, path);
        let builder = match method {
            "GET" => self.http.get(&url),
            "POST" => self.http.post(&url),
            _ => return Err(format!("unsupported HTTP method: {method}")),
        };
        let response = if let Some(payload) = body {
            builder.json(&payload).send()
        } else {
            builder.send()
        }
        .map_err(|error| format!("request to {url} failed: {error}"))?;
        let status = response.status();
        let value: Value = response
            .json()
            .map_err(|error| format!("{url} returned invalid JSON: {error}"))?;
        if !status.is_success() {
            return Err(format!("HTTP {status}: {}", compact_json(&value)));
        }
        Ok(value)
    }
}

fn normalize_session(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        new_session_id()
    } else {
        value.to_string()
    }
}

fn resolve_session(value: Option<&str>) -> String {
    value.map_or_else(new_session_id, normalize_session)
}

fn new_session_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("cli-{nanos:x}-{:x}", std::process::id())
}

fn require_text<'a>(value: &'a str, label: &str) -> Result<&'a str, String> {
    let value = value.trim();
    if value.is_empty() {
        Err(format!("{label} must not be empty"))
    } else {
        Ok(value)
    }
}

fn joined(values: &[String], label: &str) -> Result<String, String> {
    let value = values.join(" ");
    require_text(&value, label)?;
    Ok(value)
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid JSON>".to_string())
}

fn evidence_count(value: &Value) -> usize {
    value
        .get("evidence")
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

fn status_of(value: &Value) -> &str {
    value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

fn render(value: &Value, raw_json: bool) {
    if raw_json {
        println!(
            "{}",
            serde_json::to_string_pretty(value).unwrap_or_else(|_| compact_json(value))
        );
        return;
    }

    if let Some(answer) = value.get("answer").and_then(Value::as_str) {
        if let Some(tool) = value.pointer("/route/tool").and_then(Value::as_str) {
            let state_research = value.pointer("/route/mode").and_then(Value::as_str)
                == Some("state_parallel_search");
            if state_research {
                let branches = value
                    .pointer("/route/branch_width")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let rounds = value
                    .pointer("/route/rounds")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                println!("╭─ Parallel state research");
                println!("│ {tool} · B{branches} · {rounds} rounds");
            } else {
                println!("╭─ Tool call");
                println!("│ {tool}");
            }
            if let Some(arguments) = value.pointer("/route/arguments") {
                println!("│ {}", compact_json(arguments));
            }
            if let Some(result) = value.get("tool_result") {
                println!(
                    "│ {} · {} evidence",
                    status_of(result),
                    evidence_count(result)
                );
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
                    println!(
                        "│ {completed} chunks · concurrency {concurrency} · {candidates} candidates"
                    );
                }
            }
            println!("╰─");
        }
        println!();
        println!("{answer}");
        return;
    }

    println!("status: {}", status_of(value));

    if let Some(tools) = value.get("tools").and_then(Value::as_array) {
        let names: Vec<&str> = tools.iter().filter_map(Value::as_str).collect();
        if !names.is_empty() {
            println!("tools: {}", names.join(", "));
        }
    }

    if let Some(memory_id) = value.get("memory_id").and_then(Value::as_str) {
        println!("memory: {memory_id}");
    }
    if let Some(message) = value.get("message").and_then(Value::as_str) {
        println!("{message}");
    }

    if let Some(evidence) = value.get("evidence").and_then(Value::as_array) {
        println!("evidence: {}", evidence.len());
        for item in evidence {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("?");
            let title = item
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("Untitled");
            println!("  [{id}] {title}");
            if let Some(content) = item.get("content").and_then(Value::as_str) {
                println!("      {}", shorten(content, 180));
            }
            if let Some(uri) = item.get("uri").and_then(Value::as_str) {
                println!("      {uri}");
            }
        }
    } else if let Some(tool_result) = value.get("tool_result") {
        println!("evidence: {}", evidence_count(tool_result));
    }
}

fn doctor_checks(health: &Value) -> Vec<(&'static str, bool, String)> {
    let controller_ready = status_of(health) == "ready";
    let models = health.get("model").and_then(Value::as_array);
    let model_ready = models.is_some_and(|items| {
        !items.is_empty()
            && items
                .iter()
                .all(|item| item.get("status").and_then(Value::as_str) == Some("ready"))
    });
    let tools = health
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let has_tool = |name: &str| tools.iter().any(|item| item.as_str() == Some(name));
    let required_tools = ["web_search", "knowledge_search", "long_text_qa"];
    let tools_ready = required_tools.iter().all(|name| has_tool(name));
    let state_ready = health
        .pointer("/state_parallel_search/enabled")
        .and_then(Value::as_bool)
        == Some(true);
    vec![
        (
            "controller",
            controller_ready,
            format!("status={}", status_of(health)),
        ),
        (
            "model_sidecar",
            model_ready,
            format!("ready={}", models.map_or(0, Vec::len)),
        ),
        ("required_tools", tools_ready, required_tools.join(",")),
        (
            "state_parallel_search",
            state_ready,
            if state_ready { "enabled" } else { "disabled" }.to_string(),
        ),
    ]
}

fn doctor(client: &AgentClient, raw_json: bool) -> Result<(), String> {
    let health = client.health()?;
    let checks = doctor_checks(&health);
    let ready = checks.iter().all(|(_, passed, _)| *passed);
    if raw_json {
        let payload = json!({
            "status": if ready { "ready" } else { "failed" },
            "endpoint": client.endpoint,
            "checks": checks
                .iter()
                .map(|(name, passed, detail)| json!({
                    "name": name,
                    "passed": passed,
                    "detail": detail,
                }))
                .collect::<Vec<_>>(),
            "health": health,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).unwrap_or_else(|_| compact_json(&payload))
        );
    } else {
        println!("RWKV Agent doctor");
        println!("endpoint: {}", client.endpoint);
        for (name, passed, detail) in &checks {
            println!("[{}] {name}: {detail}", if *passed { "ok" } else { "fail" });
        }
    }
    if ready {
        Ok(())
    } else {
        Err("doctor found an incomplete Agent deployment".to_string())
    }
}

fn shorten(value: &str, limit: usize) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= limit {
        return normalized;
    }
    normalized
        .chars()
        .take(limit.saturating_sub(1))
        .collect::<String>()
        + "…"
}

fn with_spinner<F>(label: &str, operation: F) -> Result<Value, String>
where
    F: FnOnce() -> Result<Value, String>,
{
    if !io::stderr().is_terminal() {
        return operation();
    }
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message(label.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));
    let result = operation();
    spinner.finish_and_clear();
    result
}

fn execute(cli: &Cli, client: &mut AgentClient) -> Result<(), String> {
    match cli.command.as_ref() {
        Some(Command::Doctor) => doctor(client, cli.json)?,
        Some(Command::Health) => render(&client.health()?, cli.json),
        Some(Command::Ask { message }) => render(
            &with_spinner("Thinking…", || client.ask(&joined(message, "message")?))?,
            cli.json,
        ),
        Some(Command::Research {
            question,
            branches,
            rounds,
        }) => render(
            &with_spinner("Researching with parallel states…", || {
                client.research(&joined(question, "question")?, *branches, *rounds)
            })?,
            cli.json,
        ),
        Some(Command::Chat) | None => chat(client, cli.json)?,
        Some(Command::Tool { command }) => {
            let value = match command {
                ToolCommand::WebSearch { query } => with_spinner("Searching the web…", || {
                    client.tool("web_search", json!({"query": joined(query, "query")?}))
                })?,
                ToolCommand::KnowledgeSearch { query } => {
                    with_spinner("Searching knowledge…", || {
                        client.tool(
                            "knowledge_search",
                            json!({"query": joined(query, "query")?}),
                        )
                    })?
                }
                ToolCommand::LongTextQa { question } => {
                    with_spinner("Analyzing document chunks…", || {
                        client.tool(
                            "long_text_qa",
                            json!({
                                "question": joined(question, "question")?,
                            }),
                        )
                    })?
                }
            };
            render(&value, cli.json);
        }
    }
    Ok(())
}

fn banner(client: &AgentClient) {
    println!("╭──────────────────────────────────────────────╮");
    println!("│  RWKV Agent                                  │");
    println!("│  tools + parallel research + knowledge       │");
    println!("╰──────────────────────────────────────────────╯");
    println!("  session: {}", client.session);
    println!("  endpoint: {}", client.endpoint);
    println!("  /help for commands · Ctrl-D to exit");
    println!();
}

fn print_help() {
    println!(
        "\
Commands
  /help                 Show this help
  /status               Check backend and model health
  /tools                List available function calls
  /session [name]       Show or switch conversation session
  /knowledge <query>    Search local indexed knowledge
  /web <query>          Search the live web
  /research <question>  Search with four forked states for two rounds
  /longtext <question>  Ask the long text pasted earlier in this session
  /json on|off          Toggle complete JSON responses
  /clear                Clear the terminal
  /exit, /quit          Leave the Agent

Input ending in \\ continues on the next line."
    );
}

fn show_tools() {
    println!("Available function calls:");
    println!("  web_search(query)         live discovery + extraction");
    println!("  knowledge_search(query)   local indexed knowledge");
    println!("  long_text_qa(question)    parallel QA over pasted session text");
    println!("  /research <question>      B4 × 2-round state-native Web research");
    println!();
    println!("Paste a long text directly, wait for acceptance, then ask questions.");
}

fn slash_argument<'a>(line: &'a str, command: &str) -> Result<&'a str, String> {
    require_text(line[command.len()..].trim(), "argument")
}

fn handle_slash(line: &str, client: &mut AgentClient, raw_json: &mut bool) -> Result<bool, String> {
    let (command, _) = line.split_once(' ').unwrap_or((line, ""));
    match command {
        "/exit" | "/quit" => return Ok(false),
        "/help" => print_help(),
        "/tools" => show_tools(),
        "/clear" => {
            print!("\x1b[2J\x1b[H");
            io::stdout()
                .flush()
                .map_err(|error| format!("stdout error: {error}"))?;
        }
        "/status" => render(
            &with_spinner("Checking Agent…", || client.health())?,
            *raw_json,
        ),
        "/session" => {
            let requested = line[command.len()..].trim();
            if requested.is_empty() {
                println!("session: {}", client.session);
            } else {
                client.set_session(requested);
                println!("session switched: {}", client.session);
            }
        }
        "/json" => match line[command.len()..].trim() {
            "on" => {
                *raw_json = true;
                println!("JSON output: on");
            }
            "off" => {
                *raw_json = false;
                println!("JSON output: off");
            }
            "" => println!("JSON output: {}", if *raw_json { "on" } else { "off" }),
            _ => return Err("usage: /json on|off".to_string()),
        },
        "/knowledge" => {
            let query = slash_argument(line, command)?;
            let value = with_spinner("Searching knowledge…", || {
                client.tool("knowledge_search", json!({"query": query}))
            })?;
            render(&value, *raw_json);
        }
        "/web" => {
            let query = slash_argument(line, command)?;
            let value = with_spinner("Searching the web…", || {
                client.tool("web_search", json!({"query": query}))
            })?;
            render(&value, *raw_json);
        }
        "/research" => {
            let question = slash_argument(line, command)?;
            let value = with_spinner("Researching with parallel states…", || {
                client.research(question, 4, 2)
            })?;
            render(&value, *raw_json);
        }
        "/longtext" => {
            let question = slash_argument(line, command)?;
            let value = with_spinner("Analyzing document chunks…", || {
                client.tool(
                    "long_text_qa",
                    json!({
                        "question": question,
                    }),
                )
            })?;
            render(&value, *raw_json);
        }
        _ => return Err(format!("unknown command: {command} (try /help)")),
    }
    Ok(true)
}

fn process_chat_line(
    message: &str,
    client: &mut AgentClient,
    raw_json: &mut bool,
) -> Result<bool, String> {
    let message = message.trim();
    if message.is_empty() {
        return Ok(true);
    }
    if message.starts_with('/') {
        return handle_slash(message, client, raw_json);
    }
    let value = with_spinner("Thinking…", || client.ask(message))?;
    render(&value, *raw_json);
    Ok(true)
}

fn history_path() -> Option<PathBuf> {
    dirs::home_dir().map(|path| path.join(".rwkv-agent").join("history.txt"))
}

fn should_store_history(message: &str) -> bool {
    message.chars().count() < PASTED_TEXT_CAPTURE_CHARS
}

fn chat_tty(client: &mut AgentClient, raw_json: &mut bool) -> Result<(), String> {
    let mut editor =
        DefaultEditor::new().map_err(|error| format!("cannot start line editor: {error}"))?;
    let history = history_path();
    if let Some(path) = history.as_ref() {
        let _ = editor.load_history(path);
    }

    loop {
        let mut message = match editor.readline("❯ ") {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(error) => return Err(format!("input error: {error}")),
        };

        while message.trim_end().ends_with('\\') {
            message.truncate(message.trim_end().len().saturating_sub(1));
            match editor.readline("… ") {
                Ok(next) => {
                    message.push('\n');
                    message.push_str(&next);
                }
                Err(ReadlineError::Interrupted) => {
                    message.clear();
                    println!("^C");
                    break;
                }
                Err(ReadlineError::Eof) => break,
                Err(error) => return Err(format!("input error: {error}")),
            }
        }

        if message.trim().is_empty() {
            continue;
        }
        // Long pasted source text is transient Agent input, not CLI history.
        if should_store_history(&message) {
            let _ = editor.add_history_entry(message.as_str());
        }
        match process_chat_line(&message, client, raw_json) {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => eprintln!("error: {error}"),
        }
    }

    if let Some(path) = history {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create history directory: {error}"))?;
        }
        editor
            .save_history(&path)
            .map_err(|error| format!("cannot save history: {error}"))?;
    }
    Ok(())
}

fn chat_piped(client: &mut AgentClient, raw_json: &mut bool) -> Result<(), String> {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.map_err(|error| format!("stdin error: {error}"))?;
        match process_chat_line(&line, client, raw_json) {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => eprintln!("error: {error}"),
        }
    }
    Ok(())
}

fn chat(client: &mut AgentClient, initial_raw_json: bool) -> Result<(), String> {
    let mut raw_json = initial_raw_json;
    if io::stdin().is_terminal() {
        banner(client);
        chat_tty(client, &mut raw_json)
    } else {
        chat_piped(client, &mut raw_json)
    }
}

fn main() {
    let cli = Cli::parse();
    let result = AgentClient::new(&cli.endpoint, cli.session.as_deref(), cli.timeout)
        .and_then(|mut client| execute(&cli, &mut client));
    if let Err(error) = result {
        eprintln!("rwkv-agent: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn defaults_to_interactive_chat() {
        let cli = Cli::try_parse_from(["rwkv-agent"]).unwrap();
        assert!(cli.command.is_none());
    }

    #[test]
    fn parses_doctor_command() {
        let cli = Cli::try_parse_from(["rwkv-agent", "doctor"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Doctor)));
    }

    #[test]
    fn doctor_requires_controller_model_tools_and_state_search() {
        let ready = json!({
            "status": "ready",
            "tools": ["web_search", "knowledge_search", "long_text_qa", "memory"],
            "model": [{"status": "ready"}],
            "state_parallel_search": {"enabled": true},
        });
        assert!(doctor_checks(&ready).iter().all(|(_, passed, _)| *passed));

        let missing_model = json!({
            "status": "ready",
            "tools": ["web_search", "knowledge_search", "long_text_qa"],
            "model": [],
            "state_parallel_search": {"enabled": true},
        });
        assert!(
            doctor_checks(&missing_model)
                .iter()
                .any(|(name, passed, _)| *name == "model_sidecar" && !passed)
        );
    }

    #[test]
    fn parses_direct_web_tool() {
        let cli = Cli::try_parse_from([
            "rwkv-agent",
            "--session",
            "demo",
            "tool",
            "web-search",
            "Python",
            "latest",
        ])
        .unwrap();
        assert_eq!(cli.session.as_deref(), Some("demo"));
        match cli.command {
            Some(Command::Tool {
                command: ToolCommand::WebSearch { query },
            }) => assert_eq!(query, vec!["Python", "latest"]),
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn parses_question_only_long_text_tool() {
        let cli = Cli::try_parse_from([
            "rwkv-agent",
            "--session",
            "demo",
            "tool",
            "long-text-qa",
            "Who",
            "founded",
            "it?",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Tool {
                command: ToolCommand::LongTextQa { question },
            }) => assert_eq!(question, vec!["Who", "founded", "it?"]),
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn parses_stateful_research_with_bounded_controls() {
        let cli = Cli::try_parse_from([
            "rwkv-agent",
            "research",
            "--branches",
            "3",
            "--rounds",
            "2",
            "Who",
            "created",
            "RWKV?",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Research {
                question,
                branches,
                rounds,
            }) => {
                assert_eq!(question, vec!["Who", "created", "RWKV?"]);
                assert_eq!(branches, 3);
                assert_eq!(rounds, 2);
            }
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn joins_and_validates_text() {
        assert_eq!(
            joined(&["hello".into(), "world".into()], "message").unwrap(),
            "hello world"
        );
        assert!(joined(&[" ".into()], "message").is_err());
    }

    #[test]
    fn shortens_utf8_safely() {
        assert_eq!(shorten("这是一个很长的中文句子", 6), "这是一个很…");
        assert_eq!(shorten("short text", 20), "short text");
    }

    #[test]
    fn slash_argument_requires_content() {
        assert_eq!(slash_argument("/web hello", "/web").unwrap(), "hello");
        assert!(slash_argument("/web", "/web").is_err());
    }

    #[test]
    fn default_session_is_fresh_and_named_session_is_stable() {
        let first = resolve_session(None);
        std::thread::sleep(Duration::from_millis(1));
        let second = resolve_session(None);
        assert!(first.starts_with("cli-"));
        assert_ne!(first, second);
        assert_eq!(resolve_session(Some("demo")), "demo");
    }

    #[test]
    fn long_pasted_text_is_not_written_to_cli_history() {
        assert!(should_store_history("normal question"));
        assert!(!should_store_history(
            &"文".repeat(PASTED_TEXT_CAPTURE_CHARS)
        ));
    }
}
