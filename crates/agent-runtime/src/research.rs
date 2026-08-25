use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use rwkv_agent_core::{Action, CancellationToken, RunContext, TOOL_CALL_JSON_PREFIX, parse_action};
use serde_json::{Value, json};
use tokio::task::JoinSet;

use crate::data_client::DataPlaneClient;
use crate::debug_trace::DebugTraceHandle;
use crate::prompt::{self, MISSIONS};
use crate::sidecar::{BatchContinuation, SidecarClient, SidecarState};

#[derive(Clone)]
pub struct ResearchRunner {
    sidecar: SidecarClient,
    data: DataPlaneClient,
}

struct OpenedResearch<'a> {
    question: &'a str,
    session_id: &'a str,
    root: &'a SidecarState,
    branches: &'a [SidecarState],
    max_rounds: usize,
    started: Instant,
    context: &'a RunContext,
    debug_trace: Option<&'a DebugTraceHandle>,
}

impl ResearchRunner {
    pub fn new(sidecar: SidecarClient, data: DataPlaneClient) -> Self {
        Self { sidecar, data }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        question: &str,
        session_id: &str,
        branch_width: usize,
        max_rounds: usize,
        cancellation: CancellationToken,
        max_elapsed: Duration,
        debug_trace: Option<&DebugTraceHandle>,
    ) -> Result<Value, String> {
        if !(1..=4).contains(&branch_width) {
            return Err("branch_width must be between 1 and 4".into());
        }
        if !(1..=3).contains(&max_rounds) {
            return Err("max_rounds must be between 1 and 3".into());
        }
        if max_elapsed.is_zero() {
            return Err("max_elapsed must be positive".into());
        }
        let started = Instant::now();
        let context = RunContext {
            deadline: started.checked_add(max_elapsed).unwrap_or(started),
            cancellation,
        };
        context.check().map_err(|error| error.to_string())?;
        let owner_id = unique_owner("research", session_id);
        let root_prompt = prompt::research_root(question);
        trace_record(
            debug_trace,
            "model",
            "provider",
            "research_prefill_requested",
            || json!({"owner_id":owner_id,"prompt":root_prompt}),
        )
        .await;
        let root = tokio::time::timeout(
            context.remaining(),
            self.sidecar.prefill(&owner_id, &root_prompt),
        )
        .await
        .map_err(|_| "run deadline exceeded during research State prefill".to_string())??;
        trace_record(
            debug_trace,
            "state",
            "provider",
            "state_opened",
            || json!({"state":root}),
        )
        .await;
        if let Err(error) = context.check() {
            let _ = self
                .sidecar
                .release_many(
                    &root.home_url,
                    &root.owner_id,
                    std::slice::from_ref(&root.state_id),
                )
                .await;
            return Err(error.to_string());
        }
        let branch_names = (1..=branch_width)
            .map(|index| format!("branch-{index}"))
            .collect::<Vec<_>>();
        trace_record(
            debug_trace,
            "state",
            "provider",
            "fork_requested",
            || json!({"root_state_id":root.state_id,"branches":branch_names}),
        )
        .await;
        let branch_result =
            tokio::time::timeout(context.remaining(), self.sidecar.fork(&root, &branch_names))
                .await
                .unwrap_or_else(|_| {
                    Err("run deadline exceeded during research State fork".to_string())
                });
        let branches = match branch_result {
            Ok(branches) => branches,
            Err(error) => {
                let _ = self
                    .sidecar
                    .release_many(
                        &root.home_url,
                        &root.owner_id,
                        std::slice::from_ref(&root.state_id),
                    )
                    .await;
                return Err(error);
            }
        };
        trace_record(
            debug_trace,
            "state",
            "provider",
            "fork_completed",
            || json!({"root_state_id":root.state_id,"states":branches}),
        )
        .await;
        if branches.len() != branch_width {
            let state_ids = std::iter::once(root.state_id.clone())
                .chain(branches.iter().map(|state| state.state_id.clone()))
                .collect::<Vec<_>>();
            let _ = self
                .sidecar
                .release_many(&root.home_url, &root.owner_id, &state_ids)
                .await;
            return Err(format!(
                "fork returned {} branches; expected {branch_width}",
                branches.len()
            ));
        }
        let state_ids = std::iter::once(root.state_id.clone())
            .chain(branches.iter().map(|state| state.state_id.clone()))
            .collect::<Vec<_>>();

        let run = tokio::time::timeout(
            context.remaining(),
            self.run_opened(OpenedResearch {
                question,
                session_id,
                root: &root,
                branches: &branches,
                max_rounds,
                started,
                context: &context,
                debug_trace,
            }),
        )
        .await
        .unwrap_or_else(|_| Err("run deadline exceeded during research".into()));
        let release = self
            .sidecar
            .release_many(&root.home_url, &root.owner_id, &state_ids)
            .await;
        trace_record(
            debug_trace,
            "state",
            "provider",
            "state_release_completed",
            || match &release {
                Ok(value) => json!({"success":true,"state_ids":state_ids,"response":value}),
                Err(error) => json!({"success":false,"state_ids":state_ids,"error":error}),
            },
        )
        .await;
        match (run, release) {
            (Ok(mut response), Ok(trace)) => {
                response["trace"]["state_runtime"]["release"] = trace;
                Ok(response)
            }
            (Ok(_), Err(error)) => Err(format!(
                "research completed but State release failed: {error}"
            )),
            (Err(error), Ok(_)) => Err(error),
            (Err(run), Err(release)) => Err(format!("{run}; State release also failed: {release}")),
        }
    }

    async fn run_opened(&self, run: OpenedResearch<'_>) -> Result<Value, String> {
        let OpenedResearch {
            question,
            session_id,
            root,
            branches,
            max_rounds,
            started,
            context,
            debug_trace,
        } = run;
        let mut observations = std::collections::HashMap::<String, Value>::new();
        let mut used_queries = BTreeSet::<String>::new();
        let mut tool_results = Vec::<Value>::new();
        let mut round_traces = Vec::<Value>::new();

        for round_index in 1..=max_rounds {
            context.check().map_err(|error| error.to_string())?;
            let items = branches
                .iter()
                .enumerate()
                .map(|(index, state)| {
                    json!({
                        "state_id": state.state_id,
                        "input": prompt::branch_step(
                            question,
                            MISSIONS[index],
                            round_index,
                            observations.get(&state.state_id),
                        ),
                    })
                })
                .collect();
            trace_record(
                debug_trace,
                "model",
                "provider",
                "research_batch_continue_requested",
                || json!({"round":round_index,"owner_id":root.owner_id,"items":items,"stops":["</tool_call>"],"max_tokens":96}),
            )
            .await;
            let rows = self
                .sidecar
                .batch_continue(
                    &root.home_url,
                    &root.owner_id,
                    items,
                    &["</tool_call>".into()],
                    96,
                )
                .await?;
            trace_record(
                debug_trace,
                "model",
                "provider",
                "research_batch_continue_completed",
                || json!({"round":round_index,"rows":rows}),
            )
            .await;
            context.check().map_err(|error| error.to_string())?;
            if rows.len() != branches.len() {
                return Err(format!(
                    "branch continuation row count changed: expected {}, got {}",
                    branches.len(),
                    rows.len()
                ));
            }
            if rows
                .iter()
                .zip(branches)
                .any(|(row, branch)| row.state_id != branch.state_id)
            {
                return Err("branch continuation changed state identity or order".into());
            }

            let mut planned = Vec::new();
            for (index, row) in rows.iter().enumerate() {
                context.check().map_err(|error| error.to_string())?;
                let generated = reconstruct_tool(row);
                let parsed = parse_action(&generated).map_err(|e| e.to_string());
                let model_query = match parsed {
                    Ok(Action::Tool(call)) if call.name == "web_search" => call
                        .arguments
                        .get("query")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    _ => String::new(),
                };
                let used = used_queries.iter().cloned().collect::<Vec<_>>();
                let view = self
                    .data
                    .coordinate_query(
                        question,
                        &model_query,
                        index,
                        round_index,
                        observations.get(&row.state_id),
                        &used,
                    )
                    .await?;
                let accepted = view
                    .get("accepted")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let query = view
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if accepted && !query.is_empty() {
                    used_queries.insert(query.to_lowercase());
                }
                planned.push((row.clone(), generated, view, accepted, query));
            }

            let mut searches = JoinSet::new();
            for (index, (_, _, _, accepted, query)) in planned.iter().enumerate() {
                if *accepted {
                    let data = self.data.clone();
                    let query = query.clone();
                    let session_id = session_id.to_string();
                    let question = question.to_string();
                    searches.spawn(async move {
                        let result = data
                            .call_tool(
                                "web_search",
                                json!({"query": query}),
                                &session_id,
                                &question,
                            )
                            .await;
                        (index, result)
                    });
                }
            }
            let mut results = vec![json!({"status":"invalid","evidence":[]}); planned.len()];
            while let Some(joined) = searches.join_next().await {
                context.check().map_err(|error| error.to_string())?;
                let (index, value) = joined.map_err(|e| e.to_string())?;
                results[index] = value?;
            }
            trace_record(
                debug_trace,
                "tools",
                "research",
                "research_tools_completed",
                || json!({"round":round_index,"results":results}),
            )
            .await;
            context.check().map_err(|error| error.to_string())?;

            let mut branch_traces = Vec::new();
            for ((row, raw, view, accepted, query), result) in planned.into_iter().zip(results) {
                let observation = compact_observation(&result);
                observations.insert(row.state_id.clone(), observation);
                if result.get("status").and_then(Value::as_str) == Some("ok") {
                    tool_results.push(result.clone());
                }
                branch_traces.push(json!({
                    "state_id": row.state_id,
                    "branch": row.branch,
                    "raw": raw,
                    "route": {"strict": accepted, "tool": "web_search", "arguments": {"query": query}, "query_view": view},
                    "tool_status": result.get("status"),
                    "evidence_count": result.get("evidence").and_then(Value::as_array).map_or(0, Vec::len),
                    "seen_tokens": row.seen_tokens,
                }));
            }
            round_traces.push(json!({"round": round_index, "branches": branch_traces}));
        }

        let evidence = self
            .data
            .reduce_evidence(question, &tool_results, 8)
            .await?;
        context.check().map_err(|error| error.to_string())?;
        let (status, answer, answer_protocol, answer_completion) = if evidence.is_empty() {
            ("ok", no_evidence(question), Value::Null, Value::Null)
        } else {
            let final_prompt = prompt::research_final(question, &evidence);
            trace_record(
                debug_trace,
                "model",
                "provider",
                "research_final_continue_requested",
                || json!({"state_id":root.state_id,"input":final_prompt,"max_tokens":192}),
            )
            .await;
            let row = self
                .sidecar
                .continue_one(
                    root,
                    &final_prompt,
                    &[
                        "</answer>".into(),
                        "<tool_call>".into(),
                        "<tool_result>".into(),
                        "\n\nUser:".into(),
                        "\nSystem:".into(),
                    ],
                    192,
                )
                .await?;
            trace_record(
                debug_trace,
                "model",
                "provider",
                "research_final_continue_completed",
                || json!({"row":row}),
            )
            .await;
            context.check().map_err(|error| error.to_string())?;
            if row.state_id != root.state_id {
                return Err("research answer continuation changed root state identity".into());
            }
            let raw_answer = row
                .text
                .trim()
                .trim_start_matches("<answer>")
                .trim_end_matches("</answer>")
                .trim();
            let validation = self
                .data
                .validate_answer(question, raw_answer, &evidence)
                .await?;
            context.check().map_err(|error| error.to_string())?;
            if validation
                .get("valid")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                (
                    "ok",
                    validation
                        .get("answer")
                        .and_then(Value::as_str)
                        .unwrap_or(raw_answer)
                        .to_string(),
                    validation,
                    serde_json::to_value(row).map_err(|e| e.to_string())?,
                )
            } else {
                (
                    "insufficient_evidence",
                    insufficient(question),
                    validation,
                    serde_json::to_value(row).map_err(|e| e.to_string())?,
                )
            }
        };

        Ok(json!({
            "status": status,
            "session_id": session_id,
            "message": question,
            "route": {"mode":"state_parallel_search","tool":"web_search","branch_width":branches.len(),"rounds":max_rounds},
            "tool_result": {"status": if evidence.is_empty(){"empty"}else{"ok"},"tool":"web_search","evidence":evidence},
            "answer": answer,
            "trace": {
                "state_runtime": {"home_url":root.home_url,"root_prefill_once":true,"forked_states":branches.len(),"tensor_state_merge":false,"semantic_reduce_to_root":true},
                "rounds":round_traces,
                "answer_completion":answer_completion,
                "answer_protocol":answer_protocol,
                "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
                "control_plane":"rust",
            }
        }))
    }
}

async fn trace_record<F>(
    trace: Option<&DebugTraceHandle>,
    category: &'static str,
    component: &str,
    event_type: &str,
    payload: F,
) where
    F: FnOnce() -> Value,
{
    if let Some(trace) = trace {
        let _ = trace
            .record(category, component, event_type, None, payload())
            .await;
    }
}

fn reconstruct_tool(row: &BatchContinuation) -> String {
    let raw = row.text.trim_start();
    let opening = if raw.starts_with("<tool_call>") {
        ""
    } else if raw.starts_with("{\"name\":\"") {
        "<tool_call>"
    } else {
        TOOL_CALL_JSON_PREFIX
    };
    let closing = if raw.ends_with("</tool_call>") {
        ""
    } else {
        "</tool_call>"
    };
    format!("{opening}{raw}{closing}")
}

fn compact_observation(result: &Value) -> Value {
    let evidence = result
        .get("evidence")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .take(4)
                .map(|row| {
                    let mut value = row.clone();
                    if let Some(content) = value.get("content").and_then(Value::as_str) {
                        let short = content.chars().take(600).collect::<String>();
                        value["content"] = Value::String(short);
                    }
                    value
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({"status":result.get("status").cloned().unwrap_or_else(|| json!("empty")),"evidence":evidence})
}

fn unique_owner(prefix: &str, session_id: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let tick = NEXT.fetch_add(1, Ordering::Relaxed);
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{prefix}-{}-{epoch:x}-{tick:x}",
        session_id.chars().take(32).collect::<String>()
    )
}

fn contains_chinese(value: &str) -> bool {
    value
        .chars()
        .any(|character| ('\u{3400}'..='\u{9fff}').contains(&character))
}

fn no_evidence(question: &str) -> String {
    if contains_chinese(question) {
        "没有找到可用证据。".into()
    } else {
        "No usable evidence was found.".into()
    }
}

fn insufficient(question: &str) -> String {
    if contains_chinese(question) {
        "现有证据不足以可靠回答。".into()
    } else {
        "The available evidence is insufficient for a reliable answer.".into()
    }
}
