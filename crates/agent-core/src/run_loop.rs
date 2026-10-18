use std::future::Future;
use std::time::Instant;

use serde_json::{Value, json};

use crate::protocol::normalize_tool_result;
use crate::{
    Action, ActionKind, AgentError, AgentEvent, AgentRunRequest, AnswerDecision, CancellationToken,
    EventSink, ModelOutput, OpenStateRequest, RunContext, RunLimits, RunReport,
    StateContinueRequest, StateHandle, TOOL_CALL_JSON_PREFIX, ToolCall, ToolRegistry, ToolStep,
    parse_action, render_answer_observation, render_answer_observation_with_reminder,
    render_observation_with_progress_and_reminder,
    render_tool_observation_with_progress_reminder_and_prefix,
};

pub trait StateModel {
    fn open(
        &mut self,
        request: OpenStateRequest,
        context: RunContext,
    ) -> impl Future<Output = Result<StateHandle, String>> + Send;

    fn continue_state(
        &mut self,
        request: StateContinueRequest,
        context: RunContext,
    ) -> impl Future<Output = Result<ModelOutput, String>> + Send;

    fn fork_state(
        &mut self,
        _root: StateHandle,
        _context: RunContext,
    ) -> impl Future<Output = Result<StateHandle, String>> + Send {
        async { Err("model does not support state fork".into()) }
    }

    fn release(&mut self, state: StateHandle) -> impl Future<Output = Result<(), String>> + Send;
}

pub trait ToolExecutor {
    fn execute(
        &mut self,
        call: ToolCall,
        context: RunContext,
    ) -> impl Future<Output = Result<Value, String>> + Send;

    fn validate_answer(
        &mut self,
        _answer: &str,
        _context: RunContext,
    ) -> impl Future<Output = Result<AnswerDecision, String>> + Send {
        async { Ok(AnswerDecision::Accept) }
    }

    fn continuation_feedback(&self) -> String {
        String::new()
    }

    fn answer_retry_root_prompt(&self, _feedback: &str) -> Option<String> {
        None
    }

    fn require_tool_after_observation(&self) -> bool {
        false
    }

    fn commit_answer_after_observation(&self) -> bool {
        false
    }

    /// Return an already-grounded final answer that can be committed without
    /// another model decode. Executors should only provide this after their
    /// own state machine has recorded a successful final verification.
    fn committed_answer_after_observation(&self) -> Option<String> {
        None
    }

    fn recovery_root_prompt_after_observation(&self) -> Option<String> {
        None
    }

    /// A root-backed worker normally retains its recurrent trajectory. Return
    /// true only after the executor has detected a real no-progress loop and
    /// can restate all required evidence in the recovery prompt.
    fn refresh_worker_after_observation(&self) -> bool {
        false
    }

    /// Protocol repair is normally appended to the current recurrent state.
    /// A root-backed executor may request a fresh worker when a malformed
    /// envelope is likely to deterministically reproduce from stale decode
    /// history. The pristine root and all controller-grounded recovery
    /// evidence are retained.
    fn refresh_worker_after_protocol_rejection(&self) -> bool {
        false
    }

    /// Structural prefix for a controller-required tool turn. Executors may
    /// commit a more specific tool name at a deterministic phase boundary;
    /// arguments and the completed envelope remain strictly validated.
    fn tool_call_prefix(&self) -> String {
        TOOL_CALL_JSON_PREFIX.to_string()
    }

    /// Return a deterministic control-plane action that must run before the
    /// model receives another turn. The primary use is executing a declared
    /// TaskSpec verifier immediately after a real mutation. The call still
    /// consumes the normal tool budget, passes the strict registry, is fully
    /// traced, and receives the same Sandbox policy as model-authored calls.
    fn controller_tool_after_observation(&mut self) -> Option<ToolCall> {
        None
    }

    /// Convert a strictly framed answer payload into a controller-authored
    /// tool call. This is useful when a small model can emit complete file
    /// text reliably but cannot JSON-escape that text inside a tool argument.
    /// The returned call still consumes the ordinary tool budget, passes the
    /// registry, executes in the configured sandbox, and is fully traced.
    fn controller_tool_from_answer(&mut self, _answer: &str) -> Option<ToolCall> {
        None
    }
}

pub struct AgentLoop<'a, M, T, S> {
    model: &'a mut M,
    tools: &'a mut T,
    registry: &'a ToolRegistry,
    events: &'a mut S,
    limits: RunLimits,
    cancellation: CancellationToken,
}

impl<'a, M, T, S> AgentLoop<'a, M, T, S>
where
    M: StateModel + Send,
    T: ToolExecutor + Send,
    S: EventSink,
{
    pub fn new(
        model: &'a mut M,
        tools: &'a mut T,
        registry: &'a ToolRegistry,
        events: &'a mut S,
        limits: RunLimits,
        cancellation: CancellationToken,
    ) -> Result<Self, AgentError> {
        limits.validate()?;
        Ok(Self {
            model,
            tools,
            registry,
            events,
            limits,
            cancellation,
        })
    }

    pub async fn run(&mut self, request: AgentRunRequest) -> Result<RunReport, AgentError> {
        request.validate()?;
        self.events.emit(AgentEvent::RunStarted {
            owner_id: request.owner_id.clone(),
        });
        let deadline = Instant::now()
            .checked_add(self.limits.max_elapsed)
            .unwrap_or_else(Instant::now);
        let context = RunContext {
            deadline,
            cancellation: self.cancellation.clone(),
        };
        if let Err(error) = context.check() {
            self.emit_failure(&error);
            return Err(error);
        }
        let opened = match self
            .model
            .open(
                OpenStateRequest {
                    owner_id: request.owner_id.clone(),
                    root_prompt: request.root_prompt.clone(),
                },
                context.clone(),
            )
            .await
        {
            Ok(state) => state,
            Err(message) => {
                let error = AgentError::Model(message);
                self.emit_failure(&error);
                return Err(error);
            }
        };
        self.events.emit(AgentEvent::StateOpened {
            state_id: opened.state_id.clone(),
        });
        if opened.state_id.trim().is_empty() {
            let error = AgentError::Model("model returned an empty state_id".into());
            self.emit_failure(&error);
            return Err(error);
        }
        if opened.owner_id != request.owner_id {
            let error = AgentError::OwnerChanged {
                expected: request.owner_id.clone(),
                actual: opened.owner_id.clone(),
            };
            self.emit_failure(&error);
            let release = self.release_state(opened).await;
            return match release {
                Ok(()) => Err(error),
                Err(release) => Err(AgentError::RunAndRelease {
                    run: Box::new(error),
                    release,
                }),
            };
        }

        let (mut state, root_state) = if self.limits.fork_from_root {
            match self.model.fork_state(opened.clone(), context.clone()).await {
                Ok(worker) if worker.state_id.trim().is_empty() => {
                    let error = AgentError::Model("model returned an empty forked state_id".into());
                    self.emit_failure(&error);
                    let release = self.release_state(opened).await;
                    return match release {
                        Ok(()) => Err(error),
                        Err(release) => Err(AgentError::RunAndRelease {
                            run: Box::new(error),
                            release,
                        }),
                    };
                }
                Ok(worker) if worker.owner_id != request.owner_id => {
                    let error = AgentError::OwnerChanged {
                        expected: request.owner_id.clone(),
                        actual: worker.owner_id.clone(),
                    };
                    self.emit_failure(&error);
                    let worker_release = self.release_state(worker).await.err();
                    let root_release = self.release_state(opened).await.err();
                    return match [worker_release, root_release]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                    {
                        failures if failures.is_empty() => Err(error),
                        failures => Err(AgentError::RunAndRelease {
                            run: Box::new(error),
                            release: failures.join("; "),
                        }),
                    };
                }
                Ok(worker) => {
                    self.events.emit(AgentEvent::StateOpened {
                        state_id: worker.state_id.clone(),
                    });
                    (worker, Some(opened))
                }
                Err(message) => {
                    let error = AgentError::Model(message);
                    self.emit_failure(&error);
                    let release = self.release_state(opened).await;
                    return match release {
                        Ok(()) => Err(error),
                        Err(release) => Err(AgentError::RunAndRelease {
                            run: Box::new(error),
                            release,
                        }),
                    };
                }
            }
        } else {
            (opened, None)
        };

        let run_result = if state.state_id.trim().is_empty() {
            Err(AgentError::Model("model returned an empty state_id".into()))
        } else if state.owner_id != request.owner_id {
            Err(AgentError::OwnerChanged {
                expected: request.owner_id.clone(),
                actual: state.owner_id.clone(),
            })
        } else {
            self.run_opened(
                &mut state,
                request.initial_input,
                &request.owner_id,
                root_state.as_ref(),
                context,
            )
            .await
        };
        if let Err(error) = &run_result {
            self.emit_failure(error);
        }

        let mut release_errors = Vec::new();
        if let Err(error) = self.release_state(state).await {
            release_errors.push(error);
        }
        if let Some(root) = root_state
            && let Err(error) = self.release_state(root).await
        {
            release_errors.push(error);
        }
        let release = if release_errors.is_empty() {
            Ok(())
        } else {
            Err(release_errors.join("; "))
        };

        match (run_result, release) {
            (Ok(report), Ok(())) => Ok(report),
            (Ok(_), Err(error)) => {
                let release_error = AgentError::Release(error);
                self.events.emit(AgentEvent::RunFailed {
                    code: release_error.code().to_string(),
                    message: release_error.to_string(),
                });
                Err(release_error)
            }
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(release)) => Err(AgentError::RunAndRelease {
                run: Box::new(error),
                release,
            }),
        }
    }

    async fn run_opened(
        &mut self,
        state: &mut StateHandle,
        mut input: String,
        owner_id: &str,
        root_state: Option<&StateHandle>,
        context: RunContext,
    ) -> Result<RunReport, AgentError> {
        let mut model_turns = 0;
        let mut tool_steps = 0;
        let mut protocol_retries = 0;
        let mut answer_retries = 0;
        let mut tools = Vec::new();
        let mut last_successful_call: Option<ToolCall> = None;
        let mut last_unsuccessful_call: Option<ToolCall> = None;
        let mut scheduled_tool: Option<ToolCall> = None;
        let mut repeated_rejections = 0usize;
        loop {
            context.check()?;
            let (action, controller_authored, provider_input, raw_output, stop_reason, max_tokens) =
                if let Some(call) = scheduled_tool.take() {
                    (
                        Action::Tool(call),
                        true,
                        String::new(),
                        String::new(),
                        None,
                        0,
                    )
                } else {
                    let request_input = std::mem::take(&mut input);
                    let provider_input = if self.limits.capture_model_output {
                        request_input.clone()
                    } else {
                        String::new()
                    };
                    let max_tokens = self.limits.max_tokens_per_turn;
                    let output = self
                        .model
                        .continue_state(
                            StateContinueRequest {
                                state: state.clone(),
                                input: request_input,
                                stops: vec!["</tool_call>".into(), "</answer>".into()],
                                max_tokens,
                            },
                            context.clone(),
                        )
                        .await
                        .map_err(AgentError::Model)?;
                    // Cancellation is cooperative at Provider boundaries. Never
                    // drop an in-flight State mutation whose server-side commit
                    // status is unknown; observe its response, then stop before
                    // parsing or scheduling any further action.
                    context.check()?;
                    if output.state_id != state.state_id {
                        return Err(AgentError::StateChanged {
                            expected: state.state_id.clone(),
                            actual: output.state_id,
                        });
                    }
                    model_turns += 1;
                    let raw_output = if self.limits.capture_model_output {
                        output.text.clone()
                    } else {
                        String::new()
                    };
                    let stop_reason = output.stop_reason.clone();
                    let action = match parse_action(&output.text) {
                        Ok(action) => action,
                        Err(error) if protocol_retries < self.limits.max_protocol_retries => {
                            protocol_retries += 1;
                            self.events.emit(AgentEvent::ProtocolRejected {
                                turn: model_turns,
                                retry: protocol_retries,
                                message: error.to_string(),
                                output_preview: output.text.chars().take(1_000).collect(),
                                provider_input: provider_input.clone(),
                                raw_output: if self.limits.capture_model_output {
                                    output.text.clone()
                                } else {
                                    String::new()
                                },
                                stop_reason: output.stop_reason.clone(),
                                max_tokens,
                            });
                            let feedback = format!(
                                "The previous protocol envelope was rejected: {error}. Output a strict tool call whose arguments field is a JSON object like {{\"command\":\"...\"}}, never a JSON-encoded string."
                            );
                            if let Some(root_prompt) =
                                self.tools.answer_retry_root_prompt(&feedback)
                            {
                                let refresh_worker =
                                    self.tools.refresh_worker_after_protocol_rejection();
                                let forked = self
                                    .replace_state(
                                        state,
                                        owner_id,
                                        &root_prompt,
                                        root_state,
                                        refresh_worker,
                                        context.clone(),
                                    )
                                    .await?;
                                let tool_prefix = self.tools.tool_call_prefix();
                                input = replacement_input(&root_prompt, &tool_prefix, forked);
                            } else {
                                input = format!(
                                    "\n\nUser: Your previous protocol envelope was rejected: {error}. \
Retry {protocol_retries}/{}. Output exactly one valid <tool_call> JSON envelope or one non-empty <answer> envelope. \
For JSON, escape backslashes and control characters correctly; prefer a simpler equivalent shell command when possible. \
Do not include reasoning or repeat a completed command.\n\nAssistant:",
                                    self.limits.max_protocol_retries,
                                );
                            }
                            continue;
                        }
                        Err(error) => return Err(error.into()),
                    };
                    (
                        action,
                        false,
                        provider_input,
                        raw_output,
                        stop_reason,
                        max_tokens,
                    )
                };
            if !controller_authored {
                self.events.emit(AgentEvent::ModelCompleted {
                    turn: model_turns,
                    state_id: state.state_id.clone(),
                    action: match action {
                        Action::Tool(_) => ActionKind::Tool,
                        Action::Answer(_) => ActionKind::Answer,
                    },
                    provider_input,
                    raw_output,
                    stop_reason,
                    max_tokens,
                });
            }
            match action {
                Action::Answer(answer) => {
                    if let Some(call) = self.tools.controller_tool_from_answer(&answer) {
                        if tool_steps >= self.limits.max_tool_steps {
                            return Err(AgentError::BudgetExceeded {
                                max_steps: self.limits.max_tool_steps,
                            });
                        }
                        self.events.emit(AgentEvent::ControllerToolScheduled {
                            step: tool_steps + 1,
                            name: call.name.clone(),
                        });
                        scheduled_tool = Some(call);
                        continue;
                    }
                    match self
                        .tools
                        .validate_answer(&answer, context.clone())
                        .await
                        .map_err(AgentError::Model)?
                    {
                        AnswerDecision::Accept => {}
                        AnswerDecision::Retry {
                            feedback,
                            require_tool,
                        } if answer_retries < self.limits.max_answer_retries => {
                            answer_retries += 1;
                            self.events.emit(AgentEvent::AnswerRejected {
                                retry: answer_retries,
                                require_tool,
                                feedback: feedback.clone(),
                            });
                            if let Some(root_prompt) =
                                self.tools.answer_retry_root_prompt(&feedback)
                            {
                                let refresh_worker = self.tools.refresh_worker_after_observation();
                                let forked = self
                                    .replace_state(
                                        state,
                                        owner_id,
                                        &root_prompt,
                                        root_state,
                                        refresh_worker,
                                        context.clone(),
                                    )
                                    .await?;
                                let tool_prefix = self.tools.tool_call_prefix();
                                input = replacement_input(
                                    &root_prompt,
                                    if require_tool {
                                        &tool_prefix
                                    } else {
                                        "<answer>"
                                    },
                                    forked,
                                );
                            } else {
                                let suffix = if require_tool {
                                    format!(" {}", self.tools.tool_call_prefix())
                                } else {
                                    " <answer>".into()
                                };
                                input = format!(
                                    "\n\nUser: The proposed answer was not accepted: {feedback} \
Retry {answer_retries}/{}. Continue the original task and satisfy every requested stage and exact output. \
Do not explain unfinished work.\n\nAssistant:{suffix}",
                                    self.limits.max_answer_retries,
                                );
                            }
                            continue;
                        }
                        AnswerDecision::Retry { .. } => {
                            return Err(AgentError::AnswerRejected {
                                max_retries: self.limits.max_answer_retries,
                            });
                        }
                    }
                    self.events.emit(AgentEvent::AnswerCompleted {
                        answer: answer.clone(),
                    });
                    return Ok(RunReport {
                        answer,
                        state_id: state.state_id.clone(),
                        model_turns,
                        tool_steps,
                        tools,
                    });
                }
                Action::Tool(call) => {
                    if tool_steps >= self.limits.max_tool_steps {
                        return Err(AgentError::BudgetExceeded {
                            max_steps: self.limits.max_tool_steps,
                        });
                    }
                    context.check()?;
                    tool_steps += 1;
                    self.events.emit(AgentEvent::ToolStarted {
                        step: tool_steps,
                        name: call.name.clone(),
                    });
                    let name = call.name.clone();
                    let recorded_call = call.clone();
                    let repeated_success =
                        !controller_authored && last_successful_call.as_ref() == Some(&call);
                    let repeated_failure =
                        !controller_authored && last_unsuccessful_call.as_ref() == Some(&call);
                    if repeated_success || repeated_failure {
                        repeated_rejections += 1;
                    } else {
                        repeated_rejections = 0;
                    }
                    let result = normalize_tool_result(if repeated_success {
                        json!({
                            "status":"rejected",
                            "message":"identical successful tool call already completed; choose a distinct remaining action or return the final answer"
                        })
                    } else if repeated_failure {
                        json!({
                            "status":"rejected",
                            "message":"identical unsuccessful tool call already failed; use the actual prior error and choose a distinct corrective action"
                        })
                    } else {
                        match self.registry.validate(&call) {
                            Ok(()) => self
                                .tools
                                .execute(call, context.clone())
                                .await
                                .unwrap_or_else(
                                    |error| json!({"status": "error", "message": error}),
                                ),
                            Err(error) => {
                                json!({"status": "rejected", "message": error.to_string()})
                            }
                        }
                    });
                    let status = result
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("ok")
                        .to_string();
                    self.events.emit(AgentEvent::ToolCompleted {
                        step: tool_steps,
                        name,
                        status,
                        arguments: recorded_call.arguments.clone(),
                        result: result.clone(),
                    });
                    tools.push(ToolStep {
                        step: tool_steps,
                        call: recorded_call.clone(),
                        result: result.clone(),
                    });
                    // The completed result is retained in the run trace before
                    // cancellation is honored. No subsequent controller or
                    // model action may begin after the token is observed.
                    context.check()?;
                    if repeated_rejections >= 2 {
                        return Err(AgentError::NoProgress {
                            repetitions: repeated_rejections,
                        });
                    }
                    if result.get("status").and_then(Value::as_str) == Some("ok") {
                        last_successful_call = Some(recorded_call);
                        last_unsuccessful_call = None;
                    } else if !repeated_success && !repeated_failure {
                        last_successful_call = None;
                        last_unsuccessful_call = Some(recorded_call);
                    }
                    if tool_steps < self.limits.max_tool_steps
                        && let Some(call) = self.tools.controller_tool_after_observation()
                    {
                        self.events.emit(AgentEvent::ControllerToolScheduled {
                            step: tool_steps + 1,
                            name: call.name.clone(),
                        });
                        scheduled_tool = Some(call);
                        continue;
                    }
                    if tool_steps < self.limits.max_tool_steps
                        && let Some(root_prompt) =
                            self.tools.recovery_root_prompt_after_observation()
                    {
                        let refresh_worker = repeated_success
                            || repeated_failure
                            || self.tools.refresh_worker_after_observation();
                        let forked = self
                            .replace_state(
                                state,
                                owner_id,
                                &root_prompt,
                                root_state,
                                refresh_worker,
                                context.clone(),
                            )
                            .await?;
                        let tool_prefix = self.tools.tool_call_prefix();
                        input = replacement_input(&root_prompt, &tool_prefix, forked);
                        continue;
                    }
                    if let Some(answer) = self.tools.committed_answer_after_observation() {
                        match self
                            .tools
                            .validate_answer(&answer, context.clone())
                            .await
                            .map_err(AgentError::Model)?
                        {
                            AnswerDecision::Accept => {
                                self.events.emit(AgentEvent::AnswerCompleted {
                                    answer: answer.clone(),
                                });
                                return Ok(RunReport {
                                    answer,
                                    state_id: state.state_id.clone(),
                                    model_turns,
                                    tool_steps,
                                    tools,
                                });
                            }
                            AnswerDecision::Retry {
                                feedback,
                                require_tool,
                            } if answer_retries < self.limits.max_answer_retries => {
                                answer_retries += 1;
                                self.events.emit(AgentEvent::AnswerRejected {
                                    retry: answer_retries,
                                    require_tool,
                                    feedback: feedback.clone(),
                                });
                                if let Some(root_prompt) =
                                    self.tools.answer_retry_root_prompt(&feedback)
                                {
                                    let refresh_worker =
                                        self.tools.refresh_worker_after_observation();
                                    let forked = self
                                        .replace_state(
                                            state,
                                            owner_id,
                                            &root_prompt,
                                            root_state,
                                            refresh_worker,
                                            context.clone(),
                                        )
                                        .await?;
                                    let tool_prefix = self.tools.tool_call_prefix();
                                    input = replacement_input(
                                        &root_prompt,
                                        if require_tool {
                                            &tool_prefix
                                        } else {
                                            "<answer>"
                                        },
                                        forked,
                                    );
                                } else {
                                    let suffix = if require_tool {
                                        format!(" {}", self.tools.tool_call_prefix())
                                    } else {
                                        " <answer>".into()
                                    };
                                    input = format!(
                                        "\n\nUser: The controller could not commit the verified result: {feedback} \
Retry {answer_retries}/{}. Continue the original task and satisfy every requested stage.\n\nAssistant:{suffix}",
                                        self.limits.max_answer_retries,
                                    );
                                }
                                continue;
                            }
                            AnswerDecision::Retry { .. } => {
                                return Err(AgentError::AnswerRejected {
                                    max_retries: self.limits.max_answer_retries,
                                });
                            }
                        }
                    }
                    input = if tool_steps >= self.limits.max_tool_steps {
                        // The execution budget is closed, but the model still
                        // gets one bounded commit turn. Prefixing the answer
                        // envelope makes this independent of task semantics and
                        // avoids turning a completed trajectory into an HTTP
                        // error merely because the model tried to continue.
                        render_answer_observation(&result)
                    } else if repeated_success {
                        // Re-running an already successful call cannot add new
                        // evidence or side effects. Close the bounded loop with
                        // the best grounded answer instead of spending the
                        // remaining budget on a deterministic duplicate cycle.
                        render_answer_observation(&result)
                    } else if self.limits.answer_after_tool {
                        render_answer_observation(&result)
                    } else if self.tools.commit_answer_after_observation() {
                        render_answer_observation_with_reminder(
                            &result,
                            &self.limits.observation_reminder,
                        )
                    } else {
                        let phase_feedback = self.tools.continuation_feedback();
                        let reminder = if phase_feedback.trim().is_empty() {
                            self.limits.observation_reminder.clone()
                        } else if self.limits.observation_reminder.trim().is_empty() {
                            phase_feedback
                        } else {
                            format!(
                                "{}\nController phase (mandatory): {phase_feedback}",
                                self.limits.observation_reminder
                            )
                        };
                        if tool_steps < self.limits.max_tool_steps
                            && self.tools.require_tool_after_observation()
                        {
                            render_tool_observation_with_progress_reminder_and_prefix(
                                &result,
                                tool_steps,
                                self.limits.max_tool_steps,
                                &reminder,
                                &self.tools.tool_call_prefix(),
                            )
                        } else {
                            render_observation_with_progress_and_reminder(
                                &result,
                                tool_steps,
                                self.limits.max_tool_steps,
                                &reminder,
                            )
                        }
                    };
                }
            }
        }
    }

    async fn replace_state(
        &mut self,
        state: &mut StateHandle,
        owner_id: &str,
        replacement_prompt: &str,
        root_state: Option<&StateHandle>,
        refresh_root_worker: bool,
        context: RunContext,
    ) -> Result<bool, AgentError> {
        if root_state.is_some() && !refresh_root_worker {
            // A root-backed task forks exactly once at startup. Recovery and
            // protocol feedback are appended to that same task worker so the
            // model retains its original task, actions, and observations.
            return Ok(true);
        }
        let forked = root_state.is_some();
        let replacement = if let Some(root) = root_state {
            // A repeated successful call is a deterministic no-progress loop.
            // Refresh only the task worker from the retained root; ordinary
            // Action/Observation steps continue in the same recurrent State.
            self.model
                .fork_state(root.clone(), context)
                .await
                .map_err(AgentError::Model)?
        } else {
            self.model
                .open(
                    OpenStateRequest {
                        owner_id: owner_id.to_string(),
                        root_prompt: replacement_prompt.to_string(),
                    },
                    context,
                )
                .await
                .map_err(AgentError::Model)?
        };
        if replacement.state_id.trim().is_empty() {
            return Err(AgentError::Model(
                "model returned an empty recovery state_id".into(),
            ));
        }
        if replacement.owner_id != owner_id {
            let actual = replacement.owner_id.clone();
            let release = self.model.release(replacement.clone()).await;
            self.events.emit(AgentEvent::StateReleased {
                state_id: replacement.state_id,
                success: release.is_ok(),
                error: release.as_ref().err().cloned().unwrap_or_default(),
            });
            return match release {
                Ok(()) => Err(AgentError::OwnerChanged {
                    expected: owner_id.to_string(),
                    actual,
                }),
                Err(release) => Err(AgentError::RunAndRelease {
                    run: Box::new(AgentError::OwnerChanged {
                        expected: owner_id.to_string(),
                        actual,
                    }),
                    release,
                }),
            };
        }
        self.events.emit(AgentEvent::StateOpened {
            state_id: replacement.state_id.clone(),
        });
        let previous = std::mem::replace(state, replacement);
        self.release_state(previous)
            .await
            .map_err(AgentError::Release)?;
        Ok(forked)
    }

    async fn release_state(&mut self, state: StateHandle) -> Result<(), String> {
        let release = self.model.release(state.clone()).await;
        self.events.emit(AgentEvent::StateReleased {
            state_id: state.state_id,
            success: release.is_ok(),
            error: release.as_ref().err().cloned().unwrap_or_default(),
        });
        release
    }

    fn emit_failure(&mut self, error: &AgentError) {
        self.events.emit(AgentEvent::RunFailed {
            code: error.code().to_string(),
            message: error.to_string(),
        });
    }
}

fn replacement_input(prompt: &str, prefix: &str, forked: bool) -> String {
    if forked {
        let phase = prompt.strip_prefix("System: ").unwrap_or(prompt);
        format!("\n\nUser: Controller phase (authoritative):\n{phase}\n\nAssistant: {prefix}")
    } else {
        format!("\n\nAssistant: {prefix}")
    }
}
