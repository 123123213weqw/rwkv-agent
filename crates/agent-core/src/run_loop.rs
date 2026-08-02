use std::future::Future;
use std::time::Instant;

use serde_json::{Value, json};

use crate::protocol::normalize_tool_result;
use crate::{
    Action, ActionKind, AgentError, AgentEvent, AgentRunRequest, CancellationToken, EventSink,
    ModelOutput, OpenStateRequest, RunContext, RunLimits, RunReport, StateContinueRequest,
    StateHandle, ToolCall, ToolRegistry, ToolStep, parse_action, render_answer_observation,
    render_observation,
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

    fn release(&mut self, state: StateHandle) -> impl Future<Output = Result<(), String>> + Send;
}

pub trait ToolExecutor {
    fn execute(
        &mut self,
        call: ToolCall,
        context: RunContext,
    ) -> impl Future<Output = Result<Value, String>> + Send;
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
        let state = match self
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
            state_id: state.state_id.clone(),
        });

        let run_result = if state.state_id.trim().is_empty() {
            Err(AgentError::Model("model returned an empty state_id".into()))
        } else if state.owner_id != request.owner_id {
            Err(AgentError::OwnerChanged {
                expected: request.owner_id,
                actual: state.owner_id.clone(),
            })
        } else {
            self.run_opened(&state, request.initial_input, context)
                .await
        };
        if let Err(error) = &run_result {
            self.emit_failure(error);
        }

        let release = self.model.release(state.clone()).await;
        self.events.emit(AgentEvent::StateReleased {
            state_id: state.state_id,
            success: release.is_ok(),
            error: release.as_ref().err().cloned().unwrap_or_default(),
        });

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
        state: &StateHandle,
        mut input: String,
        context: RunContext,
    ) -> Result<RunReport, AgentError> {
        let mut model_turns = 0;
        let mut tool_steps = 0;
        let mut tools = Vec::new();
        loop {
            context.check()?;
            let output = self
                .model
                .continue_state(
                    StateContinueRequest {
                        state: state.clone(),
                        input,
                        stops: vec!["</tool_call>".into(), "</answer>".into()],
                        max_tokens: self.limits.max_tokens_per_turn,
                    },
                    context.clone(),
                )
                .await
                .map_err(AgentError::Model)?;
            if output.state_id != state.state_id {
                return Err(AgentError::StateChanged {
                    expected: state.state_id.clone(),
                    actual: output.state_id,
                });
            }
            model_turns += 1;
            let action = parse_action(&output.text)?;
            self.events.emit(AgentEvent::ModelCompleted {
                turn: model_turns,
                state_id: state.state_id.clone(),
                action: match action {
                    Action::Tool(_) => ActionKind::Tool,
                    Action::Answer(_) => ActionKind::Answer,
                },
            });
            match action {
                Action::Answer(answer) => {
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
                    let result = normalize_tool_result(match self.registry.validate(&call) {
                        Ok(()) => self
                            .tools
                            .execute(call, context.clone())
                            .await
                            .unwrap_or_else(|error| json!({"status": "error", "message": error})),
                        Err(error) => {
                            json!({"status": "rejected", "message": error.to_string()})
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
                    });
                    tools.push(ToolStep {
                        step: tool_steps,
                        call: recorded_call,
                        result: result.clone(),
                    });
                    input = if self.limits.answer_after_tool {
                        render_answer_observation(&result)
                    } else {
                        render_observation(&result)
                    };
                }
            }
        }
    }

    fn emit_failure(&mut self, error: &AgentError) {
        self.events.emit(AgentEvent::RunFailed {
            code: error.code().to_string(),
            message: error.to_string(),
        });
    }
}
