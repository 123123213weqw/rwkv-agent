use std::collections::VecDeque;
use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use rwkv_agent_core::{
    AgentError, AgentEvent, AgentLoop, AgentRunRequest, AnswerDecision, CancellationToken,
    ModelOutput, OpenStateRequest, RunContext, RunLimits, StateContinueRequest, StateHandle,
    StateModel, ToolCall, ToolDefinition, ToolExecutor, ToolRegistry, VecEventSink,
};
use serde_json::{Value, json};

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[derive(Default)]
struct MockModel {
    outputs: VecDeque<Result<String, String>>,
    opened: usize,
    forked: usize,
    released: usize,
    release_error: Option<String>,
    inputs: Vec<String>,
    returned_state_id: Option<String>,
    cancel_after_continue: Option<CancellationToken>,
}

impl MockModel {
    fn scripted<I, S>(outputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            outputs: outputs.into_iter().map(|value| Ok(value.into())).collect(),
            ..Self::default()
        }
    }
}

impl StateModel for MockModel {
    fn open(
        &mut self,
        request: OpenStateRequest,
        _context: RunContext,
    ) -> impl Future<Output = Result<StateHandle, String>> + Send {
        self.opened += 1;
        std::future::ready(Ok(StateHandle {
            endpoint: "mock://model".into(),
            owner_id: request.owner_id,
            state_id: format!("state-{}", self.opened),
        }))
    }

    fn continue_state(
        &mut self,
        request: StateContinueRequest,
        _context: RunContext,
    ) -> impl Future<Output = Result<ModelOutput, String>> + Send {
        self.inputs.push(request.input);
        if let Some(token) = self.cancel_after_continue.take() {
            token.cancel();
        }
        let output = self
            .outputs
            .pop_front()
            .unwrap_or_else(|| Err("mock output exhausted".into()));
        let state_id = self
            .returned_state_id
            .clone()
            .unwrap_or(request.state.state_id);
        std::future::ready(output.map(|text| ModelOutput {
            state_id,
            text,
            stop_reason: None,
        }))
    }

    fn fork_state(
        &mut self,
        root: StateHandle,
        _context: RunContext,
    ) -> impl Future<Output = Result<StateHandle, String>> + Send {
        self.forked += 1;
        std::future::ready(Ok(StateHandle {
            endpoint: root.endpoint,
            owner_id: root.owner_id,
            state_id: format!("fork-{}", self.forked),
        }))
    }

    fn release(&mut self, _state: StateHandle) -> impl Future<Output = Result<(), String>> + Send {
        self.released += 1;
        std::future::ready(self.release_error.take().map_or(Ok(()), Err))
    }
}

#[derive(Default)]
struct MockTools {
    calls: Vec<ToolCall>,
    results: VecDeque<Result<Value, String>>,
    answer_decisions: VecDeque<AnswerDecision>,
    recovery_prompt: Option<String>,
    recover_after_first_tool: bool,
    committed_answer: Option<String>,
    controller_tools: VecDeque<ToolCall>,
    answer_tools: VecDeque<ToolCall>,
    cancel_after_execute: Option<CancellationToken>,
    refresh_protocol: bool,
}

impl ToolExecutor for MockTools {
    fn execute(
        &mut self,
        call: ToolCall,
        _context: RunContext,
    ) -> impl Future<Output = Result<Value, String>> + Send {
        self.calls.push(call);
        if let Some(token) = self.cancel_after_execute.take() {
            token.cancel();
        }
        std::future::ready(
            self.results
                .pop_front()
                .unwrap_or_else(|| Ok(json!({"status": "ok"}))),
        )
    }

    fn validate_answer(
        &mut self,
        _answer: &str,
        _context: RunContext,
    ) -> impl Future<Output = Result<AnswerDecision, String>> + Send {
        std::future::ready(Ok(self
            .answer_decisions
            .pop_front()
            .unwrap_or(AnswerDecision::Accept)))
    }

    fn answer_retry_root_prompt(&self, _feedback: &str) -> Option<String> {
        self.recovery_prompt.clone()
    }

    fn recovery_root_prompt_after_observation(&self) -> Option<String> {
        (self.recover_after_first_tool && self.calls.len() == 1)
            .then(|| "fresh next-phase prompt".into())
    }

    fn committed_answer_after_observation(&self) -> Option<String> {
        self.committed_answer.clone()
    }

    fn controller_tool_after_observation(&mut self) -> Option<ToolCall> {
        self.controller_tools.pop_front()
    }

    fn controller_tool_from_answer(&mut self, _answer: &str) -> Option<ToolCall> {
        self.answer_tools.pop_front()
    }

    fn refresh_worker_after_protocol_rejection(&self) -> bool {
        self.refresh_protocol
    }
}

#[test]
fn strict_answer_payload_can_schedule_a_traced_controller_tool() {
    let write = ToolCall {
        name: "run_command".into(),
        arguments: json!({"command":"printf fixed > result.txt"})
            .as_object()
            .unwrap()
            .clone(),
    };
    let mut model = MockModel::scripted([
        "<answer>replacement payload</answer>",
        "<answer>done</answer>",
    ]);
    let mut tools = MockTools {
        answer_tools: [write.clone()].into(),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(report.model_turns, 2);
    assert_eq!(report.tool_steps, 1);
    assert_eq!(report.tools[0].call, write);
    assert_eq!(tools.calls.len(), 1);
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::ControllerToolScheduled { step: 1, name } if name == "run_command"
    )));
}

fn registry() -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry
        .register(ToolDefinition::one_string(
            "run_command",
            "Run a bounded command",
            "command",
        ))
        .unwrap();
    registry
}

fn request() -> AgentRunRequest {
    AgentRunRequest {
        owner_id: "owner-1".into(),
        root_prompt: "System and task".into(),
        initial_input: "\n\nAssistant: <tool_call>".into(),
    }
}

fn limits(max_tool_steps: usize) -> RunLimits {
    RunLimits {
        max_tool_steps,
        max_protocol_retries: 2,
        max_answer_retries: 3,
        observation_reminder: String::new(),
        max_tokens_per_turn: 96,
        max_elapsed: Duration::from_secs(5),
        answer_after_tool: false,
        fork_from_root: false,
        capture_model_output: false,
    }
}

#[test]
fn executor_can_commit_a_verified_answer_without_an_extra_model_decode() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat result.txt"}}</tool_call>"#,
    ]);
    let mut tools = MockTools {
        results: [Ok(json!({"status":"ok","stdout":"result=42\n"}))].into(),
        committed_answer: Some("result=42".into()),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "result=42");
    assert_eq!(report.model_turns, 1);
    assert_eq!(report.tool_steps, 1);
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.released, 1);
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::AnswerCompleted { answer } if answer == "result=42"
    )));
}

#[test]
fn reusable_root_forks_one_task_worker_and_keeps_it_across_feedback() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat input.txt"}}</tool_call>"#,
        "<answer>done</answer>",
    ]);
    let mut tools = MockTools {
        recover_after_first_tool: true,
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut run_limits = limits(2);
    run_limits.fork_from_root = true;
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        run_limits,
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(model.opened, 1);
    assert_eq!(model.forked, 1);
    assert_eq!(model.released, 2);
    assert!(model.inputs[1].contains("fresh next-phase prompt"));
    assert!(model.inputs[1].ends_with("Assistant: <tool_call>{\"name\":\""));
}

#[test]
fn root_worker_refreshes_only_after_identical_success_loop() {
    let repeated =
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat input.txt"}}</tool_call>"#;
    let next = r#"<tool_call>{"name":"run_command","arguments":{"command":"sed -n 1p input.txt"}}</tool_call>"#;
    let mut model = MockModel::scripted([repeated, repeated, next, "<answer>done</answer>"]);
    let mut tools = MockTools {
        recover_after_first_tool: true,
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut run_limits = limits(3);
    run_limits.fork_from_root = true;
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        run_limits,
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(report.tool_steps, 3);
    assert_eq!(tools.calls.len(), 2);
    assert_eq!(model.opened, 1);
    assert_eq!(model.forked, 2);
    assert_eq!(model.released, 3);
    assert_eq!(report.tools[1].result["status"], "rejected");
}

#[test]
fn root_worker_refreshes_after_an_identical_failed_call_without_reexecution() {
    let repeated = r#"<tool_call>{"name":"run_command","arguments":{"command":"cat missing.txt"}}</tool_call>"#;
    let next =
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat input.txt"}}</tool_call>"#;
    let mut model = MockModel::scripted([repeated, repeated, next, "<answer>done</answer>"]);
    let mut tools = MockTools {
        results: [
            Err("missing.txt does not exist".into()),
            Ok(json!({"status":"ok","stdout":"input\n"})),
        ]
        .into(),
        recover_after_first_tool: true,
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut run_limits = limits(3);
    run_limits.fork_from_root = true;
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        run_limits,
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(report.tool_steps, 3);
    assert_eq!(tools.calls.len(), 2);
    assert_eq!(model.forked, 2);
    assert_eq!(model.released, 3);
    assert!(
        report.tools[1].result["message"]
            .as_str()
            .unwrap()
            .contains("identical unsuccessful tool call")
    );
}

#[test]
fn second_repeated_call_fails_bounded_instead_of_spending_the_remaining_budget() {
    let repeated =
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat input.txt"}}</tool_call>"#;
    let mut model = MockModel::scripted([repeated, repeated, repeated]);
    let mut tools = MockTools {
        recover_after_first_tool: true,
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut run_limits = limits(18);
    run_limits.fork_from_root = true;
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        run_limits,
        CancellationToken::default(),
    )
    .unwrap();

    let error = block_on(agent.run(request())).unwrap_err();

    assert!(matches!(error, AgentError::NoProgress { repetitions: 2 }));
    assert_eq!(tools.calls.len(), 1);
    assert_eq!(model.forked, 2);
    assert_eq!(model.released, 3);
    assert_eq!(
        events
            .events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolCompleted { .. }))
            .count(),
        3
    );
}

#[test]
fn workspace_protocol_repair_can_refresh_a_root_worker() {
    let malformed =
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf \U"}}</tool_call>"#;
    let valid =
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf ok"}}</tool_call>"#;
    let mut model = MockModel::scripted([malformed, valid, "<answer>done</answer>"]);
    let mut tools = MockTools {
        recovery_prompt: Some("protocol repair".into()),
        refresh_protocol: true,
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut run_limits = limits(2);
    run_limits.fork_from_root = true;

    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        run_limits,
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(model.forked, 2);
    assert_eq!(model.released, 3);
    assert!(model.inputs[1].contains("protocol repair"));
}

#[test]
fn real_loop_shape_reuses_one_state_observes_tools_and_releases() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat note.txt"}}</tool_call>"#,
        r#"<tool_call>{"name":"run_command","arguments":{"command":"wc -l note.txt"}}</tool_call>"#,
        "<answer>note.txt has one line.</answer>",
    ]);
    let mut tools = MockTools {
        results: [
            Ok(json!({"status": "ok", "stdout": "hello\n"})),
            Ok(json!({"status": "ok", "stdout": "1 note.txt\n"})),
        ]
        .into(),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(4),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "note.txt has one line.");
    assert_eq!(report.state_id, "state-1");
    assert_eq!(report.model_turns, 3);
    assert_eq!(report.tool_steps, 2);
    assert_eq!(report.tools.len(), 2);
    assert_eq!(report.tools[0].call.name, "run_command");
    assert_eq!(report.tools[0].result["stdout"], "hello\n");
    assert_eq!(report.tools[1].result["stdout"], "1 note.txt\n");
    assert_eq!(model.opened, 1);
    assert_eq!(model.released, 1);
    assert_eq!(tools.calls.len(), 2);
    assert_eq!(model.inputs.len(), 3);
    assert!(model.inputs[1].contains("hello\\n"));
    assert!(model.inputs[2].contains("1 note.txt\\n"));
    assert_eq!(
        events.events.last(),
        Some(&AgentEvent::StateReleased {
            state_id: "state-1".into(),
            success: true,
            error: String::new(),
        })
    );
}

#[test]
fn controller_scheduled_verifier_is_strict_traced_and_uses_no_model_turn() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf fixed > output.txt"}}</tool_call>"#,
    ]);
    let verifier = ToolCall {
        name: "run_command".into(),
        arguments: json!({"command":"test -s output.txt && echo VERIFIED"})
            .as_object()
            .unwrap()
            .clone(),
    };
    let mut tools = MockTools {
        results: [
            Ok(json!({"status":"ok"})),
            Ok(json!({"status":"ok","stdout":"VERIFIED\n"})),
        ]
        .into(),
        controller_tools: [verifier.clone()].into(),
        committed_answer: Some("verified completion".into()),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(3),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "verified completion");
    assert_eq!(report.model_turns, 1);
    assert_eq!(report.tool_steps, 2);
    assert_eq!(report.tools[1].call, verifier);
    assert_eq!(tools.calls.len(), 2);
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::ControllerToolScheduled {
            step: 2,
            name
        } if name == "run_command"
    )));
}

#[test]
fn executor_failure_becomes_observation_and_model_can_recover() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"false"}}</tool_call>"#,
        "<answer>the command failed.</answer>",
    ]);
    let mut tools = MockTools {
        results: [Err("exit 1".into())].into(),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "the command failed.");
    assert!(model.inputs[1].contains(r#""status":"error""#));
    assert!(model.inputs[1].contains("exit 1"));
    assert_eq!(model.released, 1);
}

#[test]
fn answer_validator_can_replace_a_stuck_state_and_release_both_states() {
    let mut model = MockModel::scripted([
        "<answer>described but not executed</answer>",
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf done > out.txt"}}</tool_call>"#,
        "<answer>done</answer>",
    ]);
    let mut tools = MockTools {
        answer_decisions: [
            AnswerDecision::Retry {
                feedback: "workspace has not changed".into(),
                require_tool: true,
            },
            AnswerDecision::Accept,
        ]
        .into(),
        recovery_prompt: Some("fresh recovery prompt".into()),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(3),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(report.state_id, "state-2");
    assert_eq!(model.opened, 2);
    assert_eq!(model.released, 2);
    assert_eq!(tools.calls.len(), 1);
    assert_eq!(
        events
            .events
            .iter()
            .filter(|event| matches!(event, AgentEvent::StateReleased { success: true, .. }))
            .count(),
        2
    );
}

#[test]
fn executor_can_handoff_an_incomplete_phase_to_a_fresh_state() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat input.txt"}}</tool_call>"#,
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf done > out.txt"}}</tool_call>"#,
        "<answer>done</answer>",
    ]);
    let mut tools = MockTools {
        results: [
            Ok(json!({"status":"ok","stdout":"input\n"})),
            Ok(json!({"status":"ok"})),
        ]
        .into(),
        recover_after_first_tool: true,
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(3),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "done");
    assert_eq!(report.state_id, "state-2");
    assert_eq!(report.tool_steps, 2);
    assert_eq!(model.opened, 2);
    assert_eq!(model.released, 2);
}

#[test]
fn malformed_envelope_gets_one_bounded_repair_without_losing_state() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf \U"}}</tool_call>"#,
        r#"<tool_call>{"name":"run_command","arguments":{"command":"printf ok"}}</tool_call>"#,
        "<answer>repaired.</answer>",
    ]);
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "repaired.");
    assert_eq!(report.tool_steps, 1);
    assert_eq!(model.released, 1);
    assert!(
        events
            .events
            .iter()
            .any(|event| matches!(event, AgentEvent::ProtocolRejected { retry: 1, .. }))
    );
}

#[test]
fn consecutive_identical_success_is_rejected_without_reexecuting_side_effect() {
    let call =
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat note.txt"}}</tool_call>"#;
    let mut model = MockModel::scripted([call, call, "<answer>done.</answer>"]);
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(3),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(tools.calls.len(), 1);
    assert_eq!(report.tool_steps, 2);
    assert_eq!(report.tools[1].result["status"], "rejected");
    assert!(model.inputs[2].ends_with("Assistant: <answer>"));
    assert!(
        report.tools[1].result["message"]
            .as_str()
            .unwrap()
            .contains("identical successful tool call")
    );
}

#[test]
fn answer_validator_can_require_more_tool_work_in_the_same_state() {
    let mut model = MockModel::scripted([
        "<answer>described but not executed.</answer>",
        r#"<tool_call>{"name":"run_command","arguments":{"command":"write result"}}</tool_call>"#,
        "<answer>executed.</answer>",
    ]);
    let mut tools = MockTools {
        answer_decisions: [
            AnswerDecision::Retry {
                feedback: "no workspace file changed".into(),
                require_tool: true,
            },
            AnswerDecision::Accept,
        ]
        .into(),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(3),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "executed.");
    assert_eq!(report.tool_steps, 1);
    assert!(model.inputs[1].ends_with("Assistant: <tool_call>{\"name\":\""));
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::AnswerRejected {
            retry: 1,
            require_tool: true,
            ..
        }
    )));
}

#[test]
fn ordinary_tool_policy_commits_answer_after_one_observation() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat note.txt"}}</tool_call>"#,
        "<answer>done.</answer>",
    ]);
    let mut tools = MockTools {
        results: [Ok(json!({"status":"ok","stdout":"done\n"}))].into(),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut policy = limits(4);
    policy.answer_after_tool = true;
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        policy,
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.tool_steps, 1);
    assert!(model.inputs[1].ends_with("Assistant: <answer>"));
    assert_eq!(model.released, 1);
}

#[test]
fn non_object_tool_result_is_normalized_as_error_in_event_and_observation() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"pwd"}}</tool_call>"#,
        "<answer>bad result rejected.</answer>",
    ]);
    let mut tools = MockTools {
        results: [Ok(json!(7))].into(),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        CancellationToken::default(),
    )
    .unwrap();

    block_on(agent.run(request())).unwrap();

    assert!(model.inputs[1].contains(r#""status":"error""#));
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCompleted { status, .. } if status == "error"
    )));
}

#[test]
fn invalid_tool_arguments_are_rejected_without_execution() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":7}}</tool_call>"#,
        "<answer>arguments were rejected.</answer>",
    ]);
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "arguments were rejected.");
    assert!(tools.calls.is_empty());
    assert!(model.inputs[1].contains(r#""status":"rejected""#));
    assert_eq!(model.released, 1);
}

#[test]
fn budget_error_releases_state_and_never_executes_extra_tool() {
    let call = r#"<tool_call>{"name":"run_command","arguments":{"command":"pwd"}}</tool_call>"#;
    let mut model = MockModel::scripted([call, call]);
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(1),
        CancellationToken::default(),
    )
    .unwrap();

    let error = block_on(agent.run(request())).unwrap_err();

    assert!(matches!(error, AgentError::BudgetExceeded { max_steps: 1 }));
    assert_eq!(tools.calls.len(), 1);
    assert!(model.inputs[1].ends_with("Assistant: <answer>"));
    assert_eq!(model.released, 1);
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::RunFailed { code, .. } if code == "budget_exceeded"
    )));
}

#[test]
fn final_tool_step_gets_one_answer_commit_turn() {
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"cat value.txt"}}</tool_call>"#,
        "<answer>value is 7</answer>",
    ]);
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(1),
        CancellationToken::default(),
    )
    .unwrap();

    let report = block_on(agent.run(request())).unwrap();

    assert_eq!(report.answer, "value is 7");
    assert_eq!(report.tool_steps, 1);
    assert_eq!(tools.calls.len(), 1);
    assert!(model.inputs[1].ends_with("Assistant: <answer>"));
    assert_eq!(model.released, 1);
}

#[test]
fn protocol_and_state_identity_errors_release_state() {
    for (text, mismatch, expected_code) in [
        ("reasoning only", false, "protocol_error"),
        ("<answer>done</answer>", true, "state_changed"),
    ] {
        let outputs = if mismatch {
            vec![text]
        } else {
            vec![text, text, text]
        };
        let mut model = MockModel::scripted(outputs);
        model.returned_state_id = mismatch.then(|| "state-other".into());
        let mut tools = MockTools::default();
        let registry = registry();
        let mut events = VecEventSink::default();
        let mut agent = AgentLoop::new(
            &mut model,
            &mut tools,
            &registry,
            &mut events,
            limits(1),
            CancellationToken::default(),
        )
        .unwrap();

        let error = block_on(agent.run(request())).unwrap_err();

        assert_eq!(error.code(), expected_code);
        assert_eq!(model.released, 1);
    }
}

#[test]
fn model_open_error_emits_failure_without_fabricating_a_state_release() {
    let mut model = MockModel {
        outputs: [Err("open unavailable".into())].into(),
        ..MockModel::default()
    };
    struct OpenFailModel<'a>(&'a mut MockModel);
    impl StateModel for OpenFailModel<'_> {
        fn open(
            &mut self,
            _request: OpenStateRequest,
            _context: RunContext,
        ) -> impl Future<Output = Result<StateHandle, String>> + Send {
            self.0.opened += 1;
            std::future::ready(Err("open unavailable".into()))
        }

        fn continue_state(
            &mut self,
            _request: StateContinueRequest,
            _context: RunContext,
        ) -> impl Future<Output = Result<ModelOutput, String>> + Send {
            std::future::ready(Err("unreachable".into()))
        }

        fn release(
            &mut self,
            _state: StateHandle,
        ) -> impl Future<Output = Result<(), String>> + Send {
            self.0.released += 1;
            std::future::ready(Ok(()))
        }
    }
    let mut open_fail = OpenFailModel(&mut model);
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut open_fail,
        &mut tools,
        &registry,
        &mut events,
        limits(1),
        CancellationToken::default(),
    )
    .unwrap();

    let error = block_on(agent.run(request())).unwrap_err();

    assert!(matches!(error, AgentError::Model(_)));
    assert_eq!(model.opened, 1);
    assert_eq!(model.released, 0);
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::RunFailed { code, .. } if code == "model_error"
    )));
    assert!(
        !events
            .events
            .iter()
            .any(|event| matches!(event, AgentEvent::StateReleased { .. }))
    );
}

#[test]
fn cancellation_after_tool_is_seen_before_next_model_turn_and_releases() {
    let cancellation = CancellationToken::default();
    let mut model = MockModel::scripted([
        r#"<tool_call>{"name":"run_command","arguments":{"command":"pwd"}}</tool_call>"#,
        "<answer>must not be reached</answer>",
    ]);
    let mut tools = MockTools {
        cancel_after_execute: Some(cancellation.clone()),
        ..MockTools::default()
    };
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(2),
        cancellation,
    )
    .unwrap();

    let error = block_on(agent.run(request())).unwrap_err();

    assert!(matches!(error, AgentError::Cancelled));
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.released, 1);
}

#[test]
fn cancellation_during_model_continue_is_seen_at_boundary_and_releases() {
    let cancellation = CancellationToken::default();
    let mut model = MockModel {
        outputs: VecDeque::from([Ok("<answer>must not commit</answer>".into())]),
        cancel_after_continue: Some(cancellation.clone()),
        ..MockModel::default()
    };
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(1),
        cancellation,
    )
    .unwrap();

    let error = block_on(agent.run(request())).unwrap_err();

    assert!(matches!(error, AgentError::Cancelled));
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.released, 1);
    assert!(
        !events
            .events
            .iter()
            .any(|event| matches!(event, AgentEvent::AnswerCompleted { .. }))
    );
}

#[test]
fn release_failure_is_never_hidden() {
    let mut model = MockModel::scripted(["<answer>done</answer>"]);
    model.release_error = Some("release unavailable".into());
    let mut tools = MockTools::default();
    let registry = registry();
    let mut events = VecEventSink::default();
    let mut agent = AgentLoop::new(
        &mut model,
        &mut tools,
        &registry,
        &mut events,
        limits(1),
        CancellationToken::default(),
    )
    .unwrap();

    let error = block_on(agent.run(request())).unwrap_err();

    assert!(matches!(error, AgentError::Release(_)));
    assert_eq!(model.released, 1);
    assert!(
        events
            .events
            .iter()
            .any(|event| matches!(event, AgentEvent::StateReleased { success: false, .. }))
    );
}

#[test]
fn concurrent_independent_loops_do_not_share_state_or_events() {
    let answers = Arc::new(Mutex::new(Vec::new()));
    let handles = (0..4)
        .map(|index| {
            let answers = Arc::clone(&answers);
            std::thread::spawn(move || {
                let answer = format!("answer-{index}");
                let mut model = MockModel::scripted([format!("<answer>{answer}</answer>")]);
                let mut tools = MockTools::default();
                let registry = registry();
                let mut events = VecEventSink::default();
                let mut request = request();
                request.owner_id = format!("owner-{index}");
                let mut agent = AgentLoop::new(
                    &mut model,
                    &mut tools,
                    &registry,
                    &mut events,
                    limits(1),
                    CancellationToken::default(),
                )
                .unwrap();
                let report = block_on(agent.run(request)).unwrap();
                answers.lock().unwrap().push(report.answer);
                assert_eq!(model.released, 1);
                assert_eq!(events.events.len(), 5);
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
    let mut actual = answers.lock().unwrap().clone();
    actual.sort();
    assert_eq!(actual, ["answer-0", "answer-1", "answer-2", "answer-3"]);
}
