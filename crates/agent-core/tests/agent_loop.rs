use std::collections::VecDeque;
use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use rwkv_agent_core::{
    AgentError, AgentEvent, AgentLoop, AgentRunRequest, CancellationToken, ModelOutput,
    OpenStateRequest, RunContext, RunLimits, StateContinueRequest, StateHandle, StateModel,
    ToolCall, ToolDefinition, ToolExecutor, ToolRegistry, VecEventSink,
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
    released: usize,
    release_error: Option<String>,
    inputs: Vec<String>,
    returned_state_id: Option<String>,
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
            state_id: "state-1".into(),
        }))
    }

    fn continue_state(
        &mut self,
        request: StateContinueRequest,
        _context: RunContext,
    ) -> impl Future<Output = Result<ModelOutput, String>> + Send {
        self.inputs.push(request.input);
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

    fn release(&mut self, _state: StateHandle) -> impl Future<Output = Result<(), String>> + Send {
        self.released += 1;
        std::future::ready(self.release_error.take().map_or(Ok(()), Err))
    }
}

#[derive(Default)]
struct MockTools {
    calls: Vec<ToolCall>,
    results: VecDeque<Result<Value, String>>,
    cancel_after_execute: Option<CancellationToken>,
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
        max_tokens_per_turn: 96,
        max_elapsed: Duration::from_secs(5),
        answer_after_tool: false,
    }
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
    assert_eq!(model.released, 1);
    assert!(events.events.iter().any(|event| matches!(
        event,
        AgentEvent::RunFailed { code, .. } if code == "budget_exceeded"
    )));
}

#[test]
fn protocol_and_state_identity_errors_release_state() {
    for (text, mismatch, expected_code) in [
        ("reasoning only", false, "protocol_error"),
        ("<answer>done</answer>", true, "state_changed"),
    ] {
        let mut model = MockModel::scripted([text]);
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
