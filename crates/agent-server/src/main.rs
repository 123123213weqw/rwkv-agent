use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use rwkv_agent_runtime::{
    AgentService, CloudModelRef, CloudPluginConfig, CloudPluginFallback, CommandPolicy,
    DebugTraceConfig, DebugTraceMode, PrivacyClass, RuntimeConfig, WorkerZone,
};
use rwkv_agent_server::router;

#[derive(Parser, Debug)]
#[command(name = "rwkv-agent-server-rs")]
struct Args {
    #[arg(long, env = "RWKV_AGENT_HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(long, env = "RWKV_AGENT_PORT", default_value_t = 8122)]
    port: u16,
    #[arg(
        long,
        env = "RWKV_AGENT_RUNTIME_REVISION",
        default_value = env!("CARGO_PKG_VERSION")
    )]
    runtime_revision: String,
    #[arg(
        long,
        env = "RWKV_AGENT_MODEL_URLS",
        default_value = "http://127.0.0.1:8417"
    )]
    model_urls: String,
    #[arg(
        long,
        env = "RWKV_AGENT_DATA_PLANE_URL",
        default_value = "http://127.0.0.1:8121"
    )]
    data_plane_url: String,
    #[arg(
        long,
        env = "RWKV_AGENT_SESSION_DIR",
        default_value = "var/rust-agent-sessions"
    )]
    session_dir: PathBuf,
    #[arg(
        long,
        env = "RWKV_AGENT_TOOL_GATE_THRESHOLD",
        default_value_t = -3.2,
        allow_hyphen_values = true
    )]
    tool_gate_threshold: f64,
    #[arg(
        long,
        env = "RWKV_AGENT_PASTED_TEXT_GATE_THRESHOLD",
        default_value_t = -5.5,
        allow_hyphen_values = true
    )]
    pasted_text_gate_threshold: f64,
    #[arg(long, env = "RWKV_AGENT_CHAT_STATE_CAPACITY", default_value_t = 3)]
    chat_state_capacity: usize,
    #[arg(long, env = "RWKV_AGENT_MAX_TOOL_STEPS", default_value_t = 6)]
    max_tool_steps: usize,
    #[arg(
        long,
        env = "RWKV_AGENT_MAX_MODEL_TOKENS_PER_TURN",
        default_value_t = 192
    )]
    max_model_tokens_per_turn: u32,
    #[arg(long, env = "RWKV_AGENT_DIRECT_CHAT_MAX_TOKENS", default_value_t = 96)]
    direct_chat_max_tokens: u32,
    #[arg(
        long,
        env = "RWKV_AGENT_LONG_TEXT_CAPTURE_CHARS",
        default_value_t = 4000
    )]
    long_text_capture_chars: usize,
    #[arg(long, env = "RWKV_AGENT_MAX_RUN_SECONDS", default_value_t = 600)]
    max_run_seconds: u64,
    #[arg(long, env = "RWKV_AGENT_SHUTDOWN_GRACE_SECONDS", default_value_t = 200)]
    shutdown_grace_seconds: u64,
    #[arg(long, env = "RWKV_AGENT_CLOUD_PLUGIN", default_value_t = false)]
    cloud_plugin: bool,
    #[arg(
        long,
        env = "RWKV_AGENT_CLOUD_PLUGIN_URL",
        default_value = "http://127.0.0.1:8130"
    )]
    cloud_plugin_url: String,
    #[arg(
        long,
        env = "RWKV_AGENT_CLOUD_PLUGIN_FALLBACK",
        default_value = "local"
    )]
    cloud_plugin_fallback: String,
    #[arg(
        long,
        env = "RWKV_AGENT_CLOUD_PLUGIN_PRIVACY",
        default_value = "local_only"
    )]
    cloud_plugin_privacy: String,
    #[arg(
        long,
        env = "RWKV_AGENT_CLOUD_PLUGIN_LATENCY_SLO_MS",
        default_value_t = 5000
    )]
    cloud_plugin_latency_slo_ms: u64,
    #[arg(long, env = "RWKV_AGENT_CLOUD_PLUGIN_PREFERRED_ZONE")]
    cloud_plugin_preferred_zone: Option<String>,
    #[arg(long, env = "RWKV_AGENT_CLOUD_MODEL_ID")]
    cloud_model_id: Option<String>,
    #[arg(long, env = "RWKV_AGENT_CLOUD_MODEL_REVISION")]
    cloud_model_revision: Option<String>,
    #[arg(long, env = "RWKV_AGENT_CLOUD_TOKENIZER")]
    cloud_tokenizer: Option<String>,
    #[arg(long, env = "RWKV_AGENT_CLOUD_STATE_ABI")]
    cloud_state_abi: Option<String>,
    #[arg(long, env = "RWKV_AGENT_ENABLE_COMMAND", default_value_t = false)]
    enable_command: bool,
    #[arg(long, env = "RWKV_AGENT_COMMAND_WORKSPACE")]
    command_workspace: Option<PathBuf>,
    #[arg(long, env = "RWKV_AGENT_COMMAND_TIMEOUT_SECONDS", default_value_t = 10)]
    command_timeout_seconds: u64,
    #[arg(long, env = "RWKV_AGENT_DEBUG_MODE", default_value = "off")]
    debug_mode: String,
    #[arg(long, env = "RWKV_AGENT_DEBUG_DIR", default_value = "var/debug-traces")]
    debug_dir: PathBuf,
    #[arg(long, env = "RWKV_AGENT_DEBUG_RETENTION_HOURS", default_value_t = 24)]
    debug_retention_hours: u64,
    #[arg(
        long,
        env = "RWKV_AGENT_DEBUG_MAX_BYTES",
        default_value_t = 2_147_483_648
    )]
    debug_max_bytes: u64,
    #[arg(long, env = "RWKV_AGENT_DEBUG_API", default_value_t = false)]
    debug_api: bool,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    if args.enable_command && args.command_workspace.is_none() {
        return Err("--enable-command requires --command-workspace".into());
    }
    let address: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| format!("invalid listen address: {error}"))?;
    if args.debug_api && !address.ip().is_loopback() {
        return Err("--debug-api requires a loopback listen address".into());
    }
    let debug_mode = args.debug_mode.parse::<DebugTraceMode>()?;
    let cloud_plugin_fallback = args.cloud_plugin_fallback.parse::<CloudPluginFallback>()?;
    let cloud_plugin_privacy = args.cloud_plugin_privacy.parse::<PrivacyClass>()?;
    let cloud_plugin_preferred_zone = args
        .cloud_plugin_preferred_zone
        .map(|value| value.parse::<WorkerZone>())
        .transpose()?;
    let cloud_model_ref = match (
        args.cloud_model_id,
        args.cloud_model_revision,
        args.cloud_tokenizer,
        args.cloud_state_abi,
    ) {
        (Some(model_id), Some(revision), Some(tokenizer), Some(state_abi)) => Some(CloudModelRef {
            model_id,
            revision,
            tokenizer,
            state_abi,
        }),
        (None, None, None, None) => None,
        _ => {
            return Err(
                "cloud model identity requires model-id, revision, tokenizer and state-abi".into(),
            );
        }
    };
    let config = RuntimeConfig {
        runtime_revision: args.runtime_revision,
        model_urls: args
            .model_urls
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        data_plane_url: args.data_plane_url,
        session_dir: args.session_dir,
        tool_gate_threshold: args.tool_gate_threshold,
        pasted_text_gate_threshold: args.pasted_text_gate_threshold,
        long_text_capture_chars: args.long_text_capture_chars,
        chat_state_capacity: args.chat_state_capacity,
        max_tool_steps: args.max_tool_steps,
        max_model_tokens_per_turn: args.max_model_tokens_per_turn,
        direct_chat_max_tokens: args.direct_chat_max_tokens,
        max_run_elapsed: Duration::from_secs(args.max_run_seconds),
        shutdown_grace: Duration::from_secs(args.shutdown_grace_seconds),
        cloud_plugin: CloudPluginConfig {
            enabled: args.cloud_plugin,
            endpoint: args.cloud_plugin_url,
            fallback: cloud_plugin_fallback,
            default_privacy: cloud_plugin_privacy,
            latency_slo: Duration::from_millis(args.cloud_plugin_latency_slo_ms),
            preferred_zone: cloud_plugin_preferred_zone,
            model_ref: cloud_model_ref,
            ..CloudPluginConfig::default()
        },
        command: CommandPolicy {
            enabled: args.enable_command,
            workspace: args.command_workspace,
            timeout: Duration::from_secs(args.command_timeout_seconds),
            max_output_bytes: 64 * 1024,
        },
        debug_trace: DebugTraceConfig {
            mode: debug_mode,
            directory: args.debug_dir,
            retention: Duration::from_secs(
                args.debug_retention_hours
                    .checked_mul(60 * 60)
                    .ok_or_else(|| "debug retention hours overflow".to_string())?,
            ),
            max_bytes: args.debug_max_bytes,
            api_enabled: args.debug_api,
            ..DebugTraceConfig::default()
        },
    };
    let service = AgentService::new(config).await?;
    let app = router(service.clone());
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string())?;
    service.shutdown().await?;
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = async {
                if let Some(signal) = &mut terminate {
                    signal.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {},
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}
