use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use rwkv_agent_runtime::{AgentService, CommandPolicy, RuntimeConfig};
use rwkv_agent_server::router;

#[derive(Parser, Debug)]
#[command(name = "rwkv-agent-server-rs")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8122)]
    port: u16,
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
    #[arg(long, default_value = "var/rust-agent-sessions")]
    session_dir: PathBuf,
    #[arg(long, default_value_t = -3.2, allow_hyphen_values = true)]
    tool_gate_threshold: f64,
    #[arg(long, default_value_t = -5.5, allow_hyphen_values = true)]
    pasted_text_gate_threshold: f64,
    #[arg(long, default_value_t = 3)]
    chat_state_capacity: usize,
    #[arg(long, default_value_t = 6)]
    max_tool_steps: usize,
    #[arg(long, default_value_t = 4000)]
    long_text_capture_chars: usize,
    #[arg(long, default_value_t = false)]
    enable_command: bool,
    #[arg(long)]
    command_workspace: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    command_timeout_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    if args.enable_command && args.command_workspace.is_none() {
        return Err("--enable-command requires --command-workspace".into());
    }
    let config = RuntimeConfig {
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
        command: CommandPolicy {
            enabled: args.enable_command,
            workspace: args.command_workspace,
            timeout: Duration::from_secs(args.command_timeout_seconds),
            max_output_bytes: 64 * 1024,
        },
    };
    let service = AgentService::new(config).await?;
    let app = router(service.clone());
    let address: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| format!("invalid listen address: {error}"))?;
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string())?;
    service.shutdown().await;
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
