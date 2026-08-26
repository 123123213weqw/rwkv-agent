use std::net::SocketAddr;
use std::time::Duration;

use clap::Parser;
use rwkv_statepool_cloud_plugin::{PluginConfig, PluginState, router};

#[derive(Parser, Debug)]
#[command(name = "rwkv-statepool-cloud-plugin")]
struct Args {
    #[arg(long, env = "RWKV_STATEPOOL_HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(long, env = "RWKV_STATEPOOL_PORT", default_value_t = 8130)]
    port: u16,
    #[arg(long, env = "RWKV_STATEPOOL_WORKER_TTL_SECONDS", default_value_t = 30)]
    worker_ttl_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    let address: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| format!("invalid listen address: {error}"))?;
    let state = PluginState::new(PluginConfig {
        worker_ttl: Duration::from_secs(args.worker_ttl_seconds),
        ..PluginConfig::default()
    })?;
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string())
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
