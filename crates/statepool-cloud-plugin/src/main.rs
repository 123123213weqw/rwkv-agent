use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use rwkv_statepool_cloud_plugin::metadata::{
    InMemoryMetadataStore, MetadataStore, PostgresMetadataStore,
};
use rwkv_statepool_cloud_plugin::state_store::{
    LocalFsStateStore, S3StateStore, S3StateStoreConfig, StateStore,
};
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
    #[arg(
        long,
        env = "RWKV_STATEPOOL_LEASE_MAX_TTL_SECONDS",
        default_value_t = 120
    )]
    lease_max_ttl_seconds: u64,
    #[arg(
        long,
        env = "RWKV_STATEPOOL_MAX_STATE_BYTES",
        default_value_t = 536_870_912
    )]
    max_state_bytes: u64,
    #[arg(
        long,
        env = "RWKV_STATEPOOL_STATE_DIR",
        default_value = "var/statepool/states"
    )]
    state_dir: PathBuf,
    #[arg(long, env = "RWKV_STATEPOOL_POSTGRES_URL")]
    postgres_url: Option<String>,
    #[arg(long, env = "RWKV_STATEPOOL_S3_BUCKET")]
    s3_bucket: Option<String>,
    #[arg(long, env = "RWKV_STATEPOOL_S3_REGION", default_value = "us-east-1")]
    s3_region: String,
    #[arg(long, env = "RWKV_STATEPOOL_S3_ENDPOINT")]
    s3_endpoint: Option<String>,
    #[arg(long, env = "RWKV_STATEPOOL_S3_ACCESS_KEY_ID")]
    s3_access_key_id: Option<String>,
    #[arg(long, env = "RWKV_STATEPOOL_S3_SECRET_ACCESS_KEY")]
    s3_secret_access_key: Option<String>,
    #[arg(
        long,
        env = "RWKV_STATEPOOL_S3_PREFIX",
        default_value = "rwkv-statepool"
    )]
    s3_prefix: String,
    #[arg(long, env = "RWKV_STATEPOOL_S3_ALLOW_HTTP", default_value_t = false)]
    s3_allow_http: bool,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    let address: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| format!("invalid listen address: {error}"))?;
    let config = PluginConfig {
        worker_ttl: Duration::from_secs(args.worker_ttl_seconds),
        lease_max_ttl: Duration::from_secs(args.lease_max_ttl_seconds),
        max_state_bytes: args.max_state_bytes,
        state_dir: args.state_dir,
        ..PluginConfig::default()
    };
    let uses_s3 = args.s3_bucket.is_some();
    let state_store: Arc<dyn StateStore> = if let Some(bucket) = args.s3_bucket {
        Arc::new(
            S3StateStore::new(S3StateStoreConfig {
                bucket,
                region: args.s3_region,
                endpoint: args.s3_endpoint,
                access_key_id: args.s3_access_key_id,
                secret_access_key: args.s3_secret_access_key,
                prefix: args.s3_prefix,
                allow_http: args.s3_allow_http,
            })
            .map_err(|error| error.0)?,
        )
    } else {
        Arc::new(LocalFsStateStore::new(config.state_dir.clone()).map_err(|error| error.0)?)
    };
    let uses_postgres = args.postgres_url.is_some();
    let metadata: Arc<dyn MetadataStore> = if let Some(database_url) = args.postgres_url {
        Arc::new(
            PostgresMetadataStore::connect(&database_url)
                .await
                .map_err(|error| error.message)?,
        )
    } else {
        Arc::new(InMemoryMetadataStore::default())
    };
    let object_store_backend = if uses_s3 { "s3" } else { "localfs" };
    let metadata_backend = if uses_postgres {
        "postgresql_lease_cas"
    } else {
        "in_memory_lease_cas"
    };
    let state = PluginState::with_backends(
        config,
        metadata,
        state_store,
        metadata_backend,
        object_store_backend,
    )?;
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
