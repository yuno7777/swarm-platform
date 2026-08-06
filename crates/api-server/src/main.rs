//! `swarm-api` — the platform's HTTP front door.
//!
//! Runs as its own OS process. Start it, then drive it with `swarmctl`, curl, or the
//! dashboard:
//!
//! ```text
//! cargo run -p swarm-api-server -- --bind 127.0.0.1:8080
//! curl -X POST localhost:8080/v1/jobs -H 'content-type: application/json' \
//!      -d '{"objective":"Compare Raft and Paxos","execution_strategy":"debate"}'
//! ```
#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;

use swarm_api_server::{router, AppState};
use swarm_coordinator::{Coordinator, CoordinatorConfig};
use swarm_model_gateway::{Gateway, GatewayConfig, MockProvider, ModelProvider};
use swarm_persistence::FileJournal;

#[derive(Parser, Debug)]
#[command(name = "swarm-api", version, about = "HTTP API for the swarm platform")]
struct Cli {
    /// Address to serve on.
    #[arg(long, default_value = "127.0.0.1:8080", env = "SWARM_BIND")]
    bind: SocketAddr,

    /// Platform configuration file; see `config/example.toml`.
    #[arg(long, env = "SWARM_CONFIG")]
    config: Option<PathBuf>,

    /// Append-only journal file. With one, jobs survive a restart of this process;
    /// without one, they live only as long as it does.
    #[arg(long, env = "SWARM_JOURNAL")]
    journal: Option<PathBuf>,

    /// Default log level when `RUST_LOG` is unset.
    #[arg(long, default_value = "info")]
    log: String,
}

/// The parts of the platform a configuration file may set.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PlatformConfig {
    coordinator: CoordinatorConfig,
    gateway: GatewayConfig,
}

impl PlatformConfig {
    fn load(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log.clone().into()),
        )
        .with_target(false)
        .init();

    let config = PlatformConfig::load(cli.config.as_deref())?;

    // Phase 1 infrastructure: deterministic mock provider, in-process queue and
    // memory. Phase 2 swaps these for real providers and durable backends without
    // touching anything above the traits.
    let provider: Arc<dyn ModelProvider> = Arc::new(MockProvider::new("mock"));
    let gateway = Arc::new(Gateway::new(config.gateway).and_provider(provider));
    let mut coordinator = Coordinator::local_with(config.coordinator, gateway);

    if let Some(path) = &cli.journal {
        let journal = Arc::new(
            FileJournal::open(path)
                .with_context(|| format!("opening journal {}", path.display()))?,
        );
        coordinator = coordinator.with_journal(journal);

        // Recover before serving, so a client never sees a job that is about to be
        // resurrected under it.
        let recovered = coordinator.recover().context("recovering from journal")?;
        if recovered > 0 {
            println!("recovered {recovered} jobs from {}", path.display());
        }
    }
    let coordinator = Arc::new(coordinator);

    let listener = tokio::net::TcpListener::bind(cli.bind)
        .await
        .with_context(|| format!("binding {}", cli.bind))?;
    let bound = listener.local_addr()?;

    tracing::info!(address = %bound, "swarm-api listening");
    println!("swarm-api listening on http://{bound}");

    axum::serve(listener, router(AppState::new(coordinator)))
        .with_graceful_shutdown(shutdown())
        .await
        .context("serving")?;

    tracing::info!("shut down cleanly");
    Ok(())
}

/// Resolve on Ctrl-C, so in-flight requests finish instead of being cut off.
async fn shutdown() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to listen for shutdown signal");
        std::future::pending::<()>().await;
    }
    tracing::info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn a_partial_config_file_inherits_defaults() {
        let config: PlatformConfig = toml::from_str("[coordinator]\nlease_ms = 1234\n").unwrap();
        assert_eq!(config.coordinator.lease_ms, 1234);
        assert_eq!(
            config.coordinator.max_cluster_agents,
            CoordinatorConfig::default().max_cluster_agents
        );
        assert_eq!(config.gateway, GatewayConfig::default());
    }
}
