//! `swarmctl` — submit jobs to a swarm and watch them run.
//!
//! In Phase 1 the CLI hosts the coordinator in-process against the mock provider, so
//! the whole platform runs with `cargo run -p swarm-admin-cli`. From Phase 2 the same
//! commands talk to a remote coordinator over gRPC; the flags do not change.
#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use swarm_coordinator::{Coordinator, CoordinatorConfig, JobEvent, JobEventKind, SchedulerKind};
use swarm_domain::{ExecutionStrategy, FinalResult, JobRequest, Priority};
use swarm_model_gateway::{Gateway, GatewayConfig, MockProvider, ModelProvider};

#[derive(Parser, Debug)]
#[command(
    name = "swarmctl",
    version,
    about = "Submit and inspect distributed multi-agent swarm jobs"
)]
struct Cli {
    /// Log level for the platform's structured logs.
    #[arg(long, global = true, default_value = "warn")]
    log: String,

    /// Platform configuration file. Defaults to `$SWARM_CONFIG`, then to built-in
    /// defaults; see `config/example.toml`.
    #[arg(long, global = true, env = "SWARM_CONFIG")]
    config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Command,
}

/// The parts of the platform a configuration file may set.
///
/// Every field is optional: a file that sets one knob inherits defaults for the rest.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PlatformConfig {
    coordinator: CoordinatorConfig,
    gateway: GatewayConfig,
}

impl PlatformConfig {
    fn load(path: Option<&std::path::Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a job and stream its progress.
    Run(RunArgs),
    /// Run one of the built-in demonstration workflows.
    Demo {
        /// Which demo to run.
        #[arg(value_enum, default_value_t = Demo::Research)]
        which: Demo,
        /// Agent budget for the demo.
        #[arg(long, default_value_t = 6)]
        agents: usize,
    },
    /// List the execution and scheduling strategies this build supports.
    Strategies,
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// What the swarm should accomplish.
    #[arg(long)]
    objective: String,

    /// Background material handed to every agent.
    #[arg(long)]
    context: Option<String>,

    /// DAG shape to compile the objective into.
    #[arg(long, default_value = "parallel")]
    strategy: String,

    /// Maximum — and target — number of agents.
    #[arg(long, default_value_t = 6)]
    agents: usize,

    /// Scheduling priority: low, normal, high, or critical.
    #[arg(long, default_value = "normal")]
    priority: String,

    /// Placement strategy.
    #[arg(long, value_enum, default_value_t = Placement::CapabilityWeighted)]
    placement: Placement,

    /// Spend ceiling in USD.
    #[arg(long)]
    max_cost: Option<f64>,

    /// Wall-clock ceiling in seconds.
    #[arg(long)]
    max_runtime: Option<u64>,

    /// Print the final result as JSON instead of prose.
    #[arg(long)]
    json: bool,

    /// Do not stream per-task events.
    #[arg(long)]
    quiet: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Placement {
    RoundRobin,
    LeastLoaded,
    CapabilityWeighted,
    Adaptive,
}

impl From<Placement> for SchedulerKind {
    fn from(placement: Placement) -> Self {
        match placement {
            Placement::RoundRobin => Self::RoundRobin,
            Placement::LeastLoaded => Self::LeastLoaded,
            Placement::CapabilityWeighted => Self::CapabilityWeighted,
            Placement::Adaptive => Self::Adaptive,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Demo {
    /// A planner fans research out, critics check it, a merger concludes.
    Research,
    /// A planner splits a feature, coders implement, reviewers and tests verify.
    Codegen,
    /// Agents answer independently, critique each other, a judge decides.
    Debate,
    /// All three, in order.
    All,
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

    match cli.command {
        Command::Run(args) => run(args, config).await,
        Command::Demo { which, agents } => demo(which, agents, config).await,
        Command::Strategies => {
            strategies();
            Ok(())
        }
    }
}

/// Build the Phase 1 in-process platform: mock provider, in-memory queue and memory.
fn platform(placement: SchedulerKind, config: &PlatformConfig) -> Arc<Coordinator> {
    let provider: Arc<dyn ModelProvider> = Arc::new(MockProvider::new("mock"));
    let gateway = Arc::new(Gateway::new(config.gateway.clone()).and_provider(provider));
    Arc::new(Coordinator::local_with(
        CoordinatorConfig {
            scheduler: placement,
            ..config.coordinator.clone()
        },
        gateway,
    ))
}

async fn run(args: RunArgs, config: PlatformConfig) -> Result<()> {
    let strategy: ExecutionStrategy = args
        .strategy
        .parse()
        .with_context(|| format!("unknown strategy `{}`", args.strategy))?;
    let priority: Priority = args
        .priority
        .parse()
        .with_context(|| format!("unknown priority `{}`", args.priority))?;

    let mut request = JobRequest::new(&args.objective)
        .with_strategy(strategy)
        .with_max_agents(args.agents)
        .with_priority(priority);
    request.context = args.context;
    request.max_cost = args.max_cost;
    request.max_runtime_seconds = args.max_runtime;

    let coordinator = platform(args.placement.into(), &config);
    execute(&coordinator, request, args.json, args.quiet).await
}

async fn demo(which: Demo, agents: usize, config: PlatformConfig) -> Result<()> {
    let demos: Vec<(&str, JobRequest)> = match which {
        Demo::Research => vec![("Parallel research swarm", research_demo(agents))],
        Demo::Codegen => vec![("Distributed code-generation swarm", codegen_demo(agents))],
        Demo::Debate => vec![("Debate and consensus swarm", debate_demo(agents))],
        Demo::All => vec![
            ("Parallel research swarm", research_demo(agents)),
            ("Distributed code-generation swarm", codegen_demo(agents)),
            ("Debate and consensus swarm", debate_demo(agents)),
        ],
    };

    for (title, request) in demos {
        println!("\n\x1b[1m=== {title} ===\x1b[0m");
        println!("objective: {}\n", request.objective);
        let coordinator = platform(config.coordinator.scheduler, &config);
        execute(&coordinator, request, false, false).await?;
    }
    Ok(())
}

fn research_demo(agents: usize) -> JobRequest {
    JobRequest::new(
        "Survey how distributed systems achieve consensus, replicate state, \
         detect failures, and recover from partitions",
    )
    .with_strategy(ExecutionStrategy::Hierarchical)
    .with_max_agents(agents)
}

fn codegen_demo(agents: usize) -> JobRequest {
    JobRequest::new(
        "Implement a rate limiter: token bucket core, a middleware layer, \
         persistence, and a benchmark harness",
    )
    .with_strategy(ExecutionStrategy::PlannerExecutor)
    .with_max_agents(agents)
}

fn debate_demo(agents: usize) -> JobRequest {
    JobRequest::new("Is leader-based consensus simpler to operate than leaderless consensus?")
        .with_strategy(ExecutionStrategy::Debate)
        .with_max_agents(agents)
}

/// Submit, stream events, and report.
async fn execute(
    coordinator: &Arc<Coordinator>,
    request: JobRequest,
    as_json: bool,
    quiet: bool,
) -> Result<()> {
    let job_id = coordinator.submit(request)?;
    let state = coordinator.state(job_id)?;
    if !as_json {
        println!(
            "job {job_id}\nstrategy {} · {} tasks planned",
            state.execution_strategy, state.counts.total
        );
    }

    let printer = (!quiet && !as_json).then(|| {
        let mut events = coordinator
            .subscribe(job_id)
            .expect("the job was just submitted");
        tokio::spawn(async move {
            while let Ok(event) = events.recv().await {
                print_event(&event);
            }
        })
    });

    // Ctrl-C cancels the job rather than killing the process mid-write.
    let result = tokio::select! {
        result = coordinator.run(job_id) => result?,
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\ninterrupted; cancelling job {job_id}");
            coordinator.cancel(job_id).await?;
            tokio::time::sleep(Duration::from_millis(200)).await;
            coordinator
                .final_result(job_id)?
                .context("job was cancelled before producing a result")?
        }
    };

    if let Some(printer) = printer {
        printer.abort();
    }

    if as_json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        report(&result);
    }
    Ok(())
}

fn print_event(event: &JobEvent) {
    let (colour, label) = match event.kind {
        JobEventKind::TaskCompleted => ("\x1b[32m", "done"),
        JobEventKind::TaskFailed | JobEventKind::TaskDeadLettered => ("\x1b[31m", "fail"),
        JobEventKind::TaskRetrying => ("\x1b[33m", "retry"),
        JobEventKind::TaskStarted => ("\x1b[36m", "start"),
        JobEventKind::TaskQueued => ("\x1b[90m", "queue"),
        JobEventKind::AgentSpawned => ("\x1b[35m", "agent"),
        JobEventKind::JobFinished => ("\x1b[1m", "job"),
        _ => ("\x1b[90m", "info"),
    };
    println!(
        "{colour}[{:>3.0}%] {label:>5}\x1b[0m {}",
        event.progress * 100.0,
        event.detail
    );
}

fn report(result: &FinalResult) {
    let statistics = &result.execution_statistics;
    println!("\n\x1b[1mResult\x1b[0m ({})", result.status);
    println!("{}\n", result.summary);

    if !result.unresolved_conflicts.is_empty() {
        println!("\x1b[33mUnresolved conflicts:\x1b[0m");
        for conflict in &result.unresolved_conflicts {
            println!("  - {}", conflict.description);
        }
        println!();
    }

    println!("\x1b[1mExecution\x1b[0m");
    println!(
        "  tasks         {}/{} succeeded ({} retried, {} failed)",
        statistics.tasks_succeeded,
        statistics.tasks_total,
        statistics.tasks_retried,
        statistics.tasks_failed
    );
    println!("  agents        {}", statistics.agents_spawned);
    println!(
        "  latency       p50 {}ms · p95 {}ms · p99 {}ms",
        statistics.median_task_latency_ms,
        statistics.p95_task_latency_ms,
        statistics.p99_task_latency_ms
    );
    println!("  wall clock    {}ms", statistics.wall_clock_ms);
    println!(
        "  tokens        {} in · {} out",
        statistics.tokens_in, statistics.tokens_out
    );
    println!("  cost          ${:.4}", statistics.cost_usd);
    println!("  confidence    {:.2}", result.confidence_score);
    println!("  outputs       {}", result.outputs.len());
    println!("  evidence      {}", result.supporting_evidence.len());
}

fn strategies() {
    println!("\x1b[1mExecution strategies\x1b[0m (--strategy)");
    for strategy in ExecutionStrategy::all() {
        println!("  {}", strategy.as_str());
    }
    println!("\n\x1b[1mPlacement strategies\x1b[0m (--placement)");
    for scheduler in SchedulerKind::all() {
        println!("  {}", scheduler.build().name());
    }
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
    fn strategy_and_priority_flags_parse_the_way_the_help_claims() {
        assert_eq!(
            "map-reduce".parse::<ExecutionStrategy>().unwrap(),
            ExecutionStrategy::MapReduce
        );
        assert_eq!("critical".parse::<Priority>().unwrap(), Priority::Critical);
        assert!("nonsense".parse::<ExecutionStrategy>().is_err());
    }

    #[test]
    fn every_demo_builds_a_valid_request() {
        for request in [research_demo(6), codegen_demo(6), debate_demo(6)] {
            request.validate().unwrap();
        }
    }

    #[test]
    fn placement_flags_cover_every_scheduler() {
        for placement in [
            Placement::RoundRobin,
            Placement::LeastLoaded,
            Placement::CapabilityWeighted,
            Placement::Adaptive,
        ] {
            let kind: SchedulerKind = placement.into();
            assert!(SchedulerKind::all().contains(&kind));
        }
    }
}
