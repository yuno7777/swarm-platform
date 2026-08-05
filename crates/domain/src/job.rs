//! Job submission, job lifecycle, and the shape of a finished job's result.

use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::agent::Capability;
use crate::error::{Result, SwarmError};
use crate::ids::{CorrelationId, JobId, TaskId};

/// Scheduling priority. Higher priorities are dequeued first and may preempt.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Best-effort background work.
    Low,
    /// The default.
    #[default]
    Normal,
    /// Ahead of normal work.
    High,
    /// Ahead of everything; may preempt running low-priority tasks.
    Critical,
}

impl Priority {
    /// Numeric weight used by queue ordering and the database column.
    #[must_use]
    pub const fn weight(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Critical => "critical",
        };
        f.write_str(label)
    }
}

impl FromStr for Priority {
    type Err = SwarmError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "normal" | "medium" => Ok(Self::Normal),
            "high" => Ok(Self::High),
            "critical" | "urgent" => Ok(Self::Critical),
            other => Err(SwarmError::Config(format!("unknown priority `{other}`"))),
        }
    }
}

/// How a job's work is organised into a task graph.
///
/// The strategy picks a *shape*, not an implementation: each one compiles to a
/// different DAG topology (see `swarm-coordinator::decompose`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStrategy {
    /// One task after another; each sees the previous output.
    Sequential,
    /// Plan, fan out independent work, merge.
    #[default]
    Parallel,
    /// Plan, sub-plan, fan out, merge, verify.
    Hierarchical,
    /// Independent answers, mutual critique, judge picks.
    Debate,
    /// Map over partitions, then reduce.
    MapReduce,
    /// A planner emits work for dedicated executors, then a verifier checks it.
    PlannerExecutor,
    /// A supervisor assigns and reviews worker output.
    SupervisorWorker,
    /// Independent answers, then an explicit consensus round.
    Consensus,
    /// Starts with a plan only; agents insert tasks as they discover work.
    Adaptive,
}

impl ExecutionStrategy {
    /// Every strategy, used by the CLI and by exhaustive tests.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Sequential,
            Self::Parallel,
            Self::Hierarchical,
            Self::Debate,
            Self::MapReduce,
            Self::PlannerExecutor,
            Self::SupervisorWorker,
            Self::Consensus,
            Self::Adaptive,
        ]
    }

    /// Stable label for metrics, persistence, and CLI output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::Hierarchical => "hierarchical",
            Self::Debate => "debate",
            Self::MapReduce => "map_reduce",
            Self::PlannerExecutor => "planner_executor",
            Self::SupervisorWorker => "supervisor_worker",
            Self::Consensus => "consensus",
            Self::Adaptive => "adaptive",
        }
    }

    /// Whether finishing this strategy requires resolving votes between agents.
    #[must_use]
    pub const fn needs_consensus(self) -> bool {
        matches!(self, Self::Debate | Self::Consensus)
    }
}

impl fmt::Display for ExecutionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ExecutionStrategy {
    type Err = SwarmError;

    /// Accepts `map-reduce`, `map_reduce`, and `mapreduce` alike.
    fn from_str(s: &str) -> Result<Self> {
        let normalized: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        Self::all()
            .iter()
            .copied()
            .find(|candidate| {
                candidate.as_str().replace('_', "") == normalized
                    || candidate.as_str() == normalized
            })
            .ok_or_else(|| {
                SwarmError::Config(format!(
                    "unknown execution strategy `{s}` (expected one of: {})",
                    Self::all()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }
}

/// Default agent budget when a submitter does not say.
const fn default_max_agents() -> usize {
    8
}

/// Hard ceiling on `max_agents` for any single job, independent of cluster quotas.
pub const MAX_AGENTS_PER_JOB: usize = 1_000;

/// Longest objective we accept, to keep prompts and log lines bounded.
pub const MAX_OBJECTIVE_LEN: usize = 8_192;

/// A job as submitted by a client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRequest {
    /// What the swarm should accomplish.
    pub objective: String,
    /// Optional background material handed to every agent.
    #[serde(default)]
    pub context: Option<String>,
    /// Scheduling priority.
    #[serde(default)]
    pub priority: Priority,
    /// Wall-clock deadline; past it the job fails rather than continuing to spend.
    #[serde(default)]
    pub deadline: Option<DateTime<Utc>>,
    /// Upper bound on agents this job may hold at once.
    #[serde(default = "default_max_agents")]
    pub max_agents: usize,
    /// Upper bound on model spend, in USD.
    #[serde(default)]
    pub max_cost: Option<f64>,
    /// Upper bound on runtime.
    #[serde(default)]
    pub max_runtime_seconds: Option<u64>,
    /// Capabilities the job needs beyond what its strategy implies.
    #[serde(default)]
    pub required_capabilities: Vec<Capability>,
    /// DAG shape to compile the objective into.
    #[serde(default)]
    pub execution_strategy: ExecutionStrategy,
    /// Client-supplied key that makes resubmission safe.
    #[serde(default)]
    pub idempotency_key: Option<String>,
}

impl JobRequest {
    /// A request with defaults for everything but the objective.
    #[must_use]
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            context: None,
            priority: Priority::default(),
            deadline: None,
            max_agents: default_max_agents(),
            max_cost: None,
            max_runtime_seconds: None,
            required_capabilities: Vec::new(),
            execution_strategy: ExecutionStrategy::default(),
            idempotency_key: None,
        }
    }

    /// Set the DAG shape.
    #[must_use]
    pub fn with_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.execution_strategy = strategy;
        self
    }

    /// Set the agent ceiling.
    #[must_use]
    pub fn with_max_agents(mut self, max_agents: usize) -> Self {
        self.max_agents = max_agents;
        self
    }

    /// Attach background context.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the spend ceiling in USD.
    #[must_use]
    pub fn with_max_cost(mut self, usd: f64) -> Self {
        self.max_cost = Some(usd);
        self
    }

    /// Reject requests that cannot be executed, before any resources are committed.
    ///
    /// This is a trust boundary: the API server calls it on untrusted input.
    pub fn validate(&self) -> Result<()> {
        if self.objective.trim().is_empty() {
            return Err(SwarmError::Validation("objective must not be empty".into()));
        }
        if self.objective.len() > MAX_OBJECTIVE_LEN {
            return Err(SwarmError::Validation(format!(
                "objective is {} bytes, limit is {MAX_OBJECTIVE_LEN}",
                self.objective.len()
            )));
        }
        if self.max_agents == 0 {
            return Err(SwarmError::Validation(
                "max_agents must be at least 1".into(),
            ));
        }
        if self.max_agents > MAX_AGENTS_PER_JOB {
            return Err(SwarmError::QuotaExceeded(format!(
                "max_agents {} exceeds the per-job ceiling of {MAX_AGENTS_PER_JOB}",
                self.max_agents
            )));
        }
        if let Some(cost) = self.max_cost {
            if !cost.is_finite() || cost <= 0.0 {
                return Err(SwarmError::Validation(
                    "max_cost must be a positive, finite number of USD".into(),
                ));
            }
        }
        if self.max_runtime_seconds == Some(0) {
            return Err(SwarmError::Validation(
                "max_runtime_seconds must be greater than zero".into(),
            ));
        }
        if let Some(key) = &self.idempotency_key {
            if key.trim().is_empty() || key.len() > 256 {
                return Err(SwarmError::Validation(
                    "idempotency_key must be 1..=256 non-blank characters".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Lifecycle state of a job. Transitions are enforced in [`crate::state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Accepted by the API, not yet admitted.
    Submitted,
    /// Passed quota and budget admission control.
    Admitted,
    /// Refused by admission control.
    Rejected,
    /// Being decomposed into a task graph.
    Planning,
    /// Tasks are executing.
    Running,
    /// Scheduling suspended by an operator or a budget guard.
    Paused,
    /// All tasks terminal; merging and verifying results.
    Aggregating,
    /// Finished with a verified result.
    Completed,
    /// Finished, but some branches never succeeded.
    PartiallyCompleted,
    /// Could not be completed.
    Failed,
    /// Stopped on request.
    Cancelled,
}

impl JobStatus {
    /// Stable label for metrics and persistence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Admitted => "admitted",
            Self::Rejected => "rejected",
            Self::Planning => "planning",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Aggregating => "aggregating",
            Self::Completed => "completed",
            Self::PartiallyCompleted => "partially_completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether a job in this state has produced everything it ever will.
    #[must_use]
    pub const fn is_finished(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::PartiallyCompleted
                | Self::Failed
                | Self::Cancelled
                | Self::Rejected
        )
    }
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A job record: the request plus everything the platform tracks about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    /// Job identity.
    pub id: JobId,
    /// The original request, kept verbatim for audit and retry.
    pub request: JobRequest,
    /// Current lifecycle state.
    pub status: JobStatus,
    /// Ties every action for this job together in logs and traces.
    pub correlation_id: CorrelationId,
    /// Submission time.
    pub created_at: DateTime<Utc>,
    /// When the first task was queued.
    pub started_at: Option<DateTime<Utc>>,
    /// When the job reached a finished state.
    pub finished_at: Option<DateTime<Utc>>,
    /// Why the job is in its current state, when the reason is not obvious.
    pub status_reason: Option<String>,
}

impl Job {
    /// Create a `Submitted` job from a request.
    #[must_use]
    pub fn new(request: JobRequest) -> Self {
        Self {
            id: JobId::new(),
            request,
            status: JobStatus::Submitted,
            correlation_id: CorrelationId::new(),
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
            status_reason: None,
        }
    }

    /// Whether the job's wall-clock deadline or runtime budget has passed.
    #[must_use]
    pub fn is_past_deadline(&self, now: DateTime<Utc>) -> bool {
        if let Some(deadline) = self.request.deadline {
            if now >= deadline {
                return true;
            }
        }
        if let (Some(limit), Some(started)) = (self.request.max_runtime_seconds, self.started_at) {
            let elapsed = (now - started).num_seconds();
            if elapsed >= 0 && elapsed as u64 >= limit {
                return true;
            }
        }
        false
    }
}

/// A citation supporting a claim: where it came from and how strongly it supports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// URL, file path, tool name, or upstream task id.
    pub source: String,
    /// The claim this evidence is offered for.
    pub claim: String,
    /// How strongly the source supports the claim, `0.0..=1.0`.
    pub support: f32,
    /// Line range, quote offset, or timestamp inside the source.
    pub locator: Option<String>,
}

/// One structured piece of a job's final output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutput {
    /// Task that produced it.
    pub task_id: TaskId,
    /// Task kind, so consumers can tell a critique from a finding.
    pub kind: String,
    /// Human-readable heading.
    pub title: String,
    /// The output text.
    pub content: String,
    /// Machine-readable form of the same output.
    pub data: serde_json::Value,
    /// Confidence attached by the producing agent, `0.0..=1.0`.
    pub confidence: f32,
}

/// A disagreement the platform detected and could not resolve automatically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conflict {
    /// What the disagreement is about.
    pub description: String,
    /// The competing claims, verbatim.
    pub claims: Vec<String>,
    /// Tasks that produced the competing claims.
    pub task_ids: Vec<TaskId>,
    /// How the platform tried to resolve it.
    pub attempted_resolution: String,
}

/// Counters describing how a job actually ran. Also the benchmark record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    /// Total wall-clock time.
    pub wall_clock_ms: u64,
    /// Tasks in the final graph, including dynamically inserted ones.
    pub tasks_total: usize,
    /// Tasks that completed and passed validation.
    pub tasks_succeeded: usize,
    /// Tasks that ended dead-lettered or cancelled.
    pub tasks_failed: usize,
    /// Retry attempts across all tasks.
    pub tasks_retried: usize,
    /// Agents created for this job.
    pub agents_spawned: usize,
    /// Model calls issued, including retries but excluding cache hits.
    pub model_requests: u64,
    /// Cache hits served without a provider call.
    pub model_cache_hits: u64,
    /// Prompt tokens.
    pub tokens_in: u64,
    /// Completion tokens.
    pub tokens_out: u64,
    /// Estimated model spend in USD.
    pub cost_usd: f64,
    /// Median task latency.
    pub median_task_latency_ms: u64,
    /// 95th percentile task latency.
    pub p95_task_latency_ms: u64,
    /// 99th percentile task latency.
    pub p99_task_latency_ms: u64,
    /// Total time tasks spent waiting in the queue.
    pub queue_wait_ms_total: u64,
    /// Scheduling decisions made.
    pub scheduling_decisions: u64,
    /// Events emitted on the job's stream.
    pub messages_sent: u64,
}

impl ExecutionStatistics {
    /// Fill the latency percentiles from a set of observed task latencies.
    ///
    /// Uses nearest-rank on a sorted copy: no interpolation, no surprises.
    pub fn set_latency_percentiles(&mut self, latencies_ms: &[u64]) {
        if latencies_ms.is_empty() {
            return;
        }
        let mut sorted = latencies_ms.to_vec();
        sorted.sort_unstable();
        self.median_task_latency_ms = percentile(&sorted, 0.50);
        self.p95_task_latency_ms = percentile(&sorted, 0.95);
        self.p99_task_latency_ms = percentile(&sorted, 0.99);
    }

    /// Coordination overhead: the share of wall-clock time not spent executing tasks.
    ///
    /// The headline number for the scaling study — it is what grows with swarm size.
    #[must_use]
    pub fn coordination_overhead(&self, total_task_time_ms: u64, parallelism: usize) -> f32 {
        if self.wall_clock_ms == 0 || parallelism == 0 {
            return 0.0;
        }
        let ideal = total_task_time_ms as f32 / parallelism as f32;
        let actual = self.wall_clock_ms as f32;
        ((actual - ideal) / actual).clamp(0.0, 1.0)
    }
}

/// Nearest-rank percentile of a pre-sorted slice.
fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}

/// The verified end product of a job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalResult {
    /// Which job this is the result of.
    pub job_id: JobId,
    /// Terminal status the job reached.
    pub status: JobStatus,
    /// Prose summary of the outcome.
    pub summary: String,
    /// Structured outputs, in dependency order.
    pub outputs: Vec<StructuredOutput>,
    /// Deduplicated evidence behind the summary.
    pub supporting_evidence: Vec<Evidence>,
    /// Aggregate confidence, `0.0..=1.0`.
    pub confidence_score: f32,
    /// Disagreements that survived aggregation.
    pub unresolved_conflicts: Vec<Conflict>,
    /// How the job ran.
    pub execution_statistics: ExecutionStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_orders_low_to_critical() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn strategies_parse_from_every_spelling_operators_use() {
        assert_eq!(
            "map-reduce".parse::<ExecutionStrategy>().unwrap(),
            ExecutionStrategy::MapReduce
        );
        assert_eq!(
            "map_reduce".parse::<ExecutionStrategy>().unwrap(),
            ExecutionStrategy::MapReduce
        );
        assert_eq!(
            "MapReduce".parse::<ExecutionStrategy>().unwrap(),
            ExecutionStrategy::MapReduce
        );
        assert!("teamwork".parse::<ExecutionStrategy>().is_err());
    }

    #[test]
    fn every_strategy_roundtrips_through_its_label() {
        for &strategy in ExecutionStrategy::all() {
            assert_eq!(
                strategy.as_str().parse::<ExecutionStrategy>().unwrap(),
                strategy
            );
        }
    }

    #[test]
    fn validation_rejects_unusable_requests() {
        assert!(JobRequest::new("do a thing").validate().is_ok());
        assert!(JobRequest::new("   ").validate().is_err());
        assert!(JobRequest::new("x").with_max_agents(0).validate().is_err());
        assert!(JobRequest::new("x")
            .with_max_agents(MAX_AGENTS_PER_JOB + 1)
            .validate()
            .is_err());
        assert!(JobRequest::new("x").with_max_cost(0.0).validate().is_err());
        assert!(JobRequest::new("x")
            .with_max_cost(f64::NAN)
            .validate()
            .is_err());
        assert!(JobRequest::new("x".repeat(MAX_OBJECTIVE_LEN + 1))
            .validate()
            .is_err());
    }

    #[test]
    fn requests_deserialize_with_only_an_objective() {
        let request: JobRequest =
            serde_json::from_str(r#"{"objective":"summarise Raft"}"#).unwrap();
        assert_eq!(request.max_agents, default_max_agents());
        assert_eq!(request.priority, Priority::Normal);
        assert_eq!(request.execution_strategy, ExecutionStrategy::Parallel);
        request.validate().unwrap();
    }

    #[test]
    fn deadline_and_runtime_budget_both_expire_a_job() {
        let mut job = Job::new(JobRequest::new("x"));
        let now = Utc::now();
        assert!(!job.is_past_deadline(now));

        job.request.deadline = Some(now - chrono::Duration::seconds(1));
        assert!(job.is_past_deadline(now));

        job.request.deadline = None;
        job.request.max_runtime_seconds = Some(60);
        job.started_at = Some(now - chrono::Duration::seconds(61));
        assert!(job.is_past_deadline(now));

        job.started_at = Some(now - chrono::Duration::seconds(10));
        assert!(!job.is_past_deadline(now));
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let mut stats = ExecutionStatistics::default();
        let latencies: Vec<u64> = (1..=100).collect();
        stats.set_latency_percentiles(&latencies);
        assert_eq!(stats.median_task_latency_ms, 50);
        assert_eq!(stats.p95_task_latency_ms, 95);
        assert_eq!(stats.p99_task_latency_ms, 99);

        // Single sample: every percentile is that sample.
        stats.set_latency_percentiles(&[7]);
        assert_eq!(stats.median_task_latency_ms, 7);
        assert_eq!(stats.p99_task_latency_ms, 7);
    }

    #[test]
    fn coordination_overhead_is_zero_for_a_perfectly_parallel_run() {
        let stats = ExecutionStatistics {
            wall_clock_ms: 1_000,
            ..ExecutionStatistics::default()
        };
        // 10 tasks of 1000ms across 10 agents: ideal is 1000ms, so no overhead.
        assert_eq!(stats.coordination_overhead(10_000, 10), 0.0);
        // Same work on 10 agents but taking 2s: half the wall clock was coordination.
        let slow = ExecutionStatistics {
            wall_clock_ms: 2_000,
            ..ExecutionStatistics::default()
        };
        assert!((slow.coordination_overhead(10_000, 10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn finished_statuses_are_exactly_the_terminal_ones() {
        assert!(JobStatus::Completed.is_finished());
        assert!(JobStatus::Cancelled.is_finished());
        assert!(JobStatus::Rejected.is_finished());
        assert!(!JobStatus::Running.is_finished());
        assert!(!JobStatus::Paused.is_finished());
    }
}
