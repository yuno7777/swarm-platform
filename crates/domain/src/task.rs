//! Tasks: the unit of scheduling, execution, retry, and validation.

use std::collections::HashSet;
use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::agent::Capability;
use crate::ids::{AgentId, JobId, TaskId};
use crate::job::Evidence;

/// What a task is for. Fixes its capabilities, prompt, and default validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Decompose the objective.
    Plan,
    /// Decompose one branch of an existing plan.
    Subplan,
    /// Do a unit of the actual work.
    Work,
    /// Process one partition of a map-reduce.
    Map,
    /// Combine map outputs.
    Reduce,
    /// Answer independently, for debate or consensus.
    Answer,
    /// Critique peers' answers.
    Critique,
    /// Check an output against criteria.
    Verify,
    /// Choose between competing answers.
    Judge,
    /// Cast a vote in a consensus round.
    Vote,
    /// Merge validated outputs.
    Merge,
    /// Condense outputs into prose.
    Summarize,
    /// Assign and review worker output.
    Supervise,
}

impl TaskKind {
    /// Capabilities a task of this kind requires of its agent.
    #[must_use]
    pub const fn required_capabilities(self) -> &'static [Capability] {
        match self {
            Self::Plan | Self::Subplan | Self::Supervise => &[Capability::Planning],
            Self::Work | Self::Map | Self::Answer => &[Capability::Research],
            Self::Reduce | Self::Merge => &[Capability::Merging],
            Self::Critique => &[Capability::Review],
            Self::Verify | Self::Vote => &[Capability::Verification],
            Self::Judge => &[Capability::Review, Capability::Verification],
            Self::Summarize => &[Capability::Summarization],
        }
    }

    /// Stable label for prompts, metrics, and persistence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Subplan => "subplan",
            Self::Work => "work",
            Self::Map => "map",
            Self::Reduce => "reduce",
            Self::Answer => "answer",
            Self::Critique => "critique",
            Self::Verify => "verify",
            Self::Judge => "judge",
            Self::Vote => "vote",
            Self::Merge => "merge",
            Self::Summarize => "summarize",
            Self::Supervise => "supervise",
        }
    }

    /// Whether this kind's output belongs in the final result rather than being
    /// intermediate scaffolding.
    #[must_use]
    pub const fn is_reportable(self) -> bool {
        matches!(
            self,
            Self::Work
                | Self::Map
                | Self::Reduce
                | Self::Answer
                | Self::Judge
                | Self::Merge
                | Self::Summarize
        )
    }
}

impl fmt::Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Lifecycle state of a task. Transitions are enforced in [`crate::state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// In the graph, not yet eligible.
    Created,
    /// Eligible except that an upstream task has not finished.
    WaitingForDependency,
    /// In the queue, waiting for a worker.
    Queued,
    /// Claimed by a worker under a lease, not yet started.
    Leased,
    /// Executing.
    Running,
    /// Finished and passed validation.
    Completed,
    /// Execution failed.
    Failed,
    /// Exceeded its timeout.
    TimedOut,
    /// Will be re-queued once its backoff elapses.
    RetryScheduled,
    /// Stopped to free capacity for higher-priority work.
    Preempted,
    /// Abandoned because its job was cancelled.
    Cancelled,
    /// Out of attempts; parked for human inspection.
    DeadLettered,
}

impl TaskState {
    /// Stable label for metrics and persistence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::WaitingForDependency => "waiting_for_dependency",
            Self::Queued => "queued",
            Self::Leased => "leased",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::RetryScheduled => "retry_scheduled",
            Self::Preempted => "preempted",
            Self::Cancelled => "cancelled",
            Self::DeadLettered => "dead_lettered",
        }
    }

    /// Whether the task occupies an agent right now.
    #[must_use]
    pub const fn is_in_flight(self) -> bool {
        matches!(self, Self::Leased | Self::Running)
    }
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How many times and how patiently a failed task is retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Total attempts allowed, including the first.
    pub max_attempts: u32,
    /// First backoff, doubled per attempt.
    pub backoff_base_ms: u64,
    /// Ceiling on the backoff.
    pub backoff_max_ms: u64,
    /// Whether to spread retries out to avoid synchronised thundering herds.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_base_ms: 500,
            backoff_max_ms: 60_000,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries — for tasks whose side effects are not idempotent.
    #[must_use]
    pub const fn no_retries() -> Self {
        Self {
            max_attempts: 1,
            backoff_base_ms: 0,
            backoff_max_ms: 0,
            jitter: false,
        }
    }

    /// Whether another attempt is permitted after `attempts_used` attempts.
    #[must_use]
    pub const fn allows_retry(&self, attempts_used: u32) -> bool {
        attempts_used < self.max_attempts
    }

    /// Backoff before attempt number `attempt` (1-based), capped and optionally jittered.
    ///
    /// `nonce` supplies the jitter so this function stays pure and testable; callers
    /// pass a clock or counter value.
    #[must_use]
    pub const fn backoff_ms(&self, attempt: u32, nonce: u64) -> u64 {
        if self.backoff_base_ms == 0 {
            return 0;
        }
        let shift = if attempt > 20 {
            20
        } else {
            attempt.saturating_sub(1)
        };
        let raw = self.backoff_base_ms.saturating_mul(1u64 << shift);
        let capped = if raw > self.backoff_max_ms {
            self.backoff_max_ms
        } else {
            raw
        };
        if !self.jitter || capped == 0 {
            return capped;
        }
        // Full jitter in [capped/2, capped]: keeps a floor on the wait while still
        // decorrelating retries across the swarm.
        let half = capped / 2;
        half + (nonce % (capped - half + 1))
    }
}

/// A mechanical check applied to a task's output before it counts as complete.
///
/// Deterministic rules run first and gate everything; LLM critics only see output
/// that already passed here (ADR-7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationRule {
    /// Output must contain non-whitespace text.
    NonEmpty,
    /// Output must contain at least this many whitespace-separated words.
    MinWords(usize),
    /// Output must contain this substring, case-insensitively.
    MustMention(String),
    /// Output must parse as a JSON object containing these keys.
    RequiredJsonKeys(Vec<String>),
}

impl ValidationRule {
    /// Check `output`, returning a human-readable failure description when it fails.
    #[must_use]
    pub fn check(&self, output: &str) -> Option<String> {
        match self {
            Self::NonEmpty => output
                .trim()
                .is_empty()
                .then(|| "output is empty".to_owned()),
            Self::MinWords(minimum) => {
                let words = output.split_whitespace().count();
                (words < *minimum).then(|| format!("output has {words} words, needs {minimum}"))
            }
            Self::MustMention(needle) => {
                let found = output.to_lowercase().contains(&needle.to_lowercase());
                (!found).then(|| format!("output does not mention `{needle}`"))
            }
            Self::RequiredJsonKeys(keys) => match serde_json::from_str::<serde_json::Value>(output)
            {
                Ok(serde_json::Value::Object(map)) => {
                    let missing: Vec<&str> = keys
                        .iter()
                        .filter(|key| !map.contains_key(key.as_str()))
                        .map(String::as_str)
                        .collect();
                    (!missing.is_empty())
                        .then(|| format!("JSON output is missing keys: {}", missing.join(", ")))
                }
                Ok(_) => Some("output is JSON but not an object".to_owned()),
                Err(e) => Some(format!("output is not valid JSON: {e}")),
            },
        }
    }
}

/// Run every rule and collect the failures.
#[must_use]
pub fn validate_output(output: &str, rules: &[ValidationRule]) -> Vec<String> {
    rules.iter().filter_map(|rule| rule.check(output)).collect()
}

/// One node of a job's task DAG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskNode {
    /// Task identity.
    pub id: TaskId,
    /// Owning job.
    pub job_id: JobId,
    /// Short heading.
    pub title: String,
    /// The instruction handed to the agent.
    pub description: String,
    /// What this task is for.
    pub kind: TaskKind,
    /// Tasks that must complete first.
    pub dependencies: Vec<TaskId>,
    /// Capabilities an agent must have to run it.
    pub required_capabilities: Vec<Capability>,
    /// Relative size estimate, drives fan-out and token budgets.
    pub estimated_complexity: u32,
    /// Predicted token consumption, for cost admission control.
    pub estimated_tokens: Option<u64>,
    /// Retry behaviour.
    pub retry_policy: RetryPolicy,
    /// Execution timeout.
    pub timeout_seconds: u64,
    /// Lifecycle state.
    pub state: TaskState,
    /// Attempts already used.
    pub attempt: u32,
    /// Stage index in the compiled strategy, used for ordering outputs.
    pub stage: u32,
    /// Key that makes duplicate delivery of this task harmless.
    pub idempotency_key: String,
    /// Mechanical checks the output must pass.
    pub validation: Vec<ValidationRule>,
}

impl TaskNode {
    /// A `Created` task of `kind` belonging to `job_id`.
    ///
    /// The idempotency key is derived from the job, the title, and the stage, so the
    /// same logical task always produces the same key even if it is rebuilt after a
    /// coordinator restart.
    #[must_use]
    pub fn new(
        job_id: JobId,
        kind: TaskKind,
        title: impl Into<String>,
        description: impl Into<String>,
        stage: u32,
    ) -> Self {
        let title = title.into();
        let id = TaskId::new();
        Self {
            // The slug is for humans reading logs; the hash is what actually
            // guarantees uniqueness, because slugs are truncated and two long titles
            // can easily share their first 48 characters.
            idempotency_key: format!(
                "{job_id}:{stage}:{}:{}:{:016x}",
                kind.as_str(),
                slug(&title),
                crate::hash::stable_hash(title.as_bytes())
            ),
            id,
            job_id,
            title,
            description: description.into(),
            kind,
            dependencies: Vec::new(),
            required_capabilities: kind.required_capabilities().to_vec(),
            estimated_complexity: 1,
            estimated_tokens: None,
            retry_policy: RetryPolicy::default(),
            timeout_seconds: 300,
            state: TaskState::Created,
            attempt: 0,
            stage,
            validation: vec![ValidationRule::NonEmpty],
        }
    }

    /// Declare dependencies.
    #[must_use]
    pub fn with_dependencies(mut self, dependencies: Vec<TaskId>) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// Override the mechanical checks.
    #[must_use]
    pub fn with_validation(mut self, validation: Vec<ValidationRule>) -> Self {
        self.validation = validation;
        self
    }

    /// Set the size estimate and the derived token estimate.
    #[must_use]
    pub fn with_complexity(mut self, complexity: u32) -> Self {
        self.estimated_complexity = complexity.max(1);
        self.estimated_tokens = Some(u64::from(self.estimated_complexity) * 400);
        self
    }

    /// Override retry behaviour.
    #[must_use]
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Override the execution timeout.
    #[must_use]
    pub fn with_timeout_seconds(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Require extra capabilities on top of the kind's defaults.
    #[must_use]
    pub fn requiring(mut self, extra: &[Capability]) -> Self {
        for capability in extra {
            if !self.required_capabilities.contains(capability) {
                self.required_capabilities.push(*capability);
            }
        }
        self
    }

    /// Whether the task will never change state again.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TaskState::Completed | TaskState::Cancelled | TaskState::DeadLettered
        )
    }

    /// Whether every dependency is in `completed`.
    #[must_use]
    pub fn dependencies_satisfied(&self, completed: &HashSet<TaskId>) -> bool {
        self.dependencies.iter().all(|dep| completed.contains(dep))
    }

    /// The idempotency key for a specific attempt.
    ///
    /// Attempts get distinct keys so a retry is allowed to do work, while a *duplicate
    /// delivery of the same attempt* is recognised and skipped.
    #[must_use]
    pub fn attempt_key(&self) -> String {
        format!("{}#{}", self.idempotency_key, self.attempt)
    }
}

/// Lowercase, hyphenated, ASCII-only form of `input`, truncated for use in keys.
fn slug(input: &str) -> String {
    let mut out = String::with_capacity(input.len().min(48));
    let mut last_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
        if out.len() >= 48 {
            break;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// The outcome of one successful task attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task that produced this.
    pub task_id: TaskId,
    /// Owning job.
    pub job_id: JobId,
    /// Agent that produced it.
    pub agent_id: AgentId,
    /// Attempt number, 1-based.
    pub attempt: u32,
    /// Task kind, copied so consumers do not need the graph.
    pub kind: TaskKind,
    /// Task title, copied for the same reason.
    pub title: String,
    /// Raw output text.
    pub output: String,
    /// Parsed structured form of the output.
    pub structured: serde_json::Value,
    /// Citations the agent offered.
    pub evidence: Vec<Evidence>,
    /// Short summary of the agent's approach (never raw chain-of-thought).
    pub reasoning_summary: String,
    /// Agent-reported confidence, `0.0..=1.0`.
    pub confidence: f32,
    /// Mechanical checks that failed; empty means the output is acceptable.
    pub validation_failures: Vec<String>,
    /// Prompt tokens consumed.
    pub tokens_in: u64,
    /// Completion tokens produced.
    pub tokens_out: u64,
    /// Estimated spend for this attempt.
    pub cost_usd: f64,
    /// Execution time.
    pub duration_ms: u64,
    /// Whether the result was served from a prior identical attempt.
    pub deduplicated: bool,
    /// When the attempt finished.
    pub finished_at: DateTime<Utc>,
}

impl TaskResult {
    /// Whether every mechanical check passed.
    #[must_use]
    pub fn passed_validation(&self) -> bool {
        self.validation_failures.is_empty()
    }
}

/// A failed attempt, kept for the failure inspection API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskFailure {
    /// Task that failed.
    pub task_id: TaskId,
    /// Task title, for readable output.
    pub title: String,
    /// Attempt number, 1-based.
    pub attempt: u32,
    /// Machine-readable error class.
    pub error_kind: String,
    /// Human-readable detail.
    pub error_message: String,
    /// Whether the task is out of attempts.
    pub dead_lettered: bool,
    /// Mechanical checks that failed, when that was the cause.
    pub validation_failures: Vec<String>,
    /// When it happened.
    pub at: DateTime<Utc>,
}

/// A resumable snapshot of partial progress inside a task attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Task the checkpoint belongs to.
    pub task_id: TaskId,
    /// Attempt that wrote it.
    pub attempt: u32,
    /// Monotonic sequence within the attempt.
    pub seq: u32,
    /// Short label, e.g. `started`, `partial_answer`.
    pub label: String,
    /// The resumable state.
    pub state: serde_json::Value,
    /// When it was written.
    pub created_at: DateTime<Utc>,
}

impl Checkpoint {
    /// Build a checkpoint stamped with the current time.
    #[must_use]
    pub fn new(
        task_id: TaskId,
        attempt: u32,
        seq: u32,
        label: impl Into<String>,
        state: serde_json::Value,
    ) -> Self {
        Self {
            task_id,
            attempt,
            seq,
            label: label.into(),
            state,
            created_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> TaskNode {
        TaskNode::new(
            JobId::new(),
            TaskKind::Work,
            "Investigate Raft leader election",
            "Explain how Raft elects a leader.",
            1,
        )
    }

    #[test]
    fn backoff_doubles_then_caps() {
        let policy = RetryPolicy {
            max_attempts: 10,
            backoff_base_ms: 100,
            backoff_max_ms: 800,
            jitter: false,
        };
        assert_eq!(policy.backoff_ms(1, 0), 100);
        assert_eq!(policy.backoff_ms(2, 0), 200);
        assert_eq!(policy.backoff_ms(3, 0), 400);
        assert_eq!(policy.backoff_ms(4, 0), 800);
        assert_eq!(policy.backoff_ms(9, 0), 800, "must stay capped");
        assert_eq!(policy.backoff_ms(99, 0), 800, "must not overflow");
    }

    #[test]
    fn jitter_stays_within_half_the_backoff_window() {
        let policy = RetryPolicy {
            max_attempts: 5,
            backoff_base_ms: 1_000,
            backoff_max_ms: 60_000,
            jitter: true,
        };
        for nonce in 0..1_000 {
            let wait = policy.backoff_ms(1, nonce);
            assert!(
                (500..=1_000).contains(&wait),
                "jittered wait {wait} left the window"
            );
        }
    }

    #[test]
    fn retry_budget_is_respected() {
        let policy = RetryPolicy::default();
        assert!(policy.allows_retry(0));
        assert!(policy.allows_retry(2));
        assert!(!policy.allows_retry(3));
        assert!(!RetryPolicy::no_retries().allows_retry(1));
    }

    #[test]
    fn validation_rules_catch_what_they_claim_to() {
        assert!(ValidationRule::NonEmpty.check("text").is_none());
        assert!(ValidationRule::NonEmpty.check("   \n").is_some());

        assert!(ValidationRule::MinWords(3).check("one two three").is_none());
        assert!(ValidationRule::MinWords(3).check("one two").is_some());

        let mentions = ValidationRule::MustMention("Raft".into());
        assert!(mentions.check("we used raft consensus").is_none());
        assert!(mentions.check("we used paxos").is_some());

        let keys = ValidationRule::RequiredJsonKeys(vec!["summary".into(), "findings".into()]);
        assert!(keys.check(r#"{"summary":"s","findings":[]}"#).is_none());
        assert!(keys.check(r#"{"summary":"s"}"#).is_some());
        assert!(keys.check("[1,2]").is_some());
        assert!(keys.check("not json").is_some());
    }

    #[test]
    fn validate_output_reports_every_failure_not_just_the_first() {
        let failures = validate_output(
            "short",
            &[
                ValidationRule::MinWords(10),
                ValidationRule::MustMention("Raft".into()),
            ],
        );
        assert_eq!(failures.len(), 2);
    }

    #[test]
    fn idempotency_keys_are_stable_per_task_and_distinct_per_attempt() {
        let job_id = JobId::new();
        let first = TaskNode::new(job_id, TaskKind::Work, "Do the thing", "…", 1);
        let rebuilt = TaskNode::new(job_id, TaskKind::Work, "Do the thing", "…", 1);
        assert_ne!(first.id, rebuilt.id, "ids differ");
        assert_eq!(
            first.idempotency_key, rebuilt.idempotency_key,
            "the same logical task must keep its key across a rebuild"
        );

        let mut retried = first.clone();
        retried.attempt = 1;
        assert_ne!(first.attempt_key(), retried.attempt_key());
    }

    #[test]
    fn titles_sharing_a_long_prefix_still_get_distinct_keys() {
        // Slugs are truncated, so uniqueness cannot rest on them alone.
        let job_id = JobId::new();
        let prefix = "Investigate consensus, replication, and recovery in distributed systems";
        let first = TaskNode::new(
            job_id,
            TaskKind::Work,
            format!("{prefix} — aspect 1"),
            "",
            1,
        );
        let second = TaskNode::new(
            job_id,
            TaskKind::Work,
            format!("{prefix} — aspect 2"),
            "",
            1,
        );
        assert_ne!(first.idempotency_key, second.idempotency_key);
    }

    #[test]
    fn slugs_are_bounded_and_free_of_stray_separators() {
        assert_eq!(
            slug("Investigate Raft leader election"),
            "investigate-raft-leader-election"
        );
        assert_eq!(slug("  spaces  "), "spaces");
        assert_eq!(slug("!!!"), "");
        assert!(slug(&"word ".repeat(100)).len() <= 48);
    }

    #[test]
    fn dependencies_gate_readiness() {
        let mut node = task();
        let upstream = TaskId::new();
        node.dependencies = vec![upstream];

        let mut completed = HashSet::new();
        assert!(!node.dependencies_satisfied(&completed));
        completed.insert(upstream);
        assert!(node.dependencies_satisfied(&completed));
    }

    #[test]
    fn work_kinds_are_reportable_and_scaffolding_is_not() {
        assert!(TaskKind::Work.is_reportable());
        assert!(TaskKind::Merge.is_reportable());
        assert!(!TaskKind::Plan.is_reportable());
        assert!(!TaskKind::Vote.is_reportable());
    }

    #[test]
    fn complexity_drives_the_token_estimate() {
        let node = task().with_complexity(5);
        assert_eq!(node.estimated_complexity, 5);
        assert_eq!(node.estimated_tokens, Some(2_000));
        assert_eq!(task().with_complexity(0).estimated_complexity, 1);
    }
}
