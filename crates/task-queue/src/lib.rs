//! The distributed task queue abstraction.
//!
//! Delivery is **at-least-once**. Exactly-once delivery is not achievable across a
//! queue, a crashing worker, and a remote model API, so instead every queued task
//! carries an idempotency key and execution is made effectively-once one layer up
//! (see `swarm-agent-runtime`).
//!
//! Ownership of a task is a **lease with an expiry**, not a worker identity. Recovery
//! is therefore a single sweep — [`TaskQueue::requeue_expired`] — that needs only a
//! clock, not a consistent view of which workers are alive.
//!
//! [`InMemoryQueue`] implements the full contract (priorities, delays, leases, expiry,
//! deduplication, retry backoff, dead letters) rather than being a test double, so the
//! Redis Streams and NATS JetStream implementations added in Phase 2 can be validated
//! against exactly the same behaviour.
#![forbid(unsafe_code)]

pub mod memory;

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use swarm_domain::{
    Capability, JobId, LeaseId, LeaseState, Priority, Result, RetryPolicy, TaskId, TaskNode,
};

pub use memory::InMemoryQueue;

/// Identity of the process pulling from the queue, used for lease attribution.
pub type WorkerId = String;

/// The queue every task lands in unless the caller names another.
pub const DEFAULT_QUEUE: &str = "default";

/// A task as it sits in the queue: enough to execute it, plus the metadata that makes
/// duplicate delivery and retry safe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueuedTask {
    /// Task this entry represents.
    pub task_id: TaskId,
    /// Owning job, so a cancelled job can be purged in one pass.
    pub job_id: JobId,
    /// Named queue, e.g. `default`, `high-memory`, `gpu`.
    pub queue: String,
    /// Scheduling priority.
    pub priority: Priority,
    /// The task definition, serialized.
    pub payload: serde_json::Value,
    /// Stable across redeliveries of the same attempt; the basis of deduplication.
    pub idempotency_key: String,
    /// Attempts already started. Incremented when a lease is granted.
    pub attempt: u32,
    /// Retry budget and backoff.
    pub retry: RetryPolicy,
    /// Capabilities a worker must offer to receive this task.
    pub required_capabilities: Vec<Capability>,
    /// When it first entered the queue.
    pub enqueued_at: DateTime<Utc>,
    /// Not visible to consumers before this time; how delays and backoff are expressed.
    pub available_at: DateTime<Utc>,
    /// Visibility timeout granted with the lease.
    pub lease_duration_ms: u64,
}

impl QueuedTask {
    /// Build a queue entry from a task node.
    ///
    /// The idempotency key is the task's *attempt* key, so a retry is allowed to run
    /// while a duplicate delivery of the same attempt is recognised and dropped.
    pub fn from_task(task: &TaskNode, priority: Priority, lease_duration_ms: u64) -> Result<Self> {
        let now = Utc::now();
        Ok(Self {
            task_id: task.id,
            job_id: task.job_id,
            queue: DEFAULT_QUEUE.to_owned(),
            priority,
            payload: serde_json::to_value(task)
                .map_err(|e| swarm_domain::SwarmError::Queue(format!("payload encode: {e}")))?,
            idempotency_key: task.attempt_key(),
            attempt: task.attempt,
            retry: task.retry_policy,
            required_capabilities: task.required_capabilities.clone(),
            enqueued_at: now,
            available_at: now,
            lease_duration_ms,
        })
    }

    /// Decode the task definition.
    pub fn task(&self) -> Result<TaskNode> {
        serde_json::from_value(self.payload.clone())
            .map_err(|e| swarm_domain::SwarmError::Queue(format!("payload decode: {e}")))
    }

    /// Route to a named queue.
    #[must_use]
    pub fn on_queue(mut self, queue: impl Into<String>) -> Self {
        self.queue = queue.into();
        self
    }

    /// Hide the entry until `at`.
    #[must_use]
    pub fn available_at(mut self, at: DateTime<Utc>) -> Self {
        self.available_at = at;
        self
    }

    /// Whether a consumer may see this entry at `now`.
    #[must_use]
    pub fn is_visible(&self, now: DateTime<Utc>) -> bool {
        self.available_at <= now
    }

    /// Whether a worker offering `capabilities` may take this entry.
    ///
    /// An empty offer means "no filter" — used by the single-node prototype and by
    /// admin tooling.
    #[must_use]
    pub fn matches_capabilities(&self, capabilities: &[Capability]) -> bool {
        capabilities.is_empty()
            || self
                .required_capabilities
                .iter()
                .all(|required| capabilities.contains(required))
    }
}

/// A claim on a task, valid until it expires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskLease {
    /// Lease identity, presented on acknowledge, reject, and extend.
    pub lease_id: LeaseId,
    /// The claimed task.
    pub task: QueuedTask,
    /// Who holds it.
    pub worker: WorkerId,
    /// When it was granted.
    pub granted_at: DateTime<Utc>,
    /// When the claim lapses and the task becomes available again.
    pub expires_at: DateTime<Utc>,
    /// Lease lifecycle state.
    pub state: LeaseState,
}

impl TaskLease {
    /// Whether the lease has lapsed at `now`.
    #[must_use]
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        self.expires_at <= now
    }

    /// How long the holder has been working, in milliseconds.
    #[must_use]
    pub fn held_for_ms(&self, now: DateTime<Utc>) -> u64 {
        (now - self.granted_at).num_milliseconds().max(0) as u64
    }
}

/// A task that exhausted its retry budget, parked for inspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeadLetter {
    /// The task as it last looked.
    pub task: QueuedTask,
    /// Attempts consumed.
    pub attempts: u32,
    /// Why the final attempt failed.
    pub last_error: String,
    /// When it was parked.
    pub at: DateTime<Utc>,
}

/// A point-in-time view of queue health, exported as Prometheus metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueStats {
    /// Entries visible to consumers now.
    pub ready: usize,
    /// Entries waiting for a delay or a retry backoff.
    pub delayed: usize,
    /// Entries currently leased.
    pub leased: usize,
    /// Entries parked in the dead-letter queue.
    pub dead_lettered: usize,
    /// Enqueue calls that resulted in a new entry.
    pub enqueued_total: u64,
    /// Leases granted.
    pub dequeued_total: u64,
    /// Leases acknowledged successfully.
    pub acknowledged_total: u64,
    /// Entries re-queued after a failure or an expiry.
    pub retried_total: u64,
    /// Leases that lapsed before being acknowledged.
    pub expired_total: u64,
    /// Enqueue calls dropped because the attempt key had been seen before.
    pub deduplicated_total: u64,
}

impl QueueStats {
    /// Entries the queue still owes a worker: everything not finished or parked.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.ready + self.delayed + self.leased
    }
}

/// A durable, leased, priority-aware task queue.
///
/// Implementations must guarantee:
/// - a task is delivered to at most one live lease at a time;
/// - a lease that expires makes its task available again;
/// - `acknowledge` on an expired or unknown lease fails rather than silently
///   succeeding, so a slow worker cannot ack work another worker already redid;
/// - entries survive process restart (durable backends only).
#[async_trait]
pub trait TaskQueue: Send + Sync {
    /// Add a task. Re-enqueuing an already-seen attempt key is a no-op.
    async fn enqueue(&self, task: QueuedTask) -> Result<()>;

    /// Claim the highest-priority visible task, if any.
    async fn dequeue(&self, worker: WorkerId) -> Result<Option<TaskLease>> {
        self.dequeue_from(worker, &[], &[]).await
    }

    /// Claim the highest-priority visible task from `queues` that the worker's
    /// `capabilities` can satisfy. Empty slices mean "no filter".
    async fn dequeue_from(
        &self,
        worker: WorkerId,
        queues: &[String],
        capabilities: &[Capability],
    ) -> Result<Option<TaskLease>>;

    /// Mark the leased task done and drop the entry.
    async fn acknowledge(&self, lease: LeaseId) -> Result<()>;

    /// Report failure: retry with backoff, or dead-letter if the budget is spent.
    async fn reject(&self, lease: LeaseId, reason: String) -> Result<()>;

    /// Return a task to the queue without consuming an attempt.
    ///
    /// Used when a worker cannot start the task at all — no capable agent is free, the
    /// node is draining — so the task is not penalised for the scheduler's problem.
    async fn release(&self, lease: LeaseId) -> Result<()>;

    /// Push a lease's expiry further out because the holder reported progress.
    async fn extend_lease(&self, lease: LeaseId, duration: Duration) -> Result<()>;

    /// Sweep lapsed leases back into the queue. Returns how many were recovered.
    async fn requeue_expired(&self) -> Result<usize>;

    /// Current queue health.
    async fn stats(&self) -> Result<QueueStats>;

    /// Tasks that exhausted their retries.
    async fn dead_letters(&self) -> Result<Vec<DeadLetter>>;

    /// Re-queue a dead-lettered task with a fresh attempt budget.
    async fn replay_dead_letter(&self, task_id: TaskId) -> Result<bool>;

    /// Drop every entry and lease belonging to `job_id`. Returns how many were removed.
    async fn purge_job(&self, job_id: JobId) -> Result<usize>;
}
