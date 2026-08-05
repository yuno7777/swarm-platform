//! Shared memory for agents.
//!
//! Hundreds of agents read and write the same job state concurrently, so the central
//! guarantee here is that **no write is silently lost**. Every record is versioned;
//! blind writes bump the version, and [`MemoryStore::compare_and_swap`] fails loudly
//! with the current version attached when another agent got there first. Callers
//! either merge and retry or escalate — they are never allowed to be unaware.
//!
//! Memory is partitioned by namespace ([`ns`]) rather than by one flat keyspace, so a
//! job's working memory, an agent's private scratch space, execution records, and
//! long-lived semantic memory have different lifetimes and different access rules.
#![forbid(unsafe_code)]

pub mod memory;

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use swarm_domain::{AgentId, JobId, MemoryId, Result};

pub use memory::InMemoryStore;

/// Version used to mean "this key must not exist yet" in a compare-and-swap.
///
/// This is how a caller claims exclusive ownership of a key — the basis of the
/// execution records that keep duplicate task delivery from doing duplicate work.
pub const VERSION_ABSENT: u64 = 0;

/// Namespace naming. Keeping this in one place stops agents from inventing their own
/// layout and quietly colliding.
pub mod ns {
    use swarm_domain::{AgentId, JobId};

    /// Scratch space for a job's in-progress reasoning, cleared when the job ends.
    #[must_use]
    pub fn working(job_id: JobId) -> String {
        format!("job/{job_id}/working")
    }

    /// Durable facts about a job: plan, decisions, final outputs.
    #[must_use]
    pub fn job(job_id: JobId) -> String {
        format!("job/{job_id}/memory")
    }

    /// Completed task results, keyed by attempt idempotency key.
    #[must_use]
    pub fn results(job_id: JobId) -> String {
        format!("job/{job_id}/results")
    }

    /// Execution records claimed before side effects, to make retries safe.
    #[must_use]
    pub fn executions(job_id: JobId) -> String {
        format!("job/{job_id}/executions")
    }

    /// Resumable checkpoints written during a task attempt.
    #[must_use]
    pub fn checkpoints(job_id: JobId) -> String {
        format!("job/{job_id}/checkpoints")
    }

    /// An agent's private memory, invisible to its peers.
    #[must_use]
    pub fn agent(agent_id: AgentId) -> String {
        format!("agent/{agent_id}")
    }

    /// Cross-job semantic memory (pgvector-backed from Phase 4).
    #[must_use]
    pub fn semantic() -> String {
        "semantic".to_owned()
    }
}

/// A versioned record in shared memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Record identity.
    pub id: MemoryId,
    /// Job the record belongs to, when it is job-scoped.
    pub job_id: Option<JobId>,
    /// Agent that owns the record, when it is agent-scoped.
    pub agent_id: Option<AgentId>,
    /// Namespace, see [`ns`].
    pub namespace: String,
    /// Key within the namespace.
    pub key: String,
    /// The stored value.
    pub value: serde_json::Value,
    /// Monotonic version, incremented on every successful write.
    pub version: u64,
    /// First write.
    pub created_at: DateTime<Utc>,
    /// Most recent write.
    pub updated_at: DateTime<Utc>,
    /// When the record stops being visible.
    pub expires_at: Option<DateTime<Utc>>,
}

impl MemoryRecord {
    /// Whether the record has passed its TTL at `now`.
    #[must_use]
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.is_some_and(|expiry| expiry <= now)
    }

    /// Deserialize the value into a concrete type.
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.value.clone()).map_err(|e| {
            swarm_domain::SwarmError::Memory(format!(
                "record {}/{} does not deserialize: {e}",
                self.namespace, self.key
            ))
        })
    }
}

/// A pending write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryWrite {
    /// Target namespace.
    pub namespace: String,
    /// Target key.
    pub key: String,
    /// Value to store.
    pub value: serde_json::Value,
    /// Job scope, for cascade deletion and per-job listing.
    pub job_id: Option<JobId>,
    /// Agent scope.
    pub agent_id: Option<AgentId>,
    /// Who is writing, recorded in the audit trail.
    pub actor: String,
    /// Optional lifetime.
    pub ttl: Option<Duration>,
}

impl MemoryWrite {
    /// A write of `value` to `namespace`/`key` by `actor`.
    pub fn new(
        namespace: impl Into<String>,
        key: impl Into<String>,
        value: serde_json::Value,
        actor: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
            value,
            job_id: None,
            agent_id: None,
            actor: actor.into(),
            ttl: None,
        }
    }

    /// A write whose value is serialized from `value`.
    pub fn json<T: Serialize>(
        namespace: impl Into<String>,
        key: impl Into<String>,
        value: &T,
        actor: impl Into<String>,
    ) -> Result<Self> {
        let value = serde_json::to_value(value)
            .map_err(|e| swarm_domain::SwarmError::Memory(format!("encode: {e}")))?;
        Ok(Self::new(namespace, key, value, actor))
    }

    /// Scope the record to a job.
    #[must_use]
    pub fn for_job(mut self, job_id: JobId) -> Self {
        self.job_id = Some(job_id);
        self
    }

    /// Scope the record to an agent.
    #[must_use]
    pub fn by_agent(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Give the record a lifetime.
    #[must_use]
    pub fn expiring_in(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

/// What happened to a record, for the audit history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryOperation {
    /// First write of a key.
    Create,
    /// Unconditional overwrite.
    Put,
    /// Successful compare-and-swap.
    CompareAndSwap,
    /// Explicit removal.
    Delete,
    /// Removed by compaction after its TTL.
    Expire,
}

/// One entry of a record's change history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Namespace of the record.
    pub namespace: String,
    /// Key of the record.
    pub key: String,
    /// Version produced by this operation.
    pub version: u64,
    /// What was done.
    pub operation: MemoryOperation,
    /// Who did it.
    pub actor: String,
    /// When.
    pub at: DateTime<Utc>,
}

/// Versioned key/value storage shared by every agent.
///
/// Implementations must guarantee:
/// - `version` increases by exactly one per successful write;
/// - `compare_and_swap` is atomic and returns [`swarm_domain::SwarmError::VersionConflict`]
///   carrying the actual version when it loses;
/// - expired records are invisible to `get` and `list` even before compaction runs.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Read a record, or `None` if it is absent or expired.
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryRecord>>;

    /// Write unconditionally, bumping the version.
    async fn put(&self, write: MemoryWrite) -> Result<MemoryRecord>;

    /// Write only if the stored version is exactly `expected_version`.
    ///
    /// Pass [`VERSION_ABSENT`] to require that the key does not exist yet.
    async fn compare_and_swap(
        &self,
        write: MemoryWrite,
        expected_version: u64,
    ) -> Result<MemoryRecord>;

    /// Claim a key that must not already exist.
    ///
    /// Returns `Ok(None)` when another writer already holds it — the caller should
    /// treat that as "someone else is doing or has done this work".
    async fn claim(&self, write: MemoryWrite) -> Result<Option<MemoryRecord>> {
        match self.compare_and_swap(write, VERSION_ABSENT).await {
            Ok(record) => Ok(Some(record)),
            Err(swarm_domain::SwarmError::VersionConflict { .. }) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// Remove a record. Returns whether it existed.
    async fn delete(&self, namespace: &str, key: &str) -> Result<bool>;

    /// List live records in `namespace` whose key starts with `prefix`.
    async fn list(&self, namespace: &str, prefix: &str) -> Result<Vec<MemoryRecord>>;

    /// Full change history for a key.
    async fn audit(&self, namespace: &str, key: &str) -> Result<Vec<AuditEntry>>;

    /// Drop expired records. Returns how many were reclaimed.
    async fn compact(&self) -> Result<usize>;

    /// Remove every record belonging to a job. Returns how many were removed.
    async fn purge_job(&self, job_id: JobId) -> Result<usize>;
}
