//! The single error type shared by every crate in the platform.

use thiserror::Error;

/// Convenience alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, SwarmError>;

/// Every failure mode the platform can produce, as structured data rather than a string.
///
/// Variants are grouped by the layer that raises them. [`SwarmError::is_retryable`]
/// decides whether a task attempt that ended in this error is worth another try.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SwarmError {
    /// A state machine rejected an edge that does not exist.
    #[error("invalid {entity} transition: {from} -> {to}")]
    InvalidTransition {
        /// Entity kind (`task`, `job`, `agent`, `node`, `lease`).
        entity: &'static str,
        /// State the entity was in.
        from: String,
        /// State the caller tried to move it to.
        to: String,
    },

    /// A task graph mutation would have introduced a cycle.
    #[error("task graph is cyclic (cycle involves task {task})")]
    CyclicGraph {
        /// One task known to participate in the cycle.
        task: String,
    },

    /// A task declared a dependency on a task that is not in the graph.
    #[error("task {task} depends on unknown task {dependency}")]
    UnknownDependency {
        /// The dependent task.
        task: String,
        /// The missing dependency.
        dependency: String,
    },

    /// Lookup of a known-kind entity failed.
    #[error("unknown {kind} `{id}`")]
    NotFound {
        /// Entity kind, for the message.
        kind: &'static str,
        /// Identifier that was not found.
        id: String,
    },

    /// An identifier string could not be parsed.
    #[error("invalid identifier `{value}`: {detail}")]
    InvalidId {
        /// The rejected input.
        value: String,
        /// Parser detail.
        detail: String,
    },

    /// A request or output failed validation.
    #[error("validation failed: {0}")]
    Validation(String),

    /// The task queue could not satisfy an operation.
    #[error("queue error: {0}")]
    Queue(String),

    /// A compare-and-swap lost: another writer got there first.
    #[error("version conflict on {namespace}/{key}: expected {expected}, found {actual}")]
    VersionConflict {
        /// Memory namespace.
        namespace: String,
        /// Memory key.
        key: String,
        /// Version the caller believed was current.
        expected: u64,
        /// Version actually stored.
        actual: u64,
    },

    /// Shared memory failure that is not a version conflict.
    #[error("memory error: {0}")]
    Memory(String),

    /// A model provider returned an error.
    #[error("provider `{provider}` failed: {detail}")]
    Provider {
        /// Provider name.
        provider: String,
        /// Provider-supplied detail.
        detail: String,
    },

    /// A provider refused the call because of rate limiting.
    #[error("provider `{provider}` rate limited (retry after {retry_after_ms}ms)")]
    RateLimited {
        /// Provider name.
        provider: String,
        /// Suggested wait before retrying.
        retry_after_ms: u64,
    },

    /// A circuit breaker is open, so the call was not attempted.
    #[error("circuit open for provider `{provider}`")]
    CircuitOpen {
        /// Provider name.
        provider: String,
    },

    /// A cost ceiling would have been exceeded.
    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),

    /// An agent/token/concurrency quota would have been exceeded.
    #[error("quota exceeded: {0}")]
    QuotaExceeded(String),

    /// An operation did not finish inside its deadline.
    #[error("timed out after {millis}ms")]
    Timeout {
        /// Elapsed budget.
        millis: u64,
    },

    /// The operation was cancelled by an operator or a parent job.
    #[error("cancelled: {0}")]
    Cancelled(String),

    /// Configuration was missing or nonsensical.
    #[error("configuration error: {0}")]
    Config(String),

    /// A bug or an unclassified failure.
    #[error("internal error: {0}")]
    Internal(String),
}

impl SwarmError {
    /// Whether retrying the failed operation could plausibly succeed.
    ///
    /// Used by the scheduler to decide between `RetryScheduled` and `DeadLettered`.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Queue(_)
                | Self::Memory(_)
                | Self::VersionConflict { .. }
                | Self::Provider { .. }
                | Self::RateLimited { .. }
                | Self::CircuitOpen { .. }
                | Self::Timeout { .. }
                | Self::Internal(_)
        )
    }

    /// Short machine-readable label, used for metrics dimensions and `error_kind` columns.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::InvalidTransition { .. } => "invalid_transition",
            Self::CyclicGraph { .. } => "cyclic_graph",
            Self::UnknownDependency { .. } => "unknown_dependency",
            Self::NotFound { .. } => "not_found",
            Self::InvalidId { .. } => "invalid_id",
            Self::Validation(_) => "validation",
            Self::Queue(_) => "queue",
            Self::VersionConflict { .. } => "version_conflict",
            Self::Memory(_) => "memory",
            Self::Provider { .. } => "provider",
            Self::RateLimited { .. } => "rate_limited",
            Self::CircuitOpen { .. } => "circuit_open",
            Self::BudgetExceeded(_) => "budget_exceeded",
            Self::QuotaExceeded(_) => "quota_exceeded",
            Self::Timeout { .. } => "timeout",
            Self::Cancelled(_) => "cancelled",
            Self::Config(_) => "config",
            Self::Internal(_) => "internal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_failures_are_retryable_and_logic_errors_are_not() {
        assert!(SwarmError::Timeout { millis: 10 }.is_retryable());
        assert!(SwarmError::Provider {
            provider: "mock".into(),
            detail: "503".into()
        }
        .is_retryable());
        assert!(!SwarmError::Validation("missing section".into()).is_retryable());
        assert!(!SwarmError::Cancelled("operator".into()).is_retryable());
        assert!(!SwarmError::BudgetExceeded("over $5".into()).is_retryable());
    }

    #[test]
    fn kind_is_stable_for_metrics() {
        assert_eq!(SwarmError::Queue("full".into()).kind(), "queue");
        assert_eq!(
            SwarmError::InvalidTransition {
                entity: "task",
                from: "Completed".into(),
                to: "Running".into()
            }
            .kind(),
            "invalid_transition"
        );
    }
}
