//! Explicit state machines for jobs, tasks, leases, agents, and nodes.
//!
//! Every legal edge is declared in one place, illegal edges are refused with a typed
//! error, and every accepted edge produces a [`Transition`] record for the audit
//! trail. Nothing in the platform mutates a lifecycle field directly.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::agent::{AgentStatus, NodeStatus};
use crate::error::{Result, SwarmError};
use crate::ids::CorrelationId;
use crate::job::JobStatus;
use crate::task::TaskState;

/// A lifecycle whose legal transitions are known at compile time.
pub trait StateMachine: Copy + PartialEq + std::fmt::Debug + 'static {
    /// Entity label used in errors, metrics, and the transition journal.
    const ENTITY: &'static str;

    /// States reachable in one step from `self`.
    fn allowed(self) -> &'static [Self];

    /// Whether no further transition is possible.
    fn is_terminal(self) -> bool {
        self.allowed().is_empty()
    }

    /// Whether `to` is reachable in one step.
    fn can_transition_to(self, to: Self) -> bool {
        self.allowed().contains(&to)
    }

    /// Take the edge to `to`, or fail if it does not exist.
    fn transition(self, to: Self) -> Result<Self> {
        if self.can_transition_to(to) {
            Ok(to)
        } else {
            Err(SwarmError::InvalidTransition {
                entity: Self::ENTITY,
                from: format!("{self:?}"),
                to: format!("{to:?}"),
            })
        }
    }
}

/// Lifecycle of a task lease held by a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseState {
    /// Freshly granted at dequeue.
    Held,
    /// Extended because the worker reported progress.
    Extended,
    /// Acknowledged; the task is done.
    Released,
    /// Negatively acknowledged; the task retries or dead-letters.
    Rejected,
    /// Visibility timeout elapsed; the task is available to another worker.
    Expired,
}

impl LeaseState {
    /// Whether the lease still confers the right to acknowledge the task.
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Held | Self::Extended)
    }
}

impl StateMachine for LeaseState {
    const ENTITY: &'static str = "lease";

    fn allowed(self) -> &'static [Self] {
        match self {
            Self::Held => &[
                Self::Extended,
                Self::Released,
                Self::Rejected,
                Self::Expired,
            ],
            Self::Extended => &[
                Self::Extended,
                Self::Released,
                Self::Rejected,
                Self::Expired,
            ],
            Self::Released | Self::Rejected | Self::Expired => &[],
        }
    }
}

impl StateMachine for TaskState {
    const ENTITY: &'static str = "task";

    fn allowed(self) -> &'static [Self] {
        match self {
            Self::Created => &[Self::Queued, Self::WaitingForDependency, Self::Cancelled],
            Self::WaitingForDependency => &[Self::Queued, Self::Cancelled],
            Self::Queued => &[Self::Leased, Self::Cancelled, Self::Preempted],
            // Leased -> Queued is the recovery edge taken when a lease expires
            // before the worker starts.
            Self::Leased => &[Self::Running, Self::Queued, Self::Cancelled],
            Self::Running => &[
                Self::Completed,
                Self::Failed,
                Self::TimedOut,
                Self::WaitingForDependency,
                Self::Cancelled,
            ],
            Self::Failed | Self::TimedOut => &[Self::RetryScheduled, Self::DeadLettered],
            Self::RetryScheduled | Self::Preempted => &[Self::Queued, Self::Cancelled],
            Self::Completed | Self::Cancelled | Self::DeadLettered => &[],
        }
    }
}

impl StateMachine for JobStatus {
    const ENTITY: &'static str = "job";

    fn allowed(self) -> &'static [Self] {
        match self {
            Self::Submitted => &[Self::Admitted, Self::Rejected, Self::Cancelled],
            Self::Admitted => &[Self::Planning, Self::Failed, Self::Cancelled],
            Self::Planning => &[Self::Running, Self::Failed, Self::Cancelled],
            Self::Running => &[
                Self::Paused,
                Self::Aggregating,
                Self::Failed,
                Self::Cancelled,
            ],
            Self::Paused => &[Self::Running, Self::Cancelled],
            Self::Aggregating => &[
                Self::Completed,
                Self::PartiallyCompleted,
                Self::Failed,
                Self::Cancelled,
            ],
            // Failed and PartiallyCompleted are resting, not terminal: a retry
            // re-queues only the failed subgraph.
            Self::Failed | Self::PartiallyCompleted => &[Self::Running, Self::Cancelled],
            Self::Completed | Self::Cancelled | Self::Rejected => &[],
        }
    }
}

impl StateMachine for AgentStatus {
    const ENTITY: &'static str = "agent";

    fn allowed(self) -> &'static [Self] {
        match self {
            Self::Created => &[Self::Registered, Self::Terminated],
            Self::Registered => &[Self::Idle, Self::Assigned, Self::Terminated],
            Self::Idle => &[Self::Assigned, Self::Terminated],
            Self::Assigned => &[Self::Running, Self::Idle, Self::Terminated],
            Self::Running => &[
                Self::Waiting,
                Self::Completed,
                Self::Failed,
                Self::Terminated,
            ],
            Self::Waiting => &[Self::Running, Self::Failed, Self::Terminated],
            // Completed -> Idle is the warm-pool edge: agents are reused, not rebuilt.
            Self::Completed => &[Self::Idle, Self::Terminated],
            Self::Failed => &[Self::Retrying, Self::Idle, Self::Terminated],
            Self::Retrying => &[Self::Running, Self::Failed, Self::Terminated],
            Self::Terminated => &[],
        }
    }
}

impl StateMachine for NodeStatus {
    const ENTITY: &'static str = "node";

    fn allowed(self) -> &'static [Self] {
        match self {
            Self::Joining => &[Self::Ready, Self::Removed],
            Self::Ready => &[Self::Degraded, Self::Draining, Self::Unreachable],
            Self::Degraded => &[Self::Ready, Self::Draining, Self::Unreachable],
            Self::Draining => &[Self::Removed, Self::Ready],
            // A node may come back with the same identity, so Unreachable is not final.
            Self::Unreachable => &[Self::Ready, Self::Removed],
            Self::Removed => &[],
        }
    }
}

/// An accepted state change, recorded for audit.
///
/// This is the row shape of `*_state_transitions` in Postgres.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition<S> {
    /// Entity kind the transition belongs to.
    ///
    /// Owned rather than `&'static str` because transitions are journaled and read
    /// back: a borrowed field would make the record impossible to deserialize.
    pub entity: String,
    /// Previous state; `None` for the initial state of a new entity.
    pub from: Option<S>,
    /// New state.
    pub to: S,
    /// Who caused it, e.g. `coordinator:node-1`, `user:alice`, `system`.
    pub actor: String,
    /// Machine-readable cause, e.g. `lease_expired`, `validation_failed`.
    pub reason: Option<String>,
    /// The leader term that authorised it, when a leader did.
    pub term: Option<u64>,
    /// Correlates this change with the request or message that triggered it.
    pub correlation_id: CorrelationId,
    /// When it happened.
    pub at: DateTime<Utc>,
}

impl<S: StateMachine> Transition<S> {
    /// Record an accepted transition.
    #[must_use]
    pub fn new(
        from: Option<S>,
        to: S,
        actor: impl Into<String>,
        reason: Option<String>,
        correlation_id: CorrelationId,
    ) -> Self {
        Self {
            entity: S::ENTITY.to_owned(),
            from,
            to,
            actor: actor.into(),
            reason,
            term: None,
            correlation_id,
            at: Utc::now(),
        }
    }

    /// Stamp the leader term that authorised the change (fencing, ADR-4).
    #[must_use]
    pub fn with_term(mut self, term: u64) -> Self {
        self.term = Some(term);
        self
    }
}

/// Apply a transition to `current`, returning the new state and its audit record.
///
/// This is the only function the rest of the platform should use to change a
/// lifecycle field: it cannot succeed without producing the audit record.
pub fn apply<S: StateMachine>(
    current: S,
    to: S,
    actor: impl Into<String>,
    reason: Option<String>,
    correlation_id: CorrelationId,
) -> Result<(S, Transition<S>)> {
    let next = current.transition(to)?;
    Ok((
        next,
        Transition::new(Some(current), next, actor, reason, correlation_id),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_happy_path_is_walkable() {
        let mut state = TaskState::Created;
        for next in [
            TaskState::Queued,
            TaskState::Leased,
            TaskState::Running,
            TaskState::Completed,
        ] {
            state = state.transition(next).unwrap();
        }
        assert!(state.is_terminal());
    }

    #[test]
    fn completed_tasks_cannot_be_resurrected() {
        let err = TaskState::Completed
            .transition(TaskState::Running)
            .unwrap_err();
        match err {
            SwarmError::InvalidTransition { entity, from, to } => {
                assert_eq!(entity, "task");
                assert_eq!(from, "Completed");
                assert_eq!(to, "Running");
            }
            other => panic!("expected InvalidTransition, got {other:?}"),
        }
    }

    #[test]
    fn lease_expiry_recovers_a_task_to_the_queue() {
        // The recovery path that makes worker crashes survivable.
        let state = TaskState::Leased.transition(TaskState::Queued).unwrap();
        assert_eq!(state, TaskState::Queued);
    }

    #[test]
    fn failure_leads_to_retry_or_dead_letter_and_nothing_else() {
        assert_eq!(
            TaskState::Failed.allowed(),
            &[TaskState::RetryScheduled, TaskState::DeadLettered]
        );
        assert!(TaskState::Failed.transition(TaskState::Queued).is_err());
        assert!(TaskState::TimedOut.transition(TaskState::Running).is_err());
    }

    #[test]
    fn retrying_a_failed_job_is_allowed_but_a_completed_one_is_not() {
        assert!(JobStatus::Failed.transition(JobStatus::Running).is_ok());
        assert!(JobStatus::PartiallyCompleted
            .transition(JobStatus::Running)
            .is_ok());
        assert!(JobStatus::Completed.transition(JobStatus::Running).is_err());
        assert!(JobStatus::Completed.is_terminal());
    }

    #[test]
    fn agents_return_to_the_warm_pool_after_completing() {
        assert!(AgentStatus::Completed.transition(AgentStatus::Idle).is_ok());
        assert!(AgentStatus::Terminated
            .transition(AgentStatus::Idle)
            .is_err());
    }

    #[test]
    fn unreachable_nodes_may_rejoin_but_removed_ones_may_not() {
        assert!(NodeStatus::Unreachable
            .transition(NodeStatus::Ready)
            .is_ok());
        assert!(NodeStatus::Removed.transition(NodeStatus::Ready).is_err());
    }

    #[test]
    fn only_active_leases_can_be_acknowledged() {
        assert!(LeaseState::Held.is_active());
        assert!(LeaseState::Extended.is_active());
        assert!(!LeaseState::Expired.is_active());
        assert!(LeaseState::Expired
            .transition(LeaseState::Released)
            .is_err());
    }

    #[test]
    fn terminal_states_are_exactly_those_with_no_outgoing_edges() {
        for state in [
            TaskState::Completed,
            TaskState::Cancelled,
            TaskState::DeadLettered,
        ] {
            assert!(state.is_terminal(), "{state:?} should be terminal");
        }
        for state in [
            TaskState::Created,
            TaskState::Queued,
            TaskState::Leased,
            TaskState::Running,
            TaskState::Failed,
            TaskState::TimedOut,
            TaskState::RetryScheduled,
            TaskState::Preempted,
            TaskState::WaitingForDependency,
        ] {
            assert!(!state.is_terminal(), "{state:?} should not be terminal");
        }
    }

    #[test]
    fn no_state_machine_declares_a_self_loop_it_does_not_mean() {
        // Extended -> Extended is the one intentional self-loop in the platform.
        for state in [
            TaskState::Created,
            TaskState::Queued,
            TaskState::Running,
            TaskState::Completed,
        ] {
            assert!(
                !state.can_transition_to(state),
                "{state:?} should not loop to itself"
            );
        }
        assert!(LeaseState::Extended.can_transition_to(LeaseState::Extended));
    }

    #[test]
    fn apply_produces_an_audit_record_with_actor_and_reason() {
        let correlation_id = CorrelationId::new();
        let (state, record) = apply(
            TaskState::Leased,
            TaskState::Queued,
            "coordinator:node-1",
            Some("lease_expired".to_owned()),
            correlation_id,
        )
        .unwrap();

        assert_eq!(state, TaskState::Queued);
        assert_eq!(record.entity, "task");
        assert_eq!(record.from, Some(TaskState::Leased));
        assert_eq!(record.to, TaskState::Queued);
        assert_eq!(record.actor, "coordinator:node-1");
        assert_eq!(record.reason.as_deref(), Some("lease_expired"));
        assert_eq!(record.correlation_id, correlation_id);
        assert!(record.term.is_none());
        assert_eq!(record.with_term(7).term, Some(7));
    }

    #[test]
    fn apply_refuses_illegal_edges_and_produces_no_record() {
        let result = apply(
            TaskState::Completed,
            TaskState::Queued,
            "system",
            None,
            CorrelationId::new(),
        );
        assert!(result.is_err());
    }
}
