//! Core domain model of the swarm platform.
//!
//! This crate holds the types every other crate agrees on — identifiers, jobs, tasks,
//! agents, the task DAG, and the state machines that govern all of them.
//!
//! It deliberately contains **no async code and no I/O**. Nothing here can touch a
//! database, a queue, or a model provider, which means the logic most likely to be
//! subtly wrong (transition legality, cycle detection, readiness, retry backoff) is
//! made of pure functions that test in microseconds. Infrastructure crates depend on
//! this one; it depends on none of them.
//!
//! ```
//! use swarm_domain::{ExecutionStrategy, JobRequest, StateMachine, TaskState};
//!
//! let request = JobRequest::new("Compare Raft and Paxos")
//!     .with_strategy(ExecutionStrategy::Debate)
//!     .with_max_agents(6);
//! request.validate()?;
//!
//! // Illegal lifecycle edges are refused rather than silently ignored.
//! assert!(TaskState::Completed.transition(TaskState::Running).is_err());
//! # Ok::<(), swarm_domain::SwarmError>(())
//! ```
#![forbid(unsafe_code)]

pub mod agent;
pub mod error;
pub mod graph;
pub mod hash;
pub mod ids;
pub mod job;
pub mod state;
pub mod task;

pub use agent::{
    AgentDecision, AgentDescriptor, AgentStatus, AgentType, Capability, ModelConfig, NodeResources,
    NodeStatus,
};
pub use error::{Result, SwarmError};
pub use graph::{GraphCounts, TaskGraph};
pub use ids::{
    AgentId, AttemptId, CorrelationId, JobId, LeaseId, MemoryId, MessageId, NodeId, RoundId, TaskId,
};
pub use job::{
    Conflict, Evidence, ExecutionStatistics, ExecutionStrategy, FinalResult, Job, JobRequest,
    JobStatus, Priority, StructuredOutput, MAX_AGENTS_PER_JOB, MAX_OBJECTIVE_LEN,
};
pub use state::{LeaseState, StateMachine, Transition};
pub use task::{
    validate_output, Checkpoint, RetryPolicy, TaskFailure, TaskKind, TaskNode, TaskResult,
    TaskState, ValidationRule,
};
