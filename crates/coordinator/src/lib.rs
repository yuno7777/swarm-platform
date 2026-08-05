//! Job orchestration: decompose, schedule, execute, aggregate.
//!
//! The coordinator owns a job's lifecycle from submission to final result. In Phase 1
//! it runs the workers in-process; the same engine drives remote workers in Phase 2
//! because everything it touches — the queue, shared memory, the model gateway — is
//! already behind a trait.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use swarm_coordinator::Coordinator;
//! # use swarm_domain::{ExecutionStrategy, JobRequest};
//! # use swarm_model_gateway::{Gateway, MockProvider};
//! # async fn demo() -> swarm_domain::Result<()> {
//! let gateway = Arc::new(Gateway::with_provider(Arc::new(MockProvider::new("mock"))));
//! let coordinator = Coordinator::local(gateway);
//!
//! let result = coordinator
//!     .submit_and_run(
//!         JobRequest::new("Compare Raft and Paxos")
//!             .with_strategy(ExecutionStrategy::Debate)
//!             .with_max_agents(6),
//!     )
//!     .await?;
//!
//! println!("{}", result.summary);
//! # Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]

pub mod aggregate;
pub mod decompose;
pub mod engine;
pub mod schedule;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};

use swarm_agent_runtime::Agent;
use swarm_domain::{
    ExecutionStatistics, ExecutionStrategy, FinalResult, GraphCounts, Job, JobId, JobRequest,
    JobStatus, NodeId, Result, StateMachine, SwarmError, TaskFailure, TaskGraph, TaskId, TaskNode,
    TaskResult, TaskState, Transition,
};
use swarm_model_gateway::Gateway;
use swarm_shared_memory::{InMemoryStore, MemoryStore};
use swarm_task_queue::{InMemoryQueue, TaskQueue};

pub use aggregate::{ConsensusMode, ConsensusReport};
pub use schedule::{Scheduler, SchedulerKind, SchedulingOutcome};

/// Limits and knobs for one coordinator process.
///
/// `#[serde(default)]` so a configuration file may set only the fields it cares about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoordinatorConfig {
    /// Identity of the node this coordinator runs on.
    pub node_id: NodeId,
    /// Placement strategy.
    pub scheduler: SchedulerKind,
    /// Agents this coordinator may hold across all jobs.
    pub max_cluster_agents: usize,
    /// Visibility timeout granted with each task lease.
    pub lease_ms: u64,
    /// Buffered events per job before slow subscribers start missing some.
    pub event_buffer: usize,
    /// Pause between scheduling ticks when there is nothing to dispatch.
    pub tick_ms: u64,
    /// Safety valve: maximum scheduling ticks for one job.
    pub max_ticks: u64,
    /// How long an idle agent is kept warm before being reclaimed.
    pub agent_idle_timeout_ms: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId::new(),
            scheduler: SchedulerKind::default(),
            max_cluster_agents: 512,
            lease_ms: 60_000,
            event_buffer: 1_024,
            tick_ms: 5,
            max_ticks: 100_000,
            agent_idle_timeout_ms: 30_000,
        }
    }
}

/// What happened, as streamed to clients and the dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobEventKind {
    /// The job passed admission control.
    JobAdmitted,
    /// The task graph was compiled and validated.
    JobPlanned,
    /// An agent was created for this job.
    AgentSpawned,
    /// An idle agent was reclaimed.
    AgentTerminated,
    /// A task became eligible and entered the queue.
    TaskQueued,
    /// An agent started executing a task.
    TaskStarted,
    /// A task completed and passed validation.
    TaskCompleted,
    /// A task attempt failed.
    TaskFailed,
    /// A failed task will be retried after its backoff.
    TaskRetrying,
    /// A task ran out of attempts.
    TaskDeadLettered,
    /// A task was abandoned because the job was cancelled or its input was lost.
    TaskCancelled,
    /// Scheduling was suspended.
    JobPaused,
    /// Scheduling resumed.
    JobResumed,
    /// The job was cancelled.
    JobCancelled,
    /// The job reached a terminal state.
    JobFinished,
}

/// One entry on a job's event stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobEvent {
    /// Monotonic per job; lets a reconnecting client resume without gaps.
    pub sequence_number: u64,
    /// Job the event belongs to.
    pub job_id: JobId,
    /// Task the event is about, when it is about one.
    pub task_id: Option<TaskId>,
    /// What happened.
    pub kind: JobEventKind,
    /// Human-readable detail.
    pub detail: String,
    /// Fraction of tasks terminal at the time of the event.
    pub progress: f32,
    /// When it happened.
    pub at: DateTime<Utc>,
}

/// Operator control over a running job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Scheduling proceeds.
    Running,
    /// Scheduling is suspended; running tasks are left alone.
    Paused,
    /// The job is being torn down.
    Cancelled,
}

/// A compact view of a job's progress, for APIs and the dashboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobStateView {
    /// Job identity.
    pub job_id: JobId,
    /// Current status.
    pub status: JobStatus,
    /// Strategy it was compiled with.
    pub execution_strategy: ExecutionStrategy,
    /// Task counts by lifecycle group.
    pub counts: GraphCounts,
    /// Fraction of tasks terminal, `0.0..=1.0`.
    pub progress: f32,
    /// Agents currently allocated to the job.
    pub active_agents: usize,
    /// Prompt plus completion tokens spent so far.
    pub tokens_used: u64,
    /// Estimated spend so far.
    pub cost_usd: f64,
    /// Submission time.
    pub created_at: DateTime<Utc>,
    /// Completion time, once finished.
    pub finished_at: Option<DateTime<Utc>>,
}

/// Everything the engine mutates while a job runs.
///
/// Fields are individually locked rather than sharing one mutex so a slow event
/// subscriber cannot block scheduling. No lock is ever held across an `await`.
pub(crate) struct JobHandle {
    pub(crate) job: Mutex<Job>,
    pub(crate) graph: Mutex<TaskGraph>,
    pub(crate) agents: Mutex<Vec<Agent>>,
    pub(crate) results: Mutex<Vec<TaskResult>>,
    pub(crate) failures: Mutex<Vec<TaskFailure>>,
    pub(crate) transitions: Mutex<Vec<Transition<TaskState>>>,
    pub(crate) statistics: Mutex<ExecutionStatistics>,
    pub(crate) latencies: Mutex<Vec<u64>>,
    pub(crate) final_result: Mutex<Option<FinalResult>>,
    pub(crate) events: broadcast::Sender<JobEvent>,
    pub(crate) control: watch::Sender<Control>,
    pub(crate) sequence: AtomicU64,
}

impl JobHandle {
    fn new(job: Job, graph: TaskGraph, event_buffer: usize) -> Self {
        let (events, _) = broadcast::channel(event_buffer.max(16));
        let (control, _) = watch::channel(Control::Running);
        Self {
            job: Mutex::new(job),
            graph: Mutex::new(graph),
            agents: Mutex::new(Vec::new()),
            results: Mutex::new(Vec::new()),
            failures: Mutex::new(Vec::new()),
            transitions: Mutex::new(Vec::new()),
            statistics: Mutex::new(ExecutionStatistics::default()),
            latencies: Mutex::new(Vec::new()),
            final_result: Mutex::new(None),
            events,
            control,
            sequence: AtomicU64::new(0),
        }
    }

    /// Publish an event. Delivery is best-effort; the sequence number lets a client
    /// that missed one notice and replay from the journal (Phase 2).
    pub(crate) fn emit(
        &self,
        kind: JobEventKind,
        task_id: Option<TaskId>,
        detail: impl Into<String>,
        progress: f32,
    ) {
        let event = JobEvent {
            sequence_number: self.sequence.fetch_add(1, Ordering::Relaxed),
            job_id: lock(&self.job).id,
            task_id,
            kind,
            detail: detail.into(),
            progress,
            at: Utc::now(),
        };
        lock(&self.statistics).messages_sent += 1;
        let _ = self.events.send(event);
    }

    /// Move the job to a new status, recording the reason.
    pub(crate) fn set_status(&self, to: JobStatus, reason: Option<String>) -> Result<()> {
        let mut job = lock(&self.job);
        let next = job.status.transition(to)?;
        job.status = next;
        job.status_reason = reason;
        if next == JobStatus::Running && job.started_at.is_none() {
            job.started_at = Some(Utc::now());
        }
        if next.is_finished() {
            job.finished_at = Some(Utc::now());
        }
        Ok(())
    }

    pub(crate) fn control_state(&self) -> Control {
        *self.control.borrow()
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Accepts jobs, plans them, drives them to completion, and merges the result.
pub struct Coordinator {
    pub(crate) config: CoordinatorConfig,
    pub(crate) queue: Arc<dyn TaskQueue>,
    pub(crate) memory: Arc<dyn MemoryStore>,
    pub(crate) gateway: Arc<Gateway>,
    pub(crate) scheduler: Box<dyn Scheduler>,
    pub(crate) jobs: DashMap<JobId, Arc<JobHandle>>,
    pub(crate) cluster_agents: AtomicUsize,
}

impl std::fmt::Debug for Coordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coordinator")
            .field("node_id", &self.config.node_id)
            .field("scheduler", &self.scheduler.name())
            .field("jobs", &self.jobs.len())
            .field(
                "cluster_agents",
                &self.cluster_agents.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Coordinator {
    /// Build a coordinator over the given infrastructure.
    #[must_use]
    pub fn new(
        config: CoordinatorConfig,
        queue: Arc<dyn TaskQueue>,
        memory: Arc<dyn MemoryStore>,
        gateway: Arc<Gateway>,
    ) -> Self {
        Self {
            scheduler: config.scheduler.build(),
            config,
            queue,
            memory,
            gateway,
            jobs: DashMap::new(),
            cluster_agents: AtomicUsize::new(0),
        }
    }

    /// A single-process coordinator with in-memory infrastructure.
    ///
    /// The Phase 1 deployment, and the one every test uses.
    #[must_use]
    pub fn local(gateway: Arc<Gateway>) -> Self {
        Self::new(
            CoordinatorConfig::default(),
            Arc::new(InMemoryQueue::new()),
            Arc::new(InMemoryStore::new()),
            gateway,
        )
    }

    /// A local coordinator with non-default settings.
    #[must_use]
    pub fn local_with(config: CoordinatorConfig, gateway: Arc<Gateway>) -> Self {
        Self::new(
            config,
            Arc::new(InMemoryQueue::new()),
            Arc::new(InMemoryStore::new()),
            gateway,
        )
    }

    /// This coordinator's configuration.
    #[must_use]
    pub const fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// The model gateway, for usage and cost reporting.
    #[must_use]
    pub fn gateway(&self) -> &Arc<Gateway> {
        &self.gateway
    }

    /// The task queue, for depth and dead-letter inspection.
    #[must_use]
    pub fn queue(&self) -> &Arc<dyn TaskQueue> {
        &self.queue
    }

    /// Shared memory, for reading intermediate agent state.
    #[must_use]
    pub fn memory(&self) -> &Arc<dyn MemoryStore> {
        &self.memory
    }

    /// Validate, admit, and plan a job. Returns immediately with its id.
    ///
    /// Admission control runs here, before any resource is committed: an
    /// over-quota job is rejected rather than being allowed to starve the cluster.
    pub fn submit(&self, request: JobRequest) -> Result<JobId> {
        request.validate()?;

        let mut job = Job::new(request);
        let job_id = job.id;

        if job.request.max_agents > self.config.max_cluster_agents {
            job.status = JobStatus::Rejected;
            return Err(SwarmError::QuotaExceeded(format!(
                "job asks for {} agents; this cluster allows {}",
                job.request.max_agents, self.config.max_cluster_agents
            )));
        }

        job.status = job.status.transition(JobStatus::Admitted)?;
        job.status = job.status.transition(JobStatus::Planning)?;
        let graph = decompose::decompose(&job)?;

        let handle = Arc::new(JobHandle::new(job, graph, self.config.event_buffer));
        handle.emit(JobEventKind::JobAdmitted, None, "admitted", 0.0);
        {
            let graph = lock(&handle.graph);
            handle.emit(
                JobEventKind::JobPlanned,
                None,
                format!(
                    "{} tasks in {} stages",
                    graph.len(),
                    graph.layers().map_or(0, |layers| layers.len())
                ),
                0.0,
            );
        }
        self.jobs.insert(job_id, handle);
        Ok(job_id)
    }

    /// Drive a planned job to completion.
    pub async fn run(&self, job_id: JobId) -> Result<FinalResult> {
        let handle = self.handle(job_id)?;
        engine::run_job(self, &handle).await
    }

    /// Submit and run in one call.
    pub async fn submit_and_run(&self, request: JobRequest) -> Result<FinalResult> {
        let job_id = self.submit(request)?;
        self.run(job_id).await
    }

    /// Subscribe to a job's live event stream.
    ///
    /// The channel is bounded: a subscriber that falls too far behind loses events and
    /// is told so by the gap in sequence numbers, rather than growing memory forever.
    pub fn subscribe(&self, job_id: JobId) -> Result<broadcast::Receiver<JobEvent>> {
        Ok(self.handle(job_id)?.events.subscribe())
    }

    /// Stop a job. Running tasks are abandoned and queued work is purged.
    pub async fn cancel(&self, job_id: JobId) -> Result<()> {
        let handle = self.handle(job_id)?;
        // send_replace, not send: `send` fails when nobody is subscribed, which would
        // silently drop a cancellation issued before the job started running.
        handle.control.send_replace(Control::Cancelled);
        handle.emit(
            JobEventKind::JobCancelled,
            None,
            "cancelled by operator",
            0.0,
        );
        self.queue.purge_job(job_id).await?;
        Ok(())
    }

    /// Suspend scheduling. Tasks already running are allowed to finish.
    pub fn pause(&self, job_id: JobId) -> Result<()> {
        let handle = self.handle(job_id)?;
        if handle.control_state() == Control::Cancelled {
            return Err(SwarmError::Cancelled(format!("job {job_id} is cancelled")));
        }
        handle.control.send_replace(Control::Paused);
        handle.emit(JobEventKind::JobPaused, None, "paused by operator", 0.0);
        Ok(())
    }

    /// Resume a paused job.
    pub fn resume(&self, job_id: JobId) -> Result<()> {
        let handle = self.handle(job_id)?;
        if handle.control_state() == Control::Cancelled {
            return Err(SwarmError::Cancelled(format!("job {job_id} is cancelled")));
        }
        handle.control.send_replace(Control::Running);
        handle.emit(JobEventKind::JobResumed, None, "resumed by operator", 0.0);
        Ok(())
    }

    /// A snapshot of the job's progress.
    pub fn state(&self, job_id: JobId) -> Result<JobStateView> {
        let handle = self.handle(job_id)?;
        let active_agents = lock(&handle.agents).len();
        let job = lock(&handle.job);
        let graph = lock(&handle.graph);
        let statistics = lock(&handle.statistics);

        Ok(JobStateView {
            job_id,
            status: job.status,
            execution_strategy: job.request.execution_strategy,
            counts: graph.counts(),
            progress: graph.progress(),
            active_agents,
            tokens_used: statistics.tokens_in + statistics.tokens_out,
            cost_usd: statistics.cost_usd,
            created_at: job.created_at,
            finished_at: job.finished_at,
        })
    }

    /// A snapshot of every job this coordinator knows about, newest first.
    ///
    /// Job ids are UUIDv7, so sorting by id is sorting by submission time.
    #[must_use]
    pub fn list_jobs(&self) -> Vec<JobStateView> {
        let mut ids: Vec<JobId> = self.jobs.iter().map(|entry| *entry.key()).collect();
        ids.sort_unstable_by(|left, right| right.cmp(left));
        ids.into_iter()
            .filter_map(|job_id| self.state(job_id).ok())
            .collect()
    }

    /// The job's task graph, as a flat list of nodes.
    pub fn task_graph(&self, job_id: JobId) -> Result<Vec<TaskNode>> {
        let handle = self.handle(job_id)?;
        let graph = lock(&handle.graph);
        let mut nodes: Vec<TaskNode> = graph.nodes().cloned().collect();
        nodes.sort_by(|left, right| {
            left.stage
                .cmp(&right.stage)
                .then_with(|| left.title.cmp(&right.title))
        });
        Ok(nodes)
    }

    /// Results produced so far, including for a job still running.
    pub fn intermediate_results(&self, job_id: JobId) -> Result<Vec<TaskResult>> {
        Ok(lock(&self.handle(job_id)?.results).clone())
    }

    /// Every failed attempt, for the failure inspection API.
    pub fn failures(&self, job_id: JobId) -> Result<Vec<TaskFailure>> {
        Ok(lock(&self.handle(job_id)?.failures).clone())
    }

    /// The audit trail of task state transitions.
    pub fn transitions(&self, job_id: JobId) -> Result<Vec<Transition<TaskState>>> {
        Ok(lock(&self.handle(job_id)?.transitions).clone())
    }

    /// The final result, once the job has finished.
    pub fn final_result(&self, job_id: JobId) -> Result<Option<FinalResult>> {
        Ok(lock(&self.handle(job_id)?.final_result).clone())
    }

    /// Descriptors of the agents allocated to a job.
    pub fn agents(&self, job_id: JobId) -> Result<Vec<swarm_domain::AgentDescriptor>> {
        Ok(lock(&self.handle(job_id)?.agents)
            .iter()
            .map(|agent| agent.descriptor.clone())
            .collect())
    }

    /// Agents currently allocated across every job on this coordinator.
    #[must_use]
    pub fn cluster_agent_count(&self) -> usize {
        self.cluster_agents.load(Ordering::Relaxed)
    }

    pub(crate) fn handle(&self, job_id: JobId) -> Result<Arc<JobHandle>> {
        self.jobs
            .get(&job_id)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or_else(|| SwarmError::NotFound {
                kind: "job",
                id: job_id.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_model_gateway::MockProvider;

    fn coordinator() -> Coordinator {
        Coordinator::local(Arc::new(Gateway::with_provider(Arc::new(
            MockProvider::new("mock"),
        ))))
    }

    #[test]
    fn submitting_plans_the_job_without_running_it() {
        let coordinator = coordinator();
        let job_id = coordinator
            .submit(JobRequest::new("Explain consensus algorithms"))
            .unwrap();

        let state = coordinator.state(job_id).unwrap();
        assert_eq!(state.status, JobStatus::Planning);
        assert!(state.counts.total > 0);
        assert_eq!(state.counts.completed, 0);
        assert_eq!(state.progress, 0.0);
        assert!(!coordinator.task_graph(job_id).unwrap().is_empty());
    }

    #[test]
    fn an_invalid_request_never_becomes_a_job() {
        let coordinator = coordinator();
        assert!(coordinator.submit(JobRequest::new("   ")).is_err());
        assert!(coordinator.jobs.is_empty());
    }

    #[test]
    fn a_job_over_the_cluster_quota_is_rejected_at_admission() {
        let coordinator = Coordinator::local_with(
            CoordinatorConfig {
                max_cluster_agents: 4,
                ..CoordinatorConfig::default()
            },
            Arc::new(Gateway::with_provider(Arc::new(MockProvider::new("mock")))),
        );

        let err = coordinator
            .submit(JobRequest::new("big job").with_max_agents(100))
            .unwrap_err();
        assert!(matches!(err, SwarmError::QuotaExceeded(_)));
        assert!(coordinator.jobs.is_empty());
    }

    #[test]
    fn unknown_jobs_are_reported_as_not_found() {
        let coordinator = coordinator();
        let err = coordinator.state(JobId::new()).unwrap_err();
        assert!(matches!(err, SwarmError::NotFound { kind: "job", .. }));
    }

    #[test]
    fn planning_emits_events_before_anyone_subscribes_without_failing() {
        // broadcast::send errors when there are no receivers; that must never
        // propagate as a job failure.
        let coordinator = coordinator();
        let job_id = coordinator
            .submit(JobRequest::new("no subscribers"))
            .unwrap();
        assert!(coordinator.subscribe(job_id).is_ok());
    }

    #[test]
    fn pausing_a_cancelled_job_is_refused() {
        let coordinator = coordinator();
        let job_id = coordinator.submit(JobRequest::new("objective")).unwrap();
        let handle = coordinator.handle(job_id).unwrap();
        handle.control.send_replace(Control::Cancelled);

        assert!(coordinator.pause(job_id).is_err());
        assert!(coordinator.resume(job_id).is_err());
    }
}
