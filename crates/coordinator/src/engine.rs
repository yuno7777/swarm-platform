//! The execution engine: the loop that turns a task graph into a finished job.
//!
//! Each tick promotes newly-ready tasks into the queue, recovers expired leases,
//! leases work back out for idle agents, runs those tasks concurrently, and applies
//! the outcomes. Everything the loop touches is behind a trait, so the same code
//! drives in-process agents in Phase 1 and remote worker nodes in Phase 2.
//!
//! No lock is held across an `await`: state is read into locals, the await happens,
//! and the results are applied under a fresh lock.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::future::join_all;

use swarm_agent_runtime::{agent_type_for, Agent, TaskContext, UpstreamOutput};
use swarm_domain::{
    AgentDescriptor, AgentStatus, AgentType, Capability, CorrelationId, FinalResult, JobStatus,
    Result, StateMachine, SwarmError, TaskFailure, TaskId, TaskNode, TaskResult, TaskState,
};
use swarm_task_queue::{QueuedTask, TaskLease};

use crate::aggregate::aggregate;
use crate::schedule::Scheduler;
use crate::{lock, Control, Coordinator, JobEventKind, JobHandle};

/// One task, matched with the agent that will run it.
struct Dispatch {
    lease: TaskLease,
    agent: Agent,
    task: TaskNode,
    context: TaskContext,
    queue_wait_ms: u64,
}

/// Drive a planned job to a terminal state and produce its final result.
pub(crate) async fn run_job(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
) -> Result<FinalResult> {
    let started = Instant::now();
    let (job_id, correlation_id, max_agents, status) = {
        let job = lock(&handle.job);
        (
            job.id,
            job.correlation_id,
            job.request.max_agents,
            job.status,
        )
    };

    if status.is_finished() {
        return lock(&handle.final_result).clone().ok_or_else(|| {
            SwarmError::Internal(format!("job {job_id} finished without a result"))
        });
    }

    let actor = format!("coordinator:{}", coordinator.config.node_id);
    let worker = actor.clone();
    handle.set_status(JobStatus::Running, None)?;
    let spawned = provision_agents(coordinator, handle, max_agents)?;

    let mut ticks = 0_u64;
    let mut paused = false;
    let mut cancelled = false;

    loop {
        ticks += 1;
        if ticks > coordinator.config.max_ticks {
            return Err(SwarmError::Internal(format!(
                "job {job_id} exceeded its scheduling tick budget"
            )));
        }

        match handle.control_state() {
            Control::Cancelled => {
                cancelled = true;
                break;
            }
            Control::Paused => {
                if !paused {
                    handle.set_status(JobStatus::Paused, Some("operator paused".to_owned()))?;
                    paused = true;
                }
                if !await_resume(handle).await {
                    cancelled = true;
                    break;
                }
                continue;
            }
            Control::Running => {
                if paused {
                    handle.set_status(JobStatus::Running, None)?;
                    paused = false;
                }
            }
        }

        if past_deadline(handle) {
            handle.emit(
                JobEventKind::JobFinished,
                None,
                "deadline exceeded",
                progress(handle),
            );
            break;
        }

        let promoted = promote_ready(coordinator, handle, &actor, correlation_id).await?;
        coordinator.queue.requeue_expired().await?;
        maintain_agent_pool(coordinator, handle, max_agents).await?;

        let dispatches =
            collect_dispatches(coordinator, handle, &worker, &actor, correlation_id).await?;

        if dispatches.is_empty() {
            if is_complete(handle) {
                break;
            }
            if promoted == 0
                && !make_progress_possible(coordinator, handle, &actor, correlation_id).await?
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(coordinator.config.tick_ms)).await;
            continue;
        }

        let executions = dispatches.iter().map(|dispatch| async move {
            tokio::time::timeout(
                Duration::from_secs(dispatch.task.timeout_seconds),
                dispatch.agent.execute(&dispatch.task, &dispatch.context),
            )
            .await
        });

        let outcomes = tokio::select! {
            outcomes = join_all(executions) => Some(outcomes),
            () = wait_for_cancel(handle) => None,
        };

        let Some(outcomes) = outcomes else {
            // Cancelled mid-flight. Dropping the futures cancelled the work; the leases
            // are left to expire so nothing is silently marked done.
            cancelled = true;
            break;
        };

        for (dispatch, outcome) in dispatches.into_iter().zip(outcomes) {
            apply_outcome(
                coordinator,
                handle,
                dispatch,
                outcome,
                &actor,
                correlation_id,
            )
            .await?;
        }

        if is_complete(handle) {
            break;
        }
    }

    finish(
        coordinator,
        handle,
        cancelled,
        spawned,
        started,
        &actor,
        correlation_id,
    )
    .await
}

/// Wait until the job leaves the paused state. Returns false if it was cancelled.
async fn await_resume(handle: &Arc<JobHandle>) -> bool {
    let mut control = handle.control.subscribe();
    loop {
        let state = *control.borrow_and_update();
        match state {
            Control::Paused => {
                if control.changed().await.is_err() {
                    return false;
                }
            }
            Control::Running => return true,
            Control::Cancelled => return false,
        }
    }
}

/// Resolves when the job is cancelled; never resolves otherwise.
async fn wait_for_cancel(handle: &Arc<JobHandle>) {
    let mut control = handle.control.subscribe();
    loop {
        if *control.borrow_and_update() == Control::Cancelled {
            return;
        }
        if control.changed().await.is_err() {
            // The sender is gone, so cancellation can no longer arrive.
            std::future::pending::<()>().await;
        }
    }
}

fn progress(handle: &Arc<JobHandle>) -> f32 {
    lock(&handle.graph).progress()
}

fn is_complete(handle: &Arc<JobHandle>) -> bool {
    lock(&handle.graph).is_complete()
}

fn past_deadline(handle: &Arc<JobHandle>) -> bool {
    lock(&handle.job).is_past_deadline(Utc::now())
}

/// Record a task state change, keeping the audit trail complete.
fn advance(
    handle: &Arc<JobHandle>,
    task: TaskId,
    to: TaskState,
    actor: &str,
    reason: &str,
    correlation_id: CorrelationId,
) -> Result<()> {
    let transition = {
        let mut graph = lock(&handle.graph);
        graph.set_state(task, to, actor, Some(reason.to_owned()), correlation_id)?
    };
    lock(&handle.transitions).push(transition);
    Ok(())
}

/// Create the agents a job needs, respecting both its own and the cluster's ceiling.
fn provision_agents(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    max_agents: usize,
) -> Result<usize> {
    let (job_id, required_capabilities) = {
        let job = lock(&handle.job);
        (job.id, job.request.required_capabilities.clone())
    };

    let (mut demand, widest) = {
        let graph = lock(&handle.graph);
        let mut demand: HashMap<AgentType, usize> = HashMap::new();
        for node in graph.nodes() {
            *demand.entry(agent_type_for(node.kind)).or_default() += 1;
        }
        let widest = graph.layers()?.iter().map(Vec::len).max().unwrap_or(1);
        (demand, widest)
    };

    // A job-level capability requirement provisions an agent that offers it; it is not
    // added to every task, which would make tasks needing two scarce skills
    // unschedulable.
    for capability in &required_capabilities {
        demand
            .entry(AgentType::for_capability(*capability))
            .or_insert(1);
    }

    // Deterministic order: most-needed type first, ties broken by name.
    let mut types: Vec<(AgentType, usize)> = demand.into_iter().collect();
    types.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.0.as_str().cmp(right.0.as_str()))
    });

    // A budget smaller than the number of task kinds is fine: idle agents
    // re-specialise on demand (see `respecialise`).
    let target = widest.max(types.len()).min(max_agents).max(1);
    let claimed = coordinator
        .cluster_agents
        .fetch_add(target, Ordering::SeqCst);
    if claimed + target > coordinator.config.max_cluster_agents {
        coordinator
            .cluster_agents
            .fetch_sub(target, Ordering::SeqCst);
        return Err(SwarmError::QuotaExceeded(format!(
            "cluster has {claimed} of {} agents in use; job needs {target} more",
            coordinator.config.max_cluster_agents
        )));
    }

    let mut agents = Vec::with_capacity(target);
    for index in 0..target {
        let agent_type = types[index % types.len()].0;
        let mut agent = Agent::new(
            agent_type,
            coordinator.config.node_id,
            Arc::clone(&coordinator.gateway),
            Arc::clone(&coordinator.memory),
        );
        agent.descriptor.job_id = Some(job_id);
        agent.descriptor.status = agent
            .descriptor
            .status
            .transition(AgentStatus::Registered)?
            .transition(AgentStatus::Idle)?;
        handle.emit(
            JobEventKind::AgentSpawned,
            None,
            format!("{} agent {}", agent_type.as_str(), agent.id()),
            0.0,
        );
        agents.push(agent);
    }

    let spawned = agents.len();
    *lock(&handle.agents) = agents;
    lock(&handle.statistics).agents_spawned = spawned;
    Ok(spawned)
}

/// Grow the pool when work is queueing up; reclaim agents that have gone cold.
async fn maintain_agent_pool(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    max_agents: usize,
) -> Result<()> {
    let stats = coordinator.queue.stats().await?;
    let (total, idle) = {
        let agents = lock(&handle.agents);
        (
            agents.len(),
            agents
                .iter()
                .filter(|agent| agent.descriptor.is_available())
                .count(),
        )
    };

    if stats.ready > idle && total < max_agents {
        let cluster = coordinator.cluster_agents.load(Ordering::SeqCst);
        if cluster < coordinator.config.max_cluster_agents {
            if let Some(agent_type) = most_needed_type(handle) {
                coordinator.cluster_agents.fetch_add(1, Ordering::SeqCst);
                let job_id = lock(&handle.job).id;
                let mut agent = Agent::new(
                    agent_type,
                    coordinator.config.node_id,
                    Arc::clone(&coordinator.gateway),
                    Arc::clone(&coordinator.memory),
                );
                agent.descriptor.job_id = Some(job_id);
                agent.descriptor.status = agent
                    .descriptor
                    .status
                    .transition(AgentStatus::Registered)?
                    .transition(AgentStatus::Idle)?;
                handle.emit(
                    JobEventKind::AgentSpawned,
                    None,
                    format!("scaled up: {} agent {}", agent_type.as_str(), agent.id()),
                    progress(handle),
                );
                lock(&handle.agents).push(agent);
                lock(&handle.statistics).agents_spawned += 1;
            }
        }
    }

    if stats.depth() == 0 {
        let cutoff = Utc::now()
            - chrono::Duration::milliseconds(coordinator.config.agent_idle_timeout_ms as i64);
        let mut agents = lock(&handle.agents);
        let mut reclaimed = 0;
        // Keep at least one agent so a late dynamic task still has somewhere to run.
        while agents.len() > 1 {
            let stale = agents.iter().position(|agent| {
                agent.descriptor.is_available() && agent.descriptor.last_active_at < cutoff
            });
            match stale {
                Some(index) => {
                    agents.remove(index);
                    reclaimed += 1;
                }
                None => break,
            }
        }
        if reclaimed > 0 {
            drop(agents);
            coordinator
                .cluster_agents
                .fetch_sub(reclaimed, Ordering::SeqCst);
            handle.emit(
                JobEventKind::AgentTerminated,
                None,
                format!("reclaimed {reclaimed} idle agents"),
                progress(handle),
            );
        }
    }
    Ok(())
}

/// The agent type most in demand among tasks waiting to run.
fn most_needed_type(handle: &Arc<JobHandle>) -> Option<AgentType> {
    let graph = lock(&handle.graph);
    let mut demand: HashMap<AgentType, usize> = HashMap::new();
    for node in graph.nodes() {
        if matches!(node.state, TaskState::Queued | TaskState::RetryScheduled) {
            *demand.entry(agent_type_for(node.kind)).or_default() += 1;
        }
    }
    demand
        .into_iter()
        .max_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| right.0.as_str().cmp(left.0.as_str()))
        })
        .map(|(agent_type, _)| agent_type)
}

/// Move every ready task into the queue.
async fn promote_ready(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    actor: &str,
    correlation_id: CorrelationId,
) -> Result<usize> {
    let ready = lock(&handle.graph).ready();
    if ready.is_empty() {
        return Ok(0);
    }

    let priority = lock(&handle.job).request.priority;
    let mut queued = 0;
    for task_id in ready {
        advance(
            handle,
            task_id,
            TaskState::Queued,
            actor,
            "dependencies_satisfied",
            correlation_id,
        )?;
        let task =
            lock(&handle.graph)
                .get(task_id)
                .cloned()
                .ok_or_else(|| SwarmError::NotFound {
                    kind: "task",
                    id: task_id.to_string(),
                })?;
        let entry = QueuedTask::from_task(&task, priority, coordinator.config.lease_ms)?;
        coordinator.queue.enqueue(entry).await?;
        handle.emit(
            JobEventKind::TaskQueued,
            Some(task_id),
            task.title.clone(),
            progress(handle),
        );
        queued += 1;
    }
    Ok(queued)
}

/// Lease as much work as there are idle agents to run it, and pair the two up.
async fn collect_dispatches(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    worker: &str,
    actor: &str,
    correlation_id: CorrelationId,
) -> Result<Vec<Dispatch>> {
    let idle_count = lock(&handle.agents)
        .iter()
        .filter(|agent| agent.descriptor.is_available())
        .count();
    if idle_count == 0 {
        return Ok(Vec::new());
    }

    // No capability filter is applied at the queue: this pool can re-specialise, so
    // every capability is reachable. Remote worker nodes in Phase 2 have fixed
    // capabilities and will pass their real offer here.
    let offered: [Capability; 0] = [];

    let (objective, context_text) = {
        let job = lock(&handle.job);
        (job.request.objective.clone(), job.request.context.clone())
    };

    let mut dispatches = Vec::new();
    for _ in 0..idle_count {
        let Some(lease) = coordinator
            .queue
            .dequeue_from(worker.to_owned(), &[], &offered)
            .await?
        else {
            break;
        };

        let task = lease.task.task()?;

        // A duplicate delivery of work that is already running or already finished.
        // Acknowledging it is correct — the task is accounted for elsewhere — and it
        // is the reason at-least-once delivery does not become at-least-once
        // execution at the engine level.
        let leasable = lock(&handle.graph).get(task.id).map(|node| node.state);
        if !matches!(
            leasable,
            Some(TaskState::Queued | TaskState::RetryScheduled)
        ) {
            tracing::debug!(
                task_id = %task.id,
                state = ?leasable,
                "dropping duplicate delivery of a task that is not leasable"
            );
            let _ = coordinator.queue.acknowledge(lease.lease_id).await;
            continue;
        }

        let Some(agent) = acquire_agent(coordinator, handle, &task) else {
            // No capable agent is free after all: give the task back without spending
            // one of its attempts.
            coordinator.queue.release(lease.lease_id).await?;
            break;
        };

        // Reflect the queue's authoritative attempt count in the graph.
        {
            let mut graph = lock(&handle.graph);
            if let Some(node) = graph.get_mut(task.id) {
                node.attempt = lease.task.attempt;
            }
        }
        let awaiting_retry = lock(&handle.graph)
            .get(task.id)
            .is_some_and(|node| node.state == TaskState::RetryScheduled);
        if awaiting_retry {
            advance(
                handle,
                task.id,
                TaskState::Queued,
                actor,
                "backoff_elapsed",
                correlation_id,
            )?;
        }
        advance(
            handle,
            task.id,
            TaskState::Leased,
            actor,
            "leased_to_agent",
            correlation_id,
        )?;
        advance(
            handle,
            task.id,
            TaskState::Running,
            actor,
            "agent_started",
            correlation_id,
        )?;

        let mut task = task;
        task.attempt = lease.task.attempt;
        let upstream = upstream_outputs(handle, &task);
        handle.emit(
            JobEventKind::TaskStarted,
            Some(task.id),
            format!("{} (attempt {})", task.title, task.attempt),
            progress(handle),
        );

        dispatches.push(Dispatch {
            queue_wait_ms: (lease.granted_at - lease.task.enqueued_at)
                .num_milliseconds()
                .max(0) as u64,
            context: TaskContext {
                objective: objective.clone(),
                context: context_text.clone(),
                upstream,
                allowed_tools: Vec::new(),
            },
            agent,
            task,
            lease,
        });
    }
    Ok(dispatches)
}

/// Find an agent for `task`, re-specialising an idle one if nothing fits.
fn acquire_agent(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    task: &TaskNode,
) -> Option<Agent> {
    if let Some(agent) = assign(handle, task, coordinator.scheduler.as_ref()) {
        return Some(agent);
    }
    respecialise(coordinator, handle, task)?;
    assign(handle, task, coordinator.scheduler.as_ref())
}

/// Replace an idle agent with one of the type this task needs.
///
/// Agents are cheap — a tokio task, a capability list, and a model config — so an idle
/// Research agent with no research left is worth more as the Merger the job is now
/// waiting on. This is what lets a job whose budget is smaller than its number of task
/// kinds still make progress, instead of deadlocking on a capability nobody has.
fn respecialise(coordinator: &Coordinator, handle: &Arc<JobHandle>, task: &TaskNode) -> Option<()> {
    let wanted = agent_type_for(task.kind);
    let job_id = lock(&handle.job).id;

    let mut agents = lock(&handle.agents);
    let index = agents
        .iter()
        .position(|agent| agent.descriptor.is_available())?;

    let mut replacement = Agent::new(
        wanted,
        coordinator.config.node_id,
        Arc::clone(&coordinator.gateway),
        Arc::clone(&coordinator.memory),
    );
    replacement.descriptor.job_id = Some(job_id);
    replacement.descriptor.status = replacement
        .descriptor
        .status
        .transition(AgentStatus::Registered)
        .ok()?
        .transition(AgentStatus::Idle)
        .ok()?;
    tracing::debug!(
        from = agents[index].descriptor.agent_type.as_str(),
        to = wanted.as_str(),
        "re-specialising an idle agent"
    );
    agents[index] = replacement;
    Some(())
}

/// Ask the scheduler for an agent and mark it busy.
///
/// Returns `None` when no capable agent is free, which the caller treats as "give the
/// task back" rather than "fail the task".
fn assign(handle: &Arc<JobHandle>, task: &TaskNode, scheduler: &dyn Scheduler) -> Option<Agent> {
    let mut agents = lock(&handle.agents);
    let descriptors: Vec<AgentDescriptor> = agents
        .iter()
        .map(|agent| agent.descriptor.clone())
        .collect();
    let chosen = scheduler.select(task, &descriptors)?;

    let pooled = agents
        .iter_mut()
        .find(|agent| agent.descriptor.id == chosen)?;
    let next = pooled
        .descriptor
        .status
        .transition(AgentStatus::Assigned)
        .ok()?
        .transition(AgentStatus::Running)
        .ok()?;
    pooled.descriptor.status = next;
    pooled.descriptor.current_task = Some(task.id);
    pooled.descriptor.current_load = 1.0;
    pooled.descriptor.last_active_at = Utc::now();
    Some(pooled.clone())
}

/// Outputs of the tasks this one depends on.
fn upstream_outputs(handle: &Arc<JobHandle>, task: &TaskNode) -> Vec<UpstreamOutput> {
    let dependencies = lock(&handle.graph).dependencies(task.id);
    let results = lock(&handle.results);
    dependencies
        .iter()
        .filter_map(|dependency| {
            results
                .iter()
                .find(|result| result.task_id == *dependency)
                .map(|result| UpstreamOutput {
                    task_id: result.task_id,
                    title: result.title.clone(),
                    output: result.output.clone(),
                })
        })
        .collect()
}

/// Apply one execution outcome: acknowledge or reject the lease, move the task, and
/// update the agent's reliability stats.
async fn apply_outcome(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    dispatch: Dispatch,
    outcome: std::result::Result<Result<TaskResult>, tokio::time::error::Elapsed>,
    actor: &str,
    correlation_id: CorrelationId,
) -> Result<()> {
    let Dispatch {
        lease,
        agent,
        task,
        queue_wait_ms,
        ..
    } = dispatch;
    let retryable = lease.task.retry.allows_retry(lease.task.attempt);

    match outcome {
        Ok(Ok(result)) if result.passed_validation() => {
            match coordinator.queue.acknowledge(lease.lease_id).await {
                Ok(()) => {
                    let latency = result.duration_ms;
                    advance(
                        handle,
                        task.id,
                        TaskState::Completed,
                        actor,
                        "validated",
                        correlation_id,
                    )?;
                    record_success(handle, coordinator, &agent, &result, queue_wait_ms);
                    handle.emit(
                        JobEventKind::TaskCompleted,
                        Some(task.id),
                        format!("{} in {latency}ms", task.title),
                        progress(handle),
                    );
                }
                Err(err) => {
                    // The lease expired while we were working, so another agent may
                    // already have redone this task. Fail the attempt rather than
                    // claiming a completion we no longer own.
                    tracing::warn!(task_id = %task.id, error = %err, "lost lease before acknowledgement");
                    release_agent(
                        handle,
                        coordinator,
                        &agent,
                        false,
                        result.duration_ms,
                        queue_wait_ms,
                    );
                    fail_task(
                        handle,
                        &task,
                        TaskState::Failed,
                        "lease_lost",
                        &err.to_string(),
                        retryable,
                        Vec::new(),
                        actor,
                        correlation_id,
                    )?;
                }
            }
        }
        Ok(Ok(result)) => {
            let failures = result.validation_failures.clone();
            let detail = failures.join("; ");
            coordinator
                .queue
                .reject(lease.lease_id, format!("validation failed: {detail}"))
                .await
                .unwrap_or_else(
                    |err| tracing::warn!(error = %err, "reject after validation failure"),
                );
            release_agent(
                handle,
                coordinator,
                &agent,
                false,
                result.duration_ms,
                queue_wait_ms,
            );
            fail_task(
                handle,
                &task,
                TaskState::Failed,
                "validation_failed",
                &detail,
                retryable,
                failures,
                actor,
                correlation_id,
            )?;
        }
        Ok(Err(err)) => {
            coordinator
                .queue
                .reject(lease.lease_id, err.to_string())
                .await
                .unwrap_or_else(|e| tracing::warn!(error = %e, "reject after execution error"));
            release_agent(handle, coordinator, &agent, false, 0, queue_wait_ms);
            let kind = err.kind().to_owned();
            fail_task(
                handle,
                &task,
                TaskState::Failed,
                &kind,
                &err.to_string(),
                retryable && err.is_retryable(),
                Vec::new(),
                actor,
                correlation_id,
            )?;
        }
        Err(_elapsed) => {
            let timeout_ms = task.timeout_seconds * 1_000;
            coordinator
                .queue
                .reject(lease.lease_id, format!("timed out after {timeout_ms}ms"))
                .await
                .unwrap_or_else(|e| tracing::warn!(error = %e, "reject after timeout"));
            release_agent(
                handle,
                coordinator,
                &agent,
                false,
                timeout_ms,
                queue_wait_ms,
            );
            fail_task(
                handle,
                &task,
                TaskState::TimedOut,
                "timeout",
                &format!("exceeded {}s", task.timeout_seconds),
                retryable,
                Vec::new(),
                actor,
                correlation_id,
            )?;
        }
    }
    Ok(())
}

/// Store a successful result and update everything that learns from it.
fn record_success(
    handle: &Arc<JobHandle>,
    coordinator: &Coordinator,
    agent: &Agent,
    result: &TaskResult,
    queue_wait_ms: u64,
) {
    release_agent(
        handle,
        coordinator,
        agent,
        true,
        result.duration_ms,
        queue_wait_ms,
    );

    {
        let mut statistics = lock(&handle.statistics);
        statistics.tasks_succeeded += 1;
        statistics.tokens_in += result.tokens_in;
        statistics.tokens_out += result.tokens_out;
        statistics.cost_usd += result.cost_usd;
        statistics.queue_wait_ms_total += queue_wait_ms;
        statistics.scheduling_decisions += 1;
        if result.deduplicated {
            statistics.model_cache_hits += 1;
        } else {
            statistics.model_requests += 1;
        }
    }
    lock(&handle.latencies).push(result.duration_ms);
    lock(&handle.results).push(result.clone());
}

/// Move a failed task to its next state and record the failure.
#[allow(clippy::too_many_arguments)]
fn fail_task(
    handle: &Arc<JobHandle>,
    task: &TaskNode,
    failure_state: TaskState,
    error_kind: &str,
    error_message: &str,
    retryable: bool,
    validation_failures: Vec<String>,
    actor: &str,
    correlation_id: CorrelationId,
) -> Result<()> {
    advance(
        handle,
        task.id,
        failure_state,
        actor,
        error_kind,
        correlation_id,
    )?;

    let next = if retryable {
        TaskState::RetryScheduled
    } else {
        TaskState::DeadLettered
    };
    advance(handle, task.id, next, actor, error_kind, correlation_id)?;

    lock(&handle.failures).push(TaskFailure {
        task_id: task.id,
        title: task.title.clone(),
        attempt: task.attempt,
        error_kind: error_kind.to_owned(),
        error_message: error_message.to_owned(),
        dead_lettered: next == TaskState::DeadLettered,
        validation_failures,
        at: Utc::now(),
    });

    {
        let mut statistics = lock(&handle.statistics);
        statistics.model_requests += 1;
        if next == TaskState::RetryScheduled {
            statistics.tasks_retried += 1;
        } else {
            statistics.tasks_failed += 1;
        }
    }

    let (kind, detail) = if next == TaskState::RetryScheduled {
        (
            JobEventKind::TaskRetrying,
            format!("{}: {error_message}", task.title),
        )
    } else {
        (
            JobEventKind::TaskDeadLettered,
            format!(
                "{} gave up after {} attempts: {error_message}",
                task.title, task.attempt
            ),
        )
    };
    handle.emit(kind, Some(task.id), detail, progress(handle));
    Ok(())
}

/// Return an agent to the idle pool, updating its reliability stats and feeding the
/// outcome back to the scheduler.
fn release_agent(
    handle: &Arc<JobHandle>,
    coordinator: &Coordinator,
    agent: &Agent,
    succeeded: bool,
    latency_ms: u64,
    queue_wait_ms: u64,
) {
    coordinator.scheduler.observe(crate::SchedulingOutcome {
        agent_id: agent.descriptor.id,
        succeeded,
        latency_ms,
        queue_wait_ms,
    });

    let mut agents = lock(&handle.agents);
    let Some(pooled) = agents
        .iter_mut()
        .find(|candidate| candidate.descriptor.id == agent.descriptor.id)
    else {
        return;
    };

    let terminal = if succeeded {
        AgentStatus::Completed
    } else {
        AgentStatus::Failed
    };
    if let Ok(next) = pooled.descriptor.status.transition(terminal) {
        pooled.descriptor.status = next;
    }
    if let Ok(next) = pooled.descriptor.status.transition(AgentStatus::Idle) {
        pooled.descriptor.status = next;
    }
    pooled.descriptor.current_task = None;
    pooled.descriptor.current_load = 0.0;
    pooled.descriptor.record_attempt(succeeded, latency_ms);
}

/// Decide whether waiting another tick could possibly change anything.
///
/// Without this a job whose remaining work is unreachable — because an upstream task
/// dead-lettered — would spin until the tick budget ran out.
async fn make_progress_possible(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    actor: &str,
    correlation_id: CorrelationId,
) -> Result<bool> {
    if coordinator.queue.stats().await?.depth() > 0 {
        // Work is waiting on a retry backoff or a lease that has not expired yet.
        return Ok(true);
    }

    let stranded = lock(&handle.graph).blocked_by_abandoned();
    if stranded.is_empty() {
        return Ok(false);
    }
    for task_id in stranded {
        advance(
            handle,
            task_id,
            TaskState::Cancelled,
            actor,
            "upstream_abandoned",
            correlation_id,
        )?;
        handle.emit(
            JobEventKind::TaskCancelled,
            Some(task_id),
            "upstream task was abandoned",
            progress(handle),
        );
    }
    Ok(true)
}

/// Bring the job to a terminal state and produce the final result.
async fn finish(
    coordinator: &Coordinator,
    handle: &Arc<JobHandle>,
    cancelled: bool,
    spawned: usize,
    started: Instant,
    actor: &str,
    correlation_id: CorrelationId,
) -> Result<FinalResult> {
    if cancelled {
        let pending: Vec<TaskId> = lock(&handle.graph)
            .nodes()
            .filter(|node| !node.is_terminal())
            .map(|node| node.id)
            .collect();
        for task_id in pending {
            // Running tasks must step through a failure state before they can be
            // abandoned; the state machine will not shortcut it.
            let state = lock(&handle.graph).get(task_id).map(|node| node.state);
            if state == Some(TaskState::Running) {
                advance(
                    handle,
                    task_id,
                    TaskState::Failed,
                    actor,
                    "job_cancelled",
                    correlation_id,
                )?;
                advance(
                    handle,
                    task_id,
                    TaskState::DeadLettered,
                    actor,
                    "job_cancelled",
                    correlation_id,
                )?;
                continue;
            }
            if state == Some(TaskState::Leased) {
                advance(
                    handle,
                    task_id,
                    TaskState::Cancelled,
                    actor,
                    "job_cancelled",
                    correlation_id,
                )?;
                continue;
            }
            advance(
                handle,
                task_id,
                TaskState::Cancelled,
                actor,
                "job_cancelled",
                correlation_id,
            )?;
        }
        // Bound to a local first: a guard temporary inside the argument list would
        // still be alive across the await.
        let job_id = lock(&handle.job).id;
        coordinator.queue.purge_job(job_id).await?;
    }

    let counts = lock(&handle.graph).counts();
    let status = if cancelled {
        JobStatus::Cancelled
    } else if counts.abandoned == 0 && counts.completed == counts.total {
        JobStatus::Completed
    } else if counts.completed > 0 {
        JobStatus::PartiallyCompleted
    } else {
        JobStatus::Failed
    };

    {
        let mut statistics = lock(&handle.statistics);
        statistics.wall_clock_ms = started.elapsed().as_millis() as u64;
        statistics.tasks_total = counts.total;
        statistics.agents_spawned = statistics.agents_spawned.max(spawned);
        let latencies = lock(&handle.latencies).clone();
        statistics.set_latency_percentiles(&latencies);
    }

    if !cancelled {
        handle.set_status(JobStatus::Aggregating, None)?;
    }

    let final_result = {
        let job = lock(&handle.job).clone();
        let graph = lock(&handle.graph);
        let results = lock(&handle.results).clone();
        let statistics = lock(&handle.statistics).clone();
        aggregate(&job, &graph, &results, status, statistics)
    };

    let reason = (!final_result.unresolved_conflicts.is_empty()).then(|| {
        format!(
            "{} unresolved conflicts",
            final_result.unresolved_conflicts.len()
        )
    });
    handle.set_status(status, reason)?;
    *lock(&handle.final_result) = Some(final_result.clone());

    // Free the job's agents back to the cluster budget.
    let released = lock(&handle.agents).len();
    let _ =
        coordinator
            .cluster_agents
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(current.saturating_sub(released))
            });

    handle.emit(
        JobEventKind::JobFinished,
        None,
        format!(
            "{status}: {}/{} tasks completed, ${:.4}",
            counts.completed, counts.total, final_result.execution_statistics.cost_usd
        ),
        progress(handle),
    );
    Ok(final_result)
}
