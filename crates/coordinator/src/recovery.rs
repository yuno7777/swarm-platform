//! Rebuilding a coordinator's job state from the journal.
//!
//! Replay is a fold: every record is applied in order to a per-job accumulator, and
//! the accumulator is turned into a live [`JobHandle`] at the end. Nothing here talks
//! to the queue, shared memory, or a model — recovery is pure, so it is testable and
//! cannot half-succeed.
//!
//! The one judgement call is what to do about tasks that were mid-flight when the
//! process died. They are restored to `Created`, which puts them back in the
//! scheduler's ready set. Completed tasks keep their results and never run again, so a
//! restart resumes a job instead of redoing it.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use tokio::sync::{broadcast, watch};

use swarm_domain::{
    ExecutionStatistics, FinalResult, Job, JobId, JobStatus, Result, SwarmError, TaskFailure,
    TaskGraph, TaskId, TaskNode, TaskResult, TaskState, Transition,
};
use swarm_persistence::{Journal, JournalRecord};

use crate::{Control, JobHandle};

/// Everything replay has learned about one job so far.
#[derive(Debug, Default)]
pub struct Recovering {
    job: Option<Job>,
    tasks: Vec<TaskNode>,
    task_states: HashMap<TaskId, TaskState>,
    attempts: HashMap<TaskId, u32>,
    transitions: Vec<Transition<TaskState>>,
    results: Vec<TaskResult>,
    failures: Vec<TaskFailure>,
    final_result: Option<FinalResult>,
}

/// Fold one journal record into the per-job accumulators.
pub fn apply(jobs: &mut HashMap<JobId, Recovering>, record: JournalRecord) {
    let entry = jobs.entry(record.job_id()).or_default();

    match record {
        JournalRecord::JobPlanned { job, tasks } => {
            entry.job = Some(*job);
            entry.tasks = tasks;
        }
        JournalRecord::JobStatusChanged { status, reason, .. } => {
            if let Some(job) = entry.job.as_mut() {
                job.status = status;
                job.status_reason = reason;
            }
        }
        JournalRecord::TaskTransitioned {
            transition,
            task_id,
            attempt,
            ..
        } => {
            entry.task_states.insert(task_id, transition.to);
            entry.attempts.insert(task_id, attempt);
            entry.transitions.push(*transition);
        }
        JournalRecord::TaskCompleted { result, .. } => entry.results.push(*result),
        JournalRecord::TaskFailed { failure, .. } => entry.failures.push(*failure),
        JournalRecord::JobFinished { result, .. } => entry.final_result = Some(*result),
    }
}

impl Recovering {
    /// Turn the accumulated records into a live job handle.
    pub fn build(self, event_buffer: usize, journal: Arc<dyn Journal>) -> Result<JobHandle> {
        let mut job = self.job.ok_or_else(|| {
            SwarmError::Internal(
                "journal has records for a job it never planned; log is truncated at the head"
                    .to_owned(),
            )
        })?;

        let mut graph = rebuild_graph(job.id, self.tasks)?;
        let mut resumed = 0;
        for (task_id, state) in &self.task_states {
            let restored = if is_terminal(*state) {
                *state
            } else {
                // Mid-flight when the process died: put it back in the ready set.
                resumed += 1;
                TaskState::Created
            };
            graph.restore_state(*task_id, restored)?;

            if let (Some(node), Some(attempt)) =
                (graph.get_mut(*task_id), self.attempts.get(task_id))
            {
                // Attempts already spent still count, so a task that failed twice
                // before the crash does not get a fresh retry budget.
                node.attempt = *attempt;
            }
        }

        // A job that was running cannot re-enter Running directly; rewinding it to
        // Planning is what lets the engine start it again.
        if !job.status.is_finished() {
            job.status = JobStatus::Planning;
            job.status_reason = Some(format!("recovered from journal, {resumed} tasks re-queued"));
        }

        let statistics = recompute_statistics(&self.results, &self.failures, graph.counts().total);
        let latencies: Vec<u64> = self.results.iter().map(|r| r.duration_ms).collect();

        let (events, _) = broadcast::channel(event_buffer.max(16));
        let (control, _) = watch::channel(Control::Running);

        Ok(JobHandle {
            job: Mutex::new(job),
            graph: Mutex::new(graph),
            agents: Mutex::new(Vec::new()),
            results: Mutex::new(self.results),
            failures: Mutex::new(self.failures),
            transitions: Mutex::new(self.transitions),
            statistics: Mutex::new(statistics),
            latencies: Mutex::new(latencies),
            final_result: Mutex::new(self.final_result),
            events,
            control,
            sequence: AtomicU64::new(0),
            journal,
        })
    }
}

const fn is_terminal(state: TaskState) -> bool {
    matches!(
        state,
        TaskState::Completed | TaskState::Cancelled | TaskState::DeadLettered
    )
}

/// Insert tasks in dependency order.
///
/// The journal stores nodes in whatever order the graph iterated them, which is not
/// necessarily topological, so this is a Kahn-style loop: keep inserting whatever is
/// now insertable until nothing is left.
fn rebuild_graph(job_id: JobId, tasks: Vec<TaskNode>) -> Result<TaskGraph> {
    let mut graph = TaskGraph::new(job_id);
    let mut pending = tasks;

    while !pending.is_empty() {
        let before = pending.len();
        let mut deferred = Vec::with_capacity(before);

        for task in pending {
            let ready = task
                .dependencies
                .iter()
                .all(|dependency| graph.get(*dependency).is_some());
            if ready {
                graph.insert(task)?;
            } else {
                deferred.push(task);
            }
        }

        if deferred.len() == before {
            // Nothing became insertable, so the remaining tasks reference something
            // that is not in the log. Better to say so than to serve a partial graph.
            return Err(SwarmError::Internal(format!(
                "journal for job {job_id} references {} tasks whose dependencies are missing",
                deferred.len()
            )));
        }
        pending = deferred;
    }

    graph.assert_acyclic()?;
    Ok(graph)
}

/// Rebuild the counters that are derivable from what actually happened.
///
/// Timings that only the original process observed — wall clock, queue waits — cannot
/// be recovered and stay zero rather than being invented.
fn recompute_statistics(
    results: &[TaskResult],
    failures: &[TaskFailure],
    tasks_total: usize,
) -> ExecutionStatistics {
    let mut statistics = ExecutionStatistics {
        tasks_total,
        tasks_succeeded: results.len(),
        tasks_failed: failures.iter().filter(|f| f.dead_lettered).count(),
        tasks_retried: failures.iter().filter(|f| !f.dead_lettered).count(),
        model_requests: results.iter().filter(|r| !r.deduplicated).count() as u64,
        model_cache_hits: results.iter().filter(|r| r.deduplicated).count() as u64,
        tokens_in: results.iter().map(|r| r.tokens_in).sum(),
        tokens_out: results.iter().map(|r| r.tokens_out).sum(),
        cost_usd: results.iter().map(|r| r.cost_usd).sum(),
        ..ExecutionStatistics::default()
    };

    let latencies: Vec<u64> = results.iter().map(|r| r.duration_ms).collect();
    statistics.set_latency_percentiles(&latencies);
    statistics
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use swarm_domain::{AgentId, CorrelationId, JobRequest, TaskKind};
    use swarm_persistence::MemoryJournal;

    fn job_with_chain() -> (Job, Vec<TaskNode>) {
        let job = Job::new(JobRequest::new("recoverable objective"));
        let first = TaskNode::new(job.id, TaskKind::Plan, "plan", "plan it", 0);
        let second = TaskNode::new(job.id, TaskKind::Work, "work", "do it", 1)
            .with_dependencies(vec![first.id]);
        let third = TaskNode::new(job.id, TaskKind::Merge, "merge", "merge it", 2)
            .with_dependencies(vec![second.id]);
        (job, vec![first, second, third])
    }

    fn result_for(job: &Job, task: &TaskNode) -> TaskResult {
        TaskResult {
            task_id: task.id,
            job_id: job.id,
            agent_id: AgentId::new(),
            attempt: 1,
            kind: task.kind,
            title: task.title.clone(),
            output: "an output".to_owned(),
            structured: serde_json::json!({}),
            evidence: Vec::new(),
            reasoning_summary: String::new(),
            confidence: 0.8,
            validation_failures: Vec::new(),
            tokens_in: 10,
            tokens_out: 20,
            cost_usd: 0.001,
            duration_ms: 7,
            deduplicated: false,
            finished_at: Utc::now(),
        }
    }

    fn transition(job_id: JobId, task: &TaskNode, to: TaskState, attempt: u32) -> JournalRecord {
        JournalRecord::TaskTransitioned {
            job_id,
            transition: Box::new(Transition::new(
                None,
                to,
                "coordinator:test",
                Some("test".to_owned()),
                CorrelationId::new(),
            )),
            task_id: task.id,
            attempt,
        }
    }

    fn fold(records: Vec<JournalRecord>) -> HashMap<JobId, Recovering> {
        let mut jobs = HashMap::new();
        for record in records {
            apply(&mut jobs, record);
        }
        jobs
    }

    #[test]
    fn a_planned_but_unstarted_job_comes_back_whole() {
        let (job, tasks) = job_with_chain();
        let job_id = job.id;
        let jobs = fold(vec![JournalRecord::JobPlanned {
            job: Box::new(job),
            tasks: tasks.clone(),
        }]);

        let handle = jobs
            .into_values()
            .next()
            .unwrap()
            .build(64, Arc::new(MemoryJournal::new()))
            .unwrap();

        let graph = handle.graph.lock().unwrap();
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.job_id(), job_id);
        assert_eq!(graph.ready().len(), 1, "only the plan task is ready");
        graph.assert_acyclic().unwrap();
    }

    #[test]
    fn completed_work_is_preserved_and_in_flight_work_is_re_queued() {
        let (job, tasks) = job_with_chain();
        let job_id = job.id;

        let records = vec![
            JournalRecord::JobPlanned {
                job: Box::new(job.clone()),
                tasks: tasks.clone(),
            },
            JournalRecord::JobStatusChanged {
                job_id,
                status: JobStatus::Running,
                reason: None,
                at: Utc::now(),
            },
            transition(job_id, &tasks[0], TaskState::Completed, 1),
            JournalRecord::TaskCompleted {
                job_id,
                result: Box::new(result_for(&job, &tasks[0])),
            },
            // The second task was running when the process died.
            transition(job_id, &tasks[1], TaskState::Running, 2),
        ];

        let handle = fold(records)
            .into_values()
            .next()
            .unwrap()
            .build(64, Arc::new(MemoryJournal::new()))
            .unwrap();

        let graph = handle.graph.lock().unwrap();
        assert_eq!(graph.get(tasks[0].id).unwrap().state, TaskState::Completed);
        assert_eq!(
            graph.get(tasks[1].id).unwrap().state,
            TaskState::Created,
            "the interrupted task must be schedulable again"
        );
        assert_eq!(
            graph.get(tasks[1].id).unwrap().attempt,
            2,
            "attempts already spent still count against the retry budget"
        );
        assert_eq!(
            graph.ready(),
            vec![tasks[1].id],
            "the job resumes, not restarts"
        );

        // The job is rewound to Planning so the engine can start it again.
        assert_eq!(handle.job.lock().unwrap().status, JobStatus::Planning);
        assert!(handle.job.lock().unwrap().status_reason.is_some());

        // Restored results are not lost, and the statistics reflect them.
        assert_eq!(handle.results.lock().unwrap().len(), 1);
        let statistics = handle.statistics.lock().unwrap();
        assert_eq!(statistics.tasks_succeeded, 1);
        assert_eq!(statistics.tokens_in, 10);
        assert!((statistics.cost_usd - 0.001).abs() < 1e-9);
        assert_eq!(statistics.median_task_latency_ms, 7);
        assert_eq!(
            statistics.wall_clock_ms, 0,
            "unobservable timings stay zero"
        );
    }

    #[test]
    fn a_finished_job_keeps_its_terminal_status_and_result() {
        let (mut job, tasks) = job_with_chain();
        job.status = JobStatus::Completed;
        let job_id = job.id;
        let final_result = FinalResult {
            job_id,
            status: JobStatus::Completed,
            summary: "all done".to_owned(),
            outputs: Vec::new(),
            supporting_evidence: Vec::new(),
            confidence_score: 0.9,
            unresolved_conflicts: Vec::new(),
            execution_statistics: ExecutionStatistics::default(),
        };

        let records = vec![
            JournalRecord::JobPlanned {
                job: Box::new(job),
                tasks,
            },
            JournalRecord::JobStatusChanged {
                job_id,
                status: JobStatus::Completed,
                reason: None,
                at: Utc::now(),
            },
            JournalRecord::JobFinished {
                job_id,
                result: Box::new(final_result.clone()),
            },
        ];

        let handle = fold(records)
            .into_values()
            .next()
            .unwrap()
            .build(64, Arc::new(MemoryJournal::new()))
            .unwrap();

        assert_eq!(handle.job.lock().unwrap().status, JobStatus::Completed);
        assert_eq!(
            handle
                .final_result
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .summary,
            "all done"
        );
    }

    #[test]
    fn a_log_missing_its_plan_is_refused_rather_than_half_recovered() {
        let job_id = JobId::new();
        let jobs = fold(vec![JournalRecord::JobStatusChanged {
            job_id,
            status: JobStatus::Running,
            reason: None,
            at: Utc::now(),
        }]);

        let outcome = jobs
            .into_values()
            .next()
            .unwrap()
            .build(64, Arc::new(MemoryJournal::new()));
        let Err(error) = outcome else {
            panic!("a job with no plan must not recover");
        };
        assert!(error.to_string().contains("never planned"));
    }

    #[test]
    fn tasks_are_rebuilt_in_dependency_order_whatever_order_they_were_logged_in() {
        let (job, mut tasks) = job_with_chain();
        tasks.reverse();

        let graph = rebuild_graph(job.id, tasks.clone()).unwrap();
        assert_eq!(graph.len(), 3);
        graph.assert_acyclic().unwrap();
        assert_eq!(graph.layers().unwrap().len(), 3);
    }

    #[test]
    fn a_task_whose_dependency_is_missing_is_reported() {
        let (job, tasks) = job_with_chain();
        // Drop the root; the rest can never be inserted.
        let orphaned = tasks[1..].to_vec();

        let error = rebuild_graph(job.id, orphaned).unwrap_err();
        assert!(error.to_string().contains("dependencies are missing"));
    }
}
