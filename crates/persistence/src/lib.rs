//! Durability for the coordinator.
//!
//! Everything that happens to a job is appended to a journal as it happens, so a
//! coordinator that dies can be restarted and pick the job up where it stopped.
//! Recovery is a replay, not a reconciliation: the log is the truth, and rebuilding
//! from it is deterministic.
//!
//! The journal is **append-only and ordered**. It never rewrites a record, so a crash
//! can only ever truncate the tail — and a torn final line is detected and dropped on
//! replay rather than poisoning recovery.
//!
//! [`Journal`] is deliberately synchronous. Appends are a few hundred bytes and happen
//! a handful of times per task; a Postgres-backed implementation with the same shape
//! replaces this one when a database is available.
#![forbid(unsafe_code)]

pub mod file;

use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use swarm_domain::{
    FinalResult, Job, JobId, JobStatus, Result, TaskFailure, TaskNode, TaskResult, TaskState,
    Transition,
};

pub use file::FileJournal;

/// One durable fact about a job.
///
/// Records are self-contained: replay never needs to consult anything but the log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum JournalRecord {
    /// A job was admitted and compiled into a task graph.
    JobPlanned {
        /// The job record, including the original request.
        job: Box<Job>,
        /// Every task in the compiled graph.
        tasks: Vec<TaskNode>,
    },
    /// A job's lifecycle status changed.
    JobStatusChanged {
        /// Which job.
        job_id: JobId,
        /// Its new status.
        status: JobStatus,
        /// Why, when the reason is not obvious.
        reason: Option<String>,
        /// When.
        at: DateTime<Utc>,
    },
    /// A task moved between states.
    TaskTransitioned {
        /// Which job.
        job_id: JobId,
        /// The audited transition.
        transition: Box<Transition<TaskState>>,
        /// The task that moved.
        task_id: swarm_domain::TaskId,
        /// Attempts used at the time of the move.
        attempt: u32,
    },
    /// A task produced a validated result.
    TaskCompleted {
        /// Which job.
        job_id: JobId,
        /// The result.
        result: Box<TaskResult>,
    },
    /// A task attempt failed.
    TaskFailed {
        /// Which job.
        job_id: JobId,
        /// The failure.
        failure: Box<TaskFailure>,
    },
    /// A job finished and produced its merged result.
    JobFinished {
        /// Which job.
        job_id: JobId,
        /// The final result.
        result: Box<FinalResult>,
    },
}

impl JournalRecord {
    /// The job this record belongs to.
    #[must_use]
    pub fn job_id(&self) -> JobId {
        match self {
            Self::JobPlanned { job, .. } => job.id,
            Self::JobStatusChanged { job_id, .. }
            | Self::TaskTransitioned { job_id, .. }
            | Self::TaskCompleted { job_id, .. }
            | Self::TaskFailed { job_id, .. }
            | Self::JobFinished { job_id, .. } => *job_id,
        }
    }

    /// Short label for logs and metrics.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::JobPlanned { .. } => "job_planned",
            Self::JobStatusChanged { .. } => "job_status_changed",
            Self::TaskTransitioned { .. } => "task_transitioned",
            Self::TaskCompleted { .. } => "task_completed",
            Self::TaskFailed { .. } => "task_failed",
            Self::JobFinished { .. } => "job_finished",
        }
    }
}

/// An ordered, append-only log of everything that happened.
///
/// Implementations must preserve append order and must never lose an acknowledged
/// record. Losing the *tail* to a crash is expected and handled by replay.
pub trait Journal: Send + Sync + fmt::Debug {
    /// Durably append one record.
    fn append(&self, record: &JournalRecord) -> Result<()>;

    /// Read every record back, in the order it was written.
    fn replay(&self) -> Result<Vec<JournalRecord>>;

    /// How many records the journal holds.
    fn len(&self) -> Result<usize> {
        Ok(self.replay()?.len())
    }

    /// Whether the journal holds nothing.
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// A journal that keeps records in memory.
///
/// Not durable across a process restart — it exists so tests can exercise the replay
/// path without touching a filesystem, and so a deployment can turn journalling off
/// without the coordinator needing an `Option` everywhere.
#[derive(Debug, Default)]
pub struct MemoryJournal {
    records: Mutex<Vec<JournalRecord>>,
}

impl MemoryJournal {
    /// An empty journal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Vec<JournalRecord>> {
        self.records.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Journal for MemoryJournal {
    fn append(&self, record: &JournalRecord) -> Result<()> {
        self.lock().push(record.clone());
        Ok(())
    }

    fn replay(&self) -> Result<Vec<JournalRecord>> {
        Ok(self.lock().clone())
    }

    fn len(&self) -> Result<usize> {
        Ok(self.lock().len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_domain::{CorrelationId, JobRequest, TaskId, TaskKind};

    fn planned() -> JournalRecord {
        let job = Job::new(JobRequest::new("do the thing"));
        let task = TaskNode::new(job.id, TaskKind::Work, "a task", "do it", 0);
        JournalRecord::JobPlanned {
            job: Box::new(job),
            tasks: vec![task],
        }
    }

    #[test]
    fn records_round_trip_through_json() {
        let record = planned();
        let encoded = serde_json::to_string(&record).unwrap();
        assert!(encoded.contains("\"record\":\"job_planned\""));
        let decoded: JournalRecord = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn every_record_reports_its_job_and_kind() {
        let job_id = JobId::new();
        let records = [
            planned(),
            JournalRecord::JobStatusChanged {
                job_id,
                status: JobStatus::Running,
                reason: None,
                at: Utc::now(),
            },
            JournalRecord::TaskTransitioned {
                job_id,
                transition: Box::new(Transition::new(
                    Some(TaskState::Created),
                    TaskState::Queued,
                    "coordinator",
                    None,
                    CorrelationId::new(),
                )),
                task_id: TaskId::new(),
                attempt: 0,
            },
        ];

        assert_eq!(records[1].job_id(), job_id);
        assert_eq!(records[1].kind(), "job_status_changed");
        assert_eq!(records[2].kind(), "task_transitioned");
        assert!(!records[0].kind().is_empty());
    }

    #[test]
    fn the_memory_journal_preserves_append_order() {
        let journal = MemoryJournal::new();
        assert!(journal.is_empty().unwrap());

        let job_id = JobId::new();
        for index in 0..5 {
            journal
                .append(&JournalRecord::JobStatusChanged {
                    job_id,
                    status: JobStatus::Running,
                    reason: Some(index.to_string()),
                    at: Utc::now(),
                })
                .unwrap();
        }

        let replayed = journal.replay().unwrap();
        assert_eq!(replayed.len(), 5);
        for (index, record) in replayed.iter().enumerate() {
            let JournalRecord::JobStatusChanged { reason, .. } = record else {
                panic!("wrong record kind");
            };
            assert_eq!(reason.as_deref(), Some(index.to_string().as_str()));
        }
    }
}
