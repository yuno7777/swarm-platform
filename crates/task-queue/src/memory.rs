//! In-process implementation of [`TaskQueue`].
//!
//! This is the Phase 1 backend and the reference implementation of the contract: it
//! honours priorities, delays, leases, expiry, deduplication, retry backoff, and dead
//! lettering. Durable backends are validated against the same tests.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use swarm_domain::{
    Capability, JobId, LeaseId, LeaseState, Result, StateMachine, SwarmError, TaskId,
};

use crate::{DeadLetter, QueueStats, QueuedTask, TaskLease, TaskQueue, WorkerId};

#[derive(Debug, Default)]
struct Inner {
    // ponytail: linear scan over a Vec. Fine to five figures of queue depth; swap for
    // a BTreeMap keyed by (Reverse(priority), available_at) if depth ever justifies it.
    ready: Vec<QueuedTask>,
    leases: HashMap<LeaseId, TaskLease>,
    /// Attempt keys ever admitted, so a redelivered producer message is dropped.
    seen: HashSet<String>,
    dead: Vec<DeadLetter>,
    stats: QueueStats,
}

impl Inner {
    fn pick(
        &mut self,
        now: DateTime<Utc>,
        queues: &[String],
        capabilities: &[Capability],
    ) -> Option<QueuedTask> {
        let position = self
            .ready
            .iter()
            .enumerate()
            .filter(|(_, task)| {
                task.is_visible(now)
                    && (queues.is_empty() || queues.contains(&task.queue))
                    && task.matches_capabilities(capabilities)
            })
            // Highest priority wins; ties go to whoever waited longest.
            .max_by(|(_, left), (_, right)| {
                left.priority
                    .cmp(&right.priority)
                    .then_with(|| right.enqueued_at.cmp(&left.enqueued_at))
            })
            .map(|(index, _)| index)?;
        Some(self.ready.swap_remove(position))
    }

    /// Send a failed task back for another attempt, or park it if the budget is spent.
    fn retry_or_park(&mut self, mut task: QueuedTask, reason: String, now: DateTime<Utc>) {
        if task.retry.allows_retry(task.attempt) {
            let nonce = u64::from(now.timestamp_subsec_nanos());
            let backoff = task.retry.backoff_ms(task.attempt, nonce);
            task.available_at = now + chrono::Duration::milliseconds(backoff as i64);
            self.stats.retried_total += 1;
            self.ready.push(task);
        } else {
            self.dead.push(DeadLetter {
                attempts: task.attempt,
                task,
                last_error: reason,
                at: now,
            });
        }
    }

    fn take_lease(&mut self, lease_id: LeaseId) -> Result<TaskLease> {
        self.leases.remove(&lease_id).ok_or_else(|| {
            SwarmError::Queue(format!("unknown or already-settled lease {lease_id}"))
        })
    }
}

/// A [`TaskQueue`] that keeps everything in process memory.
#[derive(Debug, Default)]
pub struct InMemoryQueue {
    inner: Mutex<Inner>,
}

impl InMemoryQueue {
    /// An empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the state, tolerating a poisoned mutex.
    ///
    /// A panic in one worker must not take the queue down for every other worker; the
    /// data behind the lock is plain records with no broken invariant to inherit.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl TaskQueue for InMemoryQueue {
    async fn enqueue(&self, task: QueuedTask) -> Result<()> {
        let mut inner = self.lock();
        if !inner.seen.insert(task.idempotency_key.clone()) {
            inner.stats.deduplicated_total += 1;
            tracing::debug!(
                task_id = %task.task_id,
                key = %task.idempotency_key,
                "dropped duplicate enqueue"
            );
            return Ok(());
        }
        inner.stats.enqueued_total += 1;
        inner.ready.push(task);
        Ok(())
    }

    async fn dequeue_from(
        &self,
        worker: WorkerId,
        queues: &[String],
        capabilities: &[Capability],
    ) -> Result<Option<TaskLease>> {
        let now = Utc::now();
        let mut inner = self.lock();
        let Some(mut task) = inner.pick(now, queues, capabilities) else {
            return Ok(None);
        };

        // A granted lease is the start of an attempt, so the budget is spent here
        // rather than on success — otherwise a task that always crashes its worker
        // would be retried forever.
        task.attempt += 1;
        let lease = TaskLease {
            lease_id: LeaseId::new(),
            expires_at: now + chrono::Duration::milliseconds(task.lease_duration_ms as i64),
            task,
            worker,
            granted_at: now,
            state: LeaseState::Held,
        };
        inner.stats.dequeued_total += 1;
        inner.leases.insert(lease.lease_id, lease.clone());
        Ok(Some(lease))
    }

    async fn acknowledge(&self, lease: LeaseId) -> Result<()> {
        let now = Utc::now();
        let mut inner = self.lock();
        let held = inner.take_lease(lease)?;

        if held.is_expired(now) {
            // Another worker may already have redone this task. Refuse the ack and put
            // the task back rather than pretending the work landed.
            inner.stats.expired_total += 1;
            inner.retry_or_park(
                held.task,
                "lease expired before acknowledgement".into(),
                now,
            );
            return Err(SwarmError::Queue(format!(
                "lease {lease} expired at {}; task was requeued",
                held.expires_at
            )));
        }

        held.state.transition(LeaseState::Released)?;
        inner.stats.acknowledged_total += 1;
        Ok(())
    }

    async fn reject(&self, lease: LeaseId, reason: String) -> Result<()> {
        let now = Utc::now();
        let mut inner = self.lock();
        let held = inner.take_lease(lease)?;
        held.state.transition(LeaseState::Rejected)?;
        inner.retry_or_park(held.task, reason, now);
        Ok(())
    }

    async fn release(&self, lease: LeaseId) -> Result<()> {
        let now = Utc::now();
        let mut inner = self.lock();
        let held = inner.take_lease(lease)?;
        let mut task = held.task;
        // Hand the attempt back: nothing was executed.
        task.attempt = task.attempt.saturating_sub(1);
        task.available_at = now;
        inner.ready.push(task);
        Ok(())
    }

    async fn extend_lease(&self, lease: LeaseId, duration: Duration) -> Result<()> {
        let now = Utc::now();
        let mut inner = self.lock();
        let held = inner
            .leases
            .get_mut(&lease)
            .ok_or_else(|| SwarmError::Queue(format!("unknown lease {lease}")))?;

        if held.is_expired(now) {
            return Err(SwarmError::Queue(format!(
                "lease {lease} already expired at {}",
                held.expires_at
            )));
        }
        held.state = held.state.transition(LeaseState::Extended)?;
        let extension = i64::try_from(duration.as_millis())
            .map_err(|_| SwarmError::Queue("lease extension does not fit in i64 ms".into()))?;
        held.expires_at = now + chrono::Duration::milliseconds(extension);
        Ok(())
    }

    async fn requeue_expired(&self) -> Result<usize> {
        let now = Utc::now();
        let mut inner = self.lock();
        let expired: Vec<LeaseId> = inner
            .leases
            .values()
            .filter(|lease| lease.is_expired(now))
            .map(|lease| lease.lease_id)
            .collect();

        for lease_id in &expired {
            let Some(lease) = inner.leases.remove(lease_id) else {
                continue;
            };
            inner.stats.expired_total += 1;
            tracing::warn!(
                task_id = %lease.task.task_id,
                worker = %lease.worker,
                attempt = lease.task.attempt,
                "lease expired, recovering task"
            );
            inner.retry_or_park(lease.task, "lease expired".into(), now);
        }
        Ok(expired.len())
    }

    async fn stats(&self) -> Result<QueueStats> {
        let now = Utc::now();
        let inner = self.lock();
        let ready = inner
            .ready
            .iter()
            .filter(|task| task.is_visible(now))
            .count();
        Ok(QueueStats {
            ready,
            delayed: inner.ready.len() - ready,
            leased: inner.leases.len(),
            dead_lettered: inner.dead.len(),
            ..inner.stats
        })
    }

    async fn dead_letters(&self) -> Result<Vec<DeadLetter>> {
        Ok(self.lock().dead.clone())
    }

    async fn replay_dead_letter(&self, task_id: TaskId) -> Result<bool> {
        let now = Utc::now();
        let mut inner = self.lock();
        let Some(position) = inner
            .dead
            .iter()
            .position(|letter| letter.task.task_id == task_id)
        else {
            return Ok(false);
        };

        let mut task = inner.dead.remove(position).task;
        task.available_at = now;
        // Pushed directly rather than through enqueue: the attempt key is already in
        // `seen`, so the deduplication check would drop the replay as a duplicate.
        inner.stats.enqueued_total += 1;
        inner.ready.push(task);
        Ok(true)
    }

    async fn purge_job(&self, job_id: JobId) -> Result<usize> {
        let mut inner = self.lock();
        let before = inner.ready.len() + inner.leases.len();
        inner.ready.retain(|task| task.job_id != job_id);
        inner.leases.retain(|_, lease| lease.task.job_id != job_id);
        Ok(before - (inner.ready.len() + inner.leases.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_domain::{JobId, Priority, RetryPolicy, TaskKind, TaskNode};

    const LEASE_MS: u64 = 30_000;

    fn task(job_id: JobId, title: &str, priority: Priority) -> QueuedTask {
        let node = TaskNode::new(job_id, TaskKind::Work, title, "do it", 0);
        QueuedTask::from_task(&node, priority, LEASE_MS).unwrap()
    }

    /// A task whose lease is already expired the moment it is granted, so expiry
    /// behaviour is testable without sleeping.
    fn instantly_expiring(job_id: JobId, title: &str) -> QueuedTask {
        let mut queued = task(job_id, title, Priority::Normal);
        queued.lease_duration_ms = 0;
        queued
    }

    fn no_backoff(mut queued: QueuedTask, max_attempts: u32) -> QueuedTask {
        queued.retry = RetryPolicy {
            max_attempts,
            backoff_base_ms: 0,
            backoff_max_ms: 0,
            jitter: false,
        };
        queued
    }

    #[tokio::test]
    async fn tasks_come_back_highest_priority_first() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        queue
            .enqueue(task(job_id, "low", Priority::Low))
            .await
            .unwrap();
        queue
            .enqueue(task(job_id, "critical", Priority::Critical))
            .await
            .unwrap();
        queue
            .enqueue(task(job_id, "normal", Priority::Normal))
            .await
            .unwrap();

        let mut order = Vec::new();
        while let Some(lease) = queue.dequeue("worker-1".into()).await.unwrap() {
            order.push(lease.task.task().unwrap().title);
            queue.acknowledge(lease.lease_id).await.unwrap();
        }
        assert_eq!(order, vec!["critical", "normal", "low"]);
    }

    #[tokio::test]
    async fn equal_priority_is_served_oldest_first() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        let mut first = task(job_id, "first", Priority::Normal);
        first.enqueued_at = Utc::now() - chrono::Duration::seconds(60);
        queue.enqueue(first).await.unwrap();
        queue
            .enqueue(task(job_id, "second", Priority::Normal))
            .await
            .unwrap();

        let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        assert_eq!(lease.task.task().unwrap().title, "first");
    }

    #[tokio::test]
    async fn delayed_tasks_stay_invisible_until_their_time() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        let delayed = task(job_id, "later", Priority::Critical)
            .available_at(Utc::now() + chrono::Duration::seconds(30));
        queue.enqueue(delayed).await.unwrap();
        queue
            .enqueue(task(job_id, "now", Priority::Low))
            .await
            .unwrap();

        let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        assert_eq!(
            lease.task.task().unwrap().title,
            "now",
            "a delayed critical task must not jump ahead of an available low one"
        );

        queue.acknowledge(lease.lease_id).await.unwrap();
        assert!(queue.dequeue("worker-1".into()).await.unwrap().is_none());

        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.ready, 0);
        assert_eq!(stats.delayed, 1);
    }

    #[tokio::test]
    async fn a_duplicate_delivery_of_the_same_attempt_is_dropped() {
        let queue = InMemoryQueue::new();
        let entry = task(JobId::new(), "once", Priority::Normal);

        queue.enqueue(entry.clone()).await.unwrap();
        queue.enqueue(entry.clone()).await.unwrap();
        queue.enqueue(entry).await.unwrap();

        assert!(queue.dequeue("worker-1".into()).await.unwrap().is_some());
        assert!(queue.dequeue("worker-1".into()).await.unwrap().is_none());

        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.enqueued_total, 1);
        assert_eq!(stats.deduplicated_total, 2);
    }

    #[tokio::test]
    async fn a_leased_task_is_invisible_to_other_workers() {
        let queue = InMemoryQueue::new();
        queue
            .enqueue(task(JobId::new(), "only", Priority::Normal))
            .await
            .unwrap();

        let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        assert!(queue.dequeue("worker-2".into()).await.unwrap().is_none());
        queue.acknowledge(lease.lease_id).await.unwrap();
    }

    #[tokio::test]
    async fn an_expired_lease_returns_the_task_to_another_worker() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        queue
            .enqueue(no_backoff(instantly_expiring(job_id, "recoverable"), 5))
            .await
            .unwrap();

        let lost = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        assert_eq!(queue.requeue_expired().await.unwrap(), 1);

        let recovered = queue.dequeue("worker-2".into()).await.unwrap().unwrap();
        assert_eq!(recovered.task.task_id, lost.task.task_id);
        assert_eq!(recovered.task.attempt, 2, "the redelivery is a new attempt");

        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.expired_total, 1);
        assert_eq!(stats.retried_total, 1);
    }

    #[tokio::test]
    async fn a_worker_cannot_acknowledge_after_its_lease_expired() {
        let queue = InMemoryQueue::new();
        queue
            .enqueue(no_backoff(instantly_expiring(JobId::new(), "slow"), 5))
            .await
            .unwrap();

        let lease = queue.dequeue("slow-worker".into()).await.unwrap().unwrap();
        let err = queue.acknowledge(lease.lease_id).await.unwrap_err();
        assert!(matches!(err, SwarmError::Queue(_)));

        // And the task is back for someone else rather than silently lost.
        assert!(queue.dequeue("fast-worker".into()).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn extending_a_lease_keeps_a_slow_but_live_worker_in_charge() {
        let queue = InMemoryQueue::new();
        let mut entry = task(JobId::new(), "long", Priority::Normal);
        entry.lease_duration_ms = 50;
        queue.enqueue(entry).await.unwrap();

        let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        queue
            .extend_lease(lease.lease_id, Duration::from_secs(60))
            .await
            .unwrap();

        assert_eq!(queue.requeue_expired().await.unwrap(), 0);
        queue.acknowledge(lease.lease_id).await.unwrap();
    }

    #[tokio::test]
    async fn extending_an_unknown_lease_is_an_error_not_a_silent_success() {
        let queue = InMemoryQueue::new();
        let err = queue
            .extend_lease(LeaseId::new(), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(err, SwarmError::Queue(_)));
    }

    #[tokio::test]
    async fn rejected_tasks_retry_until_the_budget_runs_out_then_dead_letter() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        queue
            .enqueue(no_backoff(task(job_id, "doomed", Priority::Normal), 3))
            .await
            .unwrap();

        for attempt in 1..=3 {
            let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
            assert_eq!(lease.task.attempt, attempt);
            queue
                .reject(lease.lease_id, format!("failure {attempt}"))
                .await
                .unwrap();
        }

        assert!(
            queue.dequeue("worker-1".into()).await.unwrap().is_none(),
            "a task out of attempts must not be redelivered"
        );

        let dead = queue.dead_letters().await.unwrap();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].attempts, 3);
        assert_eq!(dead[0].last_error, "failure 3");
        assert_eq!(queue.stats().await.unwrap().dead_lettered, 1);
    }

    #[tokio::test]
    async fn a_dead_letter_can_be_replayed_by_an_operator() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        queue
            .enqueue(no_backoff(task(job_id, "doomed", Priority::Normal), 1))
            .await
            .unwrap();

        let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        let task_id = lease.task.task_id;
        queue.reject(lease.lease_id, "boom".into()).await.unwrap();
        assert_eq!(queue.dead_letters().await.unwrap().len(), 1);

        assert!(queue.replay_dead_letter(task_id).await.unwrap());
        assert!(queue.dead_letters().await.unwrap().is_empty());
        assert!(queue.dequeue("worker-2".into()).await.unwrap().is_some());

        assert!(!queue.replay_dead_letter(TaskId::new()).await.unwrap());
    }

    #[tokio::test]
    async fn releasing_a_task_does_not_spend_an_attempt() {
        let queue = InMemoryQueue::new();
        queue
            .enqueue(task(JobId::new(), "unschedulable", Priority::Normal))
            .await
            .unwrap();

        let lease = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        assert_eq!(lease.task.attempt, 1);
        queue.release(lease.lease_id).await.unwrap();

        let again = queue.dequeue("worker-2".into()).await.unwrap().unwrap();
        assert_eq!(
            again.task.attempt, 1,
            "the release must not cost an attempt"
        );
    }

    #[tokio::test]
    async fn workers_only_receive_tasks_they_can_run() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        let mut coding = task(job_id, "write code", Priority::Critical);
        coding.required_capabilities = vec![Capability::Coding];
        let mut research = task(job_id, "read papers", Priority::Low);
        research.required_capabilities = vec![Capability::Research];
        queue.enqueue(coding).await.unwrap();
        queue.enqueue(research).await.unwrap();

        let lease = queue
            .dequeue_from("researcher".into(), &[], &[Capability::Research])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            lease.task.task().unwrap().title,
            "read papers",
            "capability filtering must beat priority"
        );

        assert!(queue
            .dequeue_from("designer".into(), &[], &[Capability::Summarization])
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn named_queues_isolate_workloads() {
        let queue = InMemoryQueue::new();
        let job_id = JobId::new();
        queue
            .enqueue(task(job_id, "gpu work", Priority::Normal).on_queue("gpu"))
            .await
            .unwrap();
        queue
            .enqueue(task(job_id, "cpu work", Priority::Normal))
            .await
            .unwrap();

        let gpu_queue = vec!["gpu".to_owned()];
        let lease = queue
            .dequeue_from("gpu-worker".into(), &gpu_queue, &[])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lease.task.task().unwrap().title, "gpu work");

        assert!(queue
            .dequeue_from("gpu-worker".into(), &gpu_queue, &[])
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn cancelling_a_job_purges_its_queued_and_leased_work() {
        let queue = InMemoryQueue::new();
        let doomed = JobId::new();
        let survivor = JobId::new();
        queue
            .enqueue(task(doomed, "a", Priority::Normal))
            .await
            .unwrap();
        queue
            .enqueue(task(doomed, "b", Priority::Normal))
            .await
            .unwrap();
        queue
            .enqueue(task(survivor, "c", Priority::Normal))
            .await
            .unwrap();

        let leased = queue.dequeue("worker-1".into()).await.unwrap().unwrap();
        assert_eq!(leased.task.job_id, doomed);

        assert_eq!(queue.purge_job(doomed).await.unwrap(), 2);
        let remaining = queue.dequeue("worker-2".into()).await.unwrap().unwrap();
        assert_eq!(remaining.task.job_id, survivor);
        assert!(queue.dequeue("worker-3".into()).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_thousand_tasks_across_fifty_workers_are_each_delivered_once() {
        use std::collections::HashSet;
        use std::sync::Arc;

        let queue = Arc::new(InMemoryQueue::new());
        let job_id = JobId::new();
        for index in 0..1_000 {
            queue
                .enqueue(task(job_id, &format!("task-{index}"), Priority::Normal))
                .await
                .unwrap();
        }

        let mut workers = Vec::new();
        for worker in 0..50 {
            let queue = Arc::clone(&queue);
            workers.push(tokio::spawn(async move {
                let mut claimed = Vec::new();
                while let Some(lease) = queue.dequeue(format!("worker-{worker}")).await.unwrap() {
                    claimed.push(lease.task.task_id);
                    queue.acknowledge(lease.lease_id).await.unwrap();
                }
                claimed
            }));
        }

        let mut all = Vec::new();
        for worker in workers {
            all.extend(worker.await.unwrap());
        }

        assert_eq!(all.len(), 1_000, "every task must be delivered");
        assert_eq!(
            all.iter().collect::<HashSet<_>>().len(),
            1_000,
            "no task may be delivered twice"
        );

        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.acknowledged_total, 1_000);
        assert_eq!(stats.depth(), 0);
    }
}
