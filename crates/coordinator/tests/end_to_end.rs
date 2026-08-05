//! End-to-end tests for the Phase 1 platform.
//!
//! These drive a real coordinator over the real queue, shared memory, scheduler, and
//! agent runtime — only the model provider is a deterministic mock. They are the
//! evidence for the Phase 1 exit criteria: every strategy completes, failures recover,
//! duplicate delivery does not double-execute, and a hundred agents cooperate.

use std::sync::Arc;
use std::time::Duration;

use swarm_coordinator::{Coordinator, CoordinatorConfig, JobEventKind, SchedulerKind};
use swarm_domain::{ExecutionStrategy, JobRequest, JobStatus, Priority, StateMachine, TaskState};
use swarm_model_gateway::{Gateway, GatewayConfig, MockProvider};

fn coordinator_with(provider: MockProvider, config: CoordinatorConfig) -> Coordinator {
    coordinator_with_gateway(provider, GatewayConfig::default(), config)
}

fn coordinator_with_gateway(
    provider: MockProvider,
    gateway: GatewayConfig,
    config: CoordinatorConfig,
) -> Coordinator {
    let gateway = Arc::new(
        Gateway::new(GatewayConfig {
            retry_backoff_ms: 0,
            // Each task has a distinct prompt, so caching only ever hides a bug here.
            cache_enabled: false,
            ..gateway
        })
        .and_provider(Arc::new(provider)),
    );
    Coordinator::local_with(config, gateway)
}

fn coordinator() -> Coordinator {
    coordinator_with(MockProvider::new("mock"), CoordinatorConfig::default())
}

/// Poll until the job reaches `expected`, or give up. Job control is asynchronous:
/// it takes effect at the next scheduling boundary, not instantly.
async fn wait_for_status(
    coordinator: &Coordinator,
    job_id: swarm_domain::JobId,
    expected: JobStatus,
) {
    for _ in 0..200 {
        if coordinator.state(job_id).unwrap().status == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "job never reached {expected:?}; it is {:?}",
        coordinator.state(job_id).unwrap().status
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_execution_strategy_runs_to_completion() {
    for &strategy in ExecutionStrategy::all() {
        let coordinator = coordinator();
        let result = coordinator
            .submit_and_run(
                JobRequest::new("Compare Raft and Paxos for leader election and recovery")
                    .with_strategy(strategy)
                    .with_max_agents(4),
            )
            .await
            .unwrap_or_else(|e| panic!("{strategy} failed: {e}"));

        assert_eq!(
            result.status,
            JobStatus::Completed,
            "{strategy} did not complete"
        );
        assert!(!result.summary.is_empty(), "{strategy} produced no summary");
        assert!(!result.outputs.is_empty(), "{strategy} produced no outputs");
        assert!(
            (0.0..=1.0).contains(&result.confidence_score),
            "{strategy} produced confidence {}",
            result.confidence_score
        );

        let statistics = &result.execution_statistics;
        assert_eq!(
            statistics.tasks_succeeded, statistics.tasks_total,
            "{strategy} left tasks unfinished"
        );
        assert_eq!(statistics.tasks_failed, 0, "{strategy} had failures");
        assert!(statistics.tokens_in > 0 && statistics.tokens_out > 0);
        assert!(statistics.cost_usd > 0.0, "{strategy} recorded no cost");
        assert!(statistics.agents_spawned > 0);
    }
}

#[tokio::test]
async fn a_finished_job_exposes_its_graph_results_and_audit_trail() {
    let coordinator = coordinator();
    let job_id = coordinator
        .submit(
            JobRequest::new("Investigate consensus, replication, and recovery")
                .with_strategy(ExecutionStrategy::Parallel)
                .with_max_agents(3),
        )
        .unwrap();
    coordinator.run(job_id).await.unwrap();

    let graph = coordinator.task_graph(job_id).unwrap();
    assert!(graph.iter().all(|node| node.state == TaskState::Completed));

    let results = coordinator.intermediate_results(job_id).unwrap();
    assert_eq!(results.len(), graph.len());
    assert!(results.iter().all(|result| result.passed_validation()));

    // Every task walked Created -> Queued -> Leased -> Running -> Completed, and each
    // step was journaled with an actor and a reason.
    let transitions = coordinator.transitions(job_id).unwrap();
    assert_eq!(transitions.len(), graph.len() * 4);
    assert!(transitions
        .iter()
        .all(|t| t.actor.starts_with("coordinator:")));
    assert!(transitions.iter().all(|t| t.reason.is_some()));
    assert_eq!(
        transitions
            .iter()
            .filter(|t| t.to == TaskState::Completed)
            .count(),
        graph.len()
    );

    assert!(coordinator.failures(job_id).unwrap().is_empty());
    assert_eq!(
        coordinator.state(job_id).unwrap().status,
        JobStatus::Completed
    );
    assert!(coordinator.final_result(job_id).unwrap().is_some());
}

#[tokio::test]
async fn a_flaky_provider_is_survived_by_retries() {
    // Every third model call fails, and the gateway is told not to retry, so the
    // failure reaches the task and the *task* has to recover. The job must still
    // finish, with the retries visible in the statistics.
    let coordinator = coordinator_with_gateway(
        MockProvider::new("flaky").failing_every(3),
        GatewayConfig {
            max_attempts: 1,
            ..GatewayConfig::default()
        },
        CoordinatorConfig::default(),
    );

    let result = coordinator
        .submit_and_run(
            JobRequest::new("Explain leader election under partition")
                .with_strategy(ExecutionStrategy::Parallel)
                .with_max_agents(4),
        )
        .await
        .unwrap();

    assert_eq!(result.status, JobStatus::Completed);
    assert!(
        result.execution_statistics.tasks_retried > 0,
        "the flaky provider should have forced at least one retry"
    );
    assert_eq!(result.execution_statistics.tasks_failed, 0);
}

#[tokio::test]
async fn a_dead_provider_produces_dead_letters_not_a_hang() {
    let coordinator = coordinator_with(
        MockProvider::new("dead").always_failing(),
        CoordinatorConfig::default(),
    );

    let job_id = coordinator
        .submit(
            JobRequest::new("This cannot succeed")
                .with_strategy(ExecutionStrategy::Parallel)
                .with_max_agents(2),
        )
        .unwrap();
    let result = coordinator.run(job_id).await.unwrap();

    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.confidence_score, 0.0);
    assert!(result.outputs.is_empty());

    let failures = coordinator.failures(job_id).unwrap();
    assert!(!failures.is_empty());
    assert!(failures.iter().any(|failure| failure.dead_lettered));
    assert!(!coordinator.queue().dead_letters().await.unwrap().is_empty());

    // Downstream tasks that could never run were cancelled rather than left hanging.
    let graph = coordinator.task_graph(job_id).unwrap();
    assert!(graph.iter().all(swarm_domain::TaskNode::is_terminal));
    assert!(graph
        .iter()
        .any(|node| node.state == TaskState::DeadLettered));
}

#[tokio::test]
async fn duplicate_delivery_of_a_task_does_not_execute_it_twice() {
    // The queue is at-least-once by design, so the guarantee that matters is that a
    // second delivery produces no second execution.
    use swarm_task_queue::QueuedTask;

    let provider = Arc::new(MockProvider::new("counting"));
    let gateway = Arc::new(
        Gateway::new(GatewayConfig {
            cache_enabled: false,
            ..GatewayConfig::default()
        })
        .and_provider(provider.clone()),
    );
    let coordinator = Coordinator::local(gateway);

    let job_id = coordinator
        .submit(
            JobRequest::new("Explain quorum intersection")
                .with_strategy(ExecutionStrategy::Sequential)
                .with_max_agents(1),
        )
        .unwrap();

    // Re-enqueue every task a second time, exactly as a redelivering broker would.
    let tasks = coordinator.task_graph(job_id).unwrap();
    for task in &tasks {
        let mut duplicate = QueuedTask::from_task(task, Priority::Normal, 60_000).unwrap();
        // A redelivery carries the attempt the worker will actually run.
        duplicate.attempt = 0;
        duplicate.idempotency_key = format!("{}#1", task.idempotency_key);
        coordinator.queue().enqueue(duplicate).await.unwrap();
    }

    let result = coordinator.run(job_id).await.unwrap();
    assert_eq!(result.status, JobStatus::Completed);

    let results = coordinator.intermediate_results(job_id).unwrap();
    let unique: std::collections::HashSet<_> =
        results.iter().map(|result| result.task_id).collect();
    assert_eq!(
        unique.len(),
        results.len(),
        "no task may report two results"
    );
    assert!(
        provider.call_count() <= tasks.len() as u64,
        "{} model calls for {} tasks: work was duplicated",
        provider.call_count(),
        tasks.len()
    );
}

#[tokio::test]
async fn cancelling_a_running_job_stops_it_and_purges_its_queue() {
    let coordinator = Arc::new(coordinator_with(
        MockProvider::new("slow").with_latency(Duration::from_millis(40)),
        CoordinatorConfig::default(),
    ));

    let job_id = coordinator
        .submit(
            JobRequest::new("A long job that will be interrupted midway through")
                .with_strategy(ExecutionStrategy::Hierarchical)
                .with_max_agents(4),
        )
        .unwrap();

    let runner = {
        let coordinator = Arc::clone(&coordinator);
        tokio::spawn(async move { coordinator.run(job_id).await })
    };

    tokio::time::sleep(Duration::from_millis(60)).await;
    coordinator.cancel(job_id).await.unwrap();

    let result = runner.await.unwrap().unwrap();
    assert_eq!(result.status, JobStatus::Cancelled);

    let graph = coordinator.task_graph(job_id).unwrap();
    assert!(
        graph.iter().all(swarm_domain::TaskNode::is_terminal),
        "cancellation must leave every task in a terminal state"
    );
    assert!(
        graph.iter().any(|node| node.state != TaskState::Completed),
        "the job should not have finished everything before being cancelled"
    );
    assert_eq!(coordinator.queue().stats().await.unwrap().depth(), 0);
}

#[tokio::test]
async fn pausing_holds_scheduling_until_resumed() {
    let coordinator = Arc::new(coordinator_with(
        MockProvider::new("slow").with_latency(Duration::from_millis(20)),
        CoordinatorConfig::default(),
    ));

    let job_id = coordinator
        .submit(
            JobRequest::new("A job that gets paused halfway through its work")
                .with_strategy(ExecutionStrategy::Hierarchical)
                .with_max_agents(3),
        )
        .unwrap();

    let runner = {
        let coordinator = Arc::clone(&coordinator);
        tokio::spawn(async move { coordinator.run(job_id).await })
    };

    tokio::time::sleep(Duration::from_millis(30)).await;
    coordinator.pause(job_id).unwrap();
    wait_for_status(&coordinator, job_id, JobStatus::Paused).await;

    let paused = coordinator.state(job_id).unwrap();
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(
        coordinator.state(job_id).unwrap().counts.completed,
        paused.counts.completed,
        "no task may start while the job is paused"
    );

    coordinator.resume(job_id).unwrap();
    let result = runner.await.unwrap().unwrap();
    assert_eq!(result.status, JobStatus::Completed);
}

#[tokio::test]
async fn the_event_stream_reports_the_whole_lifecycle_in_order() {
    let coordinator = coordinator();
    let job_id = coordinator
        .submit(
            JobRequest::new("Summarise the CAP theorem")
                .with_strategy(ExecutionStrategy::Parallel)
                .with_max_agents(2),
        )
        .unwrap();

    let mut events = coordinator.subscribe(job_id).unwrap();
    coordinator.run(job_id).await.unwrap();

    let mut seen = Vec::new();
    let mut previous_sequence = None;
    while let Ok(event) = events.try_recv() {
        if let Some(previous) = previous_sequence {
            assert!(
                event.sequence_number > previous,
                "event sequence numbers must increase"
            );
        }
        previous_sequence = Some(event.sequence_number);
        assert!((0.0..=1.0).contains(&event.progress));
        seen.push(event.kind);
    }

    for expected in [
        JobEventKind::AgentSpawned,
        JobEventKind::TaskQueued,
        JobEventKind::TaskStarted,
        JobEventKind::TaskCompleted,
        JobEventKind::JobFinished,
    ] {
        assert!(
            seen.contains(&expected),
            "no {expected:?} event was emitted"
        );
    }
    assert_eq!(seen.last(), Some(&JobEventKind::JobFinished));
}

#[tokio::test]
async fn a_job_past_its_deadline_stops_spending() {
    let coordinator = coordinator_with(
        MockProvider::new("slow").with_latency(Duration::from_millis(30)),
        CoordinatorConfig::default(),
    );

    let result = coordinator
        .submit_and_run(
            JobRequest::new("A job with an impossible deadline to meet in time")
                .with_strategy(ExecutionStrategy::Hierarchical)
                .with_max_agents(4),
        )
        .await;

    // With no deadline it completes; the same job with one in the past must not.
    assert_eq!(result.unwrap().status, JobStatus::Completed);

    let mut request = JobRequest::new("A job whose deadline has already passed entirely")
        .with_strategy(ExecutionStrategy::Hierarchical)
        .with_max_agents(4);
    request.deadline = Some(chrono::Utc::now() - chrono::Duration::seconds(1));

    let expired = coordinator.submit_and_run(request).await.unwrap();
    assert_ne!(expired.status, JobStatus::Completed);
    assert_eq!(expired.execution_statistics.tasks_succeeded, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hundred_agents_cooperate_on_one_job() {
    let coordinator = coordinator_with(
        MockProvider::new("mock"),
        CoordinatorConfig {
            max_cluster_agents: 256,
            ..CoordinatorConfig::default()
        },
    );

    let started = std::time::Instant::now();
    let result = coordinator
        .submit_and_run(
            JobRequest::new(
                "Survey the design space of distributed consensus, replication, \
                 membership, failure detection, and recovery",
            )
            .with_strategy(ExecutionStrategy::Parallel)
            .with_max_agents(100),
        )
        .await
        .unwrap();

    assert_eq!(result.status, JobStatus::Completed);
    assert_eq!(
        result.execution_statistics.agents_spawned, 100,
        "the job should have used its whole agent budget"
    );
    assert_eq!(result.execution_statistics.tasks_total, 102);
    assert_eq!(result.execution_statistics.tasks_succeeded, 102);
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "100 agents took {:?}",
        started.elapsed()
    );

    // Agents are released back to the cluster budget when the job ends.
    assert_eq!(coordinator.cluster_agent_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_scheduling_strategy_finishes_the_same_job() {
    for &kind in SchedulerKind::all() {
        let coordinator = coordinator_with(
            MockProvider::new("mock"),
            CoordinatorConfig {
                scheduler: kind,
                ..CoordinatorConfig::default()
            },
        );

        let result = coordinator
            .submit_and_run(
                JobRequest::new("Compare scheduling strategies under load")
                    .with_strategy(ExecutionStrategy::MapReduce)
                    .with_max_agents(6),
            )
            .await
            .unwrap_or_else(|e| panic!("{kind:?} failed: {e}"));

        assert_eq!(
            result.status,
            JobStatus::Completed,
            "{kind:?} did not finish"
        );
    }
}

#[tokio::test]
async fn concurrent_jobs_share_the_cluster_without_interfering() {
    let coordinator = Arc::new(coordinator());

    let mut runs = Vec::new();
    for index in 0..4 {
        let coordinator = Arc::clone(&coordinator);
        runs.push(tokio::spawn(async move {
            coordinator
                .submit_and_run(
                    JobRequest::new(format!("Independent objective number {index}"))
                        .with_strategy(ExecutionStrategy::Parallel)
                        .with_max_agents(3),
                )
                .await
        }));
    }

    for run in runs {
        let result = run.await.unwrap().unwrap();
        assert_eq!(result.status, JobStatus::Completed);
    }
    assert_eq!(coordinator.cluster_agent_count(), 0);
}

#[test]
fn the_task_state_machine_refuses_illegal_edges_end_to_end() {
    // A guard against the engine ever "fixing" a stuck task by forcing a state.
    assert!(TaskState::Completed.transition(TaskState::Running).is_err());
    assert!(TaskState::DeadLettered
        .transition(TaskState::Queued)
        .is_err());
    assert!(TaskState::Queued.transition(TaskState::Completed).is_err());
}
