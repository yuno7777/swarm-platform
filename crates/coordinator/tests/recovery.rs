//! Crash recovery: a coordinator that dies mid-job resumes rather than restarts.
//!
//! The "crash" is a real one — the engine future is aborted while tasks are in flight,
//! so the journal ends wherever it happened to end, exactly as it would after a
//! `kill -9`. A second coordinator is then built over the same journal file and has to
//! finish the job without redoing the work the first one already did.

use std::sync::Arc;
use std::time::Duration;

use swarm_coordinator::{Coordinator, CoordinatorConfig};
use swarm_domain::{ExecutionStrategy, JobId, JobRequest, JobStatus, TaskState};
use swarm_model_gateway::{Gateway, GatewayConfig, MockProvider, ModelProvider};
use swarm_persistence::{FileJournal, Journal};

/// Build a coordinator over `journal`, sharing one provider so model calls can be
/// counted across the crash.
fn coordinator(journal: Arc<dyn Journal>, provider: Arc<MockProvider>) -> Arc<Coordinator> {
    let gateway = Arc::new(
        Gateway::new(GatewayConfig {
            retry_backoff_ms: 0,
            cache_enabled: false,
            ..GatewayConfig::default()
        })
        .and_provider(provider as Arc<dyn ModelProvider>),
    );
    Arc::new(Coordinator::local_with(CoordinatorConfig::default(), gateway).with_journal(journal))
}

/// Wait until at least `count` tasks have completed, so the crash lands mid-job.
async fn wait_for_completed(coordinator: &Coordinator, job_id: JobId, count: usize) {
    for _ in 0..400 {
        if coordinator
            .state(job_id)
            .is_ok_and(|state| state.counts.completed >= count)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("job never completed {count} tasks");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_coordinator_that_dies_mid_job_resumes_from_the_journal() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("swarm.journal");
    let provider = Arc::new(MockProvider::new("mock").with_latency(Duration::from_millis(15)));

    // ---- first process: start the job, then die partway through ----
    let job_id = {
        let journal = Arc::new(FileJournal::open(&path).unwrap());
        let first = coordinator(journal, Arc::clone(&provider));

        let job_id = first
            .submit(
                JobRequest::new("Survey consensus, replication, and recovery in depth")
                    .with_strategy(ExecutionStrategy::Hierarchical)
                    .with_max_agents(4),
            )
            .unwrap();

        let running = Arc::clone(&first);
        let handle = tokio::spawn(async move { running.run(job_id).await });

        wait_for_completed(&first, job_id, 2).await;
        handle.abort();
        let _ = handle.await;

        let interrupted = first.state(job_id).unwrap();
        assert!(
            !interrupted.status.is_finished(),
            "the job must not be done yet"
        );
        assert!(interrupted.counts.completed >= 2);
        job_id
    };

    let calls_before_crash = provider.call_count();
    assert!(calls_before_crash > 0);

    // ---- second process: same journal file, fresh everything else ----
    let journal = Arc::new(FileJournal::open(&path).unwrap());
    assert!(journal.len().unwrap() > 0, "the journal must have survived");

    let second = coordinator(journal, Arc::clone(&provider));
    assert_eq!(second.recover().unwrap(), 1, "one job should come back");

    let recovered = second.state(job_id).unwrap();
    assert_eq!(
        recovered.status,
        JobStatus::Planning,
        "a recovered job is rewound so the engine can start it again"
    );
    assert!(
        recovered.counts.completed >= 2,
        "completed work must survive the crash"
    );

    // Snapshot what was already done, so we can prove it is not redone.
    let restored = second.intermediate_results(job_id).unwrap();
    assert!(!restored.is_empty());

    // ---- resume ----
    let result = second.run(job_id).await.unwrap();
    assert_eq!(result.status, JobStatus::Completed);
    assert!(!result.summary.is_empty());

    let graph = second.task_graph(job_id).unwrap();
    assert!(graph.iter().all(|node| node.state == TaskState::Completed));

    // Every task that had finished before the crash kept its original result: same
    // agent, same timestamp. Resumption, not repetition.
    let after = second.intermediate_results(job_id).unwrap();
    for original in &restored {
        let matching = after
            .iter()
            .find(|candidate| candidate.task_id == original.task_id)
            .expect("a restored result disappeared");
        assert_eq!(matching.agent_id, original.agent_id);
        assert_eq!(matching.finished_at, original.finished_at);
        assert_eq!(matching.output, original.output);
    }

    // And the second process only paid for what was left.
    let calls_after = provider.call_count();
    assert!(
        calls_after < calls_before_crash + graph.len() as u64,
        "resuming cost {} calls for {} tasks; work was redone",
        calls_after - calls_before_crash,
        graph.len()
    );
}

#[tokio::test]
async fn a_finished_job_comes_back_finished_with_its_result_intact() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("swarm.journal");
    let provider = Arc::new(MockProvider::new("mock"));

    let (job_id, original) = {
        let journal = Arc::new(FileJournal::open(&path).unwrap());
        let first = coordinator(journal, Arc::clone(&provider));
        let job_id = first
            .submit(
                JobRequest::new("A job that finishes before anything goes wrong")
                    .with_max_agents(3),
            )
            .unwrap();
        let result = first.run(job_id).await.unwrap();
        (job_id, result)
    };

    let second = coordinator(
        Arc::new(FileJournal::open(&path).unwrap()),
        Arc::clone(&provider),
    );
    assert_eq!(second.recover().unwrap(), 1);

    let state = second.state(job_id).unwrap();
    assert_eq!(state.status, JobStatus::Completed);
    assert_eq!(state.progress, 1.0);

    let recovered = second.final_result(job_id).unwrap().unwrap();
    assert_eq!(recovered.summary, original.summary);
    assert_eq!(recovered.outputs.len(), original.outputs.len());
    assert!((recovered.confidence_score - original.confidence_score).abs() < f32::EPSILON);

    // Recovery must not have called the model at all.
    let calls = provider.call_count();
    let _ = second.state(job_id).unwrap();
    assert_eq!(provider.call_count(), calls);
}

#[tokio::test]
async fn several_jobs_recover_independently() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("swarm.journal");
    let provider = Arc::new(MockProvider::new("mock"));

    let mut submitted = Vec::new();
    {
        let first = coordinator(
            Arc::new(FileJournal::open(&path).unwrap()),
            Arc::clone(&provider),
        );
        for index in 0..3 {
            let job_id = first
                .submit(
                    JobRequest::new(format!("Independent objective number {index}"))
                        .with_max_agents(2),
                )
                .unwrap();
            // Only the first is actually run, so the other two recover as unstarted.
            if index == 0 {
                first.run(job_id).await.unwrap();
            }
            submitted.push(job_id);
        }
    }

    let second = coordinator(
        Arc::new(FileJournal::open(&path).unwrap()),
        Arc::clone(&provider),
    );
    assert_eq!(second.recover().unwrap(), 3);

    assert_eq!(
        second.state(submitted[0]).unwrap().status,
        JobStatus::Completed
    );
    for job_id in &submitted[1..] {
        let state = second.state(*job_id).unwrap();
        assert_eq!(state.status, JobStatus::Planning);
        assert_eq!(state.counts.completed, 0);

        // An unstarted job recovered from the journal still runs normally.
        assert_eq!(
            second.run(*job_id).await.unwrap().status,
            JobStatus::Completed
        );
    }
}

#[tokio::test]
async fn recovering_an_empty_journal_is_a_no_op() {
    let directory = tempfile::tempdir().unwrap();
    let coordinator = coordinator(
        Arc::new(FileJournal::open(directory.path().join("fresh.journal")).unwrap()),
        Arc::new(MockProvider::new("mock")),
    );

    assert_eq!(coordinator.recover().unwrap(), 0);
    assert!(coordinator.list_jobs().is_empty());
}

#[tokio::test]
async fn journalling_is_off_the_critical_path_when_no_file_is_configured() {
    // The default in-memory journal keeps the write path identical, so a deployment
    // that does not want durability still exercises the same code.
    let gateway = Arc::new(Gateway::with_provider(Arc::new(MockProvider::new("mock"))));
    let coordinator = Coordinator::local(gateway);

    let result = coordinator
        .submit_and_run(JobRequest::new("no journal configured").with_max_agents(2))
        .await
        .unwrap();
    assert_eq!(result.status, JobStatus::Completed);

    // Recovery from the in-memory journal works within the process's lifetime.
    assert_eq!(coordinator.recover().unwrap(), 1);
}
