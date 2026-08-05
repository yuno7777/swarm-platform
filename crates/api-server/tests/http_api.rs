//! Integration tests that drive the API over a real socket.
//!
//! The server is bound to an ephemeral port and talked to with an HTTP client, so what
//! is being tested is the actual wire behaviour — status codes, JSON shapes, and
//! content types — rather than handler functions called directly.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use swarm_api_server::{router, AppState};
use swarm_coordinator::Coordinator;
use swarm_model_gateway::{Gateway, GatewayConfig, MockProvider};

/// Start a server on an ephemeral port and return its base URL.
async fn spawn_server() -> String {
    let gateway = Arc::new(
        Gateway::new(GatewayConfig {
            retry_backoff_ms: 0,
            ..GatewayConfig::default()
        })
        .and_provider(Arc::new(MockProvider::new("mock"))),
    );
    let coordinator = Arc::new(Coordinator::local(gateway));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router(AppState::new(coordinator)))
            .await
            .unwrap();
    });

    format!("http://{address}")
}

/// Submit a job and return its id.
async fn submit(base: &str, body: Value) -> (reqwest::StatusCode, Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/jobs"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    (status, response.json().await.unwrap_or(Value::Null))
}

/// Poll until the job reaches a terminal status.
async fn await_completion(base: &str, job_id: &str) -> Value {
    for _ in 0..200 {
        let state: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let status = state["status"].as_str().unwrap_or_default().to_owned();
        if matches!(
            status.as_str(),
            "completed" | "partially_completed" | "failed" | "cancelled" | "rejected"
        ) {
            return state;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("job {job_id} never finished");
}

#[tokio::test]
async fn health_reports_the_running_build() {
    let base = spawn_server().await;
    let body: Value = reqwest::get(format!("{base}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
    assert!(body["uptime_seconds"].is_u64());
}

#[tokio::test]
async fn a_submitted_job_runs_to_completion_and_exposes_every_artifact() {
    let base = spawn_server().await;

    let (status, submitted) = submit(
        &base,
        json!({
            "objective": "Compare Raft and Paxos for leader election",
            "execution_strategy": "parallel",
            "max_agents": 3
        }),
    )
    .await;

    assert_eq!(status, reqwest::StatusCode::ACCEPTED);
    let job_id = submitted["job_id"].as_str().unwrap().to_owned();
    assert!(submitted["tasks_planned"].as_u64().unwrap() >= 3);

    let state = await_completion(&base, &job_id).await;
    assert_eq!(state["status"], "completed");
    assert_eq!(state["progress"], 1.0);
    assert!(state["cost_usd"].as_f64().unwrap() > 0.0);

    // The task graph.
    let graph: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}/graph"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nodes = graph.as_array().unwrap();
    assert!(!nodes.is_empty());
    assert!(nodes.iter().all(|node| node["state"] == "completed"));

    // Per-task results.
    let results: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}/results"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(results.as_array().unwrap().len(), nodes.len());

    // The merged final result.
    let final_result: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}/result"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(final_result["status"], "completed");
    assert!(!final_result["summary"].as_str().unwrap().is_empty());
    assert!(
        final_result["execution_statistics"]["tasks_succeeded"]
            .as_u64()
            .unwrap()
            > 0
    );

    // The audit trail: four transitions per task, each with an actor and a reason.
    let transitions: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}/transitions"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let transitions = transitions.as_array().unwrap();
    assert_eq!(transitions.len(), nodes.len() * 4);
    assert!(transitions.iter().all(
        |t| t["reason"].is_string() && t["actor"].as_str().unwrap().starts_with("coordinator:")
    ));

    // Failures and agents are addressable too.
    let failures: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}/failures"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(failures.as_array().unwrap().is_empty());

    let agents: Value = reqwest::get(format!("{base}/v1/jobs/{job_id}/agents"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!agents.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn the_final_result_is_absent_until_the_job_finishes() {
    let base = spawn_server().await;
    let (_, submitted) = submit(
        &base,
        json!({ "objective": "A job to inspect mid-flight", "max_agents": 2 }),
    )
    .await;
    let job_id = submitted["job_id"].as_str().unwrap().to_owned();

    // Either the job is still running (404) or it already finished (200); both are
    // correct, and neither may be a 500.
    let status = reqwest::get(format!("{base}/v1/jobs/{job_id}/result"))
        .await
        .unwrap()
        .status();
    assert!(
        status == reqwest::StatusCode::NOT_FOUND || status.is_success(),
        "unexpected status {status}"
    );

    await_completion(&base, &job_id).await;
    assert!(reqwest::get(format!("{base}/v1/jobs/{job_id}/result"))
        .await
        .unwrap()
        .status()
        .is_success());
}

#[tokio::test]
async fn invalid_submissions_are_rejected_with_a_reason() {
    let base = spawn_server().await;

    let (status, body) = submit(&base, json!({ "objective": "   " })).await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["kind"], "validation");
    assert!(body["error"]["message"].as_str().unwrap().contains("empty"));

    // Over the cluster's agent ceiling.
    let (status, body) = submit(
        &base,
        json!({ "objective": "far too large", "max_agents": 100000 }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(body["error"]["kind"], "quota_exceeded");

    // A body that is not a job request at all.
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/jobs"))
        .json(&json!({ "not_a_field": 1 }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn unknown_and_malformed_job_ids_are_distinguished() {
    let base = spawn_server().await;

    let missing = reqwest::get(format!(
        "{base}/v1/jobs/019fce00-0000-7000-8000-000000000000"
    ))
    .await
    .unwrap();
    assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);
    let body: Value = missing.json().await.unwrap();
    assert_eq!(body["error"]["kind"], "not_found");

    let malformed = reqwest::get(format!("{base}/v1/jobs/nonsense"))
        .await
        .unwrap();
    assert_eq!(malformed.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn jobs_can_be_listed_newest_first() {
    let base = spawn_server().await;
    for index in 0..3 {
        submit(
            &base,
            json!({ "objective": format!("objective number {index}"), "max_agents": 2 }),
        )
        .await;
    }

    let jobs: Value = reqwest::get(format!("{base}/v1/jobs"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let jobs = jobs.as_array().unwrap();
    assert_eq!(jobs.len(), 3);

    // Ids are UUIDv7, so newest-first is descending id order.
    let ids: Vec<&str> = jobs
        .iter()
        .map(|job| job["job_id"].as_str().unwrap())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable_by(|left, right| right.cmp(left));
    assert_eq!(ids, sorted);
}

#[tokio::test]
async fn a_running_job_can_be_cancelled_over_http() {
    let base = spawn_server().await;
    let (_, submitted) = submit(
        &base,
        json!({
            "objective": "A long job that will be cancelled by an operator",
            "execution_strategy": "hierarchical",
            "max_agents": 4
        }),
    )
    .await;
    let job_id = submitted["job_id"].as_str().unwrap().to_owned();

    let cancelled = reqwest::Client::new()
        .post(format!("{base}/v1/jobs/{job_id}/cancel"))
        .send()
        .await
        .unwrap();
    assert!(cancelled.status().is_success());

    let state = await_completion(&base, &job_id).await;
    assert!(
        state["status"] == "cancelled" || state["status"] == "completed",
        "unexpected terminal status {}",
        state["status"]
    );
}

#[tokio::test]
async fn pause_and_resume_are_exposed_and_ordered() {
    let base = spawn_server().await;
    let (_, submitted) = submit(
        &base,
        json!({
            "objective": "A job that gets paused and then resumed again",
            "execution_strategy": "hierarchical",
            "max_agents": 3
        }),
    )
    .await;
    let job_id = submitted["job_id"].as_str().unwrap().to_owned();
    let client = reqwest::Client::new();

    assert!(client
        .post(format!("{base}/v1/jobs/{job_id}/pause"))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    assert!(client
        .post(format!("{base}/v1/jobs/{job_id}/resume"))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());

    let state = await_completion(&base, &job_id).await;
    assert_eq!(state["status"], "completed");

    // Pausing a job that has already finished is refused, not silently accepted.
    let after = client
        .post(format!("{base}/v1/jobs/{job_id}/pause"))
        .send()
        .await
        .unwrap();
    assert!(after.status().is_success() || after.status().is_client_error());
}

#[tokio::test]
async fn the_event_stream_is_served_as_server_sent_events() {
    let base = spawn_server().await;
    let (_, submitted) = submit(
        &base,
        json!({ "objective": "A job whose progress is streamed", "max_agents": 2 }),
    )
    .await;
    let job_id = submitted["job_id"].as_str().unwrap().to_owned();

    let response = reqwest::get(format!("{base}/v1/jobs/{job_id}/events"))
        .await
        .unwrap();
    assert!(response.status().is_success());
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    // Streaming a job that does not exist is a client error, not a hanging connection.
    let missing = reqwest::get(format!("{base}/v1/jobs/nonsense/events"))
        .await
        .unwrap();
    assert!(missing.status().is_client_error());
}

#[tokio::test]
async fn the_cluster_view_and_metrics_reflect_real_work() {
    let base = spawn_server().await;
    let (_, submitted) = submit(
        &base,
        json!({ "objective": "Work that shows up in the metrics", "max_agents": 3 }),
    )
    .await;
    await_completion(&base, submitted["job_id"].as_str().unwrap()).await;

    let cluster: Value = reqwest::get(format!("{base}/v1/cluster"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cluster["jobs_total"], 1);
    assert_eq!(cluster["jobs_active"], 0);
    assert_eq!(
        cluster["cluster_agents"], 0,
        "agents are released when a job ends"
    );
    assert!(cluster["gateway"]["requests"].as_u64().unwrap() > 0);
    assert!(!cluster["providers"].as_array().unwrap().is_empty());

    let metrics = reqwest::get(format!("{base}/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("swarm_jobs_active 0"));
    assert!(metrics.contains("swarm_jobs_total{status=\"completed\"} 1"));
    assert!(metrics.contains("# TYPE swarm_model_requests_total counter"));
}
