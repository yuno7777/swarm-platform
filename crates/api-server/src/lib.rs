//! HTTP ingress for the swarm platform.
//!
//! Wraps a [`Coordinator`] in an HTTP API so the platform can run as its own process
//! and be driven by anything that speaks HTTP — the CLI, a dashboard, or curl. Job
//! submission returns immediately with an id; progress arrives over a Server-Sent
//! Events stream, and every intermediate artifact (task graph, per-task results,
//! failures, audit trail) is separately addressable.
//!
//! Handlers do no work of their own: they translate HTTP into coordinator calls and
//! [`SwarmError`](swarm_domain::SwarmError) into status codes. All the behaviour lives
//! in `swarm-coordinator`, which is what keeps this crate swappable for the gRPC
//! ingress that arrives alongside it.
#![forbid(unsafe_code)]

pub mod error;
pub mod metrics;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::BroadcastStream;

use swarm_coordinator::{Coordinator, JobStateView};
use swarm_domain::{
    FinalResult, JobId, JobRequest, JobStatus, TaskFailure, TaskNode, TaskResult, TaskState,
    Transition,
};
use swarm_model_gateway::{ProviderHealth, Usage};
use swarm_task_queue::QueueStats;

pub use error::ApiError;

/// Shared state handed to every request.
#[derive(Clone)]
pub struct AppState {
    coordinator: Arc<Coordinator>,
    started_at: Instant,
}

impl AppState {
    /// Wrap a coordinator for serving.
    #[must_use]
    pub fn new(coordinator: Arc<Coordinator>) -> Self {
        Self {
            coordinator,
            started_at: Instant::now(),
        }
    }

    /// The coordinator behind this API.
    #[must_use]
    pub fn coordinator(&self) -> &Arc<Coordinator> {
        &self.coordinator
    }
}

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(prometheus_metrics))
        .route("/v1/cluster", get(cluster))
        .route("/v1/jobs", post(submit_job).get(list_jobs))
        .route("/v1/jobs/:id", get(job_state))
        .route("/v1/jobs/:id/graph", get(job_graph))
        .route("/v1/jobs/:id/results", get(job_results))
        .route("/v1/jobs/:id/result", get(job_final_result))
        .route("/v1/jobs/:id/failures", get(job_failures))
        .route("/v1/jobs/:id/transitions", get(job_transitions))
        .route("/v1/jobs/:id/agents", get(job_agents))
        .route("/v1/jobs/:id/events", get(job_events))
        .route("/v1/jobs/:id/cancel", post(cancel_job))
        .route("/v1/jobs/:id/pause", post(pause_job))
        .route("/v1/jobs/:id/resume", post(resume_job))
        .with_state(state)
}

/// Liveness and build information.
#[derive(Debug, Serialize, Deserialize)]
pub struct Health {
    /// Always `ok` when the process is serving.
    pub status: &'static str,
    /// Crate version of the running binary.
    pub version: &'static str,
    /// Seconds since the process started serving.
    pub uptime_seconds: u64,
}

/// The response to a successful submission.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitResponse {
    /// Identifier to poll and stream with.
    pub job_id: JobId,
    /// Status at the moment of admission.
    pub status: JobStatus,
    /// How many tasks the objective compiled into.
    pub tasks_planned: usize,
}

/// A snapshot of the whole coordinator, for operators and the dashboard.
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterView {
    /// Node this coordinator runs on.
    pub node_id: String,
    /// Placement strategy in force.
    pub scheduler: String,
    /// Agents allocated right now.
    pub cluster_agents: usize,
    /// Ceiling on concurrently allocated agents.
    pub max_cluster_agents: usize,
    /// Jobs that have not finished.
    pub jobs_active: usize,
    /// Jobs known to this coordinator.
    pub jobs_total: usize,
    /// Queue health.
    pub queue: QueueStats,
    /// Tasks parked after exhausting their retries.
    pub dead_letters: usize,
    /// Model spend so far.
    pub gateway: Usage,
    /// Per-provider health, including breaker state.
    pub providers: Vec<ProviderHealth>,
}

async fn health(State(state): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.started_at.elapsed().as_secs(),
    })
}

async fn prometheus_metrics(
    State(state): State<AppState>,
) -> ([(&'static str, &'static str); 1], String) {
    (
        [("content-type", "text/plain; version=0.0.4")],
        metrics::render(&state.coordinator).await,
    )
}

async fn cluster(State(state): State<AppState>) -> Result<Json<ClusterView>, ApiError> {
    let coordinator = &state.coordinator;
    let jobs = coordinator.list_jobs();

    Ok(Json(ClusterView {
        node_id: coordinator.config().node_id.to_string(),
        scheduler: format!("{:?}", coordinator.config().scheduler),
        cluster_agents: coordinator.cluster_agent_count(),
        max_cluster_agents: coordinator.config().max_cluster_agents,
        jobs_active: jobs.iter().filter(|job| !job.status.is_finished()).count(),
        jobs_total: jobs.len(),
        queue: coordinator.queue().stats().await?,
        dead_letters: coordinator.queue().dead_letters().await?.len(),
        gateway: coordinator.gateway().usage(),
        providers: coordinator.gateway().health().await,
    }))
}

/// Submit a job. Returns as soon as it is planned; execution continues in the
/// background and is followed over `/v1/jobs/{id}/events`.
async fn submit_job(
    State(state): State<AppState>,
    Json(request): Json<JobRequest>,
) -> Result<(StatusCode, Json<SubmitResponse>), ApiError> {
    let job_id = state.coordinator.submit(request)?;
    let planned = state.coordinator.state(job_id)?;

    let coordinator = Arc::clone(&state.coordinator);
    tokio::spawn(async move {
        if let Err(error) = coordinator.run(job_id).await {
            // The job's own status already records the outcome; this is for the
            // operator watching logs.
            tracing::error!(%job_id, %error, "job execution failed");
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(SubmitResponse {
            job_id,
            status: planned.status,
            tasks_planned: planned.counts.total,
        }),
    ))
}

async fn list_jobs(State(state): State<AppState>) -> Json<Vec<JobStateView>> {
    Json(state.coordinator.list_jobs())
}

async fn job_state(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<JobStateView>, ApiError> {
    Ok(Json(state.coordinator.state(job_id(&id)?)?))
}

async fn job_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<TaskNode>>, ApiError> {
    Ok(Json(state.coordinator.task_graph(job_id(&id)?)?))
}

async fn job_results(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<TaskResult>>, ApiError> {
    Ok(Json(state.coordinator.intermediate_results(job_id(&id)?)?))
}

/// The final result, or `404` while the job is still running.
async fn job_final_result(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<FinalResult>, ApiError> {
    let job_id = job_id(&id)?;
    state
        .coordinator
        .final_result(job_id)?
        .map(Json)
        .ok_or_else(|| ApiError::not_found("finished job", &id))
}

async fn job_failures(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<TaskFailure>>, ApiError> {
    Ok(Json(state.coordinator.failures(job_id(&id)?)?))
}

async fn job_transitions(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<Transition<TaskState>>>, ApiError> {
    Ok(Json(state.coordinator.transitions(job_id(&id)?)?))
}

async fn job_agents(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<swarm_domain::AgentDescriptor>>, ApiError> {
    Ok(Json(state.coordinator.agents(job_id(&id)?)?))
}

/// Live job events as Server-Sent Events.
///
/// Each event carries its sequence number as the SSE id, so a client that reconnects
/// can tell whether it missed anything. The underlying channel is bounded and lossy by
/// design: a subscriber that falls too far behind sees a gap rather than making the
/// coordinator buffer without limit.
async fn job_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let receiver = state.coordinator.subscribe(job_id(&id)?)?;

    let stream = BroadcastStream::new(receiver).filter_map(|event| {
        std::future::ready(match event {
            Ok(event) => Event::default()
                .id(event.sequence_number.to_string())
                .json_data(&event)
                .ok()
                .map(Ok::<_, Infallible>),
            Err(_lagged) => None,
        })
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job_id = job_id(&id)?;
    state.coordinator.cancel(job_id).await?;
    Ok(Json(json!({ "job_id": job_id, "cancelled": true })))
}

async fn pause_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job_id = job_id(&id)?;
    state.coordinator.pause(job_id)?;
    Ok(Json(json!({ "job_id": job_id, "paused": true })))
}

async fn resume_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job_id = job_id(&id)?;
    state.coordinator.resume(job_id)?;
    Ok(Json(json!({ "job_id": job_id, "paused": false })))
}

/// Parse a path segment into a job id, rejecting garbage with `400` rather than `404`.
fn job_id(raw: &str) -> Result<JobId, ApiError> {
    raw.parse::<JobId>()
        .map_err(|_| ApiError::bad_request(format!("`{raw}` is not a job id")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_ids_are_a_client_error_not_a_missing_resource() {
        let error = job_id("not-a-uuid").unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert!(job_id(&JobId::new().to_string()).is_ok());
    }
}
