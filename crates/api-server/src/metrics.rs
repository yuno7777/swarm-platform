//! Prometheus exposition.
//!
//! Rendered by hand rather than through a metrics library: the platform already keeps
//! these counters as first-class state, so a registry would only be a second copy of
//! them that could disagree with the first.

use std::fmt::Write;

use swarm_coordinator::Coordinator;
use swarm_domain::JobStatus;

/// Collect and render the current metrics in Prometheus text format.
pub async fn render(coordinator: &Coordinator) -> String {
    let queue = coordinator.queue().stats().await.unwrap_or_default();
    let dead_letters = coordinator
        .queue()
        .dead_letters()
        .await
        .map_or(0, |letters| letters.len());
    let usage = coordinator.gateway().usage();
    let jobs = coordinator.list_jobs();

    let mut out = String::with_capacity(2_048);
    let mut gauge = |name: &str, help: &str, value: f64| {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} gauge");
        let _ = writeln!(out, "{name} {value}");
    };

    gauge(
        "swarm_jobs_active",
        "Jobs that have not reached a terminal state.",
        jobs.iter().filter(|job| !job.status.is_finished()).count() as f64,
    );
    gauge(
        "swarm_queue_ready",
        "Tasks visible to consumers right now.",
        queue.ready as f64,
    );
    gauge(
        "swarm_queue_delayed",
        "Tasks waiting on a delay or retry backoff.",
        queue.delayed as f64,
    );
    gauge(
        "swarm_queue_leased",
        "Tasks currently held under a lease.",
        queue.leased as f64,
    );
    gauge(
        "swarm_dead_letter_depth",
        "Tasks parked after exhausting their retries.",
        dead_letters as f64,
    );
    gauge(
        "swarm_cluster_agents",
        "Agents allocated across every job on this coordinator.",
        coordinator.cluster_agent_count() as f64,
    );

    let mut counter = |name: &str, help: &str, value: f64| {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name} {value}");
    };

    counter(
        "swarm_tasks_enqueued_total",
        "Tasks admitted to the queue.",
        queue.enqueued_total as f64,
    );
    counter(
        "swarm_tasks_acknowledged_total",
        "Tasks acknowledged after a successful run.",
        queue.acknowledged_total as f64,
    );
    counter(
        "swarm_tasks_retried_total",
        "Task attempts re-queued after a failure or an expiry.",
        queue.retried_total as f64,
    );
    counter(
        "swarm_leases_expired_total",
        "Leases that lapsed before being acknowledged.",
        queue.expired_total as f64,
    );
    counter(
        "swarm_model_requests_total",
        "Model calls that reached a provider.",
        usage.requests as f64,
    );
    counter(
        "swarm_model_cache_hits_total",
        "Model calls served from the gateway cache.",
        usage.cache_hits as f64,
    );
    counter(
        "swarm_model_failures_total",
        "Model calls that returned an error.",
        usage.failures as f64,
    );
    counter(
        "swarm_tokens_in_total",
        "Prompt tokens consumed.",
        usage.tokens_in as f64,
    );
    counter(
        "swarm_tokens_out_total",
        "Completion tokens produced.",
        usage.tokens_out as f64,
    );
    counter(
        "swarm_cost_usd_total",
        "Estimated model spend in USD.",
        usage.cost_usd,
    );

    // Job counts, labelled by status.
    let _ = writeln!(out, "# HELP swarm_jobs_total Jobs by current status.");
    let _ = writeln!(out, "# TYPE swarm_jobs_total gauge");
    for status in [
        JobStatus::Planning,
        JobStatus::Running,
        JobStatus::Paused,
        JobStatus::Completed,
        JobStatus::PartiallyCompleted,
        JobStatus::Failed,
        JobStatus::Cancelled,
    ] {
        let count = jobs.iter().filter(|job| job.status == status).count();
        let _ = writeln!(
            out,
            "swarm_jobs_total{{status=\"{}\"}} {count}",
            status.as_str()
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use swarm_domain::JobRequest;
    use swarm_model_gateway::{Gateway, MockProvider};

    #[tokio::test]
    async fn the_exposition_is_well_formed_and_reflects_submitted_work() {
        let coordinator = Coordinator::local(Arc::new(Gateway::with_provider(Arc::new(
            MockProvider::new("mock"),
        ))));
        coordinator
            .submit(JobRequest::new("measure something").with_max_agents(2))
            .unwrap();

        let text = render(&coordinator).await;

        // Every series must be preceded by its HELP and TYPE lines.
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            let name = line.split([' ', '{']).next().unwrap();
            assert!(
                text.contains(&format!("# HELP {name} ")),
                "series `{name}` has no HELP line"
            );
            assert!(
                text.contains(&format!("# TYPE {name} ")),
                "series `{name}` has no TYPE line"
            );
        }

        assert!(text.contains("swarm_jobs_active 1"));
        assert!(text.contains("swarm_jobs_total{status=\"planning\"} 1"));
        assert!(text.contains("swarm_cost_usd_total 0"));
    }
}
