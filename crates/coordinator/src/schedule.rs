//! Placing tasks on agents.
//!
//! Four strategies share one trait so they can be swapped per deployment and compared
//! head-to-head in the benchmark suite. Every strategy filters to agents that are
//! *available* and *capable* first; they differ only in how they rank what is left.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};

use swarm_domain::{AgentDescriptor, AgentId, TaskNode};

/// Which scheduling strategy a coordinator runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerKind {
    /// Even distribution, ignoring load and history.
    RoundRobin,
    /// Whichever capable agent is least busy.
    LeastLoaded,
    /// Weighted mix of load, reliability, latency, and capability fit.
    #[default]
    CapabilityWeighted,
    /// Capability-weighted, with weights tuned from observed outcomes.
    Adaptive,
}

impl SchedulerKind {
    /// Every strategy, for CLI help and comparative benchmarks.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::RoundRobin,
            Self::LeastLoaded,
            Self::CapabilityWeighted,
            Self::Adaptive,
        ]
    }

    /// Build the matching scheduler.
    #[must_use]
    pub fn build(self) -> Box<dyn Scheduler> {
        match self {
            Self::RoundRobin => Box::new(RoundRobin::default()),
            Self::LeastLoaded => Box::new(LeastLoaded),
            Self::CapabilityWeighted => Box::new(CapabilityWeighted::default()),
            Self::Adaptive => Box::new(Adaptive::default()),
        }
    }
}

/// What actually happened after a placement, fed back to adaptive strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulingOutcome {
    /// Agent that ran the task.
    pub agent_id: AgentId,
    /// Whether it produced a valid result.
    pub succeeded: bool,
    /// How long execution took.
    pub latency_ms: u64,
    /// How long the task sat in the queue first.
    pub queue_wait_ms: u64,
}

/// Chooses which agent runs a task.
pub trait Scheduler: Send + Sync {
    /// Strategy name, for metrics and reports.
    fn name(&self) -> &'static str;

    /// Pick an agent for `task`, or `None` if none of `agents` can take it.
    fn select(&self, task: &TaskNode, agents: &[AgentDescriptor]) -> Option<AgentId>;

    /// Learn from a completed placement. Ignored by stateless strategies.
    fn observe(&self, _outcome: SchedulingOutcome) {}

    /// Current tuning weights, exposed for the dashboard and benchmark reports.
    fn weights(&self) -> Weights {
        Weights::default()
    }
}

/// Agents that are both free and capable of running `task`.
fn candidates<'a>(task: &TaskNode, agents: &'a [AgentDescriptor]) -> Vec<&'a AgentDescriptor> {
    agents
        .iter()
        .filter(|agent| agent.is_available() && agent.supports(&task.required_capabilities))
        .collect()
}

/// Relative importance of each placement signal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    /// Preference for idle agents.
    pub load: f32,
    /// Preference for historically successful agents.
    pub reliability: f32,
    /// Preference for fast agents.
    pub latency: f32,
    /// Preference for specialists over generalists.
    pub specificity: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            load: 1.0,
            reliability: 1.0,
            latency: 0.5,
            specificity: 0.5,
        }
    }
}

/// Score an agent for a task under `weights`. Higher is better.
fn score(agent: &AgentDescriptor, task: &TaskNode, weights: Weights) -> f32 {
    let idleness = 1.0 - agent.current_load.clamp(0.0, 1.0);
    // Normalised so a 1s agent scores 0.5 and a 10s agent scores ~0.09.
    let speed = 1_000.0 / (1_000.0 + agent.average_latency_ms as f32);
    // A specialist that exactly covers the requirement beats a generalist that also
    // happens to, so scarce broad agents stay free for the tasks that need them.
    let extra = agent
        .capabilities
        .len()
        .saturating_sub(task.required_capabilities.len()) as f32;
    let specificity = 1.0 / (1.0 + extra);

    weights.load * idleness
        + weights.reliability * agent.success_rate.clamp(0.0, 1.0)
        + weights.latency * speed
        + weights.specificity * specificity
}

/// Hands tasks out in rotation. The baseline every other strategy is measured against.
#[derive(Debug, Default)]
pub struct RoundRobin {
    next: AtomicU64,
}

impl Scheduler for RoundRobin {
    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn select(&self, task: &TaskNode, agents: &[AgentDescriptor]) -> Option<AgentId> {
        let candidates = candidates(task, agents);
        if candidates.is_empty() {
            return None;
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) as usize % candidates.len();
        Some(candidates[index].id)
    }
}

/// Picks the least busy capable agent, breaking ties on latency.
#[derive(Debug, Default)]
pub struct LeastLoaded;

impl Scheduler for LeastLoaded {
    fn name(&self) -> &'static str {
        "least_loaded"
    }

    fn select(&self, task: &TaskNode, agents: &[AgentDescriptor]) -> Option<AgentId> {
        candidates(task, agents)
            .into_iter()
            .min_by(|left, right| {
                left.current_load
                    .total_cmp(&right.current_load)
                    .then_with(|| left.average_latency_ms.cmp(&right.average_latency_ms))
            })
            .map(|agent| agent.id)
    }
}

/// Weighted mix of load, reliability, latency, and capability fit.
#[derive(Debug, Default)]
pub struct CapabilityWeighted {
    weights: Weights,
}

impl CapabilityWeighted {
    /// Use non-default weights.
    #[must_use]
    pub const fn with_weights(weights: Weights) -> Self {
        Self { weights }
    }
}

impl Scheduler for CapabilityWeighted {
    fn name(&self) -> &'static str {
        "capability_weighted"
    }

    fn select(&self, task: &TaskNode, agents: &[AgentDescriptor]) -> Option<AgentId> {
        candidates(task, agents)
            .into_iter()
            .max_by(|left, right| {
                score(left, task, self.weights).total_cmp(&score(right, task, self.weights))
            })
            .map(|agent| agent.id)
    }

    fn weights(&self) -> Weights {
        self.weights
    }
}

/// Capability-weighted placement whose weights follow observed behaviour.
///
/// When the swarm starts failing, reliability matters more than speed; when it is
/// healthy but slow, latency matters more. The weights move slowly (EWMA) so one bad
/// task cannot swing placement for the whole job.
#[derive(Debug)]
pub struct Adaptive {
    weights: Mutex<Weights>,
    observed_success: Mutex<f32>,
    observed_queue_wait_ms: Mutex<f32>,
}

impl Default for Adaptive {
    fn default() -> Self {
        Self {
            weights: Mutex::new(Weights::default()),
            observed_success: Mutex::new(1.0),
            observed_queue_wait_ms: Mutex::new(0.0),
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Scheduler for Adaptive {
    fn name(&self) -> &'static str {
        "adaptive"
    }

    fn select(&self, task: &TaskNode, agents: &[AgentDescriptor]) -> Option<AgentId> {
        let weights = *lock(&self.weights);
        candidates(task, agents)
            .into_iter()
            .max_by(|left, right| {
                score(left, task, weights).total_cmp(&score(right, task, weights))
            })
            .map(|agent| agent.id)
    }

    fn observe(&self, outcome: SchedulingOutcome) {
        const ALPHA: f32 = 0.2;

        let success_rate = {
            let mut observed = lock(&self.observed_success);
            *observed = (1.0 - ALPHA) * *observed + ALPHA * f32::from(u8::from(outcome.succeeded));
            *observed
        };
        let queue_wait = {
            let mut observed = lock(&self.observed_queue_wait_ms);
            *observed = (1.0 - ALPHA) * *observed + ALPHA * outcome.queue_wait_ms as f32;
            *observed
        };

        let mut weights = lock(&self.weights);
        // Failures dominate: an unreliable swarm wastes far more time than a slow one.
        weights.reliability = 1.0 + (1.0 - success_rate) * 3.0;
        // Long queues mean the bottleneck is capacity, so spread work harder.
        weights.load = 1.0 + (queue_wait / 1_000.0).min(2.0);
        weights.latency = 0.5 + (1.0 - success_rate).min(0.5);
    }

    fn weights(&self) -> Weights {
        *lock(&self.weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use swarm_domain::{
        AgentStatus, AgentType, Capability, JobId, ModelConfig, NodeId, TaskKind, TaskNode,
    };

    fn agent(agent_type: AgentType) -> AgentDescriptor {
        let mut descriptor =
            AgentDescriptor::new(agent_type, NodeId::new(), ModelConfig::default());
        descriptor.status = AgentStatus::Idle;
        descriptor
    }

    fn task(kind: TaskKind) -> TaskNode {
        TaskNode::new(JobId::new(), kind, "a task", "do it", 0)
    }

    #[test]
    fn no_strategy_ever_places_a_task_on_an_incapable_agent() {
        let agents = vec![agent(AgentType::Summarization)];
        let coding = task(TaskKind::Work); // requires Research

        for kind in SchedulerKind::all() {
            let scheduler = kind.build();
            assert!(
                scheduler.select(&coding, &agents).is_none(),
                "{} placed a task on an agent that cannot run it",
                scheduler.name()
            );
        }
    }

    #[test]
    fn no_strategy_ever_places_a_task_on_a_busy_agent() {
        let mut busy = agent(AgentType::Research);
        busy.status = AgentStatus::Running;
        let agents = vec![busy];

        for kind in SchedulerKind::all() {
            let scheduler = kind.build();
            assert!(scheduler.select(&task(TaskKind::Work), &agents).is_none());
        }
    }

    #[test]
    fn an_empty_pool_yields_no_placement() {
        for kind in SchedulerKind::all() {
            assert!(kind.build().select(&task(TaskKind::Work), &[]).is_none());
        }
    }

    #[test]
    fn round_robin_spreads_work_evenly() {
        let agents: Vec<AgentDescriptor> = (0..4).map(|_| agent(AgentType::Research)).collect();
        let scheduler = RoundRobin::default();

        let mut counts: HashMap<AgentId, usize> = HashMap::new();
        for _ in 0..40 {
            let chosen = scheduler.select(&task(TaskKind::Work), &agents).unwrap();
            *counts.entry(chosen).or_default() += 1;
        }

        assert_eq!(counts.len(), 4);
        for count in counts.values() {
            assert_eq!(*count, 10);
        }
    }

    #[test]
    fn least_loaded_picks_the_idlest_agent() {
        let mut idle = agent(AgentType::Research);
        idle.current_load = 0.1;
        let mut busy = agent(AgentType::Research);
        busy.current_load = 0.9;
        let expected = idle.id;

        let agents = vec![busy, idle];
        assert_eq!(
            LeastLoaded.select(&task(TaskKind::Work), &agents),
            Some(expected)
        );
    }

    #[test]
    fn least_loaded_breaks_ties_on_latency() {
        let mut fast = agent(AgentType::Research);
        fast.average_latency_ms = 50;
        let mut slow = agent(AgentType::Research);
        slow.average_latency_ms = 5_000;
        let expected = fast.id;

        let agents = vec![slow, fast];
        assert_eq!(
            LeastLoaded.select(&task(TaskKind::Work), &agents),
            Some(expected)
        );
    }

    #[test]
    fn capability_weighted_prefers_the_reliable_agent_over_the_flaky_one() {
        let mut reliable = agent(AgentType::Research);
        reliable.success_rate = 0.99;
        let mut flaky = agent(AgentType::Research);
        flaky.success_rate = 0.20;
        let expected = reliable.id;

        let agents = vec![flaky, reliable];
        assert_eq!(
            CapabilityWeighted::default().select(&task(TaskKind::Work), &agents),
            Some(expected)
        );
    }

    #[test]
    fn capability_weighted_saves_generalists_for_work_only_they_can_do() {
        // Both can run the task; the specialist should take it so the broader agent
        // stays free.
        let specialist = agent(AgentType::Research);
        let mut generalist = agent(AgentType::Research);
        generalist.capabilities = Capability::all().to_vec();
        let expected = specialist.id;

        let agents = vec![generalist, specialist];
        assert_eq!(
            CapabilityWeighted::default().select(&task(TaskKind::Work), &agents),
            Some(expected)
        );
    }

    #[test]
    fn adaptive_weights_shift_towards_reliability_when_the_swarm_starts_failing() {
        let scheduler = Adaptive::default();
        let baseline = scheduler.weights();

        for _ in 0..20 {
            scheduler.observe(SchedulingOutcome {
                agent_id: AgentId::new(),
                succeeded: false,
                latency_ms: 100,
                queue_wait_ms: 0,
            });
        }
        let stressed = scheduler.weights();
        assert!(
            stressed.reliability > baseline.reliability,
            "reliability weight should rise: {} -> {}",
            baseline.reliability,
            stressed.reliability
        );

        for _ in 0..50 {
            scheduler.observe(SchedulingOutcome {
                agent_id: AgentId::new(),
                succeeded: true,
                latency_ms: 100,
                queue_wait_ms: 0,
            });
        }
        assert!(scheduler.weights().reliability < stressed.reliability);
    }

    #[test]
    fn adaptive_weights_shift_towards_spreading_load_when_queues_build_up() {
        let scheduler = Adaptive::default();
        let baseline = scheduler.weights();

        for _ in 0..20 {
            scheduler.observe(SchedulingOutcome {
                agent_id: AgentId::new(),
                succeeded: true,
                latency_ms: 100,
                queue_wait_ms: 5_000,
            });
        }
        assert!(scheduler.weights().load > baseline.load);
    }

    #[test]
    fn adaptive_still_respects_capability_and_availability_after_learning() {
        let scheduler = Adaptive::default();
        for _ in 0..10 {
            scheduler.observe(SchedulingOutcome {
                agent_id: AgentId::new(),
                succeeded: false,
                latency_ms: 10_000,
                queue_wait_ms: 9_000,
            });
        }
        let agents = vec![agent(AgentType::Summarization)];
        assert!(scheduler.select(&task(TaskKind::Work), &agents).is_none());
    }
}
