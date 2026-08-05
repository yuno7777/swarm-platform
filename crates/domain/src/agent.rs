//! Agents, their capabilities, and the resources of the nodes they run on.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{AgentId, JobId, NodeId, TaskId};

/// A skill an agent can offer and a task can require.
///
/// Scheduling is capability-based rather than type-based: a task says what it needs,
/// an agent says what it has, and the scheduler matches. This keeps new agent types
/// from requiring scheduler changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Decomposing objectives and revising plans.
    Planning,
    /// Gathering and citing external information.
    Research,
    /// Writing or modifying code.
    Coding,
    /// Critiquing another agent's output.
    Review,
    /// Checking claims and outputs against criteria.
    Verification,
    /// Condensing many outputs into one.
    Summarization,
    /// Calling external tools.
    ToolUse,
    /// Quantitative analysis of structured data.
    DataAnalysis,
    /// Looking for security problems.
    SecurityReview,
    /// Merging validated outputs into a final artifact.
    Merging,
    /// No special skill required.
    General,
}

impl Capability {
    /// Every capability, for CLI help and exhaustive tests.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Planning,
            Self::Research,
            Self::Coding,
            Self::Review,
            Self::Verification,
            Self::Summarization,
            Self::ToolUse,
            Self::DataAnalysis,
            Self::SecurityReview,
            Self::Merging,
            Self::General,
        ]
    }

    /// Stable lowercase label used in metrics, protobuf, and the database.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::Research => "research",
            Self::Coding => "coding",
            Self::Review => "review",
            Self::Verification => "verification",
            Self::Summarization => "summarization",
            Self::ToolUse => "tool_use",
            Self::DataAnalysis => "data_analysis",
            Self::SecurityReview => "security_review",
            Self::Merging => "merging",
            Self::General => "general",
        }
    }
}

/// The kind of agent, which fixes its default capabilities, prompt, and model config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    /// Decomposes objectives into tasks.
    Planner,
    /// Investigates a subtopic and cites sources.
    Research,
    /// Implements code.
    Coding,
    /// Critiques other agents' output.
    Critic,
    /// Checks outputs against validation criteria.
    Verification,
    /// Condenses outputs.
    Summarization,
    /// Calls external tools.
    ToolUse,
    /// Analyses data.
    DataAnalysis,
    /// Reviews for security issues.
    SecurityReview,
    /// Merges validated outputs.
    ResultMerger,
    /// Fallback agent with no specialisation.
    General,
}

impl AgentType {
    /// Every agent type.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Planner,
            Self::Research,
            Self::Coding,
            Self::Critic,
            Self::Verification,
            Self::Summarization,
            Self::ToolUse,
            Self::DataAnalysis,
            Self::SecurityReview,
            Self::ResultMerger,
            Self::General,
        ]
    }

    /// Capabilities an agent of this type advertises.
    ///
    /// Every type also implicitly offers [`Capability::General`].
    #[must_use]
    pub const fn capabilities(self) -> &'static [Capability] {
        match self {
            Self::Planner => &[Capability::Planning, Capability::General],
            Self::Research => &[Capability::Research, Capability::General],
            Self::Coding => &[Capability::Coding, Capability::ToolUse, Capability::General],
            Self::Critic => &[Capability::Review, Capability::General],
            Self::Verification => &[
                Capability::Verification,
                Capability::Review,
                Capability::General,
            ],
            Self::Summarization => &[Capability::Summarization, Capability::General],
            Self::ToolUse => &[Capability::ToolUse, Capability::General],
            Self::DataAnalysis => &[Capability::DataAnalysis, Capability::General],
            Self::SecurityReview => &[
                Capability::SecurityReview,
                Capability::Review,
                Capability::General,
            ],
            Self::ResultMerger => &[
                Capability::Merging,
                Capability::Summarization,
                Capability::General,
            ],
            Self::General => &[Capability::General],
        }
    }

    /// The cheapest agent type that offers `capability`.
    ///
    /// Used when the scheduler needs an agent for a task and none is warm.
    #[must_use]
    pub const fn for_capability(capability: Capability) -> Self {
        match capability {
            Capability::Planning => Self::Planner,
            Capability::Research => Self::Research,
            Capability::Coding => Self::Coding,
            Capability::Review => Self::Critic,
            Capability::Verification => Self::Verification,
            Capability::Summarization => Self::Summarization,
            Capability::ToolUse => Self::ToolUse,
            Capability::DataAnalysis => Self::DataAnalysis,
            Capability::SecurityReview => Self::SecurityReview,
            Capability::Merging => Self::ResultMerger,
            Capability::General => Self::General,
        }
    }

    /// Stable lowercase label for metrics and persistence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planner => "planner",
            Self::Research => "research",
            Self::Coding => "coding",
            Self::Critic => "critic",
            Self::Verification => "verification",
            Self::Summarization => "summarization",
            Self::ToolUse => "tool_use",
            Self::DataAnalysis => "data_analysis",
            Self::SecurityReview => "security_review",
            Self::ResultMerger => "result_merger",
            Self::General => "general",
        }
    }
}

/// Lifecycle state of an agent. Transitions are enforced in [`crate::state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// Constructed but not yet announced to the coordinator.
    Created,
    /// Known to the coordinator, not yet in the idle pool.
    Registered,
    /// Warm and available for work.
    Idle,
    /// Picked by the scheduler for a specific task.
    Assigned,
    /// Actively executing a task.
    Running,
    /// Blocked on a peer, a consensus round, or a clarification.
    Waiting,
    /// Finished its task successfully.
    Completed,
    /// Its task failed.
    Failed,
    /// Retrying the same task.
    Retrying,
    /// Shut down; will not run again.
    Terminated,
}

/// Which model an agent talks to and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider name, matched against the gateway's registered providers.
    pub provider: String,
    /// Model identifier as the provider names it.
    pub model: String,
    /// Sampling temperature.
    pub temperature: f32,
    /// Upper bound on generated tokens.
    pub max_tokens: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "mock".to_owned(),
            model: "mock-small".to_owned(),
            temperature: 0.2,
            max_tokens: 2048,
        }
    }
}

/// Everything the scheduler knows about one agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDescriptor {
    /// Agent identity.
    pub id: AgentId,
    /// Agent kind.
    pub agent_type: AgentType,
    /// Advertised capabilities.
    pub capabilities: Vec<Capability>,
    /// Lifecycle state.
    pub status: AgentStatus,
    /// Node hosting the agent.
    pub node_id: NodeId,
    /// Fraction of the agent's concurrency in use, `0.0..=1.0`.
    pub current_load: f32,
    /// Rolling success rate, `0.0..=1.0`.
    pub success_rate: f32,
    /// Rolling mean task latency.
    pub average_latency_ms: u64,
    /// Model settings.
    pub model_config: ModelConfig,
    /// Job the agent is pinned to, if any.
    pub job_id: Option<JobId>,
    /// Task currently assigned, if any.
    pub current_task: Option<TaskId>,
    /// Successful task count, feeds `success_rate`.
    pub tasks_completed: u64,
    /// Failed task count, feeds `success_rate`.
    pub tasks_failed: u64,
    /// When the agent was created.
    pub created_at: DateTime<Utc>,
    /// Last time the agent started or finished work; drives idle reaping.
    pub last_active_at: DateTime<Utc>,
}

impl AgentDescriptor {
    /// Create a warm-but-unregistered agent of `agent_type` on `node_id`.
    #[must_use]
    pub fn new(agent_type: AgentType, node_id: NodeId, model_config: ModelConfig) -> Self {
        let now = Utc::now();
        Self {
            id: AgentId::new(),
            agent_type,
            capabilities: agent_type.capabilities().to_vec(),
            status: AgentStatus::Created,
            node_id,
            current_load: 0.0,
            success_rate: 1.0,
            average_latency_ms: 0,
            model_config,
            job_id: None,
            current_task: None,
            tasks_completed: 0,
            tasks_failed: 0,
            created_at: now,
            last_active_at: now,
        }
    }

    /// Whether this agent offers every capability in `required`.
    #[must_use]
    pub fn supports(&self, required: &[Capability]) -> bool {
        required
            .iter()
            .all(|needed| self.capabilities.contains(needed))
    }

    /// Whether the agent can take work right now.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self.status, AgentStatus::Idle | AgentStatus::Registered)
    }

    /// Fold one completed attempt into the rolling reliability and latency stats.
    ///
    /// Uses an exponentially weighted mean so a single old failure does not haunt an
    /// agent forever, and a recent one is felt immediately.
    pub fn record_attempt(&mut self, succeeded: bool, latency_ms: u64) {
        const ALPHA: f32 = 0.2;
        if succeeded {
            self.tasks_completed += 1;
        } else {
            self.tasks_failed += 1;
        }
        let observed = if succeeded { 1.0 } else { 0.0 };
        self.success_rate = (1.0 - ALPHA) * self.success_rate + ALPHA * observed;
        self.average_latency_ms = if self.average_latency_ms == 0 {
            latency_ms
        } else {
            // Same smoothing, in integer arithmetic.
            (self.average_latency_ms * 4 + latency_ms) / 5
        };
        self.last_active_at = Utc::now();
    }
}

/// Lifecycle state of a worker node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    /// Registering.
    Joining,
    /// Healthy and accepting work.
    Ready,
    /// Reachable but unhealthy; deprioritised by the scheduler.
    Degraded,
    /// Finishing current work, accepting none.
    Draining,
    /// Heartbeats missed; its tasks are being rescheduled.
    Unreachable,
    /// Evicted from the cluster.
    Removed,
}

/// Resources a node advertises, refreshed on every heartbeat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeResources {
    /// Logical CPU count.
    pub cpu_cores: usize,
    /// Memory not currently committed.
    pub available_memory_bytes: u64,
    /// GPUs attached.
    pub gpu_count: usize,
    /// GPU memory not currently committed.
    pub available_gpu_memory_bytes: u64,
    /// Agents alive on this node.
    pub active_agents: usize,
    /// Tasks executing on this node.
    pub active_tasks: usize,
    /// Recent CPU utilisation, `0.0..=1.0`.
    pub cpu_utilization: f32,
    /// Tasks waiting in this node's local queue.
    pub queue_depth: usize,
}

impl Default for NodeResources {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            available_memory_bytes: 1 << 30,
            gpu_count: 0,
            available_gpu_memory_bytes: 0,
            active_agents: 0,
            active_tasks: 0,
            cpu_utilization: 0.0,
            queue_depth: 0,
        }
    }
}

impl NodeResources {
    /// Headroom score in `0.0..=1.0`; higher means better placement target.
    ///
    /// Combines CPU idleness with how loaded the node's agents already are, so load
    /// balancing considers both the machine and the work already assigned to it.
    #[must_use]
    pub fn headroom(&self) -> f32 {
        let cpu_free = (1.0 - self.cpu_utilization).clamp(0.0, 1.0);
        let task_pressure = 1.0 / (1.0 + self.active_tasks as f32);
        (cpu_free * 0.5) + (task_pressure * 0.5)
    }
}

/// One agent's answer in a consensus round.
///
/// `reasoning_summary` is a short, agent-authored summary. Raw provider reasoning
/// traces are deliberately never captured here (see ADR-8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDecision {
    /// Who voted.
    pub agent_id: AgentId,
    /// The proposed answer.
    pub answer: String,
    /// Self-reported confidence, `0.0..=1.0`.
    pub confidence: f32,
    /// Supporting evidence.
    pub evidence: Vec<crate::job::Evidence>,
    /// Short summary of how the agent got there.
    pub reasoning_summary: String,
    /// When the vote was cast.
    pub timestamp: DateTime<Utc>,
    /// Position in a ranked-choice ballot, if the mode uses one.
    pub rank_order: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_agent_type_offers_general_and_its_own_capability() {
        for &agent_type in AgentType::all() {
            let caps = agent_type.capabilities();
            assert!(
                caps.contains(&Capability::General),
                "{agent_type:?} must offer General as a fallback"
            );
        }
    }

    #[test]
    fn every_capability_has_an_agent_type_that_offers_it() {
        for &capability in Capability::all() {
            let agent_type = AgentType::for_capability(capability);
            assert!(
                agent_type.capabilities().contains(&capability),
                "{agent_type:?} was chosen for {capability:?} but does not offer it"
            );
        }
    }

    #[test]
    fn supports_requires_all_capabilities_not_just_one() {
        let agent =
            AgentDescriptor::new(AgentType::Research, NodeId::new(), ModelConfig::default());
        assert!(agent.supports(&[Capability::Research]));
        assert!(agent.supports(&[Capability::Research, Capability::General]));
        assert!(!agent.supports(&[Capability::Research, Capability::Coding]));
    }

    #[test]
    fn reliability_stats_react_to_recent_outcomes() {
        let mut agent =
            AgentDescriptor::new(AgentType::General, NodeId::new(), ModelConfig::default());
        assert_eq!(agent.success_rate, 1.0);

        agent.record_attempt(false, 100);
        assert!(agent.success_rate < 1.0, "a failure must lower the rate");
        assert_eq!(agent.tasks_failed, 1);
        assert_eq!(agent.average_latency_ms, 100);

        for _ in 0..50 {
            agent.record_attempt(true, 100);
        }
        assert!(
            agent.success_rate > 0.99,
            "sustained success must recover the rate, got {}",
            agent.success_rate
        );
        assert_eq!(agent.tasks_completed, 50);
    }

    #[test]
    fn headroom_prefers_idle_nodes() {
        let idle = NodeResources::default();
        let busy = NodeResources {
            cpu_utilization: 0.95,
            active_tasks: 32,
            ..NodeResources::default()
        };
        assert!(idle.headroom() > busy.headroom());
        assert!((0.0..=1.0).contains(&idle.headroom()));
        assert!((0.0..=1.0).contains(&busy.headroom()));
    }
}
