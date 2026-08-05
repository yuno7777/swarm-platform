//! Execution of one task by one agent.
//!
//! The agent runtime is where the platform's effectively-once guarantee is actually
//! enforced. Before doing any work it looks for a stored result, then claims an
//! execution record for its attempt; a duplicate delivery therefore returns the
//! original result instead of spending tokens and producing a second side effect.
//!
//! It also writes checkpoints, so a task that dies halfway can be resumed by whichever
//! agent picks it up next rather than starting over.
//!
//! Only concise reasoning summaries are persisted — never raw provider reasoning
//! traces (ADR-8).
#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde_json::json;

use swarm_domain::{
    validate_output, AgentDescriptor, AgentId, AgentType, Checkpoint, Evidence, JobId, NodeId,
    Result, SwarmError, TaskId, TaskKind, TaskNode, TaskResult,
};
use swarm_model_gateway::{CompletionRequest, Gateway, Message};
use swarm_shared_memory::{ns, MemoryStore, MemoryWrite};

/// What an agent is told about the job around its task.
#[derive(Debug, Clone, Default)]
pub struct TaskContext {
    /// The job's overall objective.
    pub objective: String,
    /// Extra background supplied with the job.
    pub context: Option<String>,
    /// Outputs of the tasks this one depends on.
    pub upstream: Vec<UpstreamOutput>,
    /// Tools this task is permitted to call.
    ///
    /// Enforced by the tool sandbox in Phase 5; carried from Phase 1 so no task is
    /// ever executed without an explicit permission list attached.
    pub allowed_tools: Vec<String>,
}

impl TaskContext {
    /// A context for `objective` with no upstream outputs.
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            ..Self::default()
        }
    }

    /// Add an upstream task's output.
    #[must_use]
    pub fn with_upstream(mut self, upstream: Vec<UpstreamOutput>) -> Self {
        self.upstream = upstream;
        self
    }
}

/// A dependency's output, handed to the task that waited for it.
#[derive(Debug, Clone)]
pub struct UpstreamOutput {
    /// Task that produced it.
    pub task_id: TaskId,
    /// Its title.
    pub title: String,
    /// What it produced.
    pub output: String,
}

/// A stateful worker that executes tasks against a model.
#[derive(Clone)]
pub struct Agent {
    /// Scheduling-visible state: capabilities, load, reliability.
    pub descriptor: AgentDescriptor,
    gateway: Arc<Gateway>,
    memory: Arc<dyn MemoryStore>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("id", &self.descriptor.id)
            .field("type", &self.descriptor.agent_type)
            .field("status", &self.descriptor.status)
            .finish()
    }
}

impl Agent {
    /// Create an agent of `agent_type` on `node_id`.
    #[must_use]
    pub fn new(
        agent_type: AgentType,
        node_id: NodeId,
        gateway: Arc<Gateway>,
        memory: Arc<dyn MemoryStore>,
    ) -> Self {
        Self {
            descriptor: AgentDescriptor::new(
                agent_type,
                node_id,
                swarm_domain::ModelConfig::default(),
            ),
            gateway,
            memory,
        }
    }

    /// Wrap an existing descriptor, e.g. one restored from the database.
    #[must_use]
    pub fn from_descriptor(
        descriptor: AgentDescriptor,
        gateway: Arc<Gateway>,
        memory: Arc<dyn MemoryStore>,
    ) -> Self {
        Self {
            descriptor,
            gateway,
            memory,
        }
    }

    /// This agent's identity.
    #[must_use]
    pub const fn id(&self) -> AgentId {
        self.descriptor.id
    }

    /// Execute `task`, returning its result.
    ///
    /// Returns an existing result without calling the model when the task has already
    /// been completed by another delivery of the same work.
    pub async fn execute(&self, task: &TaskNode, context: &TaskContext) -> Result<TaskResult> {
        let started = Instant::now();
        let job_id = task.job_id;

        if let Some(existing) = self.stored_result(job_id, task.id).await? {
            tracing::info!(
                task_id = %task.id,
                agent_id = %self.descriptor.id,
                "duplicate delivery: returning the stored result without re-executing"
            );
            return Ok(TaskResult {
                deduplicated: true,
                ..existing
            });
        }

        // Claim this attempt before any side effect. Losing the claim means another
        // agent is executing the same attempt right now.
        let claim = self
            .memory
            .claim(
                MemoryWrite::new(
                    ns::executions(job_id),
                    task.attempt_key(),
                    json!({
                        "agent_id": self.descriptor.id,
                        "task_id": task.id,
                        "attempt": task.attempt,
                        "claimed_at": Utc::now(),
                    }),
                    format!("agent:{}", self.descriptor.id),
                )
                .for_job(job_id),
            )
            .await?;
        if claim.is_none() {
            if let Some(existing) = self.stored_result(job_id, task.id).await? {
                return Ok(TaskResult {
                    deduplicated: true,
                    ..existing
                });
            }
            return Err(SwarmError::Internal(format!(
                "attempt {} of task {} is already being executed elsewhere",
                task.attempt, task.id
            )));
        }

        self.checkpoint(
            task,
            0,
            "started",
            json!({ "agent_id": self.descriptor.id }),
        )
        .await?;

        let request = CompletionRequest::new(
            self.descriptor.model_config.model.clone(),
            vec![
                Message::system(system_prompt(self.descriptor.agent_type)),
                Message::user(render_prompt(task, context)),
            ],
        )
        .json()
        .with_temperature(self.descriptor.model_config.temperature)
        .with_max_tokens(self.descriptor.model_config.max_tokens)
        .with_idempotency_key(task.attempt_key())
        // A retry means the previous identical call produced something unusable, so
        // the cached copy of that answer must not be handed back.
        .bypassing_cache(task.attempt > 1)
        .attributed_to(job_id, task.id, self.descriptor.id);

        let response = self.gateway.complete(request).await?;
        self.checkpoint(
            task,
            1,
            "model_responded",
            json!({ "tokens_out": response.tokens_out }),
        )
        .await?;

        let parsed = ParsedOutput::from_text(&response.text);
        // Validation runs against the raw response, so `RequiredJsonKeys` sees real
        // JSON and the text rules see everything the model actually produced.
        let validation_failures = validate_output(&response.text, &task.validation);

        let result = TaskResult {
            task_id: task.id,
            job_id,
            agent_id: self.descriptor.id,
            attempt: task.attempt,
            kind: task.kind,
            title: task.title.clone(),
            output: parsed.display_text(),
            structured: parsed.structured,
            evidence: parsed.evidence,
            reasoning_summary: parsed.reasoning_summary,
            confidence: parsed.confidence,
            validation_failures,
            tokens_in: response.tokens_in,
            tokens_out: response.tokens_out,
            cost_usd: response.cost_usd,
            duration_ms: started.elapsed().as_millis() as u64,
            deduplicated: false,
            finished_at: Utc::now(),
        };

        // Only a validated result is published, so a failed attempt does not block the
        // retry from running.
        if result.passed_validation() {
            let published = self
                .memory
                .claim(
                    MemoryWrite::json(
                        ns::results(job_id),
                        task.id.to_string(),
                        &result,
                        format!("agent:{}", self.descriptor.id),
                    )?
                    .for_job(job_id),
                )
                .await?;
            if published.is_none() {
                // Another agent published first; theirs is authoritative so the job
                // sees exactly one result per task.
                if let Some(winner) = self.stored_result(job_id, task.id).await? {
                    return Ok(TaskResult {
                        deduplicated: true,
                        ..winner
                    });
                }
            }
        }

        self.checkpoint(
            task,
            2,
            "completed",
            json!({ "validated": result.passed_validation() }),
        )
        .await?;
        Ok(result)
    }

    /// The published result for a task, if one exists.
    pub async fn stored_result(
        &self,
        job_id: JobId,
        task_id: TaskId,
    ) -> Result<Option<TaskResult>> {
        let record = self
            .memory
            .get(&ns::results(job_id), &task_id.to_string())
            .await?;
        record
            .map(|record| record.parse::<TaskResult>())
            .transpose()
    }

    /// The most recent checkpoint for a task attempt, for resumption after a crash.
    pub async fn latest_checkpoint(
        &self,
        job_id: JobId,
        task_id: TaskId,
    ) -> Result<Option<Checkpoint>> {
        let records = self
            .memory
            .list(&ns::checkpoints(job_id), &format!("{task_id}/"))
            .await?;
        records
            .last()
            .map(|record| record.parse::<Checkpoint>())
            .transpose()
    }

    async fn checkpoint(
        &self,
        task: &TaskNode,
        seq: u32,
        label: &str,
        state: serde_json::Value,
    ) -> Result<()> {
        let checkpoint = Checkpoint::new(task.id, task.attempt, seq, label, state);
        self.memory
            .put(
                MemoryWrite::json(
                    ns::checkpoints(task.job_id),
                    format!("{}/{}/{seq}", task.id, task.attempt),
                    &checkpoint,
                    format!("agent:{}", self.descriptor.id),
                )?
                .for_job(task.job_id),
            )
            .await?;
        Ok(())
    }
}

/// The structured shape agents are asked to produce, with tolerant fallbacks for
/// providers that ignore the instruction.
#[derive(Debug, Clone)]
struct ParsedOutput {
    summary: String,
    findings: Vec<String>,
    structured: serde_json::Value,
    evidence: Vec<Evidence>,
    reasoning_summary: String,
    confidence: f32,
}

impl ParsedOutput {
    fn from_text(text: &str) -> Self {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
            // A provider that ignored json_mode still produced usable prose; treat the
            // whole response as the summary rather than failing the task outright.
            return Self {
                summary: text.trim().to_owned(),
                findings: Vec::new(),
                structured: json!({ "text": text }),
                evidence: Vec::new(),
                reasoning_summary: String::new(),
                confidence: 0.5,
            };
        };

        let summary = value
            .get("summary")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| text.trim())
            .to_owned();
        let findings = value
            .get("findings")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        let confidence = value
            .get("confidence")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0) as f32;
        let reasoning_summary = value
            .get("reasoning_summary")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let evidence = value
            .get("evidence")
            .and_then(serde_json::Value::as_array)
            .map(|items| items.iter().filter_map(parse_evidence).collect())
            .unwrap_or_default();

        Self {
            summary,
            findings,
            structured: value,
            evidence,
            reasoning_summary,
            confidence,
        }
    }

    /// Human-readable rendering, used by the aggregator and the CLI.
    fn display_text(&self) -> String {
        if self.findings.is_empty() {
            return self.summary.clone();
        }
        let mut text = self.summary.clone();
        text.push('\n');
        for finding in &self.findings {
            text.push_str("\n- ");
            text.push_str(finding);
        }
        text
    }
}

fn parse_evidence(value: &serde_json::Value) -> Option<Evidence> {
    let source = value.get("source")?.as_str()?.to_owned();
    Some(Evidence {
        source,
        claim: value
            .get("claim")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        support: value
            .get("support")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0) as f32,
        locator: value
            .get("locator")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned),
    })
}

/// The system prompt for each agent type.
#[must_use]
pub fn system_prompt(agent_type: AgentType) -> String {
    let role = match agent_type {
        AgentType::Planner => {
            "You break objectives into independent, verifiable subtasks. Prefer fewer, \
             larger tasks over many trivial ones."
        }
        AgentType::Research => {
            "You investigate one narrow question and report findings with sources. State \
             what you could not establish."
        }
        AgentType::Coding => {
            "You write correct, minimal code. Prefer the standard library. Explain only \
             what is not obvious from the code."
        }
        AgentType::Critic => {
            "You find the weakest claim in the work you are given and say precisely why \
             it is weak. Do not rewrite the work."
        }
        AgentType::Verification => {
            "You check claims against their stated evidence and report each as supported, \
             unsupported, or contradicted."
        }
        AgentType::Summarization => {
            "You compress inputs without inventing anything. Keep numbers and caveats."
        }
        AgentType::ToolUse => {
            "You call only the tools you were permitted and report their raw results."
        }
        AgentType::DataAnalysis => {
            "You compute over the data given and report figures with their units and \
             sample sizes."
        }
        AgentType::SecurityReview => {
            "You look for security weaknesses and rank them by exploitability and impact."
        }
        AgentType::ResultMerger => {
            "You merge validated outputs, remove duplication, and surface contradictions \
             instead of hiding them."
        }
        AgentType::General => "You complete the task you are given, accurately and briefly.",
    };
    format!(
        "{role}\n\nRespond with a single JSON object containing: summary (string), \
         findings (array of strings), confidence (number between 0 and 1), evidence \
         (array of objects with source, claim, support), and reasoning_summary (one \
         short sentence). Do not include private deliberation."
    )
}

/// Render the user-side prompt for a task.
#[must_use]
pub fn render_prompt(task: &TaskNode, context: &TaskContext) -> String {
    let mut prompt = format!(
        "Task: {}\nKind: {}\nObjective: {}\nInstruction: {}\n",
        task.title, task.kind, context.objective, task.description
    );
    if let Some(extra) = &context.context {
        prompt.push_str(&format!("Background: {extra}\n"));
    }
    if !context.upstream.is_empty() {
        prompt.push_str("\nUpstream results:\n");
        for upstream in &context.upstream {
            // Upstream text is truncated: a fan-in task with 100 dependencies must not
            // build an unbounded prompt.
            let excerpt: String = upstream.output.chars().take(1_200).collect();
            prompt.push_str(&format!("\n## {}\n{excerpt}\n", upstream.title));
        }
    }
    if !context.allowed_tools.is_empty() {
        prompt.push_str(&format!(
            "\nPermitted tools: {}\n",
            context.allowed_tools.join(", ")
        ));
    }
    prompt
}

/// The agent type best suited to a task kind.
#[must_use]
pub const fn agent_type_for(kind: TaskKind) -> AgentType {
    match kind {
        TaskKind::Plan | TaskKind::Subplan | TaskKind::Supervise => AgentType::Planner,
        TaskKind::Work | TaskKind::Map | TaskKind::Answer => AgentType::Research,
        TaskKind::Critique => AgentType::Critic,
        TaskKind::Verify | TaskKind::Vote | TaskKind::Judge => AgentType::Verification,
        TaskKind::Reduce | TaskKind::Merge => AgentType::ResultMerger,
        TaskKind::Summarize => AgentType::Summarization,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use swarm_domain::{JobId, ValidationRule};
    use swarm_model_gateway::{Gateway, MockProvider};
    use swarm_shared_memory::InMemoryStore;

    struct Harness {
        agent: Agent,
        provider: Arc<MockProvider>,
        memory: Arc<InMemoryStore>,
        job_id: JobId,
    }

    fn harness_with(provider: MockProvider) -> Harness {
        let provider = Arc::new(provider);
        let gateway = Arc::new(Gateway::with_provider(provider.clone()));
        let memory = Arc::new(InMemoryStore::new());
        let agent = Agent::new(AgentType::Research, NodeId::new(), gateway, memory.clone());
        Harness {
            agent,
            provider,
            memory,
            job_id: JobId::new(),
        }
    }

    fn harness() -> Harness {
        harness_with(MockProvider::new("mock"))
    }

    fn task(job_id: JobId) -> TaskNode {
        let mut node = TaskNode::new(
            job_id,
            TaskKind::Work,
            "Investigate Raft leader election",
            "Explain how a leader is chosen.",
            1,
        );
        node.attempt = 1;
        node
    }

    #[tokio::test]
    async fn a_task_runs_and_produces_a_validated_result() {
        let harness = harness();
        let task = task(harness.job_id);
        let context = TaskContext::new("Understand consensus algorithms");

        let result = harness.agent.execute(&task, &context).await.unwrap();

        assert_eq!(result.task_id, task.id);
        assert_eq!(result.agent_id, harness.agent.id());
        assert_eq!(result.attempt, 1);
        assert!(result.passed_validation());
        assert!(!result.deduplicated);
        assert!(!result.output.is_empty());
        assert!(result.tokens_in > 0 && result.tokens_out > 0);
        assert!(result.cost_usd > 0.0);
        assert!((0.0..=1.0).contains(&result.confidence));
        assert!(!result.evidence.is_empty());
        assert!(!result.reasoning_summary.is_empty());
    }

    #[tokio::test]
    async fn a_duplicate_delivery_returns_the_first_result_without_calling_the_model() {
        // The core effectively-once guarantee: at-least-once delivery must not become
        // at-least-once execution.
        let harness = harness();
        let task = task(harness.job_id);
        let context = TaskContext::new("objective");

        let first = harness.agent.execute(&task, &context).await.unwrap();
        assert_eq!(harness.provider.call_count(), 1);

        let second = harness.agent.execute(&task, &context).await.unwrap();
        assert_eq!(
            harness.provider.call_count(),
            1,
            "the duplicate must not reach the model"
        );
        assert!(second.deduplicated);
        assert_eq!(second.output, first.output);
        assert_eq!(second.agent_id, first.agent_id);
    }

    #[tokio::test]
    async fn a_second_agent_delivered_the_same_task_also_deduplicates() {
        let harness = harness();
        let task = task(harness.job_id);
        let context = TaskContext::new("objective");
        let first = harness.agent.execute(&task, &context).await.unwrap();

        let other = Agent::new(
            AgentType::Research,
            NodeId::new(),
            harness.agent.gateway.clone(),
            harness.memory.clone() as Arc<dyn MemoryStore>,
        );
        let second = other.execute(&task, &context).await.unwrap();

        assert!(second.deduplicated);
        assert_eq!(
            second.agent_id, first.agent_id,
            "the original author is kept"
        );
        assert_eq!(harness.provider.call_count(), 1);
    }

    #[tokio::test]
    async fn validation_failures_are_reported_and_the_result_is_not_published() {
        let harness = harness();
        let mut task = task(harness.job_id);
        task.validation = vec![ValidationRule::MustMention("Byzantine".into())];
        let context = TaskContext::new("objective");

        let result = harness.agent.execute(&task, &context).await.unwrap();
        assert!(!result.passed_validation());
        assert_eq!(result.validation_failures.len(), 1);

        // Nothing was published, so a retry is free to run.
        assert!(harness
            .agent
            .stored_result(harness.job_id, task.id)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn a_retry_of_a_failed_attempt_is_allowed_to_execute() {
        let harness = harness();
        let mut task = task(harness.job_id);
        task.validation = vec![ValidationRule::MustMention("Byzantine".into())];
        let context = TaskContext::new("objective");

        harness.agent.execute(&task, &context).await.unwrap();
        assert_eq!(harness.provider.call_count(), 1);

        task.attempt = 2;
        task.validation = vec![ValidationRule::NonEmpty];
        let retry = harness.agent.execute(&task, &context).await.unwrap();

        assert_eq!(
            harness.provider.call_count(),
            2,
            "attempt 2 must really run"
        );
        assert!(retry.passed_validation());
        assert!(!retry.deduplicated);
    }

    #[tokio::test]
    async fn provider_failure_surfaces_as_an_error_not_a_bad_result() {
        let harness = harness_with(MockProvider::new("mock").always_failing());
        let result = harness
            .agent
            .execute(&task(harness.job_id), &TaskContext::new("objective"))
            .await;
        assert!(matches!(result, Err(SwarmError::Provider { .. })));
    }

    #[tokio::test]
    async fn checkpoints_are_written_so_a_crashed_task_can_be_resumed() {
        let harness = harness();
        let task = task(harness.job_id);
        harness
            .agent
            .execute(&task, &TaskContext::new("objective"))
            .await
            .unwrap();

        let checkpoint = harness
            .agent
            .latest_checkpoint(harness.job_id, task.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint.task_id, task.id);
        assert_eq!(checkpoint.label, "completed");
        assert_eq!(checkpoint.attempt, 1);

        let all = harness
            .memory
            .list(&ns::checkpoints(harness.job_id), &format!("{}/", task.id))
            .await
            .unwrap();
        assert_eq!(all.len(), 3, "started, model_responded, completed");
    }

    #[tokio::test]
    async fn upstream_outputs_reach_the_prompt_but_stay_bounded() {
        let task = task(JobId::new());
        let context = TaskContext::new("the objective").with_upstream(vec![UpstreamOutput {
            task_id: TaskId::new(),
            title: "Prior finding".into(),
            output: "x".repeat(5_000),
        }]);

        let prompt = render_prompt(&task, &context);
        assert!(prompt.starts_with("Task: Investigate Raft leader election"));
        assert!(prompt.contains("Objective: the objective"));
        assert!(prompt.contains("## Prior finding"));
        assert!(
            prompt.len() < 3_000,
            "a long upstream output must be truncated, got {} bytes",
            prompt.len()
        );
    }

    #[tokio::test]
    async fn prose_responses_are_handled_when_a_provider_ignores_json_mode() {
        let harness = harness_with(
            MockProvider::new("mock").scripted("Task:", "Just some prose, no JSON here."),
        );
        let result = harness
            .agent
            .execute(&task(harness.job_id), &TaskContext::new("objective"))
            .await
            .unwrap();

        assert_eq!(result.output, "Just some prose, no JSON here.");
        assert_eq!(result.confidence, 0.5);
        assert!(result.evidence.is_empty());
        assert!(result.passed_validation(), "NonEmpty still holds");
    }

    #[test]
    fn every_task_kind_maps_to_an_agent_type_that_can_serve_it() {
        for kind in [
            TaskKind::Plan,
            TaskKind::Subplan,
            TaskKind::Work,
            TaskKind::Map,
            TaskKind::Reduce,
            TaskKind::Answer,
            TaskKind::Critique,
            TaskKind::Verify,
            TaskKind::Judge,
            TaskKind::Vote,
            TaskKind::Merge,
            TaskKind::Summarize,
            TaskKind::Supervise,
        ] {
            let agent_type = agent_type_for(kind);
            for capability in kind.required_capabilities() {
                assert!(
                    agent_type.capabilities().contains(capability),
                    "{agent_type:?} cannot serve {kind:?} (missing {capability:?})"
                );
            }
        }
    }

    #[test]
    fn system_prompts_ask_for_the_structure_the_parser_expects() {
        for agent_type in AgentType::all() {
            let prompt = system_prompt(*agent_type);
            assert!(prompt.contains("summary"));
            assert!(prompt.contains("confidence"));
            assert!(prompt.contains("reasoning_summary"));
        }
    }
}
