//! Turning an objective into a task DAG.
//!
//! Decomposition is a **deterministic stage compiler**, not an LLM call: a strategy
//! maps to a list of stages, each stage has a kind and a fan-out, and stage *n* depends
//! on every task in stage *n-1*. The same objective and strategy always produce the
//! same graph, which is what makes benchmarks reproducible and stops an unparseable
//! plan from failing a whole job (ADR-6).
//!
//! An LLM planner can propose a stage list instead; it is validated (acyclic, known
//! capabilities, fan-out inside quota) before anything is scheduled.

use swarm_domain::{
    ExecutionStrategy, Job, JobRequest, Result, TaskGraph, TaskId, TaskKind, TaskNode,
    ValidationRule,
};

/// One horizontal slice of a compiled strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// What the stage's tasks do.
    pub kind: TaskKind,
    /// How many tasks run in parallel in this stage.
    pub fanout: usize,
    /// Human-readable label, used as the title prefix.
    pub label: String,
}

impl Stage {
    fn new(kind: TaskKind, fanout: usize, label: impl Into<String>) -> Self {
        Self {
            kind,
            fanout: fanout.max(1),
            label: label.into(),
        }
    }
}

/// How many parallel branches an objective seems to deserve.
///
/// Counts words and explicit enumeration (commas, semicolons, newlines, "and"), then
/// clamps hard. A heuristic, but a *stable* one — and the caller's `max_agents` is the
/// real bound.
#[must_use]
pub fn estimate_complexity(objective: &str) -> u32 {
    let words = objective.split_whitespace().count();
    let enumerations = objective
        .chars()
        .filter(|c| matches!(c, ',' | ';' | '\n'))
        .count()
        + objective.to_lowercase().matches(" and ").count();

    let from_length = (words / 10) as u32;
    let from_structure = enumerations as u32;
    (2 + from_length + from_structure).clamp(2, 8)
}

/// Compile a strategy into stages.
#[must_use]
pub fn stages(strategy: ExecutionStrategy, fanout: usize) -> Vec<Stage> {
    let fanout = fanout.max(1);
    let half = (fanout / 2).max(1);

    match strategy {
        ExecutionStrategy::Sequential => {
            let mut plan = vec![Stage::new(TaskKind::Plan, 1, "Plan")];
            for step in 1..=fanout.min(4) {
                plan.push(Stage::new(TaskKind::Work, 1, format!("Step {step}")));
            }
            plan.push(Stage::new(TaskKind::Summarize, 1, "Summary"));
            plan
        }
        ExecutionStrategy::Parallel => vec![
            Stage::new(TaskKind::Plan, 1, "Plan"),
            Stage::new(TaskKind::Work, fanout, "Investigate"),
            Stage::new(TaskKind::Merge, 1, "Merge"),
        ],
        ExecutionStrategy::Hierarchical => vec![
            Stage::new(TaskKind::Plan, 1, "Plan"),
            Stage::new(TaskKind::Subplan, half, "Sub-plan"),
            Stage::new(TaskKind::Work, fanout, "Investigate"),
            Stage::new(TaskKind::Merge, 1, "Merge"),
            Stage::new(TaskKind::Verify, 1, "Verify"),
        ],
        ExecutionStrategy::Debate => vec![
            Stage::new(TaskKind::Answer, fanout, "Answer"),
            Stage::new(TaskKind::Critique, fanout, "Critique"),
            Stage::new(TaskKind::Judge, 1, "Judge"),
        ],
        ExecutionStrategy::MapReduce => vec![
            Stage::new(TaskKind::Plan, 1, "Partition"),
            Stage::new(TaskKind::Map, fanout, "Map"),
            Stage::new(TaskKind::Reduce, 1, "Reduce"),
        ],
        ExecutionStrategy::PlannerExecutor => vec![
            Stage::new(TaskKind::Plan, 1, "Plan"),
            Stage::new(TaskKind::Work, fanout, "Execute"),
            Stage::new(TaskKind::Verify, 1, "Verify"),
        ],
        ExecutionStrategy::SupervisorWorker => vec![
            Stage::new(TaskKind::Supervise, 1, "Assign"),
            Stage::new(TaskKind::Work, fanout, "Work"),
            Stage::new(TaskKind::Critique, 1, "Review"),
            Stage::new(TaskKind::Merge, 1, "Merge"),
        ],
        ExecutionStrategy::Consensus => vec![
            Stage::new(TaskKind::Answer, fanout, "Answer"),
            Stage::new(TaskKind::Vote, 1, "Vote"),
            Stage::new(TaskKind::Merge, 1, "Decide"),
        ],
        // Adaptive starts deliberately small: agents insert the rest of the work as
        // they discover it, via TaskGraph::insert_dynamic.
        ExecutionStrategy::Adaptive => vec![
            Stage::new(TaskKind::Plan, 1, "Plan"),
            Stage::new(TaskKind::Work, half, "Explore"),
            Stage::new(TaskKind::Summarize, 1, "Summary"),
        ],
    }
}

/// The mechanical checks a task of `kind` must pass.
///
/// Deliberately coarse: these gate obvious garbage so that critic and verifier agents
/// only spend tokens on plausible output (ADR-7).
#[must_use]
pub fn validation_for(kind: TaskKind) -> Vec<ValidationRule> {
    let mut rules = vec![
        ValidationRule::NonEmpty,
        ValidationRule::RequiredJsonKeys(vec!["summary".to_owned(), "confidence".to_owned()]),
    ];
    match kind {
        TaskKind::Plan | TaskKind::Subplan | TaskKind::Supervise => {
            rules.push(ValidationRule::MinWords(10));
        }
        TaskKind::Work | TaskKind::Map | TaskKind::Answer | TaskKind::Critique => {
            rules.push(ValidationRule::MinWords(15));
        }
        TaskKind::Merge | TaskKind::Reduce | TaskKind::Summarize | TaskKind::Judge => {
            rules.push(ValidationRule::MinWords(12));
        }
        TaskKind::Verify | TaskKind::Vote => {}
    }
    rules
}

/// Split an objective into `count` distinct subtopics.
///
/// Prefers the structure the author already wrote (clauses, lists); pads with numbered
/// aspects when the objective is one short phrase. Results are unique, because task
/// titles feed idempotency keys and a collision would silently drop a task.
#[must_use]
pub fn subtopics(objective: &str, count: usize) -> Vec<String> {
    let mut topics: Vec<String> = objective
        .split(['\n', ';', ',', '.'])
        .flat_map(|clause| clause.split(" and "))
        .map(|clause| shorten(clause.trim(), 10))
        .filter(|clause| clause.split_whitespace().count() >= 2)
        .collect();

    topics.dedup();
    let mut seen: Vec<String> = Vec::with_capacity(count);
    for topic in topics {
        let lowercase = topic.to_lowercase();
        if !seen
            .iter()
            .any(|existing| existing.to_lowercase() == lowercase)
        {
            seen.push(topic);
        }
        if seen.len() == count {
            return seen;
        }
    }

    let stem = shorten(objective, 6);
    let mut aspect = 1;
    while seen.len() < count {
        let candidate = format!("{stem} — aspect {aspect}");
        if !seen.contains(&candidate) {
            seen.push(candidate);
        }
        aspect += 1;
    }
    seen
}

/// First `words` words of `text`, trimmed.
fn shorten(text: &str, words: usize) -> String {
    text.split_whitespace()
        .take(words)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compile a job into its task graph.
///
/// The returned graph is guaranteed acyclic and free of duplicate idempotency keys.
pub fn decompose(job: &Job) -> Result<TaskGraph> {
    let request = &job.request;
    let complexity = estimate_complexity(&request.objective);
    // `max_agents` is both the ceiling and the target: someone who asks for 100 agents
    // wants 100-way parallelism, not a heuristic's opinion of how hard the objective
    // looks. Complexity still sets per-task size and token estimates.
    let fanout = request.max_agents.max(1);
    let plan = stages(request.execution_strategy, fanout);
    let widest = plan.iter().map(|stage| stage.fanout).max().unwrap_or(1);
    let topics = subtopics(&request.objective, widest);

    let mut graph = TaskGraph::new(job.id);
    let mut previous: Vec<TaskId> = Vec::new();

    for (stage_index, stage) in plan.iter().enumerate() {
        let mut current = Vec::with_capacity(stage.fanout);
        for slot in 0..stage.fanout {
            let focus = if stage.fanout == 1 {
                shorten(&request.objective, 8)
            } else {
                topics[slot % topics.len()].clone()
            };
            let node = TaskNode::new(
                job.id,
                stage.kind,
                format!("{} · {focus}", stage.label),
                describe(stage, &focus, request),
                stage_index as u32,
            )
            .with_dependencies(previous.clone())
            .with_complexity(complexity)
            .with_validation(validation_for(stage.kind));
            current.push(graph.insert(node)?);
        }
        previous = current;
    }

    graph.assert_acyclic()?;
    Ok(graph)
}

/// The instruction handed to the agent for one task.
fn describe(stage: &Stage, focus: &str, request: &JobRequest) -> String {
    let objective = &request.objective;
    match stage.kind {
        TaskKind::Plan => {
            format!("Break the objective into independent pieces of work. Objective: {objective}")
        }
        TaskKind::Subplan => format!("Plan the work needed for `{focus}` within: {objective}"),
        TaskKind::Work => {
            format!("Investigate `{focus}` and report what you established, with sources.")
        }
        TaskKind::Map => format!("Process the `{focus}` partition and return its partial result."),
        TaskKind::Reduce => "Combine the partial results into one consistent result.".to_owned(),
        TaskKind::Answer => format!("Answer independently, without deferring to peers: {focus}"),
        TaskKind::Critique => {
            format!("Find the weakest claim in the work on `{focus}` and say why.")
        }
        TaskKind::Verify => {
            "Check each claim against its stated evidence and flag unsupported ones.".to_owned()
        }
        TaskKind::Judge => {
            "Compare the competing answers and choose the best-supported one, with reasons."
                .to_owned()
        }
        TaskKind::Vote => {
            "Vote for the answer best supported by evidence, with a confidence.".to_owned()
        }
        TaskKind::Merge => {
            "Merge the validated outputs, removing duplication and surfacing conflicts.".to_owned()
        }
        TaskKind::Summarize => format!("Summarise what the swarm established about: {objective}"),
        TaskKind::Supervise => {
            format!("Assign the work for `{focus}` and state what good output looks like.")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use swarm_domain::JobRequest;

    fn job(objective: &str, strategy: ExecutionStrategy, max_agents: usize) -> Job {
        Job::new(
            JobRequest::new(objective)
                .with_strategy(strategy)
                .with_max_agents(max_agents),
        )
    }

    #[test]
    fn complexity_reacts_to_length_and_enumeration_but_stays_bounded() {
        assert_eq!(estimate_complexity("do it"), 2);
        assert!(estimate_complexity("compare a, b, c, and d") > 2);
        assert_eq!(estimate_complexity(&"word ".repeat(500)), 8);
        assert_eq!(estimate_complexity(""), 2);
    }

    #[test]
    fn subtopics_are_distinct_and_exactly_as_many_as_asked_for() {
        let topics = subtopics("compare Raft, Paxos, and Viewstamped Replication", 3);
        assert_eq!(topics.len(), 3);
        assert_eq!(
            topics.iter().collect::<HashSet<_>>().len(),
            3,
            "duplicate topics would collide idempotency keys"
        );

        // A short objective still yields the requested number of distinct topics.
        let padded = subtopics("do a thing", 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded.iter().collect::<HashSet<_>>().len(), 5);
    }

    #[test]
    fn every_strategy_produces_an_acyclic_graph_with_a_single_entry_point() {
        for &strategy in ExecutionStrategy::all() {
            let job = job("Compare Raft and Paxos for leader election", strategy, 6);
            let graph = decompose(&job).unwrap();

            graph.assert_acyclic().unwrap();
            assert!(!graph.is_empty(), "{strategy} produced no tasks");
            assert_eq!(
                graph.ready().len(),
                graph.layers().unwrap()[0].len(),
                "{strategy}: the first wave must be exactly the ready set"
            );
            assert!(
                graph.layers().unwrap().len() >= 2,
                "{strategy} must have more than one stage"
            );
        }
    }

    #[test]
    fn idempotency_keys_are_unique_across_the_whole_graph() {
        // A collision would make the queue silently drop a task as a duplicate.
        for &strategy in ExecutionStrategy::all() {
            let job = job(
                "Investigate consensus, replication, and recovery",
                strategy,
                8,
            );
            let graph = decompose(&job).unwrap();
            let keys: HashSet<&str> = graph
                .nodes()
                .map(|node| node.idempotency_key.as_str())
                .collect();
            assert_eq!(
                keys.len(),
                graph.len(),
                "{strategy} produced colliding idempotency keys"
            );
        }
    }

    #[test]
    fn decomposition_is_reproducible_in_shape() {
        let first = decompose(&job("Explain Raft", ExecutionStrategy::Parallel, 4)).unwrap();
        let second = decompose(&job("Explain Raft", ExecutionStrategy::Parallel, 4)).unwrap();

        assert_eq!(first.len(), second.len());
        let titles = |graph: &TaskGraph| {
            let mut titles: Vec<String> = graph.nodes().map(|node| node.title.clone()).collect();
            titles.sort();
            titles
        };
        assert_eq!(titles(&first), titles(&second));
    }

    #[test]
    fn fanout_follows_the_agent_budget() {
        for agents in [2, 8, 50] {
            let job = job("Explain consensus", ExecutionStrategy::Parallel, agents);
            let graph = decompose(&job).unwrap();
            let widest = graph
                .layers()
                .unwrap()
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            assert_eq!(
                widest, agents,
                "a {agents}-agent budget should fan out {agents} ways"
            );
        }
    }

    #[test]
    fn fanout_never_exceeds_the_agent_budget() {
        let job = job(
            "compare a, b, c, d, e, f, g, and h across many dimensions",
            ExecutionStrategy::Parallel,
            2,
        );
        let graph = decompose(&job).unwrap();
        let widest = graph
            .layers()
            .unwrap()
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        assert!(widest <= 2, "widest wave was {widest}, budget was 2");
    }

    #[test]
    fn debate_answers_are_independent_and_the_judge_waits_for_everyone() {
        let job = job("Is Raft simpler than Paxos?", ExecutionStrategy::Debate, 4);
        let graph = decompose(&job).unwrap();
        let layers = graph.layers().unwrap();

        assert!(layers[0].len() >= 2, "a debate needs competing answers");
        for answer in &layers[0] {
            assert!(
                graph.dependencies(*answer).is_empty(),
                "answers must be formed independently"
            );
        }
        let judge = layers.last().unwrap();
        assert_eq!(judge.len(), 1);
        assert_eq!(graph.get(judge[0]).unwrap().kind, TaskKind::Judge);
    }

    #[test]
    fn sequential_work_is_a_chain_not_a_fan_out() {
        let job = job("Do a, then b, then c", ExecutionStrategy::Sequential, 8);
        let graph = decompose(&job).unwrap();
        for layer in graph.layers().unwrap() {
            assert_eq!(layer.len(), 1, "sequential jobs must never widen");
        }
    }

    #[test]
    fn every_task_carries_validation_and_a_capability_requirement() {
        let job = job("Explain consensus", ExecutionStrategy::Hierarchical, 6);
        let graph = decompose(&job).unwrap();
        for node in graph.nodes() {
            assert!(!node.validation.is_empty(), "{} has no checks", node.title);
            assert!(
                !node.required_capabilities.is_empty(),
                "{} requires no capability",
                node.title
            );
            assert!(node.estimated_tokens.is_some());
            assert!(node.timeout_seconds > 0);
        }
    }
}
