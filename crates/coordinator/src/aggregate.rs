//! Merging many agents' outputs into one verified result.
//!
//! The pipeline is deterministic: order outputs by dependency stage, drop
//! near-duplicates, cluster competing answers, surface the disagreements that survive,
//! and score confidence from what actually completed. No LLM judge is involved in
//! Phase 1 — a judge agent only sees output that has already passed here.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use swarm_domain::{
    Conflict, Evidence, ExecutionStatistics, FinalResult, Job, JobStatus, StructuredOutput,
    TaskGraph, TaskId, TaskKind, TaskResult,
};

/// Similarity above which two outputs are treated as the same answer.
const SAME_ANSWER: f32 = 0.60;
/// Similarity above which one of two outputs is redundant and dropped.
const REDUNDANT: f32 = 0.85;
/// Similarity below which two answers are treated as a genuine disagreement.
const DISAGREEMENT: f32 = 0.30;
/// Cap on evidence carried into the final result.
const MAX_EVIDENCE: usize = 20;

/// How agreement between competing answers was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsensusMode {
    /// The largest cluster of similar answers wins.
    Majority,
    /// The cluster with the most total confidence wins.
    ConfidenceWeighted,
}

/// The outcome of a consensus round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsensusReport {
    /// How the winner was chosen.
    pub mode: ConsensusMode,
    /// The winning answer.
    pub winner: String,
    /// Task that produced the winning answer.
    pub winning_task: TaskId,
    /// Share of votes (or weight) behind the winner, `0.0..=1.0`.
    pub agreement_rate: f32,
    /// How many answers took part.
    pub votes: usize,
    /// How many distinct positions were found.
    pub distinct_positions: usize,
}

/// Significant words of a text, for similarity comparison.
fn tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.len() > 3)
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity of two token sets, `0.0..=1.0`.
fn similarity(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let intersection = left.intersection(right).count() as f32;
    let union = left.union(right).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Greedily group results whose outputs say the same thing.
///
/// Greedy single-pass clustering is imprecise at the margins but stable and O(n·k),
/// which matters when 500 agents each produce an answer.
fn cluster(results: &[&TaskResult], threshold: f32) -> Vec<Vec<usize>> {
    let fingerprints: Vec<HashSet<String>> = results
        .iter()
        .map(|result| tokens(&result.output))
        .collect();
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for (index, fingerprint) in fingerprints.iter().enumerate() {
        let joined = clusters.iter_mut().find(|members| {
            members
                .iter()
                .any(|member| similarity(&fingerprints[*member], fingerprint) >= threshold)
        });
        match joined {
            Some(members) => members.push(index),
            None => clusters.push(vec![index]),
        }
    }
    clusters
}

/// Decide between competing answers.
///
/// Returns `None` when there is nothing to decide (fewer than two answers).
#[must_use]
pub fn consensus(results: &[&TaskResult], mode: ConsensusMode) -> Option<ConsensusReport> {
    if results.len() < 2 {
        return None;
    }
    let clusters = cluster(results, SAME_ANSWER);

    let (winning_cluster, agreement_rate) = match mode {
        ConsensusMode::Majority => {
            let total = results.len() as f32;
            let winner = clusters.iter().max_by_key(|members| members.len())?;
            (winner, winner.len() as f32 / total)
        }
        ConsensusMode::ConfidenceWeighted => {
            let weight = |members: &Vec<usize>| -> f32 {
                members
                    .iter()
                    .map(|index| results[*index].confidence)
                    .sum::<f32>()
            };
            let total: f32 = results.iter().map(|result| result.confidence).sum();
            let winner = clusters
                .iter()
                .max_by(|left, right| weight(left).total_cmp(&weight(right)))?;
            let share = if total > 0.0 {
                weight(winner) / total
            } else {
                0.0
            };
            (winner, share)
        }
    };

    // Within the winning position, the most confident phrasing represents it.
    let best = winning_cluster
        .iter()
        .max_by(|left, right| {
            results[**left]
                .confidence
                .total_cmp(&results[**right].confidence)
        })
        .copied()?;

    Some(ConsensusReport {
        mode,
        winner: results[best].output.clone(),
        winning_task: results[best].task_id,
        agreement_rate,
        votes: results.len(),
        distinct_positions: clusters.len(),
    })
}

/// Disagreements between competing answers that consensus could not settle.
fn conflicts(results: &[&TaskResult]) -> Vec<Conflict> {
    let clusters = cluster(results, SAME_ANSWER);
    if clusters.len() < 2 {
        return Vec::new();
    }

    let representatives: Vec<usize> = clusters
        .iter()
        .filter_map(|members| members.first().copied())
        .collect();
    let fingerprints: Vec<HashSet<String>> = representatives
        .iter()
        .map(|index| tokens(&results[*index].output))
        .collect();

    let mut conflicts = Vec::new();
    for left in 0..representatives.len() {
        for right in (left + 1)..representatives.len() {
            if similarity(&fingerprints[left], &fingerprints[right]) > DISAGREEMENT {
                // Different wording, same substance: not a conflict.
                continue;
            }
            let first = results[representatives[left]];
            let second = results[representatives[right]];
            conflicts.push(Conflict {
                description: format!(
                    "`{}` and `{}` reached materially different conclusions",
                    first.title, second.title
                ),
                claims: vec![headline(&first.output), headline(&second.output)],
                task_ids: vec![first.task_id, second.task_id],
                attempted_resolution: "similarity clustering".to_owned(),
            });
        }
    }
    conflicts
}

/// First sentence or line of an output, for compact conflict reporting.
fn headline(output: &str) -> String {
    let first_line = output.lines().next().unwrap_or(output).trim();
    let sentence = first_line
        .split_once(". ")
        .map_or(first_line, |(head, _)| head);
    sentence.chars().take(240).collect()
}

/// Merge every completed task result into the job's final answer.
#[must_use]
pub fn aggregate(
    job: &Job,
    graph: &TaskGraph,
    results: &[TaskResult],
    status: JobStatus,
    statistics: ExecutionStatistics,
) -> FinalResult {
    // Dependency order, so a reader sees plan, then findings, then conclusions.
    let stage_of: HashMap<TaskId, u32> = graph.nodes().map(|node| (node.id, node.stage)).collect();
    let mut ordered: Vec<&TaskResult> = results
        .iter()
        .filter(|result| result.passed_validation() && result.kind.is_reportable())
        .collect();
    ordered.sort_by(|left, right| {
        stage_of
            .get(&left.task_id)
            .cmp(&stage_of.get(&right.task_id))
            .then_with(|| left.title.cmp(&right.title))
    });

    // Consensus is computed *before* deduplication: in a debate, repetition is the
    // signal. Dropping the second agent who said the same thing would erase exactly
    // the agreement we are trying to measure.
    let answers: Vec<&TaskResult> = ordered
        .iter()
        .copied()
        .filter(|result| matches!(result.kind, TaskKind::Answer))
        .collect();

    let kept = drop_redundant(&ordered);
    let unresolved_conflicts = conflicts(&answers);
    let agreement = job
        .request
        .execution_strategy
        .needs_consensus()
        .then(|| consensus(&answers, ConsensusMode::ConfidenceWeighted))
        .flatten();

    let outputs: Vec<StructuredOutput> = kept
        .iter()
        .map(|result| StructuredOutput {
            task_id: result.task_id,
            kind: result.kind.to_string(),
            title: result.title.clone(),
            content: result.output.clone(),
            data: result.structured.clone(),
            confidence: result.confidence,
        })
        .collect();

    let summary = summarize(&kept, agreement.as_ref(), &job.request.objective);
    let confidence_score = confidence(&kept, graph, unresolved_conflicts.len());

    FinalResult {
        job_id: job.id,
        status,
        summary,
        outputs,
        supporting_evidence: merge_evidence(&kept),
        confidence_score,
        unresolved_conflicts,
        execution_statistics: statistics,
    }
}

/// Drop outputs that merely restate an earlier one.
fn drop_redundant<'a>(ordered: &[&'a TaskResult]) -> Vec<&'a TaskResult> {
    let mut kept: Vec<&TaskResult> = Vec::with_capacity(ordered.len());
    let mut fingerprints: Vec<HashSet<String>> = Vec::with_capacity(ordered.len());

    for result in ordered {
        let fingerprint = tokens(&result.output);
        let duplicate = fingerprints
            .iter()
            .any(|existing| similarity(existing, &fingerprint) >= REDUNDANT);
        if !duplicate {
            kept.push(result);
            fingerprints.push(fingerprint);
        }
    }
    kept
}

/// Prefer a task whose whole job was to conclude; fall back to consensus, then to the
/// most confident findings.
fn summarize(kept: &[&TaskResult], agreement: Option<&ConsensusReport>, objective: &str) -> String {
    let concluding = kept.iter().rev().find(|result| {
        matches!(
            result.kind,
            TaskKind::Merge | TaskKind::Summarize | TaskKind::Reduce | TaskKind::Judge
        )
    });
    if let Some(result) = concluding {
        return result.output.clone();
    }
    if let Some(report) = agreement {
        return format!(
            "Consensus ({:.0}% agreement across {} answers): {}",
            report.agreement_rate * 100.0,
            report.votes,
            report.winner
        );
    }
    if kept.is_empty() {
        return format!("No validated output was produced for: {objective}");
    }

    let mut best: Vec<&&TaskResult> = kept.iter().collect();
    best.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut summary = format!("Findings for: {objective}\n");
    for result in best.iter().take(3) {
        summary.push_str(&format!(
            "\n- {}: {}",
            result.title,
            headline(&result.output)
        ));
    }
    summary
}

/// Confidence in the job as a whole.
///
/// Three factors, all of which should lower it: agents that were unsure, work that
/// never completed, and disagreements nobody resolved.
fn confidence(kept: &[&TaskResult], graph: &TaskGraph, conflict_count: usize) -> f32 {
    if kept.is_empty() {
        return 0.0;
    }
    let mean: f32 = kept.iter().map(|result| result.confidence).sum::<f32>() / kept.len() as f32;
    let counts = graph.counts();
    let completion = if counts.total == 0 {
        0.0
    } else {
        counts.completed as f32 / counts.total as f32
    };
    let penalty = 0.1 * conflict_count as f32;
    (mean * completion - penalty).clamp(0.0, 1.0)
}

/// Deduplicate evidence and keep the best-supported items.
fn merge_evidence(kept: &[&TaskResult]) -> Vec<Evidence> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut evidence: Vec<Evidence> = Vec::new();

    for result in kept {
        for item in &result.evidence {
            if seen.insert((item.source.clone(), item.claim.clone())) {
                evidence.push(item.clone());
            }
        }
    }
    evidence.sort_by(|left, right| right.support.total_cmp(&left.support));
    evidence.truncate(MAX_EVIDENCE);
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use swarm_domain::{AgentId, ExecutionStrategy, JobId, JobRequest, TaskNode};

    fn result(
        job_id: JobId,
        task_id: TaskId,
        kind: TaskKind,
        title: &str,
        output: &str,
        confidence: f32,
    ) -> TaskResult {
        TaskResult {
            task_id,
            job_id,
            agent_id: AgentId::new(),
            attempt: 1,
            kind,
            title: title.to_owned(),
            output: output.to_owned(),
            structured: json!({ "summary": output }),
            evidence: vec![Evidence {
                source: format!("mock://{title}"),
                claim: output.to_owned(),
                support: confidence,
                locator: None,
            }],
            reasoning_summary: "considered the sources".to_owned(),
            confidence,
            validation_failures: Vec::new(),
            tokens_in: 10,
            tokens_out: 20,
            cost_usd: 0.001,
            duration_ms: 5,
            deduplicated: false,
            finished_at: Utc::now(),
        }
    }

    /// A graph of `count` completed work tasks, so completion ratio is 1.0.
    fn completed_graph(job_id: JobId, results: &[TaskResult]) -> TaskGraph {
        let mut graph = TaskGraph::new(job_id);
        for result in results {
            let mut node = TaskNode::new(job_id, result.kind, &result.title, "do it", 1);
            node.id = result.task_id;
            node.state = swarm_domain::TaskState::Completed;
            node.idempotency_key = format!("{}-{}", result.title, result.task_id);
            graph.insert(node).unwrap();
        }
        graph
    }

    #[test]
    fn similarity_recognises_the_same_answer_in_different_words() {
        let raft = tokens("Raft elects exactly one leader per term using randomised timeouts");
        let same = tokens("Using randomised timeouts, Raft elects exactly one leader each term");
        let other = tokens("Paxos requires no leader and tolerates duelling proposers");

        assert!(similarity(&raft, &same) > SAME_ANSWER);
        assert!(similarity(&raft, &other) < DISAGREEMENT);
        assert_eq!(similarity(&raft, &raft), 1.0);
    }

    #[test]
    fn majority_consensus_picks_the_largest_agreeing_group() {
        let job_id = JobId::new();
        let answers = [
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "a",
                "Raft elects one leader per term with randomised timeouts",
                0.6,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "b",
                "Raft elects one leader each term using randomised timeouts",
                0.6,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "c",
                "Paxos tolerates duelling proposers without any leader",
                0.9,
            ),
        ];
        let refs: Vec<&TaskResult> = answers.iter().collect();

        let report = consensus(&refs, ConsensusMode::Majority).unwrap();
        assert!(report.winner.contains("Raft"));
        assert_eq!(report.votes, 3);
        assert_eq!(report.distinct_positions, 2);
        assert!((report.agreement_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn confidence_weighting_can_overturn_a_numerical_majority() {
        let job_id = JobId::new();
        let answers = [
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "a",
                "Raft elects one leader per term with randomised timeouts",
                0.2,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "b",
                "Raft elects one leader each term using randomised timeouts",
                0.2,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "c",
                "Paxos tolerates duelling proposers without any leader",
                0.99,
            ),
        ];
        let refs: Vec<&TaskResult> = answers.iter().collect();

        assert!(consensus(&refs, ConsensusMode::Majority)
            .unwrap()
            .winner
            .contains("Raft"));
        assert!(consensus(&refs, ConsensusMode::ConfidenceWeighted)
            .unwrap()
            .winner
            .contains("Paxos"));
    }

    #[test]
    fn a_single_answer_needs_no_consensus_round() {
        let job_id = JobId::new();
        let only = [result(
            job_id,
            TaskId::new(),
            TaskKind::Answer,
            "a",
            "x",
            0.5,
        )];
        let refs: Vec<&TaskResult> = only.iter().collect();
        assert!(consensus(&refs, ConsensusMode::Majority).is_none());
        assert!(consensus(&[], ConsensusMode::Majority).is_none());
    }

    #[test]
    fn genuinely_different_answers_are_reported_as_conflicts() {
        let job_id = JobId::new();
        let answers = [
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "pro",
                "Raft elects exactly one leader per term with randomised election timeouts",
                0.8,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "con",
                "Paxos permits duelling proposers and never designates any leader",
                0.8,
            ),
        ];
        let refs: Vec<&TaskResult> = answers.iter().collect();

        let found = conflicts(&refs);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].claims.len(), 2);
        assert_eq!(found[0].task_ids.len(), 2);
    }

    #[test]
    fn rephrasings_of_one_answer_are_not_a_conflict() {
        let job_id = JobId::new();
        let answers = [
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "a",
                "Raft elects exactly one leader per term using randomised timeouts",
                0.8,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "b",
                "Using randomised timeouts Raft elects exactly one leader per term",
                0.8,
            ),
        ];
        let refs: Vec<&TaskResult> = answers.iter().collect();
        assert!(conflicts(&refs).is_empty());
    }

    #[test]
    fn redundant_outputs_are_dropped_from_the_report() {
        let job_id = JobId::new();
        let results = vec![
            result(
                job_id,
                TaskId::new(),
                TaskKind::Work,
                "first",
                "Raft elects exactly one leader per term using randomised timeouts",
                0.8,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Work,
                "second",
                "Raft elects exactly one leader per term using randomised timeouts",
                0.7,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Work,
                "third",
                "Paxos needs no leader and tolerates duelling proposers entirely",
                0.9,
            ),
        ];
        let graph = completed_graph(job_id, &results);
        let job = Job::new(JobRequest::new("compare consensus algorithms"));

        let final_result = aggregate(
            &job,
            &graph,
            &results,
            JobStatus::Completed,
            ExecutionStatistics::default(),
        );
        assert_eq!(
            final_result.outputs.len(),
            2,
            "the duplicated finding must appear once"
        );
    }

    #[test]
    fn unvalidated_results_never_reach_the_final_output() {
        let job_id = JobId::new();
        let mut bad = result(
            job_id,
            TaskId::new(),
            TaskKind::Work,
            "bad",
            "too short",
            0.9,
        );
        bad.validation_failures = vec!["output has 2 words, needs 15".to_owned()];
        let good = result(
            job_id,
            TaskId::new(),
            TaskKind::Work,
            "good",
            "a properly supported finding about consensus",
            0.7,
        );
        let results = vec![bad, good];

        let graph = completed_graph(job_id, &results);
        let job = Job::new(JobRequest::new("objective"));
        let final_result = aggregate(
            &job,
            &graph,
            &results,
            JobStatus::Completed,
            ExecutionStatistics::default(),
        );

        assert_eq!(final_result.outputs.len(), 1);
        assert_eq!(final_result.outputs[0].title, "good");
    }

    #[test]
    fn the_concluding_task_becomes_the_summary() {
        let job_id = JobId::new();
        let results = vec![
            result(
                job_id,
                TaskId::new(),
                TaskKind::Work,
                "finding",
                "a finding about consensus algorithms",
                0.7,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Merge,
                "merge",
                "The merged conclusion of the swarm.",
                0.9,
            ),
        ];
        let graph = completed_graph(job_id, &results);
        let job = Job::new(JobRequest::new("objective"));

        let final_result = aggregate(
            &job,
            &graph,
            &results,
            JobStatus::Completed,
            ExecutionStatistics::default(),
        );
        assert_eq!(final_result.summary, "The merged conclusion of the swarm.");
    }

    #[test]
    fn debate_jobs_summarise_through_consensus_when_nothing_concluded() {
        let job_id = JobId::new();
        let results = vec![
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "a",
                "Raft elects exactly one leader per term using randomised timeouts",
                0.8,
            ),
            result(
                job_id,
                TaskId::new(),
                TaskKind::Answer,
                "b",
                "Using randomised timeouts Raft elects exactly one leader per term",
                0.7,
            ),
        ];
        let graph = completed_graph(job_id, &results);
        let job = Job::new(
            JobRequest::new("Does Raft elect one leader?").with_strategy(ExecutionStrategy::Debate),
        );

        let final_result = aggregate(
            &job,
            &graph,
            &results,
            JobStatus::Completed,
            ExecutionStatistics::default(),
        );
        assert!(
            final_result.summary.starts_with("Consensus ("),
            "got: {}",
            final_result.summary
        );
        assert!(final_result.summary.contains("100% agreement"));
        assert_eq!(
            final_result.outputs.len(),
            1,
            "the restatement is still dropped from the output listing"
        );
    }

    #[test]
    fn confidence_falls_with_incomplete_work_and_unresolved_conflicts() {
        let job_id = JobId::new();
        let results = vec![result(
            job_id,
            TaskId::new(),
            TaskKind::Work,
            "a",
            "a finding about consensus",
            0.8,
        )];
        let graph = completed_graph(job_id, &results);
        let job = Job::new(JobRequest::new("objective"));

        let full = aggregate(
            &job,
            &graph,
            &results,
            JobStatus::Completed,
            ExecutionStatistics::default(),
        );
        assert!((full.confidence_score - 0.8).abs() < 0.01);

        // Same results, but half the graph never finished.
        let mut partial_graph = graph.clone();
        let mut pending = TaskNode::new(job_id, TaskKind::Work, "never ran", "x", 1);
        pending.idempotency_key = "pending".to_owned();
        partial_graph.insert(pending).unwrap();

        let partial = aggregate(
            &job,
            &partial_graph,
            &results,
            JobStatus::PartiallyCompleted,
            ExecutionStatistics::default(),
        );
        assert!(
            partial.confidence_score < full.confidence_score,
            "unfinished work must lower confidence"
        );
    }

    #[test]
    fn an_empty_job_reports_zero_confidence_rather_than_pretending() {
        let job_id = JobId::new();
        let graph = TaskGraph::new(job_id);
        let job = Job::new(JobRequest::new("nothing happened"));

        let final_result = aggregate(
            &job,
            &graph,
            &[],
            JobStatus::Failed,
            ExecutionStatistics::default(),
        );
        assert_eq!(final_result.confidence_score, 0.0);
        assert!(final_result.outputs.is_empty());
        assert!(final_result.summary.contains("No validated output"));
    }

    #[test]
    fn evidence_is_deduplicated_ranked_and_capped() {
        let job_id = JobId::new();
        let mut results = Vec::new();
        for index in 0..30 {
            let mut item = result(
                job_id,
                TaskId::new(),
                TaskKind::Work,
                &format!("task {index}"),
                &format!("finding number {index} about a distinct subject entirely"),
                0.5,
            );
            item.evidence[0].support = index as f32 / 30.0;
            // Every result also cites the same shared source.
            item.evidence.push(Evidence {
                source: "mock://shared".to_owned(),
                claim: "shared claim".to_owned(),
                support: 0.99,
                locator: None,
            });
            results.push(item);
        }
        let graph = completed_graph(job_id, &results);
        let job = Job::new(JobRequest::new("objective"));

        let final_result = aggregate(
            &job,
            &graph,
            &results,
            JobStatus::Completed,
            ExecutionStatistics::default(),
        );
        assert!(final_result.supporting_evidence.len() <= MAX_EVIDENCE);
        assert_eq!(
            final_result
                .supporting_evidence
                .iter()
                .filter(|item| item.source == "mock://shared")
                .count(),
            1,
            "the shared citation must appear once"
        );
        assert!(
            final_result.supporting_evidence[0].support
                >= final_result.supporting_evidence[1].support
        );
    }
}
