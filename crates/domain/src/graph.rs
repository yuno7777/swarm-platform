//! The task dependency graph.
//!
//! A job's work is a DAG. This type owns both the topology (petgraph) and the task
//! records, keeps them consistent, and refuses any mutation that would introduce a
//! cycle — including dynamic edges added while the job is already running.

use std::collections::{HashMap, HashSet};

use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SwarmError};
use crate::ids::{CorrelationId, JobId, TaskId};
use crate::state::{apply, Transition};
use crate::task::{TaskNode, TaskState};

/// Task counts by lifecycle group, for progress reporting and scheduling decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCounts {
    /// Tasks in the graph.
    pub total: usize,
    /// Not yet eligible or waiting on upstream work.
    pub pending: usize,
    /// In the queue.
    pub queued: usize,
    /// Leased or running.
    pub in_flight: usize,
    /// Completed successfully.
    pub completed: usize,
    /// Awaiting a retry.
    pub retrying: usize,
    /// Dead-lettered or cancelled.
    pub abandoned: usize,
}

impl GraphCounts {
    /// Tasks that will never change state again.
    #[must_use]
    pub const fn terminal(&self) -> usize {
        self.completed + self.abandoned
    }
}

/// A job's tasks and the dependencies between them.
///
/// Edges point from dependency to dependent, so a topological order yields
/// dependencies first.
#[derive(Debug, Clone)]
pub struct TaskGraph {
    job_id: JobId,
    graph: DiGraph<TaskId, ()>,
    index: HashMap<TaskId, NodeIndex>,
    nodes: HashMap<TaskId, TaskNode>,
}

impl TaskGraph {
    /// An empty graph for `job_id`.
    #[must_use]
    pub fn new(job_id: JobId) -> Self {
        Self {
            job_id,
            graph: DiGraph::new(),
            index: HashMap::new(),
            nodes: HashMap::new(),
        }
    }

    /// The job this graph belongs to.
    #[must_use]
    pub const fn job_id(&self) -> JobId {
        self.job_id
    }

    /// Number of tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no tasks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up a task.
    #[must_use]
    pub fn get(&self, id: TaskId) -> Option<&TaskNode> {
        self.nodes.get(&id)
    }

    /// Look up a task for mutation of non-lifecycle fields.
    ///
    /// Use [`Self::set_state`] to change `state`; it is the only path that produces an
    /// audit record and validates the transition.
    pub fn get_mut(&mut self, id: TaskId) -> Option<&mut TaskNode> {
        self.nodes.get_mut(&id)
    }

    /// Look up a task, or fail with a typed error.
    pub fn require(&self, id: TaskId) -> Result<&TaskNode> {
        self.nodes.get(&id).ok_or_else(|| SwarmError::NotFound {
            kind: "task",
            id: id.to_string(),
        })
    }

    /// All tasks, in arbitrary order.
    pub fn nodes(&self) -> impl Iterator<Item = &TaskNode> {
        self.nodes.values()
    }

    /// Add a task. Every dependency must already be present.
    ///
    /// Requiring dependencies to pre-exist makes a cycle impossible here, which is why
    /// this is the cheap path; [`Self::add_dependency`] is the one that must check.
    pub fn insert(&mut self, node: TaskNode) -> Result<TaskId> {
        if self.nodes.contains_key(&node.id) {
            return Err(SwarmError::Internal(format!(
                "task {} is already in the graph",
                node.id
            )));
        }
        for dependency in &node.dependencies {
            if !self.nodes.contains_key(dependency) {
                return Err(SwarmError::UnknownDependency {
                    task: node.id.to_string(),
                    dependency: dependency.to_string(),
                });
            }
        }

        let id = node.id;
        let node_index = self.graph.add_node(id);
        self.index.insert(id, node_index);
        for dependency in &node.dependencies {
            let dependency_index = self.index[dependency];
            self.graph.add_edge(dependency_index, node_index, ());
        }
        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Make `task` depend on `depends_on`, refusing the edge if it would create a cycle.
    ///
    /// This is the edge an agent adds when it discovers work the plan did not
    /// anticipate (Adaptive strategy), so it is also the one place a running job could
    /// deadlock itself. The edge is rolled back before the error is returned.
    pub fn add_dependency(&mut self, task: TaskId, depends_on: TaskId) -> Result<()> {
        let task_index = self.index_of(task)?;
        let dependency_index = self.index_of(depends_on)?;
        if task == depends_on {
            return Err(SwarmError::CyclicGraph {
                task: task.to_string(),
            });
        }
        if self
            .nodes
            .get(&task)
            .is_some_and(|node| node.dependencies.contains(&depends_on))
        {
            return Ok(());
        }

        let edge = self.graph.add_edge(dependency_index, task_index, ());
        if let Err(err) = self.assert_acyclic() {
            self.graph.remove_edge(edge);
            return Err(err);
        }
        if let Some(node) = self.nodes.get_mut(&task) {
            node.dependencies.push(depends_on);
        }
        Ok(())
    }

    /// Insert `node` and make each task in `dependents` wait for it.
    ///
    /// The whole insertion is atomic: if any edge would cycle, the node is removed
    /// from the task map and the graph is left as it was.
    pub fn insert_dynamic(&mut self, node: TaskNode, dependents: &[TaskId]) -> Result<TaskId> {
        let id = self.insert(node)?;
        for &dependent in dependents {
            if let Err(err) = self.add_dependency(dependent, id) {
                // Roll back: drop the edges we added, then forget the node. The
                // petgraph node itself is left orphaned but unreachable, which costs
                // one index and keeps every other NodeIndex valid.
                for &earlier in dependents {
                    if earlier == dependent {
                        break;
                    }
                    self.remove_dependency(earlier, id);
                }
                self.nodes.remove(&id);
                self.index.remove(&id);
                return Err(err);
            }
        }
        Ok(id)
    }

    /// Replace `task`'s work with `children`, turning `task` into their join point.
    ///
    /// Used when a task turns out to be too large: the children inherit the original's
    /// dependencies, and the original now depends on the children. This preserves both
    /// acyclicity and every downstream edge, so the rest of the DAG is untouched.
    pub fn split(&mut self, task: TaskId, children: Vec<TaskNode>) -> Result<Vec<TaskId>> {
        if children.is_empty() {
            return Err(SwarmError::Validation(
                "splitting a task requires at least one child".into(),
            ));
        }
        let task_index = self.index_of(task)?;
        let inherited: Vec<TaskId> = self.dependencies(task);

        // Detach the original from its dependencies; the children take them over.
        let incoming: Vec<EdgeIndex> = self
            .graph
            .edges_directed(task_index, Direction::Incoming)
            .map(|edge| edge.id())
            .collect();
        for edge in incoming {
            self.graph.remove_edge(edge);
        }
        if let Some(node) = self.nodes.get_mut(&task) {
            node.dependencies.clear();
        }

        let mut child_ids = Vec::with_capacity(children.len());
        for mut child in children {
            child.dependencies = inherited.clone();
            let child_id = self.insert(child)?;
            self.add_dependency(task, child_id)?;
            child_ids.push(child_id);
        }
        Ok(child_ids)
    }

    /// Direct dependencies of `task`.
    #[must_use]
    pub fn dependencies(&self, task: TaskId) -> Vec<TaskId> {
        self.index.get(&task).map_or_else(Vec::new, |&index| {
            self.graph
                .neighbors_directed(index, Direction::Incoming)
                .map(|neighbour| self.graph[neighbour])
                .collect()
        })
    }

    /// Tasks that depend directly on `task`.
    #[must_use]
    pub fn dependents(&self, task: TaskId) -> Vec<TaskId> {
        self.index.get(&task).map_or_else(Vec::new, |&index| {
            self.graph
                .neighbors_directed(index, Direction::Outgoing)
                .map(|neighbour| self.graph[neighbour])
                .collect()
        })
    }

    /// Fail if the graph contains a cycle.
    ///
    /// Called before a job is scheduled and after every dynamic edge.
    pub fn assert_acyclic(&self) -> Result<()> {
        petgraph::algo::toposort(&self.graph, None)
            .map(|_| ())
            .map_err(|cycle| SwarmError::CyclicGraph {
                task: self.graph[cycle.node_id()].to_string(),
            })
    }

    /// Tasks in dependency-first order.
    pub fn topological_order(&self) -> Result<Vec<TaskId>> {
        petgraph::algo::toposort(&self.graph, None)
            .map(|order| order.into_iter().map(|index| self.graph[index]).collect())
            .map_err(|cycle| SwarmError::CyclicGraph {
                task: self.graph[cycle.node_id()].to_string(),
            })
    }

    /// Tasks grouped into waves that can each run fully in parallel.
    ///
    /// Wave `n` contains every task whose longest dependency chain has length `n`.
    /// The number of waves is the critical path, and the widest wave is the most
    /// parallelism the job can ever use — both inputs to agent-count admission.
    pub fn layers(&self) -> Result<Vec<Vec<TaskId>>> {
        let order = self.topological_order()?;
        let mut depth: HashMap<TaskId, usize> = HashMap::with_capacity(order.len());
        let mut layers: Vec<Vec<TaskId>> = Vec::new();

        for task in order {
            let task_depth = self
                .dependencies(task)
                .iter()
                .filter_map(|dependency| depth.get(dependency))
                .max()
                .map_or(0, |deepest| deepest + 1);
            depth.insert(task, task_depth);
            if layers.len() <= task_depth {
                layers.resize(task_depth + 1, Vec::new());
            }
            layers[task_depth].push(task);
        }
        Ok(layers)
    }

    /// Tasks whose dependencies are all complete and which are waiting to be queued.
    ///
    /// Returned in topological order so the queue receives upstream work first, which
    /// keeps priority inversions from forming inside a single wave.
    pub fn ready(&self) -> Vec<TaskId> {
        let completed: HashSet<TaskId> = self
            .nodes
            .values()
            .filter(|node| node.state == TaskState::Completed)
            .map(|node| node.id)
            .collect();

        let mut ready: Vec<TaskId> = self
            .nodes
            .values()
            .filter(|node| {
                matches!(
                    node.state,
                    TaskState::Created | TaskState::WaitingForDependency
                ) && node.dependencies_satisfied(&completed)
            })
            .map(|node| node.id)
            .collect();

        // Stable order: by stage, then by id, so runs are reproducible.
        ready.sort_unstable_by_key(|id| {
            let node = &self.nodes[id];
            (node.stage, *id)
        });
        ready
    }

    /// Tasks that can never run because an upstream task was abandoned.
    ///
    /// Without this, a dead-lettered task leaves its whole subtree stuck in
    /// `WaitingForDependency` and the job never reaches a terminal state.
    #[must_use]
    pub fn blocked_by_abandoned(&self) -> Vec<TaskId> {
        let abandoned: HashSet<TaskId> = self
            .nodes
            .values()
            .filter(|node| matches!(node.state, TaskState::DeadLettered | TaskState::Cancelled))
            .map(|node| node.id)
            .collect();
        if abandoned.is_empty() {
            return Vec::new();
        }

        // Walk forward from each abandoned task; everything reachable is unreachable.
        let mut blocked = Vec::new();
        let mut frontier: Vec<TaskId> = abandoned.iter().copied().collect();
        let mut seen: HashSet<TaskId> = abandoned.clone();
        while let Some(task) = frontier.pop() {
            for dependent in self.dependents(task) {
                if !seen.insert(dependent) {
                    continue;
                }
                if self
                    .nodes
                    .get(&dependent)
                    .is_some_and(|node| !node.is_terminal())
                {
                    blocked.push(dependent);
                    frontier.push(dependent);
                }
            }
        }
        blocked
    }

    /// Move `task` to `to`, validating the edge and returning the audit record.
    ///
    /// The only sanctioned way to change a task's state.
    pub fn set_state(
        &mut self,
        task: TaskId,
        to: TaskState,
        actor: impl Into<String>,
        reason: Option<String>,
        correlation_id: CorrelationId,
    ) -> Result<Transition<TaskState>> {
        let node = self
            .nodes
            .get_mut(&task)
            .ok_or_else(|| SwarmError::NotFound {
                kind: "task",
                id: task.to_string(),
            })?;
        let (next, transition) = apply(node.state, to, actor, reason, correlation_id)?;
        node.state = next;
        Ok(transition)
    }

    /// Task counts by lifecycle group.
    #[must_use]
    pub fn counts(&self) -> GraphCounts {
        let mut counts = GraphCounts {
            total: self.nodes.len(),
            ..GraphCounts::default()
        };
        for node in self.nodes.values() {
            match node.state {
                TaskState::Created | TaskState::WaitingForDependency | TaskState::Preempted => {
                    counts.pending += 1;
                }
                TaskState::Queued => counts.queued += 1,
                TaskState::Leased | TaskState::Running => counts.in_flight += 1,
                TaskState::Completed => counts.completed += 1,
                TaskState::Failed | TaskState::TimedOut | TaskState::RetryScheduled => {
                    counts.retrying += 1;
                }
                TaskState::Cancelled | TaskState::DeadLettered => counts.abandoned += 1,
            }
        }
        counts
    }

    /// Whether every task has reached a terminal state.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.nodes.is_empty() && self.nodes.values().all(TaskNode::is_terminal)
    }

    /// Fraction of tasks that are terminal, `0.0..=1.0`.
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.counts().terminal() as f32 / self.nodes.len() as f32
    }

    fn index_of(&self, task: TaskId) -> Result<NodeIndex> {
        self.index
            .get(&task)
            .copied()
            .ok_or_else(|| SwarmError::NotFound {
                kind: "task",
                id: task.to_string(),
            })
    }

    fn remove_dependency(&mut self, task: TaskId, depends_on: TaskId) {
        let (Ok(task_index), Ok(dependency_index)) =
            (self.index_of(task), self.index_of(depends_on))
        else {
            return;
        };
        if let Some(edge) = self.graph.find_edge(dependency_index, task_index) {
            self.graph.remove_edge(edge);
        }
        if let Some(node) = self.nodes.get_mut(&task) {
            node.dependencies
                .retain(|dependency| *dependency != depends_on);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskKind;

    fn node(job_id: JobId, title: &str, stage: u32) -> TaskNode {
        TaskNode::new(job_id, TaskKind::Work, title, "do it", stage)
    }

    /// plan -> {a, b} -> merge
    fn diamond() -> (TaskGraph, Vec<TaskId>) {
        let job_id = JobId::new();
        let mut graph = TaskGraph::new(job_id);
        let plan = graph
            .insert(TaskNode::new(job_id, TaskKind::Plan, "plan", "plan it", 0))
            .unwrap();
        let a = graph
            .insert(node(job_id, "a", 1).with_dependencies(vec![plan]))
            .unwrap();
        let b = graph
            .insert(node(job_id, "b", 1).with_dependencies(vec![plan]))
            .unwrap();
        let merge = graph
            .insert(
                TaskNode::new(job_id, TaskKind::Merge, "merge", "merge it", 2)
                    .with_dependencies(vec![a, b]),
            )
            .unwrap();
        (graph, vec![plan, a, b, merge])
    }

    fn complete(graph: &mut TaskGraph, task: TaskId) {
        for state in [
            TaskState::Queued,
            TaskState::Leased,
            TaskState::Running,
            TaskState::Completed,
        ] {
            graph
                .set_state(task, state, "test", None, CorrelationId::new())
                .unwrap();
        }
    }

    #[test]
    fn a_dag_is_acyclic_and_topologically_ordered() {
        let (graph, ids) = diamond();
        graph.assert_acyclic().unwrap();

        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 4);
        let position = |id: TaskId| order.iter().position(|candidate| *candidate == id).unwrap();
        assert!(position(ids[0]) < position(ids[1]));
        assert!(position(ids[0]) < position(ids[2]));
        assert!(position(ids[1]) < position(ids[3]));
        assert!(position(ids[2]) < position(ids[3]));
    }

    #[test]
    fn layers_expose_the_critical_path_and_the_widest_wave() {
        let (graph, _) = diamond();
        let layers = graph.layers().unwrap();
        assert_eq!(layers.len(), 3, "critical path is plan -> work -> merge");
        assert_eq!(layers[0].len(), 1);
        assert_eq!(layers[1].len(), 2, "the two work tasks run together");
        assert_eq!(layers[2].len(), 1);
    }

    #[test]
    fn unknown_dependencies_are_refused() {
        let job_id = JobId::new();
        let mut graph = TaskGraph::new(job_id);
        let orphan = TaskId::new();
        let err = graph
            .insert(node(job_id, "a", 0).with_dependencies(vec![orphan]))
            .unwrap_err();
        assert!(matches!(err, SwarmError::UnknownDependency { .. }));
        assert!(graph.is_empty());
    }

    #[test]
    fn a_cycle_is_refused_and_the_graph_is_left_usable() {
        let (mut graph, ids) = diamond();
        // merge already depends on a, so a depending on merge would close a loop.
        let err = graph.add_dependency(ids[1], ids[3]).unwrap_err();
        assert!(matches!(err, SwarmError::CyclicGraph { .. }));

        graph.assert_acyclic().unwrap();
        assert_eq!(graph.dependencies(ids[1]), vec![ids[0]]);
        assert_eq!(graph.len(), 4);
    }

    #[test]
    fn self_dependency_is_refused() {
        let (mut graph, ids) = diamond();
        assert!(graph.add_dependency(ids[0], ids[0]).is_err());
    }

    #[test]
    fn readiness_unlocks_as_dependencies_complete() {
        let (mut graph, ids) = diamond();
        let [plan, a, b, merge] = [ids[0], ids[1], ids[2], ids[3]];

        assert_eq!(graph.ready(), vec![plan]);
        complete(&mut graph, plan);

        let mut ready = graph.ready();
        ready.sort_unstable();
        let mut expected = vec![a, b];
        expected.sort_unstable();
        assert_eq!(ready, expected);

        complete(&mut graph, a);
        assert_eq!(graph.ready(), vec![b], "merge still waits on b");
        complete(&mut graph, b);
        assert_eq!(graph.ready(), vec![merge]);

        complete(&mut graph, merge);
        assert!(graph.ready().is_empty());
        assert!(graph.is_complete());
        assert_eq!(graph.progress(), 1.0);
    }

    #[test]
    fn dynamic_insertion_makes_a_running_task_wait_for_new_work() {
        let (mut graph, ids) = diamond();
        let job_id = graph.job_id();
        let discovered = graph
            .insert_dynamic(node(job_id, "discovered", 1), &[ids[3]])
            .unwrap();

        graph.assert_acyclic().unwrap();
        assert!(graph.dependencies(ids[3]).contains(&discovered));
        assert_eq!(graph.len(), 5);
        assert_eq!(graph.layers().unwrap().len(), 3);
    }

    #[test]
    fn dynamic_insertion_that_would_cycle_leaves_nothing_behind() {
        let (mut graph, ids) = diamond();
        let job_id = graph.job_id();
        // The new task depends on merge, and merge is asked to depend on it.
        let cyclic = node(job_id, "cyclic", 3).with_dependencies(vec![ids[3]]);
        let err = graph.insert_dynamic(cyclic, &[ids[3]]).unwrap_err();

        assert!(matches!(err, SwarmError::CyclicGraph { .. }));
        assert_eq!(graph.len(), 4, "the rejected task must not linger");
        graph.assert_acyclic().unwrap();
    }

    #[test]
    fn splitting_a_task_preserves_its_place_in_the_dag() {
        let (mut graph, ids) = diamond();
        let [plan, a, _b, merge] = [ids[0], ids[1], ids[2], ids[3]];
        let job_id = graph.job_id();

        let children = graph
            .split(a, vec![node(job_id, "a1", 1), node(job_id, "a2", 1)])
            .unwrap();

        assert_eq!(children.len(), 2);
        graph.assert_acyclic().unwrap();
        for child in &children {
            assert_eq!(
                graph.dependencies(*child),
                vec![plan],
                "children inherit the original's dependencies"
            );
        }
        let mut a_deps = graph.dependencies(a);
        a_deps.sort_unstable();
        let mut expected = children.clone();
        expected.sort_unstable();
        assert_eq!(a_deps, expected, "the original becomes their join point");
        assert!(
            graph.dependents(a).contains(&merge),
            "downstream edges survive the split"
        );
    }

    #[test]
    fn splitting_into_nothing_is_refused() {
        let (mut graph, ids) = diamond();
        assert!(graph.split(ids[1], Vec::new()).is_err());
    }

    #[test]
    fn dead_lettered_tasks_expose_their_stranded_subtree() {
        let (mut graph, ids) = diamond();
        let [plan, a, b, merge] = [ids[0], ids[1], ids[2], ids[3]];
        complete(&mut graph, plan);
        complete(&mut graph, b);

        for state in [
            TaskState::Queued,
            TaskState::Leased,
            TaskState::Running,
            TaskState::Failed,
            TaskState::DeadLettered,
        ] {
            graph
                .set_state(a, state, "test", None, CorrelationId::new())
                .unwrap();
        }

        assert_eq!(
            graph.blocked_by_abandoned(),
            vec![merge],
            "merge can never run, so the job must not wait for it"
        );
        assert!(graph.ready().is_empty());
    }

    #[test]
    fn state_changes_are_validated_and_audited() {
        let (mut graph, ids) = diamond();
        let transition = graph
            .set_state(
                ids[0],
                TaskState::Queued,
                "coordinator:test",
                Some("dependencies_satisfied".to_owned()),
                CorrelationId::new(),
            )
            .unwrap();
        assert_eq!(transition.from, Some(TaskState::Created));
        assert_eq!(transition.to, TaskState::Queued);
        assert_eq!(graph.get(ids[0]).unwrap().state, TaskState::Queued);

        // Queued -> Completed is not an edge that exists.
        assert!(graph
            .set_state(
                ids[0],
                TaskState::Completed,
                "test",
                None,
                CorrelationId::new()
            )
            .is_err());
        assert_eq!(graph.get(ids[0]).unwrap().state, TaskState::Queued);
    }

    #[test]
    fn counts_add_up_to_the_total() {
        let (mut graph, ids) = diamond();
        complete(&mut graph, ids[0]);
        graph
            .set_state(
                ids[1],
                TaskState::Queued,
                "test",
                None,
                CorrelationId::new(),
            )
            .unwrap();

        let counts = graph.counts();
        assert_eq!(counts.total, 4);
        assert_eq!(counts.completed, 1);
        assert_eq!(counts.queued, 1);
        assert_eq!(
            counts.pending
                + counts.queued
                + counts.in_flight
                + counts.completed
                + counts.retrying
                + counts.abandoned,
            counts.total
        );
        assert!((graph.progress() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn a_thousand_tasks_stay_fast_and_acyclic() {
        let job_id = JobId::new();
        let mut graph = TaskGraph::new(job_id);
        let root = graph
            .insert(TaskNode::new(job_id, TaskKind::Plan, "plan", "plan", 0))
            .unwrap();
        let mut previous_layer = vec![root];

        for stage in 1..=10u32 {
            let mut layer = Vec::new();
            for index in 0..100 {
                let task = graph
                    .insert(
                        node(job_id, &format!("s{stage}-t{index}"), stage)
                            .with_dependencies(previous_layer.clone()),
                    )
                    .unwrap();
                layer.push(task);
            }
            previous_layer = layer;
        }

        assert_eq!(graph.len(), 1_001);
        graph.assert_acyclic().unwrap();
        assert_eq!(graph.layers().unwrap().len(), 11);
        assert_eq!(graph.ready(), vec![root]);
    }
}
