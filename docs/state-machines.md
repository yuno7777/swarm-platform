# State machines

Five entities have explicit state machines: **job**, **task**, **lease**, **agent**,
**node**. All are implemented in `crates/domain/src/state.rs` behind one trait:

```rust
pub trait StateMachine: Copy + PartialEq {
    fn allowed(self) -> &'static [Self];
    fn is_terminal(self) -> bool;
    fn transition(self, to: Self) -> Result<Self>;   // errors on illegal edge
}
```

Illegal transitions return `SwarmError::InvalidTransition { entity, from, to }` — they
are never silently ignored, and every legal transition is journaled as:

| column | meaning |
|--------|---------|
| `from_state`, `to_state` | the edge taken |
| `actor` | `coordinator:<id>`, `worker:<id>`, `agent:<id>`, `user:<sub>`, `system` |
| `reason` | short machine-readable cause (`lease_expired`, `validation_failed`, …) |
| `at` | UTC timestamp |
| `correlation_id` | ties the transition to the request/message that caused it |

History is append-only (`job_state_transitions`, `task_state_transitions`,
`task_attempts`), so "what happened to task X" is a query, not an investigation.

## Task

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Queued: dependencies satisfied
    Created --> WaitingForDependency: upstream incomplete
    Created --> Cancelled
    WaitingForDependency --> Queued: upstream completed
    WaitingForDependency --> Cancelled
    Queued --> Leased: worker dequeued
    Queued --> Cancelled
    Queued --> Preempted: higher-priority admission
    Leased --> Running: agent started
    Leased --> Queued: lease expired before start
    Leased --> Cancelled
    Running --> Completed: output passed validation
    Running --> Failed: agent error
    Running --> TimedOut: timeout_seconds exceeded
    Running --> WaitingForDependency: dynamic dependency inserted
    Running --> Cancelled
    Failed --> RetryScheduled: attempt < max_attempts
    Failed --> DeadLettered: attempts exhausted
    TimedOut --> RetryScheduled
    TimedOut --> DeadLettered
    Preempted --> Queued
    RetryScheduled --> Queued: backoff elapsed
    Completed --> [*]
    Cancelled --> [*]
    DeadLettered --> [*]
```

Terminal: `Completed`, `Cancelled`, `DeadLettered`.

Notes:
- `Leased → Queued` is the recovery edge; it is driven by lease expiry, never by a
  worker-liveness guess.
- Retry backoff is `min(base * 2^attempt, max)` plus jitter, stored as the queue
  entry's `available_at`.
- `Running → WaitingForDependency` supports an agent discovering it needs new work
  (Adaptive strategy); the new task is inserted and the DAG re-validated as acyclic.

## Job

```mermaid
stateDiagram-v2
    [*] --> Submitted
    Submitted --> Admitted: quota + budget check passed
    Submitted --> Rejected: admission denied
    Admitted --> Planning: decomposition started
    Planning --> Running: DAG validated, first tasks queued
    Planning --> Failed: decomposition invalid
    Running --> Paused: operator / budget guard
    Paused --> Running: resume
    Paused --> Cancelled
    Running --> Aggregating: all tasks terminal
    Running --> Cancelled
    Running --> Failed: unrecoverable task or deadline exceeded
    Aggregating --> Completed: final result verified
    Aggregating --> PartiallyCompleted: some branches dead-lettered
    Aggregating --> Failed: aggregation/validation failed
    Failed --> Running: retry_failed
    PartiallyCompleted --> Running: retry_failed
    Completed --> [*]
    Cancelled --> [*]
    Rejected --> [*]
```

Terminal: `Completed`, `Cancelled`, `Rejected`. `Failed` and `PartiallyCompleted` are
*resting* states — a retry moves them back to `Running` with only the failed subgraph
re-queued.

## Lease

```mermaid
stateDiagram-v2
    [*] --> Held: dequeue
    Held --> Extended: extend_lease while progressing
    Extended --> Extended: extend again
    Held --> Released: acknowledge (success)
    Extended --> Released
    Held --> Rejected: reject(reason) → retry or DLQ
    Extended --> Rejected
    Held --> Expired: visibility timeout
    Extended --> Expired
    Released --> [*]
    Expired --> [*]
```

A worker holding an `Expired` lease has its `acknowledge` refused, so a slow worker
cannot ack a task another worker already redid.

## Agent

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Registered: announced to coordinator
    Registered --> Idle
    Idle --> Assigned: scheduler picked it
    Assigned --> Running
    Running --> Waiting: awaiting peer / consensus / clarification
    Waiting --> Running
    Running --> Completed
    Running --> Failed
    Failed --> Retrying: task retry on same agent
    Retrying --> Running
    Completed --> Idle: warm reuse
    Failed --> Terminated
    Idle --> Terminated: idle timeout / drain / quota
    Waiting --> Terminated: job cancelled
    Terminated --> [*]
```

`Completed → Idle` is the warm-pool edge: agents are reused rather than rebuilt,
which is what keeps 500-agent runs cheap.

## Node

```mermaid
stateDiagram-v2
    [*] --> Joining
    Joining --> Ready: registered + resources reported
    Ready --> Degraded: heartbeat late / high error rate
    Degraded --> Ready: recovered
    Ready --> Draining: operator drain
    Degraded --> Draining
    Draining --> Removed: no tasks remaining
    Ready --> Unreachable: heartbeats missed threshold
    Degraded --> Unreachable
    Unreachable --> Ready: rejoined with same id
    Unreachable --> Removed: eviction timeout
    Removed --> [*]
```

`Unreachable` triggers rescheduling of that node's unfinished tasks, but the node may
still rejoin — so its identity is kept until the eviction timeout, and any late result
it reports is accepted only if the task has not already been completed by someone else
(checked by idempotency key).
