# Swarm Platform

A distributed multi-agent execution platform in Rust. It takes a high-level objective,
compiles it into a task DAG, schedules those tasks across a pool of agents, recovers
from failures, and merges the outputs into one verified result.

The design goal is a **distributed system that happens to run LLM agents**, not an LLM
app that happens to be distributed: scheduling, leasing, retries, consensus, and
recovery are explicit, deterministic Rust. Model calls sit behind a single trait at the
leaf, and the default provider is a deterministic mock — so the whole platform runs,
and every test passes, with no network and no API key.

## Status

**Phase 1 (local prototype) is complete**: 186 tests, `cargo fmt --check` clean,
`cargo clippy --all-targets -- -D warnings` clean.

| Phase | Scope | State |
|---|---|---|
| 1 | Domain, DAG, queue, shared memory, gateway, agents, scheduler, engine, CLI | ✅ done |
| 2 | Postgres, Redis Streams, gRPC, remote workers, HTTP API | 🔜 next |
| 3 | Coordinator replicas, leader election, fencing, failover | ⬜ |
| 4 | Dynamic spawning, full consensus modes, semantic memory, verification | ⬜ |
| 5 | AuthN/Z, TLS, sandboxing, OpenTelemetry, dashboard, chaos suite | ⬜ |
| 6 | 1→500 agent benchmark sweeps and scaling analysis | ⬜ |

See [docs/roadmap.md](docs/roadmap.md) for each phase's exit criteria.

## Quick start

Requires stable Rust (1.82+). Nothing else — no database, no broker, no API key.

```bash
cargo test --workspace
```

Run one of the three demo workflows:

```bash
cargo run -p swarm-admin-cli -- demo research
```

`demo all` runs all three: a parallel research swarm, a distributed code-generation
swarm, and a debate-and-consensus swarm.

Run your own objective:

```bash
cargo run -p swarm-admin-cli -- run --objective "Compare Raft and Paxos" --strategy debate --agents 8
```

Useful flags: `--strategy` (nine DAG shapes), `--agents` (budget *and* target
parallelism), `--placement` (four schedulers), `--json`, `--max-cost`, `--config`.
`swarmctl strategies` lists everything this build supports.

## What Phase 1 actually does

- **Nine execution strategies** — sequential, parallel, hierarchical, debate,
  map-reduce, planner-executor, supervisor-worker, consensus, adaptive — each
  compiled to a different DAG shape by a deterministic stage compiler.
- **A real task queue**: priorities, delays, leases with visibility timeouts, expiry
  recovery, deduplication, retry backoff, dead letters, and replay.
- **Effectively-once execution** on top of at-least-once delivery, via idempotency
  keys, pre-execution records, and compare-and-swap memory writes.
- **Versioned shared memory** where two agents racing on one key produce a winner and
  a `VersionConflict`, never a silent overwrite.
- **A provider-independent model gateway**: ordered fallback, per-provider circuit
  breakers, global and per-model concurrency limits, response caching, retries, and
  token/cost accounting on every call.
- **Four schedulers**, including an adaptive one that re-weights placement from
  observed success rates and queue waits.
- **Explicit state machines** for jobs, tasks, leases, agents, and nodes. Illegal
  transitions are refused with a typed error, and every accepted one is journaled with
  actor, reason, and correlation id.
- **Failure handling**: timeouts, retries with jittered backoff, dead-lettering,
  cancellation mid-flight, pause/resume, deadline enforcement, and stranded-subtree
  detection so a dead-lettered task cannot hang the job.

Proven end-to-end in [`crates/coordinator/tests/end_to_end.rs`](crates/coordinator/tests/end_to_end.rs),
including 100 agents cooperating on one job and a duplicate-delivery test that asserts
no task is executed twice.

## Layout

```text
crates/
  domain/          types, state machines, task DAG — no async, no I/O
  task-queue/      TaskQueue trait + full-semantics in-memory backend
  shared-memory/   MemoryStore trait + versioned/CAS in-memory backend
  model-gateway/   ModelProvider trait, deterministic mock, Gateway
  agent-runtime/   executes one task with one agent
  coordinator/     decompose · schedule · execute · aggregate
  admin-cli/       swarmctl
proto/swarm/v1/    protobuf contracts (wired up in Phase 2)
migrations/        Postgres schema (applied in Phase 2)
docs/              architecture, decisions, state machines, roadmap
config/            example configuration
```

Domain logic never imports an infrastructure crate; infrastructure implements traits
defined in the domain layer. That is what lets Phase 2 swap the in-memory queue for
Redis Streams without the engine noticing.

## Documentation

- [docs/architecture.md](docs/architecture.md) — services, data flow, failure paths, sequence diagrams
- [docs/decisions.md](docs/decisions.md) — the fifteen decisions that shaped the rest, and what each cost
- [docs/state-machines.md](docs/state-machines.md) — every lifecycle, as diagrams and transition rules
- [docs/roadmap.md](docs/roadmap.md) — phase-by-phase plan with exit criteria

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Every phase must pass all three before the next one starts.

Two notes for contributors:

- No lock is held across an `await`. Read state into locals, await, then re-lock.
- Reasoning traces are never persisted. Agents store a short `reasoning_summary`,
  structured evidence, and validation results — nothing else.
