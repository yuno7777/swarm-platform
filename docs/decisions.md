# Major engineering decisions

Short ADRs. Each records the decision, the reason, and the cost we accepted.

## ADR-1: Domain crate has no async and no I/O

`domain` holds ids, job/task/agent types, state machines, and the task graph. It
depends only on serde/chrono/uuid/petgraph/thiserror.

**Why:** the parts most likely to be wrong (transition legality, cycle detection,
ready-set computation) become pure functions with sub-millisecond tests. It also
means the state machine cannot secretly do a database write.

**Cost:** persistence lives one layer out; the coordinator must explicitly journal
transitions rather than getting it for free.

## ADR-2: At-least-once delivery, effectively-once execution

Exactly-once delivery is not available across a queue, a worker crash, and a model
API. We chose at-least-once + idempotency keys + a pre-execution record check + CAS
memory writes (see architecture §6).

**Why:** it is the only honest option, and it degrades safely: worst case a duplicate
delivery costs one wasted lookup, not a duplicated side effect.

**Cost:** every side-effecting path must be written with an execution record. That is
a discipline, enforced by tests (`duplicate_delivery_does_not_double_execute`).

## ADR-3: Leases with visibility timeouts, not worker-liveness tracking, as the unit of recovery

A task is owned by a lease with an expiry, not by a worker identity. Recovery is
`requeue_expired()`, a single sweep. Heartbeats are an *optimization* that shortens
detection latency; correctness does not depend on them.

**Why:** heartbeat-based ownership needs a consistent view of liveness, which is
exactly what a network partition denies you. Lease expiry needs only a clock.

**Cost:** a slow-but-alive worker can have its task stolen. Mitigated by lease
extension while making progress, and by idempotency when both attempts finish.

## ADR-4: Fencing tokens are mandatory, not advisory

Every leader write carries the term number. Postgres enforces
`term >= (SELECT max(term) FROM leadership_terms)` on scheduling writes.

**Why:** leader election alone does not prevent split brain — a partitioned old
leader still believes it is leader until it notices. Fencing makes its writes
harmless rather than racing.

**Cost:** one extra column and one predicate on the hot write path.

## ADR-5: Custom lease-based election first, OpenRaft behind the same trait

Phase 3 ships `LeaderElector` as a trait with a Postgres/Redis lease implementation
(≈200 lines, easy to chaos-test). OpenRaft plugs into the same trait when replicated
log state — not just leadership — is needed.

**Why:** we need *one* leader, and a fenced lease gives that with far less
operational surface than a Raft log. Followers rebuild state from Postgres on
promotion, so there is no log to replicate yet.

**Cost:** leadership depends on the store's availability. Accepted: the store is
already a hard dependency for job state.

## ADR-6: Deterministic decomposition by default, LLM planning as an override

The stage compiler in `coordinator::decompose` is pure Rust: strategy → stage list →
DAG. An LLM planner is an optional stage that *proposes* a stage list, which is then
validated (acyclic, capability-known, fan-out within quota) before use.

**Why:** reproducible benchmarks and tests need a decomposition that does not change
between runs. It also removes the most common failure mode of agent frameworks —
an unparseable plan bringing the whole job down.

**Cost:** default plans are shaped by templates, not by insight into the objective.
Adaptive strategy + dynamic task insertion are the escape hatch.

## ADR-7: Deterministic validation outranks LLM judging

Output validation runs rules first (`NonEmpty`, `MinWords`, `MustMention`,
`RequiredJsonKeys`, plus compile/test/lint for code tasks). Critic and judge agents
only run on outputs that already pass the mechanical gate, and only where genuine
judgement is needed.

**Why:** an LLM judge is expensive, non-deterministic, and cheerfully approves
garbage. A word count never does.

**Cost:** rules are coarse. They gate; they do not assess quality.

## ADR-8: No hidden chain-of-thought stored anywhere

Agents persist: final answer, a short `reasoning_summary`, structured evidence,
confidence, and validation results. Raw provider reasoning traces are never stored,
logged, or exposed through the API.

**Why:** required by provider policy and by ordinary data hygiene — reasoning traces
are the highest-volume, highest-risk, lowest-value thing you could retain.

**Cost:** post-hoc debugging leans on evidence + traces instead of transcripts.

## ADR-9: One `ModelProvider` trait; nothing above it knows a provider name

`complete`, `stream`, `health_check`. The `Gateway` adds routing, fallback,
per-model concurrency, rate limits, retries, circuit breaking, caching, and token/
cost accounting — all provider-agnostic. Provider quirks stay inside the impl.

**Why:** it keeps `if provider == "openai"` out of 40 files, and makes
`MockProvider` a first-class citizen so 95% of tests need no network.

**Cost:** provider-specific features (e.g. exotic tool formats) need a capability
flag on the trait rather than a direct call.

## ADR-10: Postgres for truth, Redis for speed, object store for bulk

Postgres holds every state transition and attempt record (append-only history, not
just latest state). Redis Streams carry the queue and ephemeral hot state. Large
artifacts go to S3/MinIO with only a reference in Postgres.

**Why:** auditability requires a relational store with constraints; queue throughput
requires something that is not a relational store; 2 MB agent outputs belong in
neither.

**Cost:** three systems to run. Compose file makes that one command locally.

## ADR-11: Queue behind a trait with an in-memory implementation that is not a toy

`InMemoryQueue` implements the full contract — priorities, delays, leases, expiry
sweep, dedupe, retry backoff, dead-letter — so Phase 1 tests exercise real
semantics, and Redis/NATS impls are validated against the same test suite.

**Why:** if the in-memory queue cheats, every test above it is theatre.

**Cost:** duplicated effort versus a thin fake. Paid back the first time the Redis
impl gets a lease bug the shared suite catches.

## ADR-12: Agents are supervised in-process tasks, not OS processes (until they need to be)

An agent is a tokio task with an identity, capability set, model config, and load
metrics. Untrusted *tool* execution — generated code — is what gets a sandbox
(container/WASM/restricted subprocess), not the agent itself.

**Why:** 100 agents per node as tasks costs kilobytes; as processes it costs
gigabytes. The security boundary that matters is around executing generated code.

**Cost:** a panicking agent can't corrupt siblings (tokio catches it) but does share
the process's memory limit. Per-agent memory caps are advisory until agents move to
child processes.

## ADR-13: Events are a broadcast channel with a Postgres journal behind it

Live streaming uses `tokio::sync::broadcast` (lossy, bounded, fast). Every event is
also appended to `job_events`, so a client that lags or reconnects replays from a
sequence number instead of losing history.

**Why:** you cannot have both unbounded buffering and low latency. Splitting live
delivery from durable replay gives each one the right guarantee.

**Cost:** an event can be delivered twice (live + replay). Events carry sequence
numbers so consumers dedupe.

## ADR-14: Cost and tokens are first-class domain concepts, not logs

Every model call returns tokens and cost; every task attempt records them; job
budgets are enforced by the gateway (reject) and the coordinator (stop scheduling).

**Why:** "how much did 500 agents cost" is a primary research question of this
project, and a budget that lives only in a dashboard is not a budget.

**Cost:** cost tables must be maintained per model.

## ADR-15: Windows-friendly local development

Phase 1 requires only `cargo test`. `protoc` is vendored via `protoc-bin-vendored`
in Phase 2 rather than assumed on PATH, and Postgres/Redis arrive through Compose.

**Why:** the primary dev machine here is Windows without protoc; a toolchain that
only builds on Linux CI is a toolchain that stops getting built.

**Cost:** one extra build dependency.
