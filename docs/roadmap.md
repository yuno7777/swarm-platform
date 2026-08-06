# Phase-by-phase implementation plan

Rule: **every phase compiles, passes `cargo fmt --check`, passes
`cargo clippy -- -D warnings`, and has tests before the next phase starts.**
Crates are created in the phase that needs them.

Status legend: ✅ done · 🔜 next · ⬜ planned

---

## Phase 1 — Local prototype ✅

Crates: `domain`, `task-queue`, `shared-memory`, `model-gateway`, `agent-runtime`,
`coordinator`, `admin-cli`.

- Core domain types + five state machines with rejection of illegal transitions
- Task DAG on petgraph: acyclic assertion, ready-set, topological layers, dynamic
  insertion, task splitting
- `TaskQueue` trait + full-semantics in-memory implementation (priority, delay,
  leases, expiry sweep, dedupe, retry backoff, dead-letter)
- `MemoryStore` trait + in-memory implementation (versioning, CAS, namespaces, TTL)
- `ModelProvider` trait + deterministic `MockProvider`; `Gateway` with fallback,
  concurrency limit, response cache, token/cost accounting
- Agent runtime with execution records (duplicate-safe), checkpoints, deterministic
  output validation
- Coordinator: stage-compiler decomposition for all 9 strategies, 4 schedulers,
  execution engine with retries/timeouts/cancel/pause/resume, event stream,
  aggregation into `FinalResult`
- `swarmctl run` / `demo` executing jobs end to end in-process

**Exit criteria:** all three demo workflows complete locally with the mock provider;
1000-task job and 100-agent job pass in tests; duplicate delivery proven not to
double-execute.

## Phase 2a — HTTP ingress ✅

New crate: `api-server`.

- `swarm-api` binary: job submission, cancel/pause/resume, job state, task graph,
  intermediate results, final result, failures, transition audit trail, agents
- Server-Sent Events stream with per-job sequence numbers
- Prometheus `/metrics` and a `/v1/cluster` operator view
- Typed error mapping: every `SwarmError` gets the right status code and a
  machine-readable `kind`
- Integration tests drive a real socket, not handler functions

**Exit criteria met:** the platform runs as its own OS process and is driven entirely
over HTTP; 16 tests cover the wire behaviour.

## Phase 2b — Crash recovery ✅

New crate: `persistence`.

- `Journal` trait: an ordered, append-only log of everything that happens to a job
- `FileJournal`: JSON lines, fsynced per append, tolerant of a torn final record and
  loud about corruption anywhere else
- `MemoryJournal`: the default, so the write path is identical whether or not
  durability is configured
- Coordinator journals every task transition, result, failure, and status change
- `Coordinator::recover()` replays the log into live job state: completed tasks keep
  their results, mid-flight tasks return to the ready set, spent attempts still count
- `swarm-api --journal <path>` recovers before it starts serving

**Exit criteria met:** a coordinator aborted mid-job resumes without redoing completed
work — proven by restored results keeping their original agent and timestamp — and
`kill -9` on the API server followed by a restart brings the job back.

## Phase 2c — Distributed execution 🔜

New crates: `protocol`, `worker`. (`protoc` via `protoc-bin-vendored`.)

- Postgres schema via sqlx migrations; every transition and attempt journaled
- `PostgresJobStore`, `PostgresMemoryStore` behind the Phase 1 traits
- Redis Streams `TaskQueue` implementation, validated against the *same* test suite
  as the in-memory one (consumer groups, XCLAIM-based lease recovery)
- `swarm-worker` binary: node registration, resource reporting, heartbeats,
  lease loop, graceful shutdown on SIGINT/SIGTERM
- `swarm-api`: axum HTTP + tonic gRPC — submit/cancel/pause/resume/retry, job state,
  task graph, intermediate outputs, failures, SSE event stream
- Real providers: OpenAI-compatible, Anthropic-compatible, Ollama, vLLM
- Docker Compose: postgres + redis + 1 coordinator + 2 workers + api

**Exit criteria:** kill a worker mid-job → task recovers from checkpoint on another
worker; restart Redis → queue survives; restart coordinator → job resumes from
Postgres.

## Phase 3 — Cluster coordination ⬜

New crate: `cluster-membership`.

- `LeaderElector` trait; fenced-lease implementation on Postgres
- `leadership_terms` table + fencing predicate on all scheduling writes
- Follower promotion: rebuild scheduler state from Postgres, sweep expired leases
- Node registry: join/heartbeat/degrade/drain/evict, work stealing between nodes
- Chaos harness: kill leader during scheduling, partition leader from Postgres,
  clock skew

**Exit criteria:** measured leader-election and failover time; zero lost and zero
double-executed tasks across 50 induced failovers; state consistent after recovery.

## Phase 4 — Advanced swarm behaviour ⬜

New crates: `consensus`, `result-aggregator` (split out of `coordinator`).

- Dynamic agent spawning: queue-depth-driven scale-up, idle timeout scale-down, warm
  pools, recursive-spawn guard, per-job and cluster caps
- Consensus modes: majority, weighted, confidence-weighted, ranked-choice,
  critic-verifier, debate+judge, Byzantine-tolerant quorum simulation
- Adaptive scheduler using live success-rate/latency/cost signals
- Semantic memory (pgvector) with read-through cache and compaction
- Verification pipeline: critic/verifier agents, contradiction detection, evidence
  ranking, confidence scoring, low-quality retry
- Task split/merge during execution

**Exit criteria:** consensus correct with 30% deliberately unreliable agents;
adaptive scheduler beats least-loaded on the mixed benchmark.

## Phase 5 — Production readiness ⬜

New crate: `telemetry`. Plus `dashboard/`.

- JWT + API keys, RBAC, per-tenant isolation and quotas, audit log
- TLS between services, optional mTLS, signed agent messages, replay protection
- Tool permission policy per capability; sandboxed tool execution (container/WASM)
- Prometheus metrics (§17 list), OpenTelemetry traces end-to-end, JSON logs
- Next.js dashboard: topology, leader, nodes, agents, DAG, messages, costs, DLQ,
  live via SSE; admin actions (cancel, retry, drain, pause, failover drill)
- Chaos test suite in CI

**Exit criteria:** one trace spans api→coordinator→queue→worker→gateway→memory→
aggregator; dashboard shows a live 100-agent job; chaos suite green.

## Phase 6 — Performance research ⬜

New crate: `benchmark-runner`.

- Sweeps: 1, 5, 10, 25, 50, 100, 250, 500 agents × 9 topologies
- Metrics per run: completion time, throughput, p50/p95/p99 latency, queue wait,
  scheduling latency, message count/throughput, CPU/mem/network, model calls, tokens,
  cost, retries, failure rate, consensus quality, agent utilization, coordination
  overhead, failover time
- CSV + JSON reports, plots for the seven required comparisons
- Criterion micro-benchmarks for scheduler, graph, queue, serialization

**Exit criteria:** reproducible reports checked into `benchmarks/`, with a written
analysis of coordination overhead and the cost/quality trade-off curve.
