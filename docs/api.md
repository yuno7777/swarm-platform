# HTTP API

Served by `swarm-api`. Everything is JSON except `/metrics` (Prometheus text) and the
event stream (`text/event-stream`).

```bash
cargo run -p swarm-api-server -- --bind 127.0.0.1:8080
```

Flags: `--bind` (or `SWARM_BIND`), `--config` (or `SWARM_CONFIG`), `--journal` (or
`SWARM_JOURNAL`), `--log`. Ctrl-C shuts down gracefully, letting in-flight requests
finish.

With `--journal ./swarm.journal`, every job state change is appended to disk and
replayed at start-up *before* the socket is bound, so a client never sees a job that
is about to be resurrected under it. Completed tasks keep their results across the
restart and are not re-run; tasks that were in flight go back into the ready set.
Without the flag, jobs live only as long as the process.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, build version, uptime |
| GET | `/metrics` | Prometheus exposition |
| GET | `/v1/cluster` | Agents, queue depth, dead letters, spend, provider health |
| POST | `/v1/jobs` | Submit a job; returns `202` and a job id |
| GET | `/v1/jobs` | Every job this coordinator knows, newest first |
| GET | `/v1/jobs/{id}` | Status, task counts, progress, tokens, cost |
| GET | `/v1/jobs/{id}/graph` | The task DAG, in stage order |
| GET | `/v1/jobs/{id}/results` | Per-task results, including while running |
| GET | `/v1/jobs/{id}/result` | The merged final result (`404` until finished) |
| GET | `/v1/jobs/{id}/failures` | Every failed attempt, with its error kind |
| GET | `/v1/jobs/{id}/transitions` | The task state-transition audit trail |
| GET | `/v1/jobs/{id}/agents` | Agents allocated to the job |
| GET | `/v1/jobs/{id}/events` | Live progress as Server-Sent Events |
| POST | `/v1/jobs/{id}/cancel` | Abandon running work and purge the queue |
| POST | `/v1/jobs/{id}/pause` | Suspend scheduling; running tasks finish |
| POST | `/v1/jobs/{id}/resume` | Resume scheduling |

## Submitting a job

Only `objective` is required; everything else has a default.

```bash
curl -X POST localhost:8080/v1/jobs \
  -H 'content-type: application/json' \
  -d '{
        "objective": "Compare Raft and Paxos for leader election",
        "execution_strategy": "debate",
        "max_agents": 6,
        "priority": "high",
        "max_cost": 5.0
      }'
```

```json
{ "job_id": "019fd423-3a7e-7081-a0b5-b3e5ed61fa48", "status": "planning", "tasks_planned": 13 }
```

Submission returns as soon as the objective has been compiled and admitted; execution
continues in the background. `max_agents` is both the ceiling and the target
parallelism. Valid strategies: `sequential`, `parallel`, `hierarchical`, `debate`,
`map_reduce`, `planner_executor`, `supervisor_worker`, `consensus`, `adaptive`.

## Following progress

```bash
curl -N localhost:8080/v1/jobs/$JOB/events
```

```text
id: 14
data: {"sequence_number":14,"job_id":"019f…","task_id":"019f…","kind":"task_completed",
       "detail":"Answer · … in 3ms","progress":0.46,"at":"2026-08-06T…Z"}
```

The SSE `id` is the event's sequence number, which is monotonic per job. The channel is
bounded and lossy on purpose: a client that falls too far behind sees a gap in the
sequence rather than making the coordinator buffer without limit. Durable replay from a
sequence number arrives with the Postgres event journal.

Event kinds: `job_admitted`, `job_planned`, `agent_spawned`, `agent_terminated`,
`task_queued`, `task_started`, `task_completed`, `task_failed`, `task_retrying`,
`task_dead_lettered`, `task_cancelled`, `job_paused`, `job_resumed`, `job_cancelled`,
`job_finished`.

## Errors

Every error carries a machine-readable `kind` and a human-readable `message`:

```json
{ "error": { "kind": "quota_exceeded", "message": "job asks for 100000 agents; this cluster allows 512" } }
```

| Status | When |
|---|---|
| 400 | Malformed id, failed validation, a graph that would cycle |
| 402 | The job or gateway budget is exhausted |
| 404 | No such job, or no final result yet |
| 409 | Cancelled job, or a memory version conflict |
| 422 | The request body is not a job request |
| 429 | Over an agent quota, or provider rate limited |
| 502 | A model provider failed or its breaker is open |
| 504 | An operation exceeded its deadline |

## Metrics

`/metrics` exposes queue depth by state, dead letters, allocated agents, task
throughput and retries, expired leases, model requests and cache hits, tokens, spend,
and job counts labelled by status. Values come from the coordinator's own counters
rather than a separate registry, so they cannot drift from the state they describe.
