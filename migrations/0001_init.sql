-- Swarm Platform initial schema.
-- Forward-only migration. Applied by sqlx (Phase 2).
--
-- Principles:
--   * Truth is append-only. Latest state is a convenience column; history is a table.
--   * Every state transition is journaled with actor, reason, and correlation id.
--   * Every leader write is fenced by a term number (see leadership_terms).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------- tenancy

CREATE TABLE tenants (
    id              UUID PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    quotas          JSONB NOT NULL DEFAULT '{}'::jsonb, -- max_agents, tokens_per_minute, budget_usd
    disabled_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- cluster

CREATE TABLE leadership_terms (
    term            BIGINT PRIMARY KEY,
    leader_id       TEXT NOT NULL,
    fencing_token   BIGINT NOT NULL,
    acquired_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    renewed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    released_at     TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT fencing_matches_term CHECK (fencing_token = term)
);

CREATE UNIQUE INDEX leadership_single_active
    ON leadership_terms (released_at) WHERE released_at IS NULL;

CREATE TABLE nodes (
    id                      UUID PRIMARY KEY,
    address                 TEXT NOT NULL,
    status                  TEXT NOT NULL, -- joining|ready|degraded|draining|unreachable|removed
    cpu_cores               INTEGER NOT NULL,
    total_memory_bytes      BIGINT NOT NULL,
    available_memory_bytes  BIGINT NOT NULL,
    gpu_count               INTEGER NOT NULL DEFAULT 0,
    available_gpu_memory_bytes BIGINT NOT NULL DEFAULT 0,
    active_agents           INTEGER NOT NULL DEFAULT 0,
    active_tasks            INTEGER NOT NULL DEFAULT 0,
    labels                  JSONB NOT NULL DEFAULT '{}'::jsonb,
    version                 TEXT NOT NULL DEFAULT 'unknown',
    registered_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_heartbeat_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX nodes_status_heartbeat ON nodes (status, last_heartbeat_at DESC);

CREATE TABLE node_state_transitions (
    id              BIGSERIAL PRIMARY KEY,
    node_id         UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    from_state      TEXT,
    to_state        TEXT NOT NULL,
    actor           TEXT NOT NULL,
    reason          TEXT,
    term            BIGINT,
    correlation_id  UUID,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX node_transitions_node ON node_state_transitions (node_id, at DESC);

-- ---------------------------------------------------------------- jobs

CREATE TABLE jobs (
    id                      UUID PRIMARY KEY,
    tenant_id               UUID NOT NULL REFERENCES tenants(id),
    objective               TEXT NOT NULL,
    context                 TEXT,
    priority                SMALLINT NOT NULL, -- 0 low .. 3 critical
    execution_strategy      TEXT NOT NULL,
    status                  TEXT NOT NULL,
    required_capabilities   TEXT[] NOT NULL DEFAULT '{}',
    max_agents              INTEGER NOT NULL,
    max_cost_usd            NUMERIC(12, 6),
    max_runtime_seconds     BIGINT,
    deadline                TIMESTAMPTZ,
    idempotency_key         TEXT,
    submitted_by            TEXT NOT NULL,
    correlation_id          UUID NOT NULL,
    trace_id                TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at              TIMESTAMPTZ,
    finished_at             TIMESTAMPTZ,
    CONSTRAINT jobs_priority_range CHECK (priority BETWEEN 0 AND 3),
    CONSTRAINT jobs_max_agents_positive CHECK (max_agents > 0)
);

CREATE UNIQUE INDEX jobs_tenant_idempotency
    ON jobs (tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX jobs_status_priority ON jobs (status, priority DESC, created_at);
CREATE INDEX jobs_tenant_created ON jobs (tenant_id, created_at DESC);

CREATE TABLE job_state_transitions (
    id              BIGSERIAL PRIMARY KEY,
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    from_state      TEXT,
    to_state        TEXT NOT NULL,
    actor           TEXT NOT NULL,
    reason          TEXT,
    term            BIGINT,
    correlation_id  UUID,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX job_transitions_job ON job_state_transitions (job_id, at DESC);

CREATE TABLE job_events (
    seq             BIGSERIAL PRIMARY KEY,
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    payload         JSONB NOT NULL,
    trace_id        TEXT,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX job_events_job_seq ON job_events (job_id, seq);

-- ---------------------------------------------------------------- tasks

CREATE TABLE tasks (
    id                      UUID PRIMARY KEY,
    job_id                  UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    title                   TEXT NOT NULL,
    description             TEXT NOT NULL,
    kind                    TEXT NOT NULL, -- plan|work|critique|verify|judge|merge|...
    state                   TEXT NOT NULL,
    required_capabilities   TEXT[] NOT NULL DEFAULT '{}',
    estimated_complexity    INTEGER NOT NULL DEFAULT 1,
    estimated_tokens        BIGINT,
    timeout_seconds         BIGINT NOT NULL DEFAULT 300,
    max_attempts            INTEGER NOT NULL DEFAULT 3,
    attempt                 INTEGER NOT NULL DEFAULT 0,
    backoff_base_ms         BIGINT NOT NULL DEFAULT 500,
    backoff_max_ms          BIGINT NOT NULL DEFAULT 60000,
    idempotency_key         TEXT NOT NULL,
    validation              JSONB NOT NULL DEFAULT '[]'::jsonb,
    stage                   INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT tasks_attempt_bound CHECK (attempt <= max_attempts + 1)
);

CREATE UNIQUE INDEX tasks_idempotency ON tasks (idempotency_key);
CREATE INDEX tasks_job_state ON tasks (job_id, state);

CREATE TABLE task_dependencies (
    task_id         UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    depends_on      UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, depends_on),
    CONSTRAINT no_self_dependency CHECK (task_id <> depends_on)
);

CREATE INDEX task_dependencies_reverse ON task_dependencies (depends_on);

CREATE TABLE task_state_transitions (
    id              BIGSERIAL PRIMARY KEY,
    task_id         UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    from_state      TEXT,
    to_state        TEXT NOT NULL,
    actor           TEXT NOT NULL,
    reason          TEXT,
    term            BIGINT,
    correlation_id  UUID,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX task_transitions_task ON task_state_transitions (task_id, at DESC);

-- ---------------------------------------------------------------- agents

CREATE TABLE agents (
    id                  UUID PRIMARY KEY,
    node_id             UUID REFERENCES nodes(id) ON DELETE SET NULL,
    job_id              UUID REFERENCES jobs(id) ON DELETE SET NULL,
    agent_type          TEXT NOT NULL,
    status              TEXT NOT NULL,
    model_provider      TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    temperature         REAL NOT NULL DEFAULT 0.2,
    max_tokens          INTEGER NOT NULL DEFAULT 2048,
    current_load        REAL NOT NULL DEFAULT 0,
    success_rate        REAL NOT NULL DEFAULT 1,
    average_latency_ms  BIGINT NOT NULL DEFAULT 0,
    tasks_completed     BIGINT NOT NULL DEFAULT 0,
    tasks_failed        BIGINT NOT NULL DEFAULT 0,
    spawn_cost_usd      NUMERIC(12, 6) NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    terminated_at       TIMESTAMPTZ
);

CREATE INDEX agents_node_status ON agents (node_id, status);
CREATE INDEX agents_job ON agents (job_id) WHERE job_id IS NOT NULL;

CREATE TABLE agent_capabilities (
    agent_id        UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    capability      TEXT NOT NULL,
    PRIMARY KEY (agent_id, capability)
);

CREATE INDEX agent_capabilities_capability ON agent_capabilities (capability);

CREATE TABLE agent_state_transitions (
    id              BIGSERIAL PRIMARY KEY,
    agent_id        UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    from_state      TEXT,
    to_state        TEXT NOT NULL,
    actor           TEXT NOT NULL,
    reason          TEXT,
    correlation_id  UUID,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- assignment & attempts

CREATE TABLE task_assignments (
    id                  UUID PRIMARY KEY,
    task_id             UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    agent_id            UUID REFERENCES agents(id) ON DELETE SET NULL,
    node_id             UUID REFERENCES nodes(id) ON DELETE SET NULL,
    lease_id            UUID NOT NULL,
    lease_state         TEXT NOT NULL, -- held|extended|released|rejected|expired
    scheduler_strategy  TEXT NOT NULL,
    queue               TEXT NOT NULL DEFAULT 'default',
    term                BIGINT,
    assigned_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_expires_at    TIMESTAMPTZ NOT NULL,
    released_at         TIMESTAMPTZ
);

CREATE UNIQUE INDEX task_assignments_lease ON task_assignments (lease_id);
CREATE INDEX task_assignments_open_leases
    ON task_assignments (lease_expires_at) WHERE released_at IS NULL;

CREATE TABLE task_attempts (
    id                  UUID PRIMARY KEY,
    task_id             UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt             INTEGER NOT NULL,
    agent_id            UUID REFERENCES agents(id) ON DELETE SET NULL,
    node_id             UUID REFERENCES nodes(id) ON DELETE SET NULL,
    outcome             TEXT, -- succeeded|failed|timed_out|cancelled|superseded
    error_kind          TEXT,
    error_message       TEXT,
    validation_passed   BOOLEAN,
    validation_detail   JSONB,
    confidence          REAL,
    tokens_in           BIGINT NOT NULL DEFAULT 0,
    tokens_out          BIGINT NOT NULL DEFAULT 0,
    cost_usd            NUMERIC(12, 6) NOT NULL DEFAULT 0,
    output_ref          TEXT, -- memory key or object-store URI
    trace_id            TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    UNIQUE (task_id, attempt)
);

CREATE INDEX task_attempts_outcome ON task_attempts (outcome, finished_at DESC);

CREATE TABLE checkpoints (
    id              UUID PRIMARY KEY,
    task_id         UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt         INTEGER NOT NULL,
    seq             INTEGER NOT NULL,
    label           TEXT NOT NULL,
    state           JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (task_id, attempt, seq)
);

CREATE INDEX checkpoints_latest ON checkpoints (task_id, seq DESC);

CREATE TABLE dead_letters (
    id              UUID PRIMARY KEY,
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    task_id         UUID REFERENCES tasks(id) ON DELETE SET NULL,
    queue           TEXT NOT NULL,
    attempts        INTEGER NOT NULL,
    last_error      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    replayed_at     TIMESTAMPTZ
);

CREATE INDEX dead_letters_open ON dead_letters (created_at DESC) WHERE replayed_at IS NULL;

-- ---------------------------------------------------------------- messaging

CREATE TABLE messages (
    id                  UUID PRIMARY KEY,
    correlation_id      UUID NOT NULL,
    causation_id        UUID,
    job_id              UUID REFERENCES jobs(id) ON DELETE CASCADE,
    task_id             UUID REFERENCES tasks(id) ON DELETE SET NULL,
    sender              TEXT NOT NULL,
    receiver            TEXT,
    topic               TEXT,
    message_type        TEXT NOT NULL,
    sequence_number     BIGINT NOT NULL,
    schema_version      INTEGER NOT NULL DEFAULT 1,
    payload             JSONB NOT NULL,
    compressed          BOOLEAN NOT NULL DEFAULT false,
    signature           BYTEA,
    idempotency_key     TEXT NOT NULL,
    trace_id            TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ,
    CONSTRAINT message_addressed CHECK (receiver IS NOT NULL OR topic IS NOT NULL)
);

CREATE UNIQUE INDEX messages_idempotency ON messages (idempotency_key);
CREATE INDEX messages_job_seq ON messages (job_id, sequence_number);
CREATE INDEX messages_correlation ON messages (correlation_id);

-- ---------------------------------------------------------------- shared memory

CREATE TABLE memory_records (
    id              UUID PRIMARY KEY,
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id        UUID REFERENCES agents(id) ON DELETE SET NULL,
    namespace       TEXT NOT NULL,
    key             TEXT NOT NULL,
    value           JSONB NOT NULL,
    version         BIGINT NOT NULL DEFAULT 1,
    content_hash    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ,
    UNIQUE (namespace, key)
);

CREATE INDEX memory_records_job ON memory_records (job_id);
CREATE INDEX memory_records_expiry ON memory_records (expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE memory_audit (
    id              BIGSERIAL PRIMARY KEY,
    record_id       UUID,
    namespace       TEXT NOT NULL,
    key             TEXT NOT NULL,
    version         BIGINT NOT NULL,
    operation       TEXT NOT NULL, -- put|cas|delete|expire|compact
    actor           TEXT NOT NULL,
    previous_value  JSONB,
    value           JSONB,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX memory_audit_key ON memory_audit (namespace, key, version DESC);

CREATE TABLE artifacts (
    id              UUID PRIMARY KEY,
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    task_id         UUID REFERENCES tasks(id) ON DELETE SET NULL,
    uri             TEXT NOT NULL,
    media_type      TEXT NOT NULL,
    size_bytes      BIGINT NOT NULL,
    content_hash    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- consensus

CREATE TABLE consensus_rounds (
    id              UUID PRIMARY KEY,
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    task_id         UUID REFERENCES tasks(id) ON DELETE SET NULL,
    mode            TEXT NOT NULL, -- majority|weighted|confidence|ranked|critic_verifier|debate_judge|bft
    quorum          INTEGER NOT NULL,
    status          TEXT NOT NULL, -- open|decided|escalated|failed
    agreement_rate  REAL,
    decision        JSONB,
    judge_agent_id  UUID REFERENCES agents(id) ON DELETE SET NULL,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ
);

CREATE TABLE consensus_votes (
    id                  UUID PRIMARY KEY,
    round_id            UUID NOT NULL REFERENCES consensus_rounds(id) ON DELETE CASCADE,
    agent_id            UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    answer              TEXT NOT NULL,
    answer_hash         TEXT NOT NULL,
    confidence          REAL NOT NULL,
    weight              REAL NOT NULL DEFAULT 1,
    rank_order          INTEGER,
    evidence            JSONB NOT NULL DEFAULT '[]'::jsonb,
    reasoning_summary   TEXT,
    rejected_reason     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (round_id, agent_id),
    CONSTRAINT confidence_range CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX consensus_votes_hash ON consensus_votes (round_id, answer_hash);

-- ---------------------------------------------------------------- model usage & audit

CREATE TABLE model_requests (
    id              UUID PRIMARY KEY,
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    task_id         UUID REFERENCES tasks(id) ON DELETE SET NULL,
    agent_id        UUID REFERENCES agents(id) ON DELETE SET NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    status          TEXT NOT NULL, -- ok|error|rate_limited|circuit_open|cached
    tokens_in       BIGINT NOT NULL DEFAULT 0,
    tokens_out      BIGINT NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms      BIGINT NOT NULL DEFAULT 0,
    cached          BOOLEAN NOT NULL DEFAULT false,
    fallback_from   TEXT,
    error           TEXT,
    idempotency_key TEXT,
    trace_id        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX model_requests_job ON model_requests (job_id, created_at DESC);
CREATE INDEX model_requests_provider ON model_requests (provider, created_at DESC);

CREATE TABLE cost_records (
    id              UUID PRIMARY KEY,
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL, -- model|agent_spawn|storage
    amount_usd      NUMERIC(12, 6) NOT NULL,
    detail          JSONB NOT NULL DEFAULT '{}'::jsonb,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX cost_records_tenant ON cost_records (tenant_id, at DESC);

CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       UUID REFERENCES tenants(id) ON DELETE SET NULL,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    subject_type    TEXT NOT NULL,
    subject_id      TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    detail          JSONB NOT NULL DEFAULT '{}'::jsonb,
    correlation_id  UUID,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX audit_log_subject ON audit_log (subject_type, subject_id, at DESC);
CREATE INDEX audit_log_actor ON audit_log (actor, at DESC);
