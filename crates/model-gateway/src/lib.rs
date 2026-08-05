//! Provider-independent access to language models.
//!
//! Everything above this crate talks to [`ModelProvider`] and never to a vendor. The
//! [`Gateway`] adds the operational concerns that are identical for every vendor —
//! routing, fallback, retries, circuit breaking, concurrency limits, response caching,
//! and token/cost accounting — so no provider-specific branch ever leaks upward.
//!
//! Cost and token counts are returned on every call rather than logged, because
//! "what did 500 agents cost" is a first-class question for this platform (ADR-14).
#![forbid(unsafe_code)]

pub mod gateway;
pub mod mock;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use swarm_domain::hash::{fnv1a, OFFSET};
use swarm_domain::{AgentId, JobId, Result, TaskId};

/// Stable hash of a byte string, re-exported so providers and the cache agree.
pub use swarm_domain::hash::stable_hash as hash_bytes;

pub use gateway::{Gateway, GatewayConfig, Usage};
pub use mock::MockProvider;

/// A stream of generated tokens.
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<TokenChunk>> + Send>>;

/// Who authored a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Instructions that frame the whole exchange.
    System,
    /// Input from the platform on behalf of a task.
    User,
    /// A previous model response.
    Assistant,
    /// Output of a tool call.
    Tool,
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Author.
    pub role: Role,
    /// Content.
    pub content: String,
}

impl Message {
    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A request for a completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model to use; the gateway routes on this.
    pub model: String,
    /// Conversation so far.
    pub messages: Vec<Message>,
    /// Cap on generated tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Sequences that end generation.
    pub stop: Vec<String>,
    /// Whether the response must be a JSON object.
    pub json_mode: bool,
    /// Skip the response cache for this call.
    ///
    /// Set when the caller already has the cached answer and knows it was unusable —
    /// a retry after a failed validation, for instance. Without this, such a retry
    /// would be served the identical answer and fail identically.
    #[serde(default)]
    pub bypass_cache: bool,
    /// Makes a retried call recognisable as the same logical request.
    pub idempotency_key: String,
    /// Job this call belongs to, for cost attribution.
    pub job_id: Option<JobId>,
    /// Task this call belongs to.
    pub task_id: Option<TaskId>,
    /// Agent making the call.
    pub agent_id: Option<AgentId>,
}

impl CompletionRequest {
    /// A request for `model` with `messages`.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            max_tokens: None,
            temperature: None,
            stop: Vec::new(),
            json_mode: false,
            bypass_cache: false,
            idempotency_key: String::new(),
            job_id: None,
            task_id: None,
            agent_id: None,
        }
    }

    /// Require a JSON object response.
    #[must_use]
    pub fn json(mut self) -> Self {
        self.json_mode = true;
        self
    }

    /// Set the sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the generated-token ceiling.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Attach the idempotency key of the calling task attempt.
    #[must_use]
    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = key.into();
        self
    }

    /// Ask the gateway to ignore any cached answer for this request.
    #[must_use]
    pub fn bypassing_cache(mut self, bypass: bool) -> Self {
        self.bypass_cache = bypass;
        self
    }

    /// Attach the job, task, and agent this call is for.
    #[must_use]
    pub fn attributed_to(mut self, job_id: JobId, task_id: TaskId, agent_id: AgentId) -> Self {
        self.job_id = Some(job_id);
        self.task_id = Some(task_id);
        self.agent_id = Some(agent_id);
        self
    }

    /// The last user message, which is what mock providers key their answers on.
    #[must_use]
    pub fn last_user_message(&self) -> &str {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map_or("", |message| message.content.as_str())
    }

    /// Rough prompt token count.
    ///
    /// Deliberately a word count rather than a tokenizer: it needs to be
    /// vendor-neutral, allocation-free, and stable across runs for benchmarking.
    #[must_use]
    pub fn estimated_prompt_tokens(&self) -> u64 {
        self.messages
            .iter()
            .map(|message| message.content.split_whitespace().count() as u64)
            .sum()
    }

    /// Cache key covering everything that changes the answer.
    ///
    /// Excludes the idempotency key and attribution, so two tasks asking the identical
    /// question share one answer.
    #[must_use]
    pub fn cache_key(&self) -> u64 {
        let mut hash = fnv1a(OFFSET, self.model.as_bytes());
        for message in &self.messages {
            hash = fnv1a(hash, &[message.role as u8]);
            hash = fnv1a(hash, message.content.as_bytes());
        }
        hash = fnv1a(hash, &[u8::from(self.json_mode)]);
        if let Some(temperature) = self.temperature {
            hash = fnv1a(hash, &temperature.to_le_bytes());
        }
        if let Some(max_tokens) = self.max_tokens {
            hash = fnv1a(hash, &max_tokens.to_le_bytes());
        }
        hash
    }
}

/// A completed model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated text.
    pub text: String,
    /// Provider that served it.
    pub provider: String,
    /// Model that served it.
    pub model: String,
    /// Prompt tokens consumed.
    pub tokens_in: u64,
    /// Completion tokens produced.
    pub tokens_out: u64,
    /// Estimated spend for this call.
    pub cost_usd: f64,
    /// Time the call took.
    pub latency_ms: u64,
    /// Whether it was served from the gateway's cache.
    pub cached: bool,
    /// Why generation stopped.
    pub finish_reason: String,
}

impl CompletionResponse {
    /// Parse the text as a JSON object, for `json_mode` requests.
    pub fn parse_json(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.text).map_err(|e| swarm_domain::SwarmError::Provider {
            provider: self.provider.clone(),
            detail: format!("response is not valid JSON: {e}"),
        })
    }
}

/// One chunk of a streamed response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenChunk {
    /// Text of this chunk.
    pub text: String,
    /// Whether this is the final chunk.
    pub last: bool,
}

/// A provider's self-reported health.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderHealth {
    /// Provider name.
    pub provider: String,
    /// Whether it is accepting calls.
    pub healthy: bool,
    /// Observed probe latency.
    pub latency_ms: u64,
    /// Whether the gateway has tripped its breaker.
    pub circuit_open: bool,
    /// Free-text detail when unhealthy.
    pub detail: Option<String>,
}

/// Price per million tokens, used for cost accounting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    /// USD per million prompt tokens.
    pub input_per_million: f64,
    /// USD per million completion tokens.
    pub output_per_million: f64,
}

impl Default for ModelPricing {
    fn default() -> Self {
        // Stand-in figures. Real per-model tables arrive with the real providers in
        // Phase 2; the point here is that every call carries a cost at all.
        Self {
            input_per_million: 3.0,
            output_per_million: 15.0,
        }
    }
}

impl ModelPricing {
    /// Cost of a call in USD.
    #[must_use]
    pub fn cost(&self, tokens_in: u64, tokens_out: u64) -> f64 {
        (tokens_in as f64 / 1_000_000.0) * self.input_per_million
            + (tokens_out as f64 / 1_000_000.0) * self.output_per_million
    }
}

/// A source of model completions.
///
/// Implementations own vendor quirks (auth, wire format, error mapping) and nothing
/// else: retries, fallback, limits, and caching all belong to the [`Gateway`].
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Provider name, used for routing, metrics, and breaker state.
    fn name(&self) -> &str;

    /// Generate a complete response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// Generate a response incrementally.
    async fn stream(&self, request: CompletionRequest) -> Result<TokenStream>;

    /// Probe whether the provider is usable.
    async fn health_check(&self) -> Result<ProviderHealth>;

    /// Whether this provider can serve `model`.
    ///
    /// The default accepts everything, which is what single-provider deployments and
    /// mocks want; real providers narrow it.
    fn supports_model(&self, _model: &str) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_keys_ignore_attribution_but_not_content() {
        let base = CompletionRequest::new("m", vec![Message::user("hello")]);
        let attributed = base.clone().with_idempotency_key("task-1#0").attributed_to(
            JobId::new(),
            TaskId::new(),
            AgentId::new(),
        );
        assert_eq!(base.cache_key(), attributed.cache_key());

        let different = CompletionRequest::new("m", vec![Message::user("hello!")]);
        assert_ne!(base.cache_key(), different.cache_key());

        assert_ne!(base.cache_key(), base.clone().json().cache_key());
        assert_ne!(
            base.cache_key(),
            base.clone().with_temperature(0.9).cache_key()
        );
        assert_ne!(
            base.cache_key(),
            CompletionRequest::new("other", vec![Message::user("hello")]).cache_key()
        );
    }

    #[test]
    fn hashing_is_stable_across_runs() {
        // Cache keys and mock responses depend on this being deterministic.
        assert_eq!(hash_bytes(b"swarm"), hash_bytes(b"swarm"));
        assert_ne!(hash_bytes(b"swarm"), hash_bytes(b"swarn"));
    }

    #[test]
    fn pricing_charges_input_and_output_separately() {
        let pricing = ModelPricing {
            input_per_million: 1.0,
            output_per_million: 10.0,
        };
        assert!((pricing.cost(1_000_000, 0) - 1.0).abs() < 1e-9);
        assert!((pricing.cost(0, 1_000_000) - 10.0).abs() < 1e-9);
        assert_eq!(pricing.cost(0, 0), 0.0);
    }

    #[test]
    fn the_last_user_message_is_what_a_provider_answers() {
        let request = CompletionRequest::new(
            "m",
            vec![
                Message::system("be brief"),
                Message::user("first"),
                Message::assistant("ok"),
                Message::user("second"),
            ],
        );
        assert_eq!(request.last_user_message(), "second");
        assert_eq!(request.estimated_prompt_tokens(), 5);
        assert_eq!(
            CompletionRequest::new("m", vec![Message::system("x")]).last_user_message(),
            ""
        );
    }
}
