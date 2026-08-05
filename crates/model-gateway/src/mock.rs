//! A deterministic in-process model provider.
//!
//! Almost every automated test in this workspace runs against this rather than a real
//! model: the same prompt always yields the same answer, so scheduling, consensus, and
//! benchmark results are reproducible and no test needs a network or an API key.
//!
//! It can also be told to fail on demand, which is how retry, fallback, and circuit
//! breaker behaviour is tested without waiting for a real provider to have a bad day.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::json;

use swarm_domain::{Result, SwarmError};

use crate::{
    hash_bytes, CompletionRequest, CompletionResponse, ModelPricing, ModelProvider, ProviderHealth,
    TokenChunk, TokenStream,
};

/// A deterministic, offline [`ModelProvider`].
#[derive(Debug)]
pub struct MockProvider {
    name: String,
    pricing: ModelPricing,
    latency: Duration,
    /// Fail the first N calls, then behave. Models a provider recovering.
    fail_first: u64,
    /// Fail every Nth call. Models an unreliable provider.
    fail_every: Option<u64>,
    /// Fail every call. Models an outage.
    always_fail: bool,
    /// Scripted (needle, response) pairs matched against the last user message.
    scripted: Vec<(String, String)>,
    calls: AtomicU64,
    failures: AtomicU64,
    healthy: AtomicBool,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new("mock")
    }
}

impl MockProvider {
    /// A healthy provider named `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pricing: ModelPricing::default(),
            latency: Duration::ZERO,
            fail_first: 0,
            fail_every: None,
            always_fail: false,
            scripted: Vec::new(),
            calls: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            healthy: AtomicBool::new(true),
        }
    }

    /// Delay every call, to make latency-sensitive behaviour observable.
    #[must_use]
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Charge a specific price, for cost-accounting tests.
    #[must_use]
    pub fn with_pricing(mut self, pricing: ModelPricing) -> Self {
        self.pricing = pricing;
        self
    }

    /// Fail the first `count` calls, then succeed.
    #[must_use]
    pub fn failing_first(mut self, count: u64) -> Self {
        self.fail_first = count;
        self
    }

    /// Fail every `n`th call.
    #[must_use]
    pub fn failing_every(mut self, n: u64) -> Self {
        self.fail_every = Some(n.max(1));
        self
    }

    /// Fail every call.
    #[must_use]
    pub fn always_failing(mut self) -> Self {
        self.always_fail = true;
        self.healthy.store(false, Ordering::Relaxed);
        self
    }

    /// Return `response` whenever the last user message contains `needle`.
    #[must_use]
    pub fn scripted(mut self, needle: impl Into<String>, response: impl Into<String>) -> Self {
        self.scripted.push((needle.into(), response.into()));
        self
    }

    /// How many calls have been made, including failures.
    #[must_use]
    pub fn call_count(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    /// How many calls have failed.
    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Flip the health probe result at runtime.
    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }

    fn should_fail(&self, call_index: u64) -> bool {
        if self.always_fail || call_index < self.fail_first {
            return true;
        }
        self.fail_every.is_some_and(|n| (call_index + 1) % n == 0)
    }

    /// Build the deterministic answer for a request.
    fn synthesize(&self, request: &CompletionRequest) -> String {
        let prompt = request.last_user_message();
        if let Some((_, scripted)) = self
            .scripted
            .iter()
            .find(|(needle, _)| prompt.contains(needle.as_str()))
        {
            return scripted.clone();
        }

        let topic = topic_of(prompt);
        let seed = hash_bytes(prompt.as_bytes());
        let confidence = 0.60 + ((seed % 35) as f64) / 100.0;
        let angles = ANGLES[(seed % ANGLES.len() as u64) as usize];
        let findings: Vec<String> = angles
            .iter()
            .map(|angle| format!("{topic}: {angle}"))
            .collect();
        let reasoning = format!(
            "Considered {} angles on `{topic}` and kept the ones that agreed.",
            findings.len()
        );

        if request.json_mode {
            let evidence: Vec<serde_json::Value> = findings
                .iter()
                .enumerate()
                .map(|(index, finding)| {
                    json!({
                        "source": format!("mock://corpus/{:x}/{index}", seed),
                        "claim": finding,
                        "support": ((seed >> (index * 3)) % 30) as f64 / 100.0 + 0.65,
                    })
                })
                .collect();
            json!({
                "summary": format!("{topic}: {}", SUMMARIES[(seed % SUMMARIES.len() as u64) as usize]),
                "findings": findings,
                "confidence": (confidence * 100.0).round() / 100.0,
                "evidence": evidence,
                "reasoning_summary": reasoning,
            })
            .to_string()
        } else {
            let mut text = format!(
                "{topic}: {}\n\n",
                SUMMARIES[(seed % SUMMARIES.len() as u64) as usize]
            );
            for finding in &findings {
                text.push_str("- ");
                text.push_str(finding);
                text.push('\n');
            }
            text.push_str(&format!("\nConfidence: {confidence:.2}\n{reasoning}"));
            text
        }
    }
}

/// Phrasings picked by prompt hash, so different tasks produce visibly different — but
/// still reproducible — output.
const SUMMARIES: [&str; 4] = [
    "the evidence points one way, with one caveat",
    "two independent sources agree on the main claim",
    "the mainstream answer holds, with a known exception",
    "the result is well supported but narrow in scope",
];

const ANGLES: [[&str; 3]; 4] = [
    [
        "the mechanism is well documented",
        "the failure mode is understood",
        "the trade-off is latency against durability",
    ],
    [
        "the primary source is unambiguous",
        "a secondary source disagrees on scope",
        "the practical impact is small",
    ],
    [
        "the approach is standard practice",
        "the cost grows with coordination",
        "measurement matters more than intuition",
    ],
    [
        "the constraint binds only under load",
        "the simple version is usually enough",
        "the edge case needs an explicit test",
    ],
];

/// Pull a short topic out of a prompt: prefers an explicit `Task:` line, since that is
/// how the agent runtime frames work, and falls back to the opening words.
fn topic_of(prompt: &str) -> String {
    let candidate = prompt
        .lines()
        .find_map(|line| line.trim().strip_prefix("Task:"))
        .unwrap_or_else(|| prompt.lines().next().unwrap_or("the objective"));
    let trimmed: String = candidate
        .split_whitespace()
        .take(12)
        .collect::<Vec<_>>()
        .join(" ");
    if trimmed.is_empty() {
        "the objective".to_owned()
    } else {
        trimmed
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let started = Instant::now();
        let call_index = self.calls.fetch_add(1, Ordering::Relaxed);

        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }
        if self.should_fail(call_index) {
            self.failures.fetch_add(1, Ordering::Relaxed);
            return Err(SwarmError::Provider {
                provider: self.name.clone(),
                detail: format!("injected failure on call {}", call_index + 1),
            });
        }

        let text = self.synthesize(&request);
        let tokens_in = request.estimated_prompt_tokens();
        let tokens_out = text.split_whitespace().count() as u64;
        Ok(CompletionResponse {
            tokens_in,
            tokens_out,
            cost_usd: self.pricing.cost(tokens_in, tokens_out),
            text,
            provider: self.name.clone(),
            model: request.model,
            latency_ms: started.elapsed().as_millis() as u64,
            cached: false,
            finish_reason: "stop".to_owned(),
        })
    }

    async fn stream(&self, request: CompletionRequest) -> Result<TokenStream> {
        let response = self.complete(request).await?;
        // split_inclusive keeps the separators, so concatenating the chunks
        // reproduces the response byte for byte.
        let chunks: Vec<String> = response
            .text
            .split_inclusive(' ')
            .map(ToOwned::to_owned)
            .collect();
        let last = chunks.len().saturating_sub(1);
        Ok(Box::pin(futures::stream::iter(
            chunks.into_iter().enumerate().map(move |(index, text)| {
                Ok(TokenChunk {
                    text,
                    last: index == last,
                })
            }),
        )))
    }

    async fn health_check(&self) -> Result<ProviderHealth> {
        Ok(ProviderHealth {
            provider: self.name.clone(),
            healthy: self.healthy.load(Ordering::Relaxed),
            latency_ms: self.latency.as_millis() as u64,
            circuit_open: false,
            detail: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use futures::StreamExt;

    fn request(prompt: &str) -> CompletionRequest {
        CompletionRequest::new("mock-small", vec![Message::user(prompt)])
    }

    #[tokio::test]
    async fn the_same_prompt_always_produces_the_same_answer() {
        let provider = MockProvider::new("mock");
        let first = provider
            .complete(request("Task: explain Raft"))
            .await
            .unwrap();
        let second = provider
            .complete(request("Task: explain Raft"))
            .await
            .unwrap();
        assert_eq!(first.text, second.text);

        let other = provider
            .complete(request("Task: explain Paxos"))
            .await
            .unwrap();
        assert_ne!(first.text, other.text);
    }

    #[tokio::test]
    async fn json_mode_returns_the_keys_agents_validate_against() {
        let provider = MockProvider::new("mock");
        let response = provider
            .complete(request("Task: explain Raft").json())
            .await
            .unwrap();

        let value = response.parse_json().unwrap();
        for key in [
            "summary",
            "findings",
            "confidence",
            "evidence",
            "reasoning_summary",
        ] {
            assert!(value.get(key).is_some(), "missing key `{key}`");
        }
        assert!(value["findings"].as_array().unwrap().len() >= 3);
        let confidence = value["confidence"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&confidence));
    }

    #[tokio::test]
    async fn answers_mention_the_task_so_content_checks_can_pass() {
        let provider = MockProvider::new("mock");
        let response = provider
            .complete(request(
                "Task: compare Raft and Paxos\nInstruction: be brief",
            ))
            .await
            .unwrap();
        assert!(response.text.contains("compare Raft and Paxos"));
    }

    #[tokio::test]
    async fn scripted_answers_win_over_synthesis() {
        let provider = MockProvider::new("mock").scripted("capital of France", "Paris");
        let response = provider
            .complete(request("What is the capital of France?"))
            .await
            .unwrap();
        assert_eq!(response.text, "Paris");
    }

    #[tokio::test]
    async fn token_and_cost_accounting_follows_the_price_list() {
        let provider = MockProvider::new("mock").with_pricing(ModelPricing {
            input_per_million: 1_000_000.0,
            output_per_million: 2_000_000.0,
        });
        let response = provider.complete(request("one two three")).await.unwrap();

        assert_eq!(response.tokens_in, 3);
        assert!(response.tokens_out > 0);
        let expected = response.tokens_in as f64 + response.tokens_out as f64 * 2.0;
        assert!((response.cost_usd - expected).abs() < 1e-6);
    }

    #[tokio::test]
    async fn injected_failures_happen_exactly_when_asked() {
        let recovering = MockProvider::new("mock").failing_first(2);
        assert!(recovering.complete(request("x")).await.is_err());
        assert!(recovering.complete(request("x")).await.is_err());
        assert!(recovering.complete(request("x")).await.is_ok());
        assert_eq!(recovering.call_count(), 3);
        assert_eq!(recovering.failure_count(), 2);

        let flaky = MockProvider::new("mock").failing_every(3);
        assert!(flaky.complete(request("x")).await.is_ok());
        assert!(flaky.complete(request("x")).await.is_ok());
        assert!(flaky.complete(request("x")).await.is_err());

        let broken = MockProvider::new("mock").always_failing();
        assert!(broken.complete(request("x")).await.is_err());
        assert!(!broken.health_check().await.unwrap().healthy);
    }

    #[tokio::test]
    async fn streaming_reassembles_into_the_same_answer() {
        let provider = MockProvider::new("mock");
        let complete = provider
            .complete(request("Task: explain Raft"))
            .await
            .unwrap();

        let mut stream = provider
            .stream(request("Task: explain Raft"))
            .await
            .unwrap();
        let mut assembled = String::new();
        let mut saw_last = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            assembled.push_str(&chunk.text);
            saw_last = chunk.last;
        }

        assert_eq!(assembled, complete.text);
        assert!(saw_last, "the final chunk must be flagged");
    }

    #[test]
    fn topics_come_from_the_task_line_when_there_is_one() {
        assert_eq!(topic_of("Task: explain Raft\nother"), "explain Raft");
        assert_eq!(topic_of("just a prompt"), "just a prompt");
        assert_eq!(topic_of(""), "the objective");
        assert_eq!(topic_of("Task:   "), "the objective");
        assert_eq!(
            topic_of(&format!("Task: {}", "word ".repeat(50)))
                .split_whitespace()
                .count(),
            12
        );
    }
}
