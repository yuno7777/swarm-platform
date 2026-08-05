//! The gateway: everything that is true of every provider.
//!
//! Routing, ordered fallback, retry with backoff, circuit breaking, global and
//! per-model concurrency limits, response caching, budget enforcement, and token/cost
//! accounting. Providers stay dumb; this is where the operational behaviour lives.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use swarm_domain::{Result, SwarmError};

use crate::{CompletionRequest, CompletionResponse, ModelProvider, ProviderHealth, TokenStream};

/// Operational limits applied to every model call.
///
/// `#[serde(default)]` so a configuration file may set only the fields it cares about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GatewayConfig {
    /// Model calls in flight across the whole process.
    pub max_concurrent: usize,
    /// Whether identical requests may be served from cache.
    pub cache_enabled: bool,
    /// Cached responses retained before the oldest are dropped.
    pub cache_capacity: usize,
    /// Attempts per provider, including the first.
    pub max_attempts: u32,
    /// Base backoff between attempts, doubled each time.
    pub retry_backoff_ms: u64,
    /// Consecutive failures before a provider's breaker opens.
    pub circuit_failure_threshold: u32,
    /// How long a breaker stays open before a probe is allowed through.
    pub circuit_reset_ms: u64,
    /// Total spend ceiling for this gateway, in USD.
    pub budget_usd: Option<f64>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 32,
            cache_enabled: true,
            cache_capacity: 4_096,
            max_attempts: 3,
            retry_backoff_ms: 50,
            circuit_failure_threshold: 5,
            circuit_reset_ms: 5_000,
            budget_usd: None,
        }
    }
}

/// Running totals for everything the gateway has spent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Calls that reached a provider.
    pub requests: u64,
    /// Calls served from cache.
    pub cache_hits: u64,
    /// Provider calls that returned an error.
    pub failures: u64,
    /// Times the gateway moved on to a lower-priority provider.
    pub fallbacks: u64,
    /// Times a call was refused because a breaker was open.
    pub circuit_rejections: u64,
    /// Prompt tokens.
    pub tokens_in: u64,
    /// Completion tokens.
    pub tokens_out: u64,
    /// Estimated spend in USD.
    pub cost_usd: f64,
}

/// Per-provider failure tracking.
#[derive(Debug)]
struct Breaker {
    consecutive_failures: AtomicU32,
    opened_at: Mutex<Option<Instant>>,
}

impl Breaker {
    fn new() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            opened_at: Mutex::new(None),
        }
    }
}

/// Routes model calls to providers and enforces the platform's operational limits.
pub struct Gateway {
    config: GatewayConfig,
    providers: Vec<Arc<dyn ModelProvider>>,
    permits: Arc<Semaphore>,
    model_permits: DashMap<String, Arc<Semaphore>>,
    model_limits: HashMap<String, usize>,
    cache: DashMap<u64, CompletionResponse>,
    cache_order: Mutex<Vec<u64>>,
    breakers: DashMap<String, Arc<Breaker>>,
    usage: Mutex<Usage>,
}

impl std::fmt::Debug for Gateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gateway")
            .field("config", &self.config)
            .field(
                "providers",
                &self
                    .providers
                    .iter()
                    .map(|provider| provider.name())
                    .collect::<Vec<_>>(),
            )
            .field("cached_responses", &self.cache.len())
            .finish()
    }
}

impl Gateway {
    /// A gateway with no providers yet.
    #[must_use]
    pub fn new(config: GatewayConfig) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(config.max_concurrent.max(1))),
            config,
            providers: Vec::new(),
            model_permits: DashMap::new(),
            model_limits: HashMap::new(),
            cache: DashMap::new(),
            cache_order: Mutex::new(Vec::new()),
            breakers: DashMap::new(),
            usage: Mutex::new(Usage::default()),
        }
    }

    /// A gateway with default limits and a single provider.
    #[must_use]
    pub fn with_provider(provider: Arc<dyn ModelProvider>) -> Self {
        Self::new(GatewayConfig::default()).and_provider(provider)
    }

    /// Append a provider. Order is fallback order: earlier providers are preferred.
    #[must_use]
    pub fn and_provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.breakers
            .insert(provider.name().to_owned(), Arc::new(Breaker::new()));
        self.providers.push(provider);
        self
    }

    /// Cap concurrent calls to one specific model.
    #[must_use]
    pub fn and_model_limit(mut self, model: impl Into<String>, limit: usize) -> Self {
        self.model_limits.insert(model.into(), limit.max(1));
        self
    }

    /// Names of the registered providers, in fallback order.
    #[must_use]
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers
            .iter()
            .map(|provider| provider.name())
            .collect()
    }

    /// A snapshot of spend so far.
    #[must_use]
    pub fn usage(&self) -> Usage {
        *lock(&self.usage)
    }

    /// Zero the counters, for benchmarking between phases of a run.
    pub fn reset_usage(&self) {
        *lock(&self.usage) = Usage::default();
    }

    /// Remaining budget in USD, if one is configured.
    #[must_use]
    pub fn remaining_budget(&self) -> Option<f64> {
        self.config
            .budget_usd
            .map(|budget| (budget - lock(&self.usage).cost_usd).max(0.0))
    }

    /// Probe every provider.
    pub async fn health(&self) -> Vec<ProviderHealth> {
        let mut report = Vec::with_capacity(self.providers.len());
        for provider in &self.providers {
            let circuit_open = self.circuit_is_open(provider.name());
            let health = provider.health_check().await.unwrap_or(ProviderHealth {
                provider: provider.name().to_owned(),
                healthy: false,
                latency_ms: 0,
                circuit_open,
                detail: Some("health check failed".to_owned()),
            });
            report.push(ProviderHealth {
                circuit_open,
                ..health
            });
        }
        report
    }

    /// Run a completion, applying every operational rule.
    ///
    /// Tries providers in registration order, skipping any whose breaker is open, and
    /// retrying transient failures within each provider before falling back.
    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        if self.providers.is_empty() {
            return Err(SwarmError::Config("gateway has no providers".into()));
        }

        let cache_key = request.cache_key();
        if self.config.cache_enabled && !request.bypass_cache {
            if let Some(hit) = self.cache.get(&cache_key) {
                let mut usage = lock(&self.usage);
                usage.cache_hits += 1;
                return Ok(CompletionResponse {
                    cached: true,
                    ..hit.clone()
                });
            }
        }

        // Budget is checked before the call, not after, so an overspend is refused
        // rather than merely reported.
        if let Some(budget) = self.config.budget_usd {
            let spent = lock(&self.usage).cost_usd;
            if spent >= budget {
                return Err(SwarmError::BudgetExceeded(format!(
                    "gateway has spent ${spent:.4} of its ${budget:.4} budget"
                )));
            }
        }

        let candidates: Vec<&Arc<dyn ModelProvider>> = self
            .providers
            .iter()
            .filter(|provider| provider.supports_model(&request.model))
            .collect();
        if candidates.is_empty() {
            return Err(SwarmError::Config(format!(
                "no provider serves model `{}`",
                request.model
            )));
        }

        let mut last_error = None;
        for (index, provider) in candidates.iter().enumerate() {
            if self.circuit_is_open(provider.name()) {
                lock(&self.usage).circuit_rejections += 1;
                last_error = Some(SwarmError::CircuitOpen {
                    provider: provider.name().to_owned(),
                });
                continue;
            }
            if index > 0 {
                lock(&self.usage).fallbacks += 1;
            }

            match self.call_with_retries(provider.as_ref(), &request).await {
                Ok(response) => {
                    self.record_success(provider.name(), &response, cache_key);
                    return Ok(response);
                }
                Err(err) => {
                    tracing::warn!(
                        provider = provider.name(),
                        error = %err,
                        "provider failed, trying next"
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| SwarmError::Provider {
            provider: "gateway".to_owned(),
            detail: "no provider was able to serve the request".to_owned(),
        }))
    }

    /// Stream a completion from the first provider whose breaker is closed.
    ///
    /// Streaming responses are not cached: a partial stream is not a usable cache
    /// entry, and buffering the whole thing would defeat the point of streaming.
    pub async fn stream(&self, request: CompletionRequest) -> Result<TokenStream> {
        for provider in &self.providers {
            if self.circuit_is_open(provider.name()) || !provider.supports_model(&request.model) {
                continue;
            }
            match provider.stream(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(err) => {
                    self.record_failure(provider.name());
                    tracing::warn!(provider = provider.name(), error = %err, "stream failed");
                }
            }
        }
        Err(SwarmError::Provider {
            provider: "gateway".to_owned(),
            detail: "no provider could open a stream".to_owned(),
        })
    }

    async fn call_with_retries(
        &self,
        provider: &dyn ModelProvider,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let result = self.call_once(provider, request).await;
            match result {
                Ok(response) => return Ok(response),
                Err(err) => {
                    self.record_failure(provider.name());
                    let exhausted = attempt >= self.config.max_attempts;
                    if exhausted || !err.is_retryable() {
                        return Err(err);
                    }
                    let backoff = self
                        .config
                        .retry_backoff_ms
                        .saturating_mul(1 << (attempt - 1).min(10));
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
    }

    async fn call_once(
        &self,
        provider: &dyn ModelProvider,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse> {
        let _global = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| SwarmError::Internal("gateway concurrency limiter closed".into()))?;

        let _per_model =
            match self.model_semaphore(&request.model) {
                Some(semaphore) => Some(semaphore.acquire_owned().await.map_err(|_| {
                    SwarmError::Internal("model concurrency limiter closed".into())
                })?),
                None => None,
            };

        lock(&self.usage).requests += 1;
        provider.complete(request.clone()).await
    }

    fn model_semaphore(&self, model: &str) -> Option<Arc<Semaphore>> {
        let limit = *self.model_limits.get(model)?;
        Some(
            self.model_permits
                .entry(model.to_owned())
                .or_insert_with(|| Arc::new(Semaphore::new(limit)))
                .clone(),
        )
    }

    fn circuit_is_open(&self, provider: &str) -> bool {
        let Some(breaker) = self.breakers.get(provider) else {
            return false;
        };
        let mut opened_at = lock(&breaker.opened_at);
        match *opened_at {
            Some(since)
                if since.elapsed() < Duration::from_millis(self.config.circuit_reset_ms) =>
            {
                true
            }
            Some(_) => {
                // Reset elapsed: let one call through to see whether it recovered.
                *opened_at = None;
                breaker.consecutive_failures.store(0, Ordering::Relaxed);
                false
            }
            None => false,
        }
    }

    fn record_failure(&self, provider: &str) {
        lock(&self.usage).failures += 1;
        let Some(breaker) = self.breakers.get(provider) else {
            return;
        };
        let failures = breaker.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if failures >= self.config.circuit_failure_threshold {
            let mut opened_at = lock(&breaker.opened_at);
            if opened_at.is_none() {
                tracing::warn!(provider, failures, "opening circuit breaker");
                *opened_at = Some(Instant::now());
            }
        }
    }

    fn record_success(&self, provider: &str, response: &CompletionResponse, cache_key: u64) {
        if let Some(breaker) = self.breakers.get(provider) {
            breaker.consecutive_failures.store(0, Ordering::Relaxed);
            *lock(&breaker.opened_at) = None;
        }

        {
            let mut usage = lock(&self.usage);
            usage.tokens_in += response.tokens_in;
            usage.tokens_out += response.tokens_out;
            usage.cost_usd += response.cost_usd;
        }

        if self.config.cache_enabled {
            self.cache.insert(cache_key, response.clone());
            // ponytail: insertion-order eviction, not LRU. Upgrade if the hit rate on
            // long jobs turns out to matter more than the bookkeeping cost.
            let mut order = lock(&self.cache_order);
            order.push(cache_key);
            while order.len() > self.config.cache_capacity {
                let oldest = order.remove(0);
                self.cache.remove(&oldest);
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, MockProvider};
    use futures::StreamExt;

    fn request(prompt: &str) -> CompletionRequest {
        CompletionRequest::new("mock-small", vec![Message::user(prompt)])
    }

    fn config() -> GatewayConfig {
        GatewayConfig {
            retry_backoff_ms: 0,
            ..GatewayConfig::default()
        }
    }

    #[tokio::test]
    async fn a_gateway_without_providers_says_so() {
        let gateway = Gateway::new(config());
        assert!(matches!(
            gateway.complete(request("x")).await.unwrap_err(),
            SwarmError::Config(_)
        ));
    }

    #[tokio::test]
    async fn a_dead_primary_falls_back_to_the_secondary() {
        let primary = Arc::new(MockProvider::new("primary").always_failing());
        let secondary = Arc::new(MockProvider::new("secondary"));
        let gateway = Gateway::new(config())
            .and_provider(primary.clone())
            .and_provider(secondary.clone());

        let response = gateway.complete(request("Task: anything")).await.unwrap();
        assert_eq!(response.provider, "secondary");
        assert_eq!(secondary.call_count(), 1);
        assert!(primary.call_count() >= 1);
        assert_eq!(gateway.usage().fallbacks, 1);
    }

    #[tokio::test]
    async fn transient_failures_are_retried_inside_one_provider() {
        let flaky = Arc::new(MockProvider::new("flaky").failing_first(2));
        let gateway = Gateway::new(config()).and_provider(flaky.clone());

        let response = gateway.complete(request("Task: retry me")).await.unwrap();
        assert_eq!(response.provider, "flaky");
        assert_eq!(flaky.call_count(), 3, "two failures then a success");
        assert_eq!(gateway.usage().failures, 2);
        assert_eq!(gateway.usage().fallbacks, 0);
    }

    #[tokio::test]
    async fn retries_give_up_after_the_configured_budget() {
        let broken = Arc::new(MockProvider::new("broken").always_failing());
        let gateway = Gateway::new(GatewayConfig {
            max_attempts: 2,
            ..config()
        })
        .and_provider(broken.clone());

        assert!(gateway.complete(request("x")).await.is_err());
        assert_eq!(broken.call_count(), 2);
    }

    #[tokio::test]
    async fn identical_requests_are_served_from_cache() {
        let provider = Arc::new(MockProvider::new("mock"));
        let gateway = Gateway::new(config()).and_provider(provider.clone());

        let first = gateway.complete(request("Task: cache me")).await.unwrap();
        assert!(!first.cached);

        let second = gateway.complete(request("Task: cache me")).await.unwrap();
        assert!(second.cached);
        assert_eq!(second.text, first.text);
        assert_eq!(
            provider.call_count(),
            1,
            "the cache must prevent the second call"
        );

        let usage = gateway.usage();
        assert_eq!(usage.cache_hits, 1);
        assert_eq!(usage.requests, 1);
        // A cache hit is free: it must not be billed twice.
        assert!((usage.cost_usd - first.cost_usd).abs() < 1e-9);
    }

    #[tokio::test]
    async fn a_caller_can_bypass_a_cached_answer_it_knows_is_unusable() {
        let provider = Arc::new(MockProvider::new("mock"));
        let gateway = Gateway::new(config()).and_provider(provider.clone());

        gateway.complete(request("Task: retry me")).await.unwrap();
        let retried = gateway
            .complete(request("Task: retry me").bypassing_cache(true))
            .await
            .unwrap();

        assert!(!retried.cached);
        assert_eq!(
            provider.call_count(),
            2,
            "the bypass must reach the provider"
        );
    }

    #[tokio::test]
    async fn caching_can_be_turned_off() {
        let provider = Arc::new(MockProvider::new("mock"));
        let gateway = Gateway::new(GatewayConfig {
            cache_enabled: false,
            ..config()
        })
        .and_provider(provider.clone());

        gateway.complete(request("Task: x")).await.unwrap();
        gateway.complete(request("Task: x")).await.unwrap();
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn the_cache_stays_bounded() {
        let provider = Arc::new(MockProvider::new("mock"));
        let gateway = Gateway::new(GatewayConfig {
            cache_capacity: 4,
            ..config()
        })
        .and_provider(provider);

        for index in 0..20 {
            gateway
                .complete(request(&format!("Task: prompt {index}")))
                .await
                .unwrap();
        }
        assert!(
            gateway.cache.len() <= 4,
            "cache must not grow without bound"
        );
    }

    #[tokio::test]
    async fn token_and_cost_totals_accumulate_across_calls() {
        let provider = Arc::new(MockProvider::new("mock"));
        let gateway = Gateway::new(config()).and_provider(provider);

        let first = gateway.complete(request("Task: one")).await.unwrap();
        let second = gateway.complete(request("Task: two")).await.unwrap();

        let usage = gateway.usage();
        assert_eq!(usage.requests, 2);
        assert_eq!(usage.tokens_in, first.tokens_in + second.tokens_in);
        assert_eq!(usage.tokens_out, first.tokens_out + second.tokens_out);
        assert!((usage.cost_usd - (first.cost_usd + second.cost_usd)).abs() < 1e-9);

        gateway.reset_usage();
        assert_eq!(gateway.usage(), Usage::default());
    }

    #[tokio::test]
    async fn a_failing_provider_gets_its_breaker_opened_and_is_then_skipped() {
        let broken = Arc::new(MockProvider::new("broken").always_failing());
        let healthy = Arc::new(MockProvider::new("healthy"));
        let gateway = Gateway::new(GatewayConfig {
            max_attempts: 1,
            circuit_failure_threshold: 2,
            circuit_reset_ms: 60_000,
            // Without this the cache would answer calls 2 and 3 and the breaker would
            // never see enough traffic to trip.
            cache_enabled: false,
            ..config()
        })
        .and_provider(broken.clone())
        .and_provider(healthy.clone());

        for _ in 0..3 {
            gateway.complete(request("Task: x")).await.unwrap();
        }

        assert_eq!(
            broken.call_count(),
            2,
            "after two failures the breaker must stop further calls"
        );
        assert_eq!(healthy.call_count(), 3);
        assert!(gateway.usage().circuit_rejections >= 1);

        let health = gateway.health().await;
        assert!(health
            .iter()
            .any(|h| h.provider == "broken" && h.circuit_open));
    }

    #[tokio::test]
    async fn a_breaker_reopens_the_path_once_the_reset_window_passes() {
        let recovering = Arc::new(MockProvider::new("recovering").failing_first(2));
        let gateway = Gateway::new(GatewayConfig {
            max_attempts: 1,
            circuit_failure_threshold: 2,
            circuit_reset_ms: 1,
            ..config()
        })
        .and_provider(recovering.clone());

        assert!(gateway.complete(request("Task: x")).await.is_err());
        assert!(gateway.complete(request("Task: x")).await.is_err());

        tokio::time::sleep(Duration::from_millis(5)).await;
        let response = gateway.complete(request("Task: x")).await.unwrap();
        assert_eq!(response.provider, "recovering");
    }

    #[tokio::test]
    async fn spending_past_the_budget_is_refused_rather_than_reported() {
        let provider = Arc::new(MockProvider::new("mock").with_pricing(crate::ModelPricing {
            input_per_million: 1_000_000.0,
            output_per_million: 1_000_000.0,
        }));
        let gateway = Gateway::new(GatewayConfig {
            budget_usd: Some(10.0),
            cache_enabled: false,
            ..config()
        })
        .and_provider(provider);

        // Each call costs (tokens_in + tokens_out) dollars at this absurd price.
        let mut refused = false;
        for index in 0..10 {
            if let Err(err) = gateway.complete(request(&format!("Task: {index}"))).await {
                assert!(matches!(err, SwarmError::BudgetExceeded(_)));
                refused = true;
                break;
            }
        }
        assert!(refused, "the budget must eventually stop the gateway");
        assert_eq!(gateway.remaining_budget(), Some(0.0));
    }

    #[tokio::test]
    async fn model_routing_skips_providers_that_do_not_serve_the_model() {
        struct OnlyBig;

        #[async_trait::async_trait]
        impl ModelProvider for OnlyBig {
            fn name(&self) -> &str {
                "only-big"
            }
            async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse> {
                Err(SwarmError::Internal("should not be called".into()))
            }
            async fn stream(&self, _: CompletionRequest) -> Result<TokenStream> {
                Err(SwarmError::Internal("should not be called".into()))
            }
            async fn health_check(&self) -> Result<ProviderHealth> {
                Err(SwarmError::Internal("no".into()))
            }
            fn supports_model(&self, model: &str) -> bool {
                model == "big"
            }
        }

        let gateway = Gateway::new(config())
            .and_provider(Arc::new(OnlyBig))
            .and_provider(Arc::new(MockProvider::new("mock")));

        let response = gateway.complete(request("Task: x")).await.unwrap();
        assert_eq!(response.provider, "mock");

        let unserved = CompletionRequest::new("enormous", vec![Message::user("x")]);
        let gateway = Gateway::new(config()).and_provider(Arc::new(OnlyBig));
        assert!(matches!(
            gateway.complete(unserved).await.unwrap_err(),
            SwarmError::Config(_)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrency_limits_are_honoured() {
        let provider = Arc::new(MockProvider::new("slow").with_latency(Duration::from_millis(20)));
        let gateway = Arc::new(
            Gateway::new(GatewayConfig {
                max_concurrent: 2,
                cache_enabled: false,
                ..config()
            })
            .and_provider(provider),
        );

        let started = Instant::now();
        let mut calls = Vec::new();
        for index in 0..6 {
            let gateway = Arc::clone(&gateway);
            calls.push(tokio::spawn(async move {
                gateway
                    .complete(request(&format!("Task: {index}")))
                    .await
                    .unwrap()
            }));
        }
        for call in calls {
            call.await.unwrap();
        }

        // Six 20ms calls, two at a time, cannot finish in under three windows.
        assert!(
            started.elapsed() >= Duration::from_millis(55),
            "six calls at concurrency 2 finished too fast: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn streaming_falls_back_like_completion_does() {
        let gateway = Gateway::new(config())
            .and_provider(Arc::new(MockProvider::new("dead").always_failing()))
            .and_provider(Arc::new(MockProvider::new("live")));

        let mut stream = gateway.stream(request("Task: stream")).await.unwrap();
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            text.push_str(&chunk.unwrap().text);
        }
        assert!(text.contains("stream"));
    }
}
