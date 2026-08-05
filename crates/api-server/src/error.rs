//! Mapping platform errors onto HTTP.
//!
//! Every failure the API can produce is a [`SwarmError`], so the mapping lives in one
//! place: a caller can tell a bad request from an over-quota one from a provider
//! outage by status code alone, and gets the machine-readable `kind` either way.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use swarm_domain::SwarmError;

/// An error rendered as an HTTP response.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    /// A `400 Bad Request` with a custom message.
    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "bad_request",
            message: message.into(),
        }
    }

    /// A `404 Not Found` for a named resource.
    #[must_use]
    pub fn not_found(kind: &'static str, id: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "not_found",
            message: format!("unknown {kind} `{id}`"),
        }
    }

    /// The status this error will be served with.
    #[must_use]
    pub const fn status(&self) -> StatusCode {
        self.status
    }
}

impl From<SwarmError> for ApiError {
    fn from(error: SwarmError) -> Self {
        let status = match &error {
            SwarmError::NotFound { .. } => StatusCode::NOT_FOUND,
            SwarmError::Validation(_)
            | SwarmError::InvalidId { .. }
            | SwarmError::UnknownDependency { .. }
            | SwarmError::CyclicGraph { .. }
            | SwarmError::InvalidTransition { .. } => StatusCode::BAD_REQUEST,
            SwarmError::QuotaExceeded(_) | SwarmError::RateLimited { .. } => {
                StatusCode::TOO_MANY_REQUESTS
            }
            // The request was well-formed and permitted; it is the budget that ran out.
            SwarmError::BudgetExceeded(_) => StatusCode::PAYMENT_REQUIRED,
            SwarmError::Cancelled(_) | SwarmError::VersionConflict { .. } => StatusCode::CONFLICT,
            SwarmError::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            SwarmError::Provider { .. } | SwarmError::CircuitOpen { .. } => StatusCode::BAD_GATEWAY,
            SwarmError::Queue(_)
            | SwarmError::Memory(_)
            | SwarmError::Config(_)
            | SwarmError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            // The error type is non_exhaustive: an unmapped variant is a server fault,
            // not a client one.
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        Self {
            status,
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::error!(kind = self.kind, message = %self.message, "request failed");
        }
        (
            self.status,
            Json(json!({
                "error": {
                    "kind": self.kind,
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_faults_and_server_faults_are_told_apart() {
        let cases = [
            (
                SwarmError::NotFound {
                    kind: "job",
                    id: "x".into(),
                },
                StatusCode::NOT_FOUND,
            ),
            (
                SwarmError::Validation("objective must not be empty".into()),
                StatusCode::BAD_REQUEST,
            ),
            (
                SwarmError::QuotaExceeded("too many agents".into()),
                StatusCode::TOO_MANY_REQUESTS,
            ),
            (
                SwarmError::BudgetExceeded("over $5".into()),
                StatusCode::PAYMENT_REQUIRED,
            ),
            (
                SwarmError::Provider {
                    provider: "mock".into(),
                    detail: "503".into(),
                },
                StatusCode::BAD_GATEWAY,
            ),
            (
                SwarmError::Internal("bug".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];

        for (error, expected) in cases {
            let api_error = ApiError::from(error);
            assert_eq!(api_error.status(), expected, "{}", api_error.message);
        }
    }

    #[test]
    fn the_machine_readable_kind_survives_the_conversion() {
        let error = ApiError::from(SwarmError::Timeout { millis: 10 });
        assert_eq!(error.kind, "timeout");
        assert_eq!(error.status(), StatusCode::GATEWAY_TIMEOUT);
    }
}
