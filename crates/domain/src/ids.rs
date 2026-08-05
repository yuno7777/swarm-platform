//! Globally unique, time-ordered identifiers.
//!
//! All ids wrap a UUIDv7, so they sort by creation time — which makes them usable
//! directly as database primary keys without a separate `created_at` index, and makes
//! log output roughly chronological.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, SwarmError};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Generate a fresh time-ordered identifier.
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }

            /// Wrap an existing UUID (for rows loaded from the database).
            #[must_use]
            pub const fn from_uuid(id: Uuid) -> Self {
                Self(id)
            }

            /// The underlying UUID.
            #[must_use]
            pub const fn as_uuid(&self) -> &Uuid {
                &self.0
            }

            /// Operator-facing form with a type prefix, e.g. `task_0190f3…`.
            #[must_use]
            pub fn prefixed(&self) -> String {
                format!("{}_{}", $prefix, self.0.as_simple())
            }

            /// The type prefix used by [`Self::prefixed`].
            #[must_use]
            pub const fn prefix() -> &'static str {
                $prefix
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl FromStr for $name {
            type Err = SwarmError;

            /// Accepts both the plain UUID form and the prefixed form.
            fn from_str(s: &str) -> Result<Self> {
                let raw = s.strip_prefix(concat!($prefix, "_")).unwrap_or(s);
                Uuid::parse_str(raw).map(Self).map_err(|e| SwarmError::InvalidId {
                    value: s.to_owned(),
                    detail: e.to_string(),
                })
            }
        }

        impl From<Uuid> for $name {
            fn from(id: Uuid) -> Self {
                Self(id)
            }
        }

        impl From<$name> for Uuid {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

define_id!(
    /// Identifies a submitted job.
    JobId,
    "job"
);
define_id!(
    /// Identifies a task inside a job's DAG.
    TaskId,
    "task"
);
define_id!(
    /// Identifies an agent instance.
    AgentId,
    "agent"
);
define_id!(
    /// Identifies a worker node.
    NodeId,
    "node"
);
define_id!(
    /// Identifies a task lease held by a worker.
    LeaseId,
    "lease"
);
define_id!(
    /// Identifies a protocol message.
    MessageId,
    "msg"
);
define_id!(
    /// Identifies a shared-memory record.
    MemoryId,
    "mem"
);
define_id!(
    /// Identifies one execution attempt of a task.
    AttemptId,
    "attempt"
);
define_id!(
    /// Ties together every action caused by one logical operation.
    CorrelationId,
    "corr"
);
define_id!(
    /// Identifies a consensus round.
    RoundId,
    "round"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_roundtrip_through_plain_and_prefixed_forms() {
        let id = TaskId::new();
        assert_eq!(TaskId::from_str(&id.to_string()).unwrap(), id);
        assert_eq!(TaskId::from_str(&id.prefixed()).unwrap(), id);
        assert!(id.prefixed().starts_with("task_"));
    }

    #[test]
    fn ids_roundtrip_through_serde_as_bare_strings() {
        let id = JobId::new();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{id}\""));
        assert_eq!(serde_json::from_str::<JobId>(&json).unwrap(), id);
    }

    #[test]
    fn garbage_is_rejected_with_the_offending_value() {
        let err = AgentId::from_str("not-a-uuid").unwrap_err();
        assert!(matches!(err, SwarmError::InvalidId { ref value, .. } if value == "not-a-uuid"));
    }

    #[test]
    fn ids_are_unique() {
        let ids: std::collections::HashSet<JobId> = (0..10_000).map(|_| JobId::new()).collect();
        assert_eq!(ids.len(), 10_000);
    }

    #[test]
    fn ids_sort_by_creation_time() {
        // UUIDv7 encodes a millisecond timestamp in its high bits, so ids created in
        // different milliseconds always sort in creation order. This is what lets us
        // use them as primary keys without a separate created_at index.
        let earlier = TaskId::new();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let later = TaskId::new();
        assert!(later > earlier, "{later} should sort after {earlier}");
    }
}
