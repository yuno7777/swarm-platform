//! In-process implementation of [`MemoryStore`].
//!
//! Backed by [`dashmap`], so concurrent agents contend per shard rather than on one
//! global lock. Compare-and-swap is made atomic by holding the shard's entry guard for
//! the whole read-modify-write.

use chrono::{DateTime, Utc};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;

use async_trait::async_trait;

use swarm_domain::{JobId, MemoryId, Result, SwarmError};

use crate::{AuditEntry, MemoryOperation, MemoryRecord, MemoryStore, MemoryWrite, VERSION_ABSENT};

type RecordKey = (String, String);

/// A [`MemoryStore`] that keeps everything in process memory.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    records: DashMap<RecordKey, MemoryRecord>,
    history: DashMap<RecordKey, Vec<AuditEntry>>,
}

impl InMemoryStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live records, including expired ones not yet compacted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the store holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn note(&self, record: &MemoryRecord, operation: MemoryOperation, actor: &str) {
        self.history
            .entry((record.namespace.clone(), record.key.clone()))
            .or_default()
            .push(AuditEntry {
                namespace: record.namespace.clone(),
                key: record.key.clone(),
                version: record.version,
                operation,
                actor: actor.to_owned(),
                at: record.updated_at,
            });
    }
}

/// Build a fresh record at version 1.
fn fresh(write: &MemoryWrite, now: DateTime<Utc>) -> MemoryRecord {
    MemoryRecord {
        id: MemoryId::new(),
        job_id: write.job_id,
        agent_id: write.agent_id,
        namespace: write.namespace.clone(),
        key: write.key.clone(),
        value: write.value.clone(),
        version: 1,
        created_at: now,
        updated_at: now,
        expires_at: expiry(write, now),
    }
}

/// Apply a write on top of an existing record, bumping its version.
fn bumped(previous: &MemoryRecord, write: &MemoryWrite, now: DateTime<Utc>) -> MemoryRecord {
    MemoryRecord {
        id: previous.id,
        job_id: write.job_id.or(previous.job_id),
        agent_id: write.agent_id.or(previous.agent_id),
        namespace: previous.namespace.clone(),
        key: previous.key.clone(),
        value: write.value.clone(),
        version: previous.version + 1,
        created_at: previous.created_at,
        updated_at: now,
        expires_at: expiry(write, now).or(previous.expires_at),
    }
}

fn expiry(write: &MemoryWrite, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    write
        .ttl
        .and_then(|ttl| chrono::Duration::from_std(ttl).ok())
        .map(|ttl| now + ttl)
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryRecord>> {
        let now = Utc::now();
        Ok(self
            .records
            .get(&(namespace.to_owned(), key.to_owned()))
            .map(|entry| entry.value().clone())
            .filter(|record| !record.is_expired(now)))
    }

    async fn put(&self, write: MemoryWrite) -> Result<MemoryRecord> {
        let now = Utc::now();
        let (record, operation) = match self
            .records
            .entry((write.namespace.clone(), write.key.clone()))
        {
            Entry::Occupied(mut occupied) => {
                // An expired record is treated as absent: the writer starts a new
                // lineage at version 1 rather than inheriting a stale version.
                let record = if occupied.get().is_expired(now) {
                    fresh(&write, now)
                } else {
                    bumped(occupied.get(), &write, now)
                };
                let operation = if record.version == 1 {
                    MemoryOperation::Create
                } else {
                    MemoryOperation::Put
                };
                occupied.insert(record.clone());
                (record, operation)
            }
            Entry::Vacant(vacant) => {
                let record = fresh(&write, now);
                vacant.insert(record.clone());
                (record, MemoryOperation::Create)
            }
        };
        self.note(&record, operation, &write.actor);
        Ok(record)
    }

    async fn compare_and_swap(
        &self,
        write: MemoryWrite,
        expected_version: u64,
    ) -> Result<MemoryRecord> {
        let now = Utc::now();
        let conflict = |actual: u64| SwarmError::VersionConflict {
            namespace: write.namespace.clone(),
            key: write.key.clone(),
            expected: expected_version,
            actual,
        };

        let (record, operation) = match self
            .records
            .entry((write.namespace.clone(), write.key.clone()))
        {
            Entry::Occupied(mut occupied) => {
                let current_version = if occupied.get().is_expired(now) {
                    VERSION_ABSENT
                } else {
                    occupied.get().version
                };
                if current_version != expected_version {
                    return Err(conflict(current_version));
                }
                let record = if current_version == VERSION_ABSENT {
                    fresh(&write, now)
                } else {
                    bumped(occupied.get(), &write, now)
                };
                occupied.insert(record.clone());
                (record, MemoryOperation::CompareAndSwap)
            }
            Entry::Vacant(vacant) => {
                if expected_version != VERSION_ABSENT {
                    return Err(conflict(VERSION_ABSENT));
                }
                let record = fresh(&write, now);
                vacant.insert(record.clone());
                (record, MemoryOperation::Create)
            }
        };
        self.note(&record, operation, &write.actor);
        Ok(record)
    }

    async fn delete(&self, namespace: &str, key: &str) -> Result<bool> {
        let removed = self
            .records
            .remove(&(namespace.to_owned(), key.to_owned()))
            .map(|(_, record)| record);
        if let Some(record) = removed {
            self.note(&record, MemoryOperation::Delete, "unknown");
            return Ok(true);
        }
        Ok(false)
    }

    async fn list(&self, namespace: &str, prefix: &str) -> Result<Vec<MemoryRecord>> {
        let now = Utc::now();
        let mut records: Vec<MemoryRecord> = self
            .records
            .iter()
            .filter(|entry| {
                let record = entry.value();
                record.namespace == namespace
                    && record.key.starts_with(prefix)
                    && !record.is_expired(now)
            })
            .map(|entry| entry.value().clone())
            .collect();
        // Stable order so callers (and tests) see reproducible listings.
        records.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        Ok(records)
    }

    async fn audit(&self, namespace: &str, key: &str) -> Result<Vec<AuditEntry>> {
        Ok(self
            .history
            .get(&(namespace.to_owned(), key.to_owned()))
            .map(|entry| entry.value().clone())
            .unwrap_or_default())
    }

    async fn compact(&self) -> Result<usize> {
        let now = Utc::now();
        let expired: Vec<RecordKey> = self
            .records
            .iter()
            .filter(|entry| entry.value().is_expired(now))
            .map(|entry| entry.key().clone())
            .collect();

        for key in &expired {
            if let Some((_, record)) = self.records.remove(key) {
                self.note(&record, MemoryOperation::Expire, "system");
            }
        }
        Ok(expired.len())
    }

    async fn purge_job(&self, job_id: JobId) -> Result<usize> {
        let owned: Vec<RecordKey> = self
            .records
            .iter()
            .filter(|entry| entry.value().job_id == Some(job_id))
            .map(|entry| entry.key().clone())
            .collect();
        for key in &owned {
            self.records.remove(key);
        }
        Ok(owned.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns;
    use serde_json::json;
    use std::sync::Arc;
    use std::time::Duration;

    fn write(key: &str, value: serde_json::Value) -> MemoryWrite {
        MemoryWrite::new("test", key, value, "agent:test")
    }

    #[tokio::test]
    async fn writes_are_versioned_from_one_upwards() {
        let store = InMemoryStore::new();

        let first = store.put(write("plan", json!({"step": 1}))).await.unwrap();
        assert_eq!(first.version, 1);
        assert_eq!(first.created_at, first.updated_at);

        let second = store.put(write("plan", json!({"step": 2}))).await.unwrap();
        assert_eq!(second.version, 2);
        assert_eq!(second.id, first.id, "the record keeps its identity");
        assert_eq!(second.created_at, first.created_at);

        let read = store.get("test", "plan").await.unwrap().unwrap();
        assert_eq!(read.version, 2);
        assert_eq!(read.value, json!({"step": 2}));
    }

    #[tokio::test]
    async fn missing_keys_read_as_none() {
        let store = InMemoryStore::new();
        assert!(store.get("test", "absent").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn compare_and_swap_lets_exactly_one_writer_win() {
        let store = InMemoryStore::new();
        let base = store.put(write("shared", json!(0))).await.unwrap();

        let winner = store
            .compare_and_swap(write("shared", json!(1)), base.version)
            .await
            .unwrap();
        assert_eq!(winner.version, 2);

        // The loser held a stale version and is told so, with the real one attached.
        let err = store
            .compare_and_swap(write("shared", json!(2)), base.version)
            .await
            .unwrap_err();
        match err {
            SwarmError::VersionConflict {
                expected, actual, ..
            } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            other => panic!("expected VersionConflict, got {other:?}"),
        }

        assert_eq!(
            store.get("test", "shared").await.unwrap().unwrap().value,
            json!(1),
            "the losing write must not have landed"
        );
    }

    #[tokio::test]
    async fn claiming_a_key_succeeds_once_and_only_once() {
        // This is the mechanism that stops a duplicate task delivery from doing the
        // work twice: the second claimant is told the work is already owned.
        let store = InMemoryStore::new();

        let claimed = store
            .claim(write("task-42", json!("running")))
            .await
            .unwrap();
        assert!(claimed.is_some());
        assert_eq!(claimed.unwrap().version, 1);

        let second = store
            .claim(write("task-42", json!("running")))
            .await
            .unwrap();
        assert!(second.is_none(), "the second claim must lose quietly");
    }

    #[tokio::test]
    async fn compare_and_swap_against_a_missing_key_reports_version_zero() {
        let store = InMemoryStore::new();
        let err = store
            .compare_and_swap(write("ghost", json!(1)), 7)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            SwarmError::VersionConflict {
                actual: 0,
                expected: 7,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn expired_records_are_invisible_before_compaction_runs() {
        let store = InMemoryStore::new();
        store
            .put(write("ephemeral", json!("x")).expiring_in(Duration::from_millis(1)))
            .await
            .unwrap();
        store.put(write("durable", json!("y"))).await.unwrap();

        tokio::time::sleep(Duration::from_millis(5)).await;

        assert!(store.get("test", "ephemeral").await.unwrap().is_none());
        assert!(store.get("test", "durable").await.unwrap().is_some());
        assert_eq!(store.list("test", "").await.unwrap().len(), 1);

        assert_eq!(store.compact().await.unwrap(), 1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.compact().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn an_expired_key_can_be_claimed_again() {
        let store = InMemoryStore::new();
        store
            .put(write("lease", json!("held")).expiring_in(Duration::from_millis(1)))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;

        let reclaimed = store
            .claim(write("lease", json!("held again")))
            .await
            .unwrap();
        assert_eq!(reclaimed.unwrap().version, 1, "a new lineage starts at 1");
    }

    #[tokio::test]
    async fn listing_is_scoped_by_namespace_and_prefix() {
        let store = InMemoryStore::new();
        for key in ["result/a", "result/b", "checkpoint/a"] {
            store.put(write(key, json!(key))).await.unwrap();
        }
        store
            .put(MemoryWrite::new("other", "result/c", json!("c"), "test"))
            .await
            .unwrap();

        let results = store.list("test", "result/").await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "result/a");
        assert_eq!(results[1].key, "result/b");
        assert_eq!(store.list("test", "").await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn history_records_every_operation_with_its_actor() {
        let store = InMemoryStore::new();
        store
            .put(MemoryWrite::new("test", "k", json!(1), "agent:one"))
            .await
            .unwrap();
        store
            .compare_and_swap(MemoryWrite::new("test", "k", json!(2), "agent:two"), 1)
            .await
            .unwrap();
        store.delete("test", "k").await.unwrap();

        let history = store.audit("test", "k").await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].operation, MemoryOperation::Create);
        assert_eq!(history[0].actor, "agent:one");
        assert_eq!(history[1].operation, MemoryOperation::CompareAndSwap);
        assert_eq!(history[1].actor, "agent:two");
        assert_eq!(history[1].version, 2);
        assert_eq!(history[2].operation, MemoryOperation::Delete);
    }

    #[tokio::test]
    async fn deleting_reports_whether_anything_was_there() {
        let store = InMemoryStore::new();
        store.put(write("k", json!(1))).await.unwrap();
        assert!(store.delete("test", "k").await.unwrap());
        assert!(!store.delete("test", "k").await.unwrap());
    }

    #[tokio::test]
    async fn purging_a_job_leaves_other_jobs_alone() {
        let store = InMemoryStore::new();
        let doomed = JobId::new();
        let survivor = JobId::new();

        store
            .put(MemoryWrite::new(ns::job(doomed), "a", json!(1), "t").for_job(doomed))
            .await
            .unwrap();
        store
            .put(MemoryWrite::new(ns::job(doomed), "b", json!(2), "t").for_job(doomed))
            .await
            .unwrap();
        store
            .put(MemoryWrite::new(ns::job(survivor), "c", json!(3), "t").for_job(survivor))
            .await
            .unwrap();

        assert_eq!(store.purge_job(doomed).await.unwrap(), 2);
        assert_eq!(store.len(), 1);
        assert!(store.get(&ns::job(survivor), "c").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn records_deserialize_into_domain_types() {
        let store = InMemoryStore::new();
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct Finding {
            claim: String,
            score: u8,
        }
        let finding = Finding {
            claim: "raft elects one leader per term".into(),
            score: 9,
        };
        store
            .put(MemoryWrite::json("test", "finding", &finding, "agent").unwrap())
            .await
            .unwrap();

        let record = store.get("test", "finding").await.unwrap().unwrap();
        assert_eq!(record.parse::<Finding>().unwrap(), finding);
        assert!(record.parse::<Vec<u8>>().is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_hundred_concurrent_writers_lose_no_updates() {
        let store = Arc::new(InMemoryStore::new());
        store.put(write("counter", json!(0))).await.unwrap();

        let mut writers = Vec::new();
        for id in 0..100 {
            let store = Arc::clone(&store);
            writers.push(tokio::spawn(async move {
                // The retry loop every agent is expected to run on conflict.
                let mut conflicts = 0;
                loop {
                    let current = store.get("test", "counter").await.unwrap().unwrap();
                    let next = current.value.as_i64().unwrap() + 1;
                    let write =
                        MemoryWrite::new("test", "counter", json!(next), format!("agent:{id}"));
                    match store.compare_and_swap(write, current.version).await {
                        Ok(_) => return conflicts,
                        Err(SwarmError::VersionConflict { .. }) => conflicts += 1,
                        Err(other) => panic!("unexpected error: {other:?}"),
                    }
                }
            }));
        }

        let mut total_conflicts = 0;
        for writer in writers {
            total_conflicts += writer.await.unwrap();
        }

        let final_record = store.get("test", "counter").await.unwrap().unwrap();
        assert_eq!(
            final_record.value.as_i64().unwrap(),
            100,
            "every increment must survive; {total_conflicts} conflicts were detected"
        );
        assert_eq!(final_record.version, 101);
    }
}
