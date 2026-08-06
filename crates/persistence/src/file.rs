//! A journal backed by an append-only file of JSON lines.
//!
//! One record per line, flushed on every append. The format is deliberately boring:
//! it is greppable, it survives a partial write by losing only the final line, and it
//! needs no schema migration to read an older log.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use swarm_domain::{Result, SwarmError};

use crate::{Journal, JournalRecord};

/// An append-only journal stored as JSON lines on disk.
#[derive(Debug)]
pub struct FileJournal {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
    fsync: bool,
}

impl FileJournal {
    /// Open (or create) a journal at `path`, keeping anything already there.
    ///
    /// Every append is flushed to the OS *and* fsynced, which is what makes "the
    /// coordinator died" survivable rather than merely unlikely.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_fsync(path, true)
    }

    /// Open a journal, choosing whether each append is fsynced.
    ///
    /// `fsync = false` trades durability against a power cut for speed. Correct for
    /// tests and benchmarks; not for a deployment that claims to survive a crash.
    pub fn with_fsync(path: impl AsRef<Path>, fsync: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    SwarmError::Config(format!("creating journal directory {parent:?}: {e}"))
                })?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .map_err(|e| SwarmError::Config(format!("opening journal {}: {e}", path.display())))?;

        Ok(Self {
            path,
            writer: Mutex::new(BufWriter::new(file)),
            fsync,
        })
    }

    /// Where this journal lives.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn lock(&self) -> MutexGuard<'_, BufWriter<File>> {
        self.writer.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Journal for FileJournal {
    fn append(&self, record: &JournalRecord) -> Result<()> {
        let mut line = serde_json::to_string(record)
            .map_err(|e| SwarmError::Internal(format!("encoding journal record: {e}")))?;
        line.push('\n');

        let mut writer = self.lock();
        writer
            .write_all(line.as_bytes())
            .and_then(|()| writer.flush())
            .map_err(|e| SwarmError::Internal(format!("writing journal: {e}")))?;

        if self.fsync {
            writer
                .get_ref()
                .sync_data()
                .map_err(|e| SwarmError::Internal(format!("syncing journal: {e}")))?;
        }
        Ok(())
    }

    fn replay(&self) -> Result<Vec<JournalRecord>> {
        // Flush anything buffered so a replay in the same process sees its own writes.
        let mut writer = self.lock();
        writer
            .flush()
            .map_err(|e| SwarmError::Internal(format!("flushing journal: {e}")))?;
        drop(writer);

        let file = File::open(&self.path).map_err(|e| {
            SwarmError::Config(format!("reading journal {}: {e}", self.path.display()))
        })?;

        let mut records = Vec::new();
        let mut lines = BufReader::new(file).lines().peekable();
        while let Some(line) = lines.next() {
            let line =
                line.map_err(|e| SwarmError::Internal(format!("reading journal line: {e}")))?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<JournalRecord>(&line) {
                Ok(record) => records.push(record),
                Err(error) if lines.peek().is_none() => {
                    // A torn final line is what a crash mid-append looks like. Dropping
                    // it loses the last fact, which the caller was never told was
                    // durable; failing the whole recovery would lose everything.
                    tracing::warn!(
                        journal = %self.path.display(),
                        %error,
                        "discarding torn final journal record"
                    );
                }
                Err(error) => {
                    return Err(SwarmError::Internal(format!(
                        "journal {} is corrupt at record {}: {error}",
                        self.path.display(),
                        records.len() + 1
                    )));
                }
            }
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::Utc;
    use swarm_domain::{JobId, JobStatus};

    use crate::MemoryJournal;

    fn status(job_id: JobId, reason: &str) -> JournalRecord {
        JournalRecord::JobStatusChanged {
            job_id,
            status: JobStatus::Running,
            reason: Some(reason.to_owned()),
            at: Utc::now(),
        }
    }

    #[test]
    fn records_survive_being_written_and_read_back() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("swarm.journal");
        let job_id = JobId::new();

        {
            let journal = FileJournal::open(&path).unwrap();
            for index in 0..3 {
                journal.append(&status(job_id, &index.to_string())).unwrap();
            }
            assert_eq!(journal.len().unwrap(), 3);
        }

        // A fresh handle — as after a restart — sees everything.
        let reopened = FileJournal::open(&path).unwrap();
        let records = reopened.replay().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].job_id(), job_id);
    }

    #[test]
    fn reopening_appends_rather_than_truncating() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("swarm.journal");
        let job_id = JobId::new();

        FileJournal::open(&path)
            .unwrap()
            .append(&status(job_id, "first"))
            .unwrap();
        FileJournal::open(&path)
            .unwrap()
            .append(&status(job_id, "second"))
            .unwrap();

        assert_eq!(FileJournal::open(&path).unwrap().len().unwrap(), 2);
    }

    #[test]
    fn a_torn_final_record_is_dropped_not_fatal() {
        // Exactly what a crash mid-append leaves behind.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("swarm.journal");
        let job_id = JobId::new();

        let journal = FileJournal::open(&path).unwrap();
        journal.append(&status(job_id, "complete")).unwrap();
        drop(journal);

        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"record\":\"job_status_ch").unwrap();
        file.flush().unwrap();

        let records = FileJournal::open(&path).unwrap().replay().unwrap();
        assert_eq!(records.len(), 1, "the intact record must survive");
    }

    #[test]
    fn corruption_in_the_middle_is_reported_rather_than_hidden() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("swarm.journal");
        let job_id = JobId::new();

        let journal = FileJournal::open(&path).unwrap();
        journal.append(&status(job_id, "first")).unwrap();
        drop(journal);

        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"not json at all\n").unwrap();
        file.write_all(b"{\"record\":\"job_status_changed\",\"job_id\":\"")
            .unwrap();
        file.write_all(job_id.to_string().as_bytes()).unwrap();
        file.write_all(
            b"\",\"status\":\"running\",\"reason\":null,\"at\":\"2026-01-01T00:00:00Z\"}\n",
        )
        .unwrap();
        file.flush().unwrap();

        let error = FileJournal::open(&path).unwrap().replay().unwrap_err();
        assert!(
            error.to_string().contains("corrupt"),
            "expected a corruption error, got: {error}"
        );
    }

    #[test]
    fn an_empty_journal_replays_to_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let journal = FileJournal::open(directory.path().join("fresh.journal")).unwrap();
        assert!(journal.is_empty().unwrap());
        assert!(journal.replay().unwrap().is_empty());
    }

    #[test]
    fn missing_parent_directories_are_created() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/deeper/swarm.journal");
        let journal = FileJournal::open(&path).unwrap();
        journal.append(&status(JobId::new(), "x")).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn the_file_and_memory_journals_agree_on_behaviour() {
        let directory = tempfile::tempdir().unwrap();
        let job_id = JobId::new();
        let file = FileJournal::open(directory.path().join("j")).unwrap();
        let memory = MemoryJournal::new();

        for index in 0..4 {
            let record = status(job_id, &index.to_string());
            file.append(&record).unwrap();
            memory.append(&record).unwrap();
        }

        assert_eq!(file.replay().unwrap(), memory.replay().unwrap());
    }
}
