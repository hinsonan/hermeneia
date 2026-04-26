use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

const PENDING_CANCEL_TTL: Duration = Duration::from_secs(300);
const MAX_PENDING_CANCELS: usize = 1024;

#[derive(Debug, Default)]
struct RegistryState {
    jobs: HashMap<String, JobMeta>,
    batches: HashMap<String, HashSet<String>>,
    pending_cancels: HashMap<String, Instant>,
}

#[derive(Debug, Clone)]
struct JobMeta {
    cancel_flag: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    batch_id: Option<String>,
    registration_id: u64,
}

#[derive(Debug, Default)]
pub struct CancelRegistry {
    state: Mutex<RegistryState>,
}

impl CancelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_job(
        self: &Arc<Self>,
        job_id: String,
        batch_id: Option<String>,
    ) -> JobRegistration {
        use std::sync::atomic::{AtomicU64, Ordering};

        static REGISTRATION_COUNTER: AtomicU64 = AtomicU64::new(1);

        let mut state = self.state.lock().expect("CancelRegistry mutex poisoned");
        Self::prune_pending_cancels(&mut state);
        let was_pending_cancel = state.pending_cancels.remove(&job_id).is_some();
        let cancel_flag = Arc::new(AtomicBool::new(was_pending_cancel));
        let cancel_notify = Arc::new(Notify::new());
        let registration_id = REGISTRATION_COUNTER.fetch_add(1, Ordering::SeqCst);

        if let Some(existing) = state.jobs.get(&job_id) {
            existing
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::SeqCst);
            existing.cancel_notify.notify_waiters();
        }

        if let Some(previous) = state.jobs.insert(
            job_id.clone(),
            JobMeta {
                cancel_flag: Arc::clone(&cancel_flag),
                cancel_notify: Arc::clone(&cancel_notify),
                batch_id: batch_id.clone(),
                registration_id,
            },
        ) {
            if let Some(previous_batch) = previous.batch_id {
                if let Some(ids) = state.batches.get_mut(&previous_batch) {
                    ids.remove(&job_id);
                    if ids.is_empty() {
                        state.batches.remove(&previous_batch);
                    }
                }
            }
        }

        if let Some(batch) = &batch_id {
            state
                .batches
                .entry(batch.clone())
                .or_default()
                .insert(job_id.clone());
        }

        if was_pending_cancel {
            cancel_notify.notify_waiters();
        }

        JobRegistration {
            registry: Arc::clone(self),
            job_id,
            registration_id,
            cancel_flag,
            cancel_notify,
            active: true,
        }
    }

    pub fn cancel_job(&self, job_id: &str) -> bool {
        let mut state = self.state.lock().expect("CancelRegistry mutex poisoned");
        Self::prune_pending_cancels(&mut state);
        if let Some(meta) = state.jobs.get(job_id) {
            meta.cancel_flag
                .store(true, std::sync::atomic::Ordering::SeqCst);
            meta.cancel_notify.notify_waiters();
            return true;
        }
        let is_new_pending = !state.pending_cancels.contains_key(job_id);
        if is_new_pending && state.pending_cancels.len() >= MAX_PENDING_CANCELS {
            if let Some(oldest_key) = state
                .pending_cancels
                .iter()
                .min_by_key(|(_, created_at)| *created_at)
                .map(|(job_id, _)| job_id.clone())
            {
                state.pending_cancels.remove(&oldest_key);
            }
        }
        state
            .pending_cancels
            .insert(job_id.to_string(), Instant::now());
        true
    }

    fn prune_pending_cancels(state: &mut RegistryState) {
        let now = Instant::now();
        state
            .pending_cancels
            .retain(|_, created_at| now.duration_since(*created_at) <= PENDING_CANCEL_TTL);
    }

    pub fn cancel_batch(&self, batch_id: &str) -> usize {
        let state = self.state.lock().expect("CancelRegistry mutex poisoned");
        let Some(job_ids) = state.batches.get(batch_id) else {
            return 0;
        };

        let mut cancelled = 0;
        for job_id in job_ids {
            if let Some(meta) = state.jobs.get(job_id) {
                meta.cancel_flag
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                meta.cancel_notify.notify_waiters();
                cancelled += 1;
            }
        }
        cancelled
    }

    pub fn cancel_all(&self) -> usize {
        let state = self.state.lock().expect("CancelRegistry mutex poisoned");
        let mut cancelled = 0;
        for meta in state.jobs.values() {
            meta.cancel_flag
                .store(true, std::sync::atomic::Ordering::SeqCst);
            meta.cancel_notify.notify_waiters();
            cancelled += 1;
        }
        cancelled
    }

    pub fn active_jobs(&self) -> usize {
        self.state
            .lock()
            .expect("CancelRegistry mutex poisoned")
            .jobs
            .len()
    }

    fn unregister_job(&self, job_id: &str, registration_id: u64) {
        let mut state = self.state.lock().expect("CancelRegistry mutex poisoned");
        let should_remove = state
            .jobs
            .get(job_id)
            .map(|meta| meta.registration_id == registration_id)
            .unwrap_or(false);
        if !should_remove {
            return;
        }

        let Some(meta) = state.jobs.remove(job_id) else {
            return;
        };

        if let Some(batch_id) = meta.batch_id {
            if let Some(ids) = state.batches.get_mut(&batch_id) {
                ids.remove(job_id);
                if ids.is_empty() {
                    state.batches.remove(&batch_id);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct JobRegistration {
    registry: Arc<CancelRegistry>,
    job_id: String,
    registration_id: u64,
    cancel_flag: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    active: bool,
}

impl JobRegistration {
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }

    pub fn cancel_notify(&self) -> Arc<Notify> {
        Arc::clone(&self.cancel_notify)
    }
}

impl Drop for JobRegistration {
    fn drop(&mut self) {
        if self.active {
            self.registry
                .unregister_job(&self.job_id, self.registration_id);
            self.active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_cancel_job_isolated() {
        let registry = Arc::new(CancelRegistry::new());
        let job_a = registry.register_job("job-a".to_string(), None);
        let job_b = registry.register_job("job-b".to_string(), None);

        assert!(registry.cancel_job("job-a"));
        assert!(job_a.cancel_flag().load(Ordering::SeqCst));
        assert!(!job_b.cancel_flag().load(Ordering::SeqCst));
    }

    #[test]
    fn test_cancel_batch_only_affects_batch_members() {
        let registry = Arc::new(CancelRegistry::new());
        let job_a = registry.register_job("job-a".to_string(), Some("batch-1".to_string()));
        let job_b = registry.register_job("job-b".to_string(), Some("batch-1".to_string()));
        let job_c = registry.register_job("job-c".to_string(), Some("batch-2".to_string()));

        assert_eq!(registry.cancel_batch("batch-1"), 2);
        assert!(job_a.cancel_flag().load(Ordering::SeqCst));
        assert!(job_b.cancel_flag().load(Ordering::SeqCst));
        assert!(!job_c.cancel_flag().load(Ordering::SeqCst));
    }

    #[test]
    fn test_drop_registration_cleans_job_and_batch_entries() {
        let registry = Arc::new(CancelRegistry::new());
        {
            let _job = registry.register_job("job-a".to_string(), Some("batch-1".to_string()));
            assert_eq!(registry.active_jobs(), 1);
            assert_eq!(registry.cancel_batch("batch-1"), 1);
        }

        assert_eq!(registry.active_jobs(), 0);
        assert_eq!(registry.cancel_batch("batch-1"), 0);
    }

    #[test]
    fn test_stale_registration_drop_does_not_unregister_newer_same_job_id() {
        let registry = Arc::new(CancelRegistry::new());

        let old = registry.register_job("job-a".to_string(), Some("batch-1".to_string()));
        let new = registry.register_job("job-a".to_string(), Some("batch-1".to_string()));

        drop(old);

        assert_eq!(registry.active_jobs(), 1);
        assert_eq!(registry.cancel_batch("batch-1"), 1);
        assert!(new.cancel_flag().load(Ordering::SeqCst));
    }

    #[test]
    fn test_cancel_job_before_registration_marks_registration_cancelled() {
        let registry = Arc::new(CancelRegistry::new());

        assert!(registry.cancel_job("job-a"));

        let job = registry.register_job("job-a".to_string(), None);

        assert!(job.cancel_flag().load(Ordering::SeqCst));
        assert_eq!(registry.active_jobs(), 1);
    }
}
