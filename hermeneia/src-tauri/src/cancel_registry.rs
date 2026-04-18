use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct RegistryState {
    jobs: HashMap<String, JobMeta>,
    batches: HashMap<String, HashSet<String>>,
}

#[derive(Debug, Clone)]
struct JobMeta {
    cancel_flag: Arc<AtomicBool>,
    batch_id: Option<String>,
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
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let mut state = self.state.lock().expect("CancelRegistry mutex poisoned");

        if let Some(existing) = state.jobs.get(&job_id) {
            existing
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        if let Some(previous) = state.jobs.insert(
            job_id.clone(),
            JobMeta {
                cancel_flag: Arc::clone(&cancel_flag),
                batch_id: batch_id.clone(),
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

        JobRegistration {
            registry: Arc::clone(self),
            job_id,
            cancel_flag,
            active: true,
        }
    }

    pub fn cancel_job(&self, job_id: &str) -> bool {
        let state = self.state.lock().expect("CancelRegistry mutex poisoned");
        if let Some(meta) = state.jobs.get(job_id) {
            meta.cancel_flag
                .store(true, std::sync::atomic::Ordering::SeqCst);
            return true;
        }
        false
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

    fn unregister_job(&self, job_id: &str) {
        let mut state = self.state.lock().expect("CancelRegistry mutex poisoned");
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
    cancel_flag: Arc<AtomicBool>,
    active: bool,
}

impl JobRegistration {
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }
}

impl Drop for JobRegistration {
    fn drop(&mut self) {
        if self.active {
            self.registry.unregister_job(&self.job_id);
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
}
