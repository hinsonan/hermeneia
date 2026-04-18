use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Condvar, Mutex};

#[derive(Debug)]
struct RuntimePoolState<T> {
    idle_workers: Vec<T>,
    total_workers: usize,
}

#[derive(Debug)]
struct RuntimePoolInner<T> {
    max_workers: usize,
    state: Mutex<RuntimePoolState<T>>,
    condvar: Condvar,
}

#[derive(Debug, Clone)]
pub struct RuntimePool<T> {
    inner: Arc<RuntimePoolInner<T>>,
}

impl<T> RuntimePool<T> {
    pub fn new(max_workers: usize) -> Self {
        assert!(max_workers > 0, "RuntimePool max_workers must be > 0");
        Self {
            inner: Arc::new(RuntimePoolInner {
                max_workers,
                state: Mutex::new(RuntimePoolState {
                    idle_workers: Vec::new(),
                    total_workers: 0,
                }),
                condvar: Condvar::new(),
            }),
        }
    }

    pub fn checkout<F, E>(&self, create_worker: F) -> std::result::Result<RuntimeLease<T>, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        let mut state = self.inner.state.lock().expect("RuntimePool mutex poisoned");

        loop {
            if let Some(worker) = state.idle_workers.pop() {
                return Ok(RuntimeLease::new(worker, Arc::clone(&self.inner)));
            }

            if state.total_workers < self.inner.max_workers {
                state.total_workers += 1;
                drop(state);

                match create_worker() {
                    Ok(worker) => {
                        return Ok(RuntimeLease::new(worker, Arc::clone(&self.inner)));
                    }
                    Err(err) => {
                        let mut state =
                            self.inner.state.lock().expect("RuntimePool mutex poisoned");
                        state.total_workers = state.total_workers.saturating_sub(1);
                        self.inner.condvar.notify_one();
                        return Err(err);
                    }
                }
            }

            state = self
                .inner
                .condvar
                .wait(state)
                .expect("RuntimePool condvar wait failed");
        }
    }

    pub fn total_workers(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("RuntimePool mutex poisoned")
            .total_workers
    }

    pub fn idle_workers(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("RuntimePool mutex poisoned")
            .idle_workers
            .len()
    }
}

#[derive(Debug)]
pub struct RuntimeLease<T> {
    worker: Option<T>,
    pool: Arc<RuntimePoolInner<T>>,
}

impl<T> RuntimeLease<T> {
    fn new(worker: T, pool: Arc<RuntimePoolInner<T>>) -> Self {
        Self {
            worker: Some(worker),
            pool,
        }
    }
}

impl<T> Deref for RuntimeLease<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.worker
            .as_ref()
            .expect("RuntimeLease unexpectedly empty")
    }
}

impl<T> DerefMut for RuntimeLease<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.worker
            .as_mut()
            .expect("RuntimeLease unexpectedly empty")
    }
}

impl<T> Drop for RuntimeLease<T> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let mut state = self.pool.state.lock().expect("RuntimePool mutex poisoned");
            state.idle_workers.push(worker);
            self.pool.condvar.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_lease_returns_worker_to_pool() {
        let pool = RuntimePool::new(1);

        {
            let lease = pool.checkout(|| Ok::<usize, ()>(42)).unwrap();
            assert_eq!(*lease, 42);
            assert_eq!(pool.total_workers(), 1);
            assert_eq!(pool.idle_workers(), 0);
        }

        assert_eq!(pool.total_workers(), 1);
        assert_eq!(pool.idle_workers(), 1);
    }

    #[test]
    fn test_checkout_reuses_existing_worker() {
        let pool = RuntimePool::new(1);
        let create_calls = AtomicUsize::new(0);

        {
            let _lease = pool
                .checkout(|| {
                    create_calls.fetch_add(1, Ordering::SeqCst);
                    Ok::<usize, ()>(7)
                })
                .unwrap();
        }

        {
            let _lease = pool
                .checkout(|| {
                    create_calls.fetch_add(1, Ordering::SeqCst);
                    Ok::<usize, ()>(9)
                })
                .unwrap();
        }

        assert_eq!(create_calls.load(Ordering::SeqCst), 1);
        assert_eq!(pool.total_workers(), 1);
    }

    #[test]
    fn test_checkout_blocks_when_pool_at_capacity() {
        let pool = Arc::new(RuntimePool::new(1));
        let first_lease = pool.checkout(|| Ok::<usize, ()>(1)).unwrap();

        let create_calls = Arc::new(AtomicUsize::new(0));
        let pool_for_thread = Arc::clone(&pool);
        let calls_for_thread = Arc::clone(&create_calls);

        let handle = thread::spawn(move || {
            let lease = pool_for_thread
                .checkout(|| {
                    calls_for_thread.fetch_add(1, Ordering::SeqCst);
                    Ok::<usize, ()>(2)
                })
                .unwrap();
            *lease
        });

        thread::sleep(Duration::from_millis(120));
        assert_eq!(create_calls.load(Ordering::SeqCst), 0);

        drop(first_lease);

        let value = handle.join().unwrap();
        assert_eq!(value, 1);
        assert_eq!(create_calls.load(Ordering::SeqCst), 0);
    }
}
