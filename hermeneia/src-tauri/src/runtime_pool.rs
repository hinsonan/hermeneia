use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug)]
pub enum RuntimePoolCheckoutError<E> {
    Worker(E),
    Cancelled,
}

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
        match self.checkout_cancellable(create_worker, || false) {
            Ok(lease) => Ok(lease),
            Err(RuntimePoolCheckoutError::Worker(err)) => Err(err),
            Err(RuntimePoolCheckoutError::Cancelled) => {
                unreachable!("checkout() cannot be cancelled")
            }
        }
    }

    pub fn checkout_cancellable<F, E, C>(
        &self,
        create_worker: F,
        is_cancelled: C,
    ) -> std::result::Result<RuntimeLease<T>, RuntimePoolCheckoutError<E>>
    where
        F: FnOnce() -> std::result::Result<T, E>,
        C: Fn() -> bool,
    {
        let mut create_worker = Some(create_worker);
        let mut state = self.inner.state.lock().expect("RuntimePool mutex poisoned");

        loop {
            if let Some(worker) = state.idle_workers.pop() {
                return Ok(RuntimeLease::new(worker, Arc::clone(&self.inner)));
            }

            if state.total_workers < self.inner.max_workers {
                state.total_workers += 1;
                let create_worker = create_worker
                    .take()
                    .expect("create_worker already consumed");
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
                        return Err(RuntimePoolCheckoutError::Worker(err));
                    }
                }
            }

            if is_cancelled() {
                return Err(RuntimePoolCheckoutError::Cancelled);
            }

            let (next_state, _) = self
                .inner
                .condvar
                .wait_timeout(state, CANCEL_POLL_INTERVAL)
                .expect("RuntimePool condvar wait failed");
            state = next_state;
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

    #[test]
    fn test_checkout_cancellable_returns_cancelled_when_waiting() {
        let pool = Arc::new(RuntimePool::new(1));
        let _first_lease = pool.checkout(|| Ok::<usize, ()>(1)).unwrap();

        let cancelled = Arc::new(AtomicUsize::new(0));
        let cancelled_for_thread = Arc::clone(&cancelled);
        let pool_for_thread = Arc::clone(&pool);

        let handle = thread::spawn(move || {
            pool_for_thread.checkout_cancellable(
                || Ok::<usize, ()>(2),
                || cancelled_for_thread.load(Ordering::SeqCst) == 1,
            )
        });

        thread::sleep(Duration::from_millis(120));
        cancelled.store(1, Ordering::SeqCst);

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(RuntimePoolCheckoutError::Cancelled)));
    }
}
