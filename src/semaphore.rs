use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

pub struct Semaphore {
    state: Arc<SemaphoreState>,
}

struct SemaphoreState {
    count: Mutex<usize>,
    condition: Condvar,
}

pub struct SemaphoreGuard {
    state: Arc<SemaphoreState>,
}

impl Semaphore {
    pub fn new(permits: usize) -> Self {
        Self {
            state: Arc::new(SemaphoreState {
                count: Mutex::new(permits),
                condition: Condvar::new(),
            }),
        }
    }

    pub fn acquire(&self) -> SemaphoreGuard {
        let mut count = self.state.count.lock();
        self.state.condition.wait_while(&mut count, |&mut c| c == 0);
        *count -= 1;
        SemaphoreGuard {
            state: self.state.clone(),
        }
    }
}

impl SemaphoreGuard {
    fn release(&self) {
        let mut count = self.state.count.lock();
        *count += 1;
        self.state.condition.notify_one();
    }
}

impl Drop for SemaphoreGuard {
    fn drop(&mut self) {
        self.release();
    }
}
