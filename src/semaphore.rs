use std::{ops::AddAssign, sync::Arc};

use parking_lot::{Condvar, Mutex};

pub(super) struct Semaphore {
    state: Arc<SemaphoreState>,
}

struct SemaphoreState {
    count: Mutex<usize>,
    condition: Condvar,
}

pub(super) struct Guard {
    state: Arc<SemaphoreState>,
}

impl Semaphore {
    pub(super) fn new(permits: usize) -> Self {
        Self {
            state: Arc::new(SemaphoreState {
                count: Mutex::new(permits),
                condition: Condvar::new(),
            }),
        }
    }

    pub(super) fn acquire(&self) -> Guard {
        {
            let mut count = self.state.count.lock();
            self.state.condition.wait_while(&mut count, |&mut c| c == 0);
            *count -= 1;
        }
        Guard {
            state: self.state.clone(),
        }
    }
}

impl Guard {
    fn release(&self) {
        self.state.count.lock().add_assign(1);
        self.state.condition.notify_one();
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.release();
    }
}
