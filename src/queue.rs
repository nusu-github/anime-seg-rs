use crate::errors::{AnimeSegError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Queue abstraction following dependency inversion principle to hide
/// implementation details (Redis, in-memory, etc.) from business logic
#[async_trait]
pub trait QueueProvider: Send + Sync {
    async fn enqueue(&self, queue_name: &str, job: Job) -> Result<()>;
    async fn enqueue_with_limit(&self, queue_name: &str, job: Job, max_size: usize) -> Result<()>;
    async fn dequeue(&self, queue_name: &str) -> Result<Option<Job>>;
    async fn queue_size(&self, queue_name: &str) -> Result<usize>;
    async fn clear_queue(&self, queue_name: &str) -> Result<()>;
}

/// Processing job with retry logic and metadata for distributed tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Job {
    pub id: String,
    pub job_type: JobType,
    pub payload: JobPayload,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub retry_count: u32,
    pub max_retries: u32,
}

/// Job type to route jobs to appropriate processing stages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobType {
    Preprocessing,
    Inference,
    Postprocessing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobPayload {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub batch_id: Option<String>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl Job {
    pub fn new(job_type: JobType, input_path: PathBuf, output_path: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            job_type,
            payload: JobPayload {
                input_path,
                output_path,
                batch_id: None,
                metadata: std::collections::HashMap::new(),
            },
            created_at: chrono::Utc::now(),
            retry_count: 0,
            max_retries: 3,
        }
    }

    pub fn with_batch_id(mut self, batch_id: String) -> Self {
        self.payload.batch_id = Some(batch_id);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.payload.metadata.insert(key, value);
        self
    }

    pub const fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    pub const fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// In-memory queue for testing and single-machine deployments.
/// Use RedisQueueProvider for production distributed setups.
#[derive(Debug)]
pub struct InMemoryQueueProvider {
    queues: Arc<Mutex<std::collections::HashMap<String, VecDeque<Job>>>>,
}

impl InMemoryQueueProvider {
    pub fn new() -> Self {
        Self {
            queues: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for InMemoryQueueProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QueueProvider for InMemoryQueueProvider {
    async fn enqueue(&self, queue_name: &str, job: Job) -> Result<()> {
        let mut queues = self.queues.lock().await;
        let queue = queues
            .entry(queue_name.to_string())
            .or_insert_with(VecDeque::new);
        queue.push_back(job);
        Ok(())
    }

    async fn enqueue_with_limit(&self, queue_name: &str, job: Job, max_size: usize) -> Result<()> {
        loop {
            {
                let mut queues = self.queues.lock().await;
                let queue = queues
                    .entry(queue_name.to_string())
                    .or_insert_with(VecDeque::new);

                if queue.len() < max_size {
                    queue.push_back(job);
                    return Ok(());
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    async fn dequeue(&self, queue_name: &str) -> Result<Option<Job>> {
        let mut queues = self.queues.lock().await;
        let queue = queues
            .entry(queue_name.to_string())
            .or_insert_with(VecDeque::new);
        Ok(queue.pop_front())
    }

    async fn queue_size(&self, queue_name: &str) -> Result<usize> {
        let queues = self.queues.lock().await;
        Ok(queues.get(queue_name).map(|q| q.len()).unwrap_or(0))
    }

    async fn clear_queue(&self, queue_name: &str) -> Result<()> {
        let mut queues = self.queues.lock().await;
        if let Some(queue) = queues.get_mut(queue_name) {
            queue.clear();
        }
        Ok(())
    }
}

/// Redis-based queue provider (placeholder for future implementation)
pub struct RedisQueueProvider {
    _redis_url: String,
}

impl RedisQueueProvider {
    pub const fn new(redis_url: String) -> Self {
        Self {
            _redis_url: redis_url,
        }
    }
}

#[async_trait]
impl QueueProvider for RedisQueueProvider {
    async fn enqueue(&self, _queue_name: &str, _job: Job) -> Result<()> {
        Err(AnimeSegError::Queue {
            queue_name: _queue_name.to_string(),
            operation: "enqueue".to_string(),
            source: Box::new(std::io::Error::other(
                "Redis implementation not yet available",
            )),
        })
    }

    async fn enqueue_with_limit(
        &self,
        _queue_name: &str,
        _job: Job,
        _max_size: usize,
    ) -> Result<()> {
        Err(AnimeSegError::Queue {
            queue_name: _queue_name.to_string(),
            operation: "enqueue_with_limit".to_string(),
            source: Box::new(std::io::Error::other(
                "Redis implementation not yet available",
            )),
        })
    }

    async fn dequeue(&self, _queue_name: &str) -> Result<Option<Job>> {
        Err(AnimeSegError::Queue {
            queue_name: _queue_name.to_string(),
            operation: "dequeue".to_string(),
            source: Box::new(std::io::Error::other(
                "Redis implementation not yet available",
            )),
        })
    }

    async fn queue_size(&self, _queue_name: &str) -> Result<usize> {
        Err(AnimeSegError::Queue {
            queue_name: _queue_name.to_string(),
            operation: "queue_size".to_string(),
            source: Box::new(std::io::Error::other(
                "Redis implementation not yet available",
            )),
        })
    }

    async fn clear_queue(&self, _queue_name: &str) -> Result<()> {
        Err(AnimeSegError::Queue {
            queue_name: _queue_name.to_string(),
            operation: "clear_queue".to_string(),
            source: Box::new(std::io::Error::other(
                "Redis implementation not yet available",
            )),
        })
    }
}

pub mod queue_names {
    pub const PREPROCESSING: &str = "preprocessing";
    pub const INFERENCE: &str = "inference";
    pub const POSTPROCESSING: &str = "postprocessing";
    pub const ERROR: &str = "error";
    pub const PROCESSING: &str = "processing";
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_job_creation() {
        let input_path = PathBuf::from("/input/test.jpg");
        let output_path = PathBuf::from("/output/test.png");

        let job = Job::new(
            JobType::Preprocessing,
            input_path.clone(),
            output_path.clone(),
        );

        assert_eq!(job.job_type, JobType::Preprocessing);
        assert_eq!(job.payload.input_path, input_path);
        assert_eq!(job.payload.output_path, output_path);
        assert_eq!(job.retry_count, 0);
        assert_eq!(job.max_retries, 3);
        assert!(job.can_retry());
        assert!(!job.id.is_empty());
    }

    #[test]
    fn test_job_with_batch_id() {
        let job = Job::new(
            JobType::Inference,
            PathBuf::from("/input/test.jpg"),
            PathBuf::from("/output/test.png"),
        )
        .with_batch_id("batch_123".to_string());

        assert_eq!(job.payload.batch_id, Some("batch_123".to_string()));
    }

    #[test]
    fn test_job_retry_logic() {
        let mut job = Job::new(
            JobType::Postprocessing,
            PathBuf::from("/input/test.jpg"),
            PathBuf::from("/output/test.png"),
        );

        assert!(job.can_retry());

        for _ in 0..3 {
            job.increment_retry();
        }

        assert!(!job.can_retry());
    }

    #[tokio::test]
    async fn test_inmemory_queue_operations() -> Result<()> {
        let queue_provider = InMemoryQueueProvider::new();
        let queue_name = "test_queue";

        assert_eq!(queue_provider.queue_size(queue_name).await?, 0);
        assert_eq!(queue_provider.dequeue(queue_name).await?, None);

        let job1 = Job::new(
            JobType::Preprocessing,
            PathBuf::from("/input/test1.jpg"),
            PathBuf::from("/output/test1.png"),
        );
        let job2 = Job::new(
            JobType::Inference,
            PathBuf::from("/input/test2.jpg"),
            PathBuf::from("/output/test2.png"),
        );

        queue_provider.enqueue(queue_name, job1.clone()).await?;
        queue_provider.enqueue(queue_name, job2.clone()).await?;

        assert_eq!(queue_provider.queue_size(queue_name).await?, 2);

        let dequeued1 = queue_provider.dequeue(queue_name).await?;
        assert_eq!(dequeued1, Some(job1));

        let dequeued2 = queue_provider.dequeue(queue_name).await?;
        assert_eq!(dequeued2, Some(job2));

        assert_eq!(queue_provider.queue_size(queue_name).await?, 0);
        assert_eq!(queue_provider.dequeue(queue_name).await?, None);

        Ok(())
    }

    #[tokio::test]
    async fn test_clear_queue() -> Result<()> {
        let queue_provider = InMemoryQueueProvider::new();
        let queue_name = "test_clear_queue";

        let job = Job::new(
            JobType::Preprocessing,
            PathBuf::from("/input/test.jpg"),
            PathBuf::from("/output/test.png"),
        );
        queue_provider.enqueue(queue_name, job).await?;

        assert_eq!(queue_provider.queue_size(queue_name).await?, 1);

        queue_provider.clear_queue(queue_name).await?;
        assert_eq!(queue_provider.queue_size(queue_name).await?, 0);

        Ok(())
    }
}
