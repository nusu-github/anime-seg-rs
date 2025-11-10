use crate::errors::{AnimeSegError, Result};
use crate::queue::{Job, JobType, QueueProvider};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, timeout};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// Configuration for batch processing to optimize GPU utilization
#[derive(Debug, Clone)]
pub struct BatchConfiguration {
    /// Maximum batch size triggers immediate processing to avoid GPU starvation
    pub max_batch_size: usize,
    /// Timeout balances throughput (larger batches) vs latency (faster response)
    pub timeout: Duration,
    /// Minimum batch size prevents inefficient small batches except on timeout
    pub min_batch_size: usize,
}

impl BatchConfiguration {
    pub const fn new(max_batch_size: usize, timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            timeout: Duration::from_millis(timeout_ms),
            min_batch_size: 1,
        }
    }

    pub fn with_min_batch_size(mut self, min_size: usize) -> Self {
        self.min_batch_size = min_size.min(self.max_batch_size);
        self
    }
}

impl Default for BatchConfiguration {
    fn default() -> Self {
        Self::new(32, 5000)
    }
}

/// Abstraction for batch processing to enable testability and flexible implementations
#[async_trait]
pub trait BatchProcessor: Send + Sync {
    async fn process_batch(&self, batch: Vec<Job>) -> Result<Vec<Job>>;
    fn batch_type(&self) -> JobType;
}

/// Configuration for batching loop
struct BatchingLoopConfig<Q: QueueProvider, P: BatchProcessor> {
    config: BatchConfiguration,
    queue_provider: Arc<Q>,
    processor: Arc<P>,
    input_queue: String,
    output_queue: String,
    max_output_queue_size: Option<usize>,
    is_running: Arc<Mutex<bool>>,
    cancellation_token: CancellationToken,
}

/// Aggregates jobs into batches with timeout support to maximize GPU throughput
pub struct Batcher<Q: QueueProvider, P: BatchProcessor> {
    config: BatchConfiguration,
    queue_provider: Arc<Q>,
    processor: Arc<P>,
    input_queue: String,
    output_queue: String,
    max_output_queue_size: Option<usize>,
    is_running: Arc<Mutex<bool>>,
    tracker: TaskTracker,
    cancellation_token: CancellationToken,
}

impl<Q: QueueProvider + 'static, P: BatchProcessor + 'static> Batcher<Q, P> {
    pub fn new(
        config: BatchConfiguration,
        queue_provider: Arc<Q>,
        processor: Arc<P>,
        input_queue: String,
        output_queue: String,
    ) -> Self {
        Self {
            config,
            queue_provider,
            processor,
            input_queue,
            output_queue,
            max_output_queue_size: None,
            is_running: Arc::new(Mutex::new(false)),
            tracker: TaskTracker::new(),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub const fn with_output_queue_limit(mut self, max_size: usize) -> Self {
        self.max_output_queue_size = Some(max_size);
        self
    }

    pub async fn start(&self) -> Result<()> {
        let mut running = self.is_running.lock();
        if *running {
            return Err(AnimeSegError::BatchProcessing {
                batch_size: 0,
                operation: "batcher start".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "Batcher is already running",
                )),
            });
        }
        *running = true;
        drop(running);

        let loop_config = BatchingLoopConfig {
            config: self.config.clone(),
            queue_provider: Arc::clone(&self.queue_provider),
            processor: Arc::clone(&self.processor),
            input_queue: self.input_queue.clone(),
            output_queue: self.output_queue.clone(),
            max_output_queue_size: self.max_output_queue_size,
            is_running: Arc::clone(&self.is_running),
            cancellation_token: self.cancellation_token.clone(),
        };

        self.tracker.spawn(async move {
            if let Err(e) = Self::run_batching_loop(loop_config).await {
                eprintln!("Batcher error: {}", e);
            }
        });

        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        {
            let mut running = self.is_running.lock();
            *running = false;
        }

        self.tracker.close();
        self.cancellation_token.cancel();
        self.tracker.wait().await;
        Ok(())
    }

    pub async fn is_running(&self) -> bool {
        *self.is_running.lock()
    }

    async fn run_batching_loop(loop_config: BatchingLoopConfig<Q, P>) -> Result<()> {
        let BatchingLoopConfig {
            config,
            queue_provider,
            processor,
            input_queue,
            output_queue,
            max_output_queue_size,
            is_running,
            cancellation_token,
        } = loop_config;

        let mut batch: Vec<Job> = Vec::with_capacity(config.max_batch_size);
        let mut interval = interval(Duration::from_millis(100));

        loop {
            if cancellation_token.is_cancelled() || !*is_running.lock() {
                if !batch.is_empty() {
                    Self::process_and_enqueue_batch(
                        &batch,
                        &processor,
                        &queue_provider,
                        &output_queue,
                        max_output_queue_size,
                    )
                    .await?;
                }
                break;
            }

            let collection_result = timeout(config.timeout, async {
                loop {
                    match queue_provider.dequeue(&input_queue).await {
                        Ok(Some(job)) => {
                            batch.push(job);

                            if batch.len() >= config.max_batch_size {
                                return Ok(true);
                            }
                        }
                        Ok(None) => {
                            if batch.len() >= config.min_batch_size {
                                return Ok(true);
                            }
                            interval.tick().await;
                        }
                        Err(e) => return Err(e),
                    }
                }
            })
            .await;

            match collection_result {
                Ok(Ok(true)) | Err(_) => {
                    if !batch.is_empty() {
                        let current_batch = std::mem::replace(
                            &mut batch,
                            Vec::with_capacity(config.max_batch_size),
                        );
                        Self::process_and_enqueue_batch(
                            &current_batch,
                            &processor,
                            &queue_provider,
                            &output_queue,
                            max_output_queue_size,
                        )
                        .await?;
                    }
                }
                Ok(Err(e)) => return Err(e),
                _ => {}
            }
        }

        Ok(())
    }

    async fn process_and_enqueue_batch(
        batch: &[Job],
        processor: &Arc<P>,
        queue_provider: &Arc<Q>,
        output_queue: &str,
        max_output_queue_size: Option<usize>,
    ) -> Result<()> {
        let batch_size = batch.len();

        match processor.process_batch(batch.to_vec()).await {
            Ok(processed_jobs) => {
                for job in processed_jobs {
                    if let Some(max_size) = max_output_queue_size {
                        queue_provider
                            .enqueue_with_limit(output_queue, job, max_size)
                            .await?;
                    } else {
                        queue_provider.enqueue(output_queue, job).await?;
                    }
                }
                Ok(())
            }
            Err(e) => {
                eprintln!("Batch processing failed for {} jobs: {}", batch_size, e);

                for mut job in batch.iter().cloned() {
                    if job.can_retry() {
                        job.increment_retry();
                        let requeue_result = if let Some(max_size) = max_output_queue_size {
                            queue_provider
                                .enqueue_with_limit(output_queue, job, max_size)
                                .await
                        } else {
                            queue_provider.enqueue(output_queue, job).await
                        };

                        if let Err(e) = requeue_result {
                            eprintln!("Failed to requeue job for retry: {}", e);
                        }
                    }
                }

                Err(e)
            }
        }
    }
}

/// Mock GPU inference batch processor for testing without actual model
#[derive(Debug)]
pub struct GpuInferenceBatchProcessor {
    batch_type: JobType,
}

impl GpuInferenceBatchProcessor {
    pub const fn new() -> Self {
        Self {
            batch_type: JobType::Inference,
        }
    }
}

impl Default for GpuInferenceBatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BatchProcessor for GpuInferenceBatchProcessor {
    async fn process_batch(&self, mut batch: Vec<Job>) -> Result<Vec<Job>> {
        let batch_size = batch.len();

        for job in &mut batch {
            if job.job_type != self.batch_type {
                return Err(AnimeSegError::BatchProcessing {
                    batch_size,
                    operation: "job type validation".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Expected {:?} jobs", self.batch_type),
                    )),
                });
            }

            job.job_type = JobType::Postprocessing;
            job.payload
                .metadata
                .insert("batch_processed".to_string(), "true".to_string());
            job.payload
                .metadata
                .insert("batch_size".to_string(), batch_size.to_string());
        }

        Ok(batch)
    }

    fn batch_type(&self) -> JobType {
        self.batch_type.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::InMemoryQueueProvider;
    use std::path::PathBuf;

    #[test]
    fn test_batch_configuration() {
        let config = BatchConfiguration::new(16, 3000);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.timeout, Duration::from_millis(3000));
        assert_eq!(config.min_batch_size, 1);

        let config = config.with_min_batch_size(4);
        assert_eq!(config.min_batch_size, 4);

        let config = config.with_min_batch_size(20);
        assert_eq!(config.min_batch_size, 16);
    }

    #[test]
    fn test_default_configuration() {
        let config = BatchConfiguration::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.timeout, Duration::from_millis(5000));
        assert_eq!(config.min_batch_size, 1);
    }

    #[tokio::test]
    async fn test_gpu_inference_batch_processor() -> Result<()> {
        let processor = GpuInferenceBatchProcessor::new();

        let jobs = vec![
            Job::new(
                JobType::Inference,
                PathBuf::from("/input/1.jpg"),
                PathBuf::from("/output/1.png"),
            ),
            Job::new(
                JobType::Inference,
                PathBuf::from("/input/2.jpg"),
                PathBuf::from("/output/2.png"),
            ),
        ];

        let processed = processor.process_batch(jobs.clone()).await?;

        assert_eq!(processed.len(), 2);
        for job in &processed {
            assert_eq!(job.job_type, JobType::Postprocessing);
            assert_eq!(
                job.payload.metadata.get("batch_processed"),
                Some(&"true".to_string())
            );
            assert_eq!(
                job.payload.metadata.get("batch_size"),
                Some(&"2".to_string())
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_batcher_lifecycle() -> Result<()> {
        let queue_provider = Arc::new(InMemoryQueueProvider::new());
        let processor = Arc::new(GpuInferenceBatchProcessor::new());
        let config = BatchConfiguration::new(10, 1000);

        let batcher = Batcher::new(
            config,
            queue_provider,
            processor,
            "inference_queue".to_string(),
            "postprocessing_queue".to_string(),
        );

        assert!(!batcher.is_running().await);

        batcher.start().await?;
        assert!(batcher.is_running().await);

        batcher.stop().await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!batcher.is_running().await);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_size_trigger() -> Result<()> {
        let queue_provider = Arc::new(InMemoryQueueProvider::new());
        let processor = Arc::new(GpuInferenceBatchProcessor::new());
        let config = BatchConfiguration::new(3, 10000);

        for i in 0..3 {
            let job = Job::new(
                JobType::Inference,
                PathBuf::from(format!("/input/{}.jpg", i)),
                PathBuf::from(format!("/output/{}.png", i)),
            );
            queue_provider.enqueue("inference_queue", job).await?;
        }

        let batcher = Batcher::new(
            config,
            Arc::clone(&queue_provider),
            processor,
            "inference_queue".to_string(),
            "postprocessing_queue".to_string(),
        );

        batcher.start().await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        assert_eq!(queue_provider.queue_size("postprocessing_queue").await?, 3);

        batcher.stop().await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_timeout_trigger() -> Result<()> {
        let queue_provider = Arc::new(InMemoryQueueProvider::new());
        let processor = Arc::new(GpuInferenceBatchProcessor::new());
        let config = BatchConfiguration::new(10, 500).with_min_batch_size(3);

        for i in 0..2 {
            let job = Job::new(
                JobType::Inference,
                PathBuf::from(format!("/input/{}.jpg", i)),
                PathBuf::from(format!("/output/{}.png", i)),
            );
            queue_provider.enqueue("inference_queue", job).await?;
        }

        let batcher = Batcher::new(
            config,
            Arc::clone(&queue_provider),
            processor,
            "inference_queue".to_string(),
            "postprocessing_queue".to_string(),
        );

        batcher.start().await?;

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(queue_provider.queue_size("postprocessing_queue").await?, 0);

        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(queue_provider.queue_size("postprocessing_queue").await?, 2);

        batcher.stop().await?;

        Ok(())
    }
}
