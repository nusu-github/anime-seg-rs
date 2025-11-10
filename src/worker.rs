use crate::errors::{AnimeSegError, Result};
use crate::queue::{Job, JobType, QueueProvider};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::{interval, timeout};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// Abstracts worker pool to enable testing with different concurrency strategies
#[async_trait]
pub trait WorkerPool: Send + Sync {
    async fn start(&self) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn is_running(&self) -> bool;
    async fn worker_count(&self) -> usize;
    async fn process_job(&self, job: Job) -> Result<Job>;
}

#[async_trait]
pub trait Worker: Send + Sync {
    async fn process(&self, job: Job) -> Result<Job>;
    fn worker_type(&self) -> WorkerType;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerType {
    Preprocessing,
    Postprocessing,
}

/// Configuration for worker pool processing loop
struct WorkerLoopConfig<Q: QueueProvider> {
    queue_provider: Arc<Q>,
    input_queue: String,
    output_queue: String,
    worker_type: WorkerType,
    max_output_queue_size: Option<usize>,
    is_running: Arc<RwLock<bool>>,
    semaphore: Arc<Semaphore>,
    timeout_duration: Duration,
    cancellation_token: CancellationToken,
}

/// CPU-bound worker pool for pre/post-processing with semaphore-based concurrency control
pub struct CpuWorkerPool<Q: QueueProvider> {
    queue_provider: Arc<Q>,
    input_queue: String,
    output_queue: String,
    worker_type: WorkerType,
    max_workers: usize,
    max_output_queue_size: Option<usize>,
    semaphore: Arc<Semaphore>,
    is_running: Arc<RwLock<bool>>,
    timeout_duration: Duration,
    tracker: TaskTracker,
    cancellation_token: CancellationToken,
}

impl<Q: QueueProvider + 'static> CpuWorkerPool<Q> {
    pub fn new(
        queue_provider: Arc<Q>,
        input_queue: String,
        output_queue: String,
        worker_type: WorkerType,
        max_workers: usize,
    ) -> Self {
        Self {
            queue_provider,
            input_queue,
            output_queue,
            worker_type,
            max_workers,
            max_output_queue_size: None,
            semaphore: Arc::new(Semaphore::new(max_workers)),
            is_running: Arc::new(RwLock::new(false)),
            timeout_duration: Duration::from_secs(30),
            tracker: TaskTracker::new(),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub const fn with_output_queue_limit(mut self, max_size: usize) -> Self {
        self.max_output_queue_size = Some(max_size);
        self
    }

    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = timeout;
        self
    }

    async fn process_jobs_loop(config: WorkerLoopConfig<Q>) -> Result<()> {
        let WorkerLoopConfig {
            queue_provider,
            input_queue,
            output_queue,
            worker_type,
            max_output_queue_size,
            is_running,
            semaphore,
            timeout_duration,
            cancellation_token,
        } = config;

        let mut check_interval = interval(Duration::from_millis(100));

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    break;
                }
                _ = check_interval.tick() => {
                    if !*is_running.read() {
                        break;
                    }
                }
            }

            let has_job = match queue_provider.queue_size(&input_queue).await {
                Ok(size) => size > 0,
                Err(_) => false,
            };

            if !has_job {
                continue;
            }

            let queue_provider_clone = Arc::clone(&queue_provider);
            let input_queue_clone = input_queue.clone();
            let output_queue_clone = output_queue.clone();
            let worker_type_clone = worker_type.clone();
            let max_output_queue_size_clone = max_output_queue_size;
            let semaphore_clone = Arc::clone(&semaphore);

            tokio::spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();

                let maybe_job = queue_provider_clone.dequeue(&input_queue_clone).await;

                if let Ok(Some(mut job)) = maybe_job {
                    let processing_queue = format!("{}_processing", input_queue_clone);
                    let _ = queue_provider_clone
                        .enqueue(&processing_queue, job.clone())
                        .await;

                    let result = timeout(timeout_duration, async {
                        let processed_job = match worker_type_clone {
                            WorkerType::Preprocessing => {
                                PreprocessingWorker::new().process(job.clone()).await
                            }
                            WorkerType::Postprocessing => {
                                PostprocessingWorker::new().process(job.clone()).await
                            }
                        };

                        match processed_job {
                            Ok(processed) => {
                                let enqueue_result =
                                    if let Some(max_size) = max_output_queue_size_clone {
                                        queue_provider_clone
                                            .enqueue_with_limit(
                                                &output_queue_clone,
                                                processed,
                                                max_size,
                                            )
                                            .await
                                    } else {
                                        queue_provider_clone
                                            .enqueue(&output_queue_clone, processed)
                                            .await
                                    };

                                if let Err(e) = enqueue_result {
                                    eprintln!("Failed to enqueue processed job: {}", e);
                                    return Err(e);
                                }
                                Ok(())
                            }
                            Err(e) => {
                                eprintln!("Failed to process job: {}", e);
                                Err(e)
                            }
                        }
                    })
                    .await;

                    let _ = queue_provider_clone.dequeue(&processing_queue).await;

                    match result {
                        Ok(process_result) => {
                            if process_result.is_err() {
                                if job.can_retry() {
                                    job.increment_retry();
                                    if let Err(retry_err) =
                                        queue_provider_clone.enqueue(&input_queue_clone, job).await
                                    {
                                        eprintln!("Failed to re-enqueue failed job: {}", retry_err);
                                    }
                                } else if let Err(error_err) =
                                    queue_provider_clone.enqueue("error", job).await
                                {
                                    eprintln!("Failed to enqueue error job: {}", error_err);
                                }
                            }
                        }
                        Err(_) => {
                            eprintln!("Worker task timed out after {:?}", timeout_duration);
                            if job.can_retry() {
                                job.increment_retry();
                                if let Err(retry_err) =
                                    queue_provider_clone.enqueue(&input_queue_clone, job).await
                                {
                                    eprintln!("Failed to re-enqueue timed out job: {}", retry_err);
                                }
                            } else if let Err(error_err) =
                                queue_provider_clone.enqueue("error", job).await
                            {
                                eprintln!(
                                    "Failed to enqueue timed out job to error queue: {}",
                                    error_err
                                );
                            }
                        }
                    }
                }
            });

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }
}

#[async_trait]
impl<Q: QueueProvider + 'static> WorkerPool for CpuWorkerPool<Q> {
    async fn start(&self) -> Result<()> {
        let mut running = self.is_running.write();
        if *running {
            return Err(AnimeSegError::WorkerPool {
                pool_id: format!("{:?}", self.worker_type),
                operation: "start".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "Worker pool is already running",
                )),
            });
        }

        *running = true;
        drop(running);

        let config = WorkerLoopConfig {
            queue_provider: Arc::clone(&self.queue_provider),
            input_queue: self.input_queue.clone(),
            output_queue: self.output_queue.clone(),
            worker_type: self.worker_type.clone(),
            max_output_queue_size: self.max_output_queue_size,
            is_running: Arc::clone(&self.is_running),
            semaphore: Arc::clone(&self.semaphore),
            timeout_duration: self.timeout_duration,
            cancellation_token: self.cancellation_token.clone(),
        };

        self.tracker.spawn(async move {
            if let Err(e) = Self::process_jobs_loop(config).await {
                eprintln!("Worker pool error: {}", e);
            }
        });

        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        {
            let mut running = self.is_running.write();
            *running = false;
        }

        self.tracker.close();
        self.cancellation_token.cancel();
        self.tracker.wait().await;
        Ok(())
    }

    async fn is_running(&self) -> bool {
        *self.is_running.read()
    }

    async fn worker_count(&self) -> usize {
        self.max_workers - self.semaphore.available_permits()
    }

    async fn process_job(&self, job: Job) -> Result<Job> {
        match self.worker_type {
            WorkerType::Preprocessing => PreprocessingWorker::new().process(job).await,
            WorkerType::Postprocessing => PostprocessingWorker::new().process(job).await,
        }
    }
}

pub struct PreprocessingWorker {
    worker_id: String,
}

impl PreprocessingWorker {
    pub fn new() -> Self {
        Self {
            worker_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl Default for PreprocessingWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Worker for PreprocessingWorker {
    async fn process(&self, mut job: Job) -> Result<Job> {
        if job.job_type != JobType::Preprocessing {
            return Err(AnimeSegError::WorkerPool {
                pool_id: self.worker_id.clone(),
                operation: "job type validation".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Expected Preprocessing job, got {:?}", job.job_type),
                )),
            });
        }

        if !job.payload.input_path.exists() {
            return Err(AnimeSegError::FileSystem {
                path: job.payload.input_path.clone(),
                operation: "file existence check".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Input file does not exist",
                ),
            });
        }

        let img =
            image::open(&job.payload.input_path).map_err(|e| AnimeSegError::ImageProcessing {
                path: job.payload.input_path.display().to_string(),
                operation: "image loading".to_string(),
                source: Box::new(e),
            })?;

        let processed_img = self.preprocess_image(&img)?;

        job.payload.metadata.insert(
            "preprocessed_dimensions".to_string(),
            format!("{}x{}", processed_img.width(), processed_img.height()),
        );
        job.payload
            .metadata
            .insert("worker_id".to_string(), self.worker_id.clone());

        job.job_type = JobType::Inference;

        Ok(job)
    }

    fn worker_type(&self) -> WorkerType {
        WorkerType::Preprocessing
    }
}

impl PreprocessingWorker {
    fn preprocess_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        let max_dimension = 2048;
        let (width, height) = img.dimensions();

        if width > max_dimension || height > max_dimension {
            let scale = (max_dimension as f32 / width.max(height) as f32).min(1.0);
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;

            Ok(img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3))
        } else {
            Ok(img.clone())
        }
    }
}

pub struct PostprocessingWorker {
    worker_id: String,
}

impl PostprocessingWorker {
    pub fn new() -> Self {
        Self {
            worker_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl Default for PostprocessingWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Worker for PostprocessingWorker {
    async fn process(&self, mut job: Job) -> Result<Job> {
        if job.job_type != JobType::Postprocessing {
            return Err(AnimeSegError::WorkerPool {
                pool_id: self.worker_id.clone(),
                operation: "job type validation".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Expected Postprocessing job, got {:?}", job.job_type),
                )),
            });
        }

        if let Some(parent) = job.payload.output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AnimeSegError::FileSystem {
                path: parent.to_path_buf(),
                operation: "output directory creation".to_string(),
                source: e,
            })?;
        }

        if let Some(temp_path_str) = job.payload.metadata.get("temp_path") {
            let temp_path = std::path::PathBuf::from(temp_path_str);
            if temp_path.exists() {
                std::fs::rename(&temp_path, &job.payload.output_path)
                    .or_else(|_| {
                        std::fs::copy(&temp_path, &job.payload.output_path)?;
                        std::fs::remove_file(&temp_path)?;
                        Ok(())
                    })
                    .map_err(|e| AnimeSegError::FileSystem {
                        path: job.payload.output_path.clone(),
                        operation: "file move".to_string(),
                        source: e,
                    })?;
            }
        } else if job.payload.input_path.exists() {
            std::fs::copy(&job.payload.input_path, &job.payload.output_path).map_err(|e| {
                AnimeSegError::FileSystem {
                    path: job.payload.output_path.clone(),
                    operation: "file copy".to_string(),
                    source: e,
                }
            })?;
        }

        job.payload.metadata.insert(
            "postprocessing_worker_id".to_string(),
            self.worker_id.clone(),
        );
        job.payload
            .metadata
            .insert("processed_at".to_string(), chrono::Utc::now().to_rfc3339());

        Ok(job)
    }

    fn worker_type(&self) -> WorkerType {
        WorkerType::Postprocessing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::{InMemoryQueueProvider, Job, JobType};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_preprocessing_worker() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");

        // テスト用の小さな画像を作成
        let test_img = image::DynamicImage::new_rgb8(100, 100);
        test_img.save(&input_path).unwrap();

        let job = Job::new(
            JobType::Preprocessing,
            input_path,
            temp_dir.path().join("output.png"),
        );

        let worker = PreprocessingWorker::new();
        let processed_job = worker.process(job).await?;

        assert_eq!(processed_job.job_type, JobType::Inference);
        assert!(processed_job
            .payload
            .metadata
            .contains_key("preprocessed_dimensions"));
        assert!(processed_job.payload.metadata.contains_key("worker_id"));

        Ok(())
    }

    #[tokio::test]
    async fn test_postprocessing_worker() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("subdir").join("output.png");

        let job = Job::new(
            JobType::Postprocessing,
            temp_dir.path().join("input.jpg"),
            output_path.clone(),
        );

        let worker = PostprocessingWorker::new();
        let processed_job = worker.process(job).await?;

        assert_eq!(processed_job.job_type, JobType::Postprocessing);
        assert!(processed_job
            .payload
            .metadata
            .contains_key("postprocessing_worker_id"));
        assert!(processed_job.payload.metadata.contains_key("processed_at"));

        // 出力ディレクトリが作成されていることを確認
        assert!(output_path.parent().unwrap().exists());

        Ok(())
    }

    #[tokio::test]
    async fn test_cpu_worker_pool_creation() -> Result<()> {
        let queue_provider = Arc::new(InMemoryQueueProvider::new());
        let pool = CpuWorkerPool::new(
            queue_provider,
            "input".to_string(),
            "output".to_string(),
            WorkerType::Preprocessing,
            4,
        );

        assert!(!pool.is_running().await);
        assert_eq!(pool.worker_count().await, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_cpu_worker_pool_lifecycle() -> Result<()> {
        let queue_provider = Arc::new(InMemoryQueueProvider::new());
        let pool = CpuWorkerPool::new(
            queue_provider,
            "input".to_string(),
            "output".to_string(),
            WorkerType::Preprocessing,
            2,
        );

        assert!(!pool.is_running().await);

        pool.start().await?;
        assert!(pool.is_running().await);

        pool.stop().await?;

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!pool.is_running().await);

        Ok(())
    }

    #[tokio::test]
    async fn test_worker_type_validation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");

        let test_img = image::DynamicImage::new_rgb8(50, 50);
        test_img.save(&input_path).unwrap();

        let job = Job::new(
            JobType::Postprocessing,
            input_path,
            temp_dir.path().join("output.png"),
        );

        let worker = PreprocessingWorker::new();
        let result = worker.process(job).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            AnimeSegError::WorkerPool { operation, .. } => {
                assert_eq!(operation, "job type validation");
            }
            _ => panic!("Expected WorkerPool error"),
        }

        Ok(())
    }
}
