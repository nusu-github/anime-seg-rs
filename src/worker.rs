use crate::errors::{AnimeSegError, Result};
use crate::queue::{Job, JobType, QueueProvider};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::timeout;

/// ワーカープールの抽象化trait
/// Clean Architectureの依存逆転原則に従い、具体的な実装を隠蔽
#[async_trait]
pub trait WorkerPool: Send + Sync {
    async fn start(&self) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn is_running(&self) -> bool;
    async fn worker_count(&self) -> usize;
    async fn process_job(&self, job: Job) -> Result<Job>;
}

/// 個別ワーカーの抽象化trait
#[async_trait]
pub trait Worker: Send + Sync {
    async fn process(&self, job: Job) -> Result<Job>;
    fn worker_type(&self) -> WorkerType;
}

/// ワーカータイプの定義
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerType {
    Preprocessing,
    Postprocessing,
}

/// CPU並列処理用ワーカープール
pub struct CpuWorkerPool<Q: QueueProvider> {
    queue_provider: Arc<Q>,
    input_queue: String,
    output_queue: String,
    worker_type: WorkerType,
    max_workers: usize,
    max_output_queue_size: Option<usize>,
    semaphore: Arc<Semaphore>,
    is_running: Arc<tokio::sync::RwLock<bool>>,
    timeout_duration: Duration,
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
            is_running: Arc::new(tokio::sync::RwLock::new(false)),
            timeout_duration: Duration::from_secs(30),
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

    async fn process_jobs_loop(
        queue_provider: Arc<Q>,
        input_queue: String,
        output_queue: String,
        worker_type: WorkerType,
        max_output_queue_size: Option<usize>,
        is_running: Arc<tokio::sync::RwLock<bool>>,
        semaphore: Arc<Semaphore>,
        timeout_duration: Duration,
    ) -> Result<()> {
        loop {
            // ワーカープールが停止要求されたら終了
            if !*is_running.read().await {
                break;
            }

            // キューにジョブがあるかチェック
            let has_job = match queue_provider.queue_size(&input_queue).await {
                Ok(size) => size > 0,
                Err(_) => false,
            };

            if !has_job {
                // ジョブがない場合は少し待つ
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }

            // セマフォのチェック（並行実行数制限）
            if semaphore.available_permits() == 0 {
                // セマフォが満杯の場合は少し待つ
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }

            let queue_provider_clone = Arc::clone(&queue_provider);
            let input_queue_clone = input_queue.clone();
            let output_queue_clone = output_queue.clone();
            let worker_type_clone = worker_type.clone();
            let max_output_queue_size_clone = max_output_queue_size;
            let semaphore_clone = Arc::clone(&semaphore);

            // 各ワーカーを独立したタスクで実行
            tokio::spawn(async move {
                // ワーカータスク内でセマフォを取得
                let _permit = semaphore_clone.acquire().await;

                // ジョブを取得
                let maybe_job = queue_provider_clone.dequeue(&input_queue_clone).await;

                if let Ok(Some(mut job)) = maybe_job {
                    // ジョブを処理中キューに移動（追跡のため）
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

                    // 処理中キューからジョブを削除
                    let _ = queue_provider_clone.dequeue(&processing_queue).await;

                    match result {
                        Ok(process_result) => {
                            if process_result.is_err() {
                                // 処理失敗時は元のジョブを再エンキュー（リトライ可能なら）
                                if job.can_retry() {
                                    job.increment_retry();
                                    if let Err(retry_err) =
                                        queue_provider_clone.enqueue(&input_queue_clone, job).await
                                    {
                                        eprintln!("Failed to re-enqueue failed job: {}", retry_err);
                                    }
                                } else {
                                    // リトライ不可能な場合はエラーキューに移動
                                    if let Err(error_err) =
                                        queue_provider_clone.enqueue("error", job).await
                                    {
                                        eprintln!("Failed to enqueue error job: {}", error_err);
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // タイムアウト発生 - ジョブを再エンキュー
                            eprintln!("Worker task timed out after {:?}", timeout_duration);
                            if job.can_retry() {
                                job.increment_retry();
                                if let Err(retry_err) =
                                    queue_provider_clone.enqueue(&input_queue_clone, job).await
                                {
                                    eprintln!("Failed to re-enqueue timed out job: {}", retry_err);
                                }
                            } else {
                                // リトライ不可能な場合はエラーキューに移動
                                if let Err(error_err) =
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
                }
            });

            // CPU使用率を抑制するための短時間スリープ
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }
}

#[async_trait]
impl<Q: QueueProvider + 'static> WorkerPool for CpuWorkerPool<Q> {
    async fn start(&self) -> Result<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(AnimeSegError::WorkerPool {
                pool_id: format!("{:?}", self.worker_type),
                operation: "開始".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "ワーカープールは既に実行中です",
                )),
            });
        }

        *running = true;
        drop(running); // RwLockを明示的に解放

        // ワーカープールを別タスクで開始
        let queue_provider = Arc::clone(&self.queue_provider);
        let input_queue = self.input_queue.clone();
        let output_queue = self.output_queue.clone();
        let worker_type = self.worker_type.clone();
        let max_output_queue_size = self.max_output_queue_size;
        let is_running = Arc::clone(&self.is_running);
        let semaphore = Arc::clone(&self.semaphore);
        let timeout_duration = self.timeout_duration;

        tokio::spawn(async move {
            if let Err(e) = Self::process_jobs_loop(
                queue_provider,
                input_queue,
                output_queue,
                worker_type,
                max_output_queue_size,
                is_running,
                semaphore,
                timeout_duration,
            )
            .await
            {
                eprintln!("Worker pool error: {}", e);
            }
        });

        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let mut running = self.is_running.write().await;
        *running = false;
        Ok(())
    }

    async fn is_running(&self) -> bool {
        *self.is_running.read().await
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

/// 前処理ワーカー実装
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
                operation: "ジョブタイプ検証".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Expected Preprocessing job, got {:?}", job.job_type),
                )),
            });
        }

        // 画像ファイルの存在確認
        if !job.payload.input_path.exists() {
            return Err(AnimeSegError::FileSystem {
                path: job.payload.input_path.clone(),
                operation: "ファイル存在確認".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "入力ファイルが存在しません",
                ),
            });
        }

        // 画像読み込み
        let img =
            image::open(&job.payload.input_path).map_err(|e| AnimeSegError::ImageProcessing {
                path: job.payload.input_path.display().to_string(),
                operation: "画像読み込み".to_string(),
                source: Box::new(e),
            })?;

        // 基本的な前処理（リサイズ、フォーマット検証）
        let processed_img = self.preprocess_image(&img)?;

        // 前処理結果をメタデータに保存
        job.payload.metadata.insert(
            "preprocessed_dimensions".to_string(),
            format!("{}x{}", processed_img.width(), processed_img.height()),
        );
        job.payload
            .metadata
            .insert("worker_id".to_string(), self.worker_id.clone());

        // ジョブタイプを次の段階に更新
        job.job_type = JobType::Inference;

        Ok(job)
    }

    fn worker_type(&self) -> WorkerType {
        WorkerType::Preprocessing
    }
}

impl PreprocessingWorker {
    fn preprocess_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        // 基本的な前処理：最大サイズ制限
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

/// 後処理ワーカー実装
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
                operation: "ジョブタイプ検証".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Expected Postprocessing job, got {:?}", job.job_type),
                )),
            });
        }

        // 出力ディレクトリの作成
        if let Some(parent) = job.payload.output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AnimeSegError::FileSystem {
                path: parent.to_path_buf(),
                operation: "出力ディレクトリ作成".to_string(),
                source: e,
            })?;
        }

        // 推論結果を最終出力ファイルに移動
        if let Some(temp_path_str) = job.payload.metadata.get("temp_path") {
            // GPU推論の結果ファイルを移動
            let temp_path = std::path::PathBuf::from(temp_path_str);
            if temp_path.exists() {
                std::fs::rename(&temp_path, &job.payload.output_path)
                    .or_else(|_| {
                        // renameが失敗した場合（異なるファイルシステムなど）はコピー&削除
                        std::fs::copy(&temp_path, &job.payload.output_path)?;
                        std::fs::remove_file(&temp_path)?;
                        Ok(())
                    })
                    .map_err(|e| AnimeSegError::FileSystem {
                        path: job.payload.output_path.clone(),
                        operation: "ファイル移動".to_string(),
                        source: e,
                    })?;
            }
        } else if job.payload.input_path.exists() {
            // フォールバック：一時ファイルがない場合は入力ファイルをコピー（テスト用）
            std::fs::copy(&job.payload.input_path, &job.payload.output_path).map_err(|e| {
                AnimeSegError::FileSystem {
                    path: job.payload.output_path.clone(),
                    operation: "ファイルコピー".to_string(),
                    source: e,
                }
            })?;
        }

        // 後処理結果をメタデータに保存
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

        // 初期状態で停止
        assert!(!pool.is_running().await);

        // 開始
        pool.start().await?;
        assert!(pool.is_running().await);

        // 停止
        pool.stop().await?;

        // 少し待ってから状態確認
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!pool.is_running().await);

        Ok(())
    }

    #[tokio::test]
    async fn test_worker_type_validation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");

        // テスト用画像作成
        let test_img = image::DynamicImage::new_rgb8(50, 50);
        test_img.save(&input_path).unwrap();

        // 間違ったジョブタイプでPreprocessingWorkerをテスト
        let job = Job::new(
            JobType::Postprocessing, // 間違ったタイプ
            input_path,
            temp_dir.path().join("output.png"),
        );

        let worker = PreprocessingWorker::new();
        let result = worker.process(job).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            AnimeSegError::WorkerPool { operation, .. } => {
                assert_eq!(operation, "ジョブタイプ検証");
            }
            _ => panic!("Expected WorkerPool error"),
        }

        Ok(())
    }
}
