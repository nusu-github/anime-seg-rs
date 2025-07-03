use crate::batch::{BatchConfiguration, BatchProcessor, Batcher};
use crate::config::Config;
use crate::errors::{AnimeSegError, Result};
use crate::queue::{queue_names, Job, JobType, QueueProvider};
use crate::traits::BatchImageSegmentationModel;
use crate::worker::{CpuWorkerPool, WorkerPool, WorkerType};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

/// 分散処理対応の画像プロセッサー
/// simple_ai_pipeline.mermaidのアーキテクチャを実装
pub struct DistributedImageProcessor<Q: QueueProvider, M: BatchImageSegmentationModel + 'static> {
    #[allow(dead_code)]
    model: Arc<M>,
    queue_provider: Arc<Q>,
    config: Config,
    preprocessing_pool: CpuWorkerPool<Q>,
    postprocessing_pool: CpuWorkerPool<Q>,
    gpu_batcher: Batcher<Q, GpuInferenceBatchProcessor<M>>,
}

impl<Q: QueueProvider + 'static, M: BatchImageSegmentationModel + 'static>
    DistributedImageProcessor<Q, M>
{
    pub fn new(model: M, queue_provider: Arc<Q>, config: Config) -> Self {
        let model_arc = Arc::new(model);

        // 前処理ワーカープール（推論キューにサイズ制限を設定）
        let preprocessing_pool = CpuWorkerPool::new(
            Arc::clone(&queue_provider),
            queue_names::PREPROCESSING.to_string(),
            queue_names::INFERENCE.to_string(),
            WorkerType::Preprocessing,
            config.preprocessing_workers,
        )
        .with_output_queue_limit(config.max_inference_queue_size)
        .with_timeout(config.standard_worker_timeout());

        // 後処理ワーカープール
        let postprocessing_pool = CpuWorkerPool::new(
            Arc::clone(&queue_provider),
            queue_names::POSTPROCESSING.to_string(),
            "completed".to_string(),
            WorkerType::Postprocessing,
            config.postprocessing_workers,
        )
        .with_timeout(config.standard_worker_timeout());

        // GPU推論バッチャー
        let batch_config =
            BatchConfiguration::new(config.batch_size as usize, config.batch_timeout_ms);
        let gpu_processor = Arc::new(GpuInferenceBatchProcessor::new(Arc::clone(&model_arc)));
        let gpu_batcher = Batcher::new(
            batch_config,
            Arc::clone(&queue_provider),
            gpu_processor,
            queue_names::INFERENCE.to_string(),
            queue_names::POSTPROCESSING.to_string(),
        );

        Self {
            model: model_arc,
            queue_provider,
            config,
            preprocessing_pool,
            postprocessing_pool,
            gpu_batcher,
        }
    }

    /// パイプライン全体を開始
    pub async fn start(&self) -> Result<()> {
        // 各コンポーネントを順番に開始
        self.preprocessing_pool.start().await?;
        self.gpu_batcher.start().await?;
        self.postprocessing_pool.start().await?;
        Ok(())
    }

    /// パイプライン全体を停止
    pub async fn stop(&self) -> Result<()> {
        // 各コンポーネントを逆順で停止
        self.postprocessing_pool.stop().await?;
        self.gpu_batcher.stop().await?;
        self.preprocessing_pool.stop().await?;
        Ok(())
    }

    /// ディレクトリを処理（エントリーポイント）
    pub async fn process_directory(&self, input_dir: &PathBuf, output_dir: &PathBuf) -> Result<()> {
        // 入力ディレクトリの存在確認
        if !input_dir.exists() {
            return Err(AnimeSegError::FileSystem {
                path: input_dir.clone(),
                operation: "ディレクトリ存在確認".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "入力ディレクトリが存在しません",
                ),
            });
        }

        // 出力ディレクトリの作成
        std::fs::create_dir_all(output_dir).map_err(|e| AnimeSegError::FileSystem {
            path: output_dir.clone(),
            operation: "ディレクトリ作成".to_string(),
            source: e,
        })?;

        // 画像ファイルを収集
        let image_files = self.collect_image_files(input_dir)?;

        if image_files.is_empty() {
            println!("処理対象の画像ファイルが見つかりません");
            return Ok(());
        }

        println!("{}個の画像ファイルを処理します", image_files.len());

        // パイプラインを開始
        self.start().await?;

        // 各画像ファイルをキューに投入
        for input_file in &image_files {
            let relative_path =
                input_file
                    .strip_prefix(input_dir)
                    .map_err(|_| AnimeSegError::FileSystem {
                        path: input_file.clone(),
                        operation: "相対パス取得".to_string(),
                        source: std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "入力ファイルが入力ディレクトリ内にありません",
                        ),
                    })?;

            let output_file = output_dir
                .join(relative_path)
                .with_extension(&self.config.format);

            let job = Job::new(JobType::Preprocessing, input_file.clone(), output_file);

            // 前処理キューには制限なしで投入（前処理は軽い処理のため）
            self.queue_provider
                .enqueue(queue_names::PREPROCESSING, job)
                .await?;
        }

        // 処理完了を待つ（簡易的な実装）
        // TODO: より洗練された完了検知メカニズムを実装
        self.wait_for_completion(&image_files.len()).await?;

        // パイプラインを停止
        self.stop().await?;

        println!("全ての画像処理が完了しました");
        Ok(())
    }

    fn collect_image_files(&self, input_path: &PathBuf) -> Result<Vec<PathBuf>> {
        let mut image_files = Vec::new();

        for entry in walkdir::WalkDir::new(input_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() && self.is_supported_image_format(path) {
                image_files.push(path.to_path_buf());
            }
        }

        Ok(image_files)
    }

    fn is_supported_image_format(&self, path: &std::path::Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(
                extension.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tiff" | "avif"
            )
        } else {
            false
        }
    }

    async fn wait_for_completion(&self, total_jobs: &usize) -> Result<()> {
        // 完了待機（完了キューとエラーキューのサイズをチェック）
        let mut completed = 0;
        let mut failed = 0;
        let check_interval = tokio::time::Duration::from_millis(500);

        while (completed + failed) < *total_jobs {
            tokio::time::sleep(check_interval).await;

            // 各キューのサイズを確認
            let preprocessing_size = self
                .queue_provider
                .queue_size(queue_names::PREPROCESSING)
                .await
                .unwrap_or(0);
            let inference_size = self
                .queue_provider
                .queue_size(queue_names::INFERENCE)
                .await
                .unwrap_or(0);
            let postprocessing_size = self
                .queue_provider
                .queue_size(queue_names::POSTPROCESSING)
                .await
                .unwrap_or(0);
            let completed_size = self
                .queue_provider
                .queue_size("completed")
                .await
                .unwrap_or(0);
            let error_size = self
                .queue_provider
                .queue_size(queue_names::ERROR)
                .await
                .unwrap_or(0);

            completed = completed_size;
            failed = error_size;

            // キューサイズ制限チェック
            let queue_status = if inference_size >= self.config.max_inference_queue_size {
                " | 推論キュー満杯"
            } else {
                ""
            };

            let progress_message = if failed > 0 {
                format!(
                    "進捗: 前処理={}, 推論待ち={}, 後処理={}, 完了={}, 失敗={}, 合計={}/{}{}",
                    preprocessing_size,
                    inference_size,
                    postprocessing_size,
                    completed,
                    failed,
                    completed + failed,
                    total_jobs,
                    queue_status
                )
            } else {
                format!(
                    "進捗: 前処理={}, 推論待ち={}, 後処理={}, 完了={}/{}{}",
                    preprocessing_size,
                    inference_size,
                    postprocessing_size,
                    completed,
                    total_jobs,
                    queue_status
                )
            };

            println!("{}", progress_message);
        }

        if failed > 0 {
            println!("警告: {}個のジョブが失敗しました", failed);
        }

        Ok(())
    }
}

/// GPU推論バッチプロセッサー（実際のモデルを使用）
pub struct GpuInferenceBatchProcessor<M: BatchImageSegmentationModel> {
    model: Arc<M>,
}

impl<M: BatchImageSegmentationModel> GpuInferenceBatchProcessor<M> {
    pub const fn new(model: Arc<M>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl<M: BatchImageSegmentationModel + 'static> BatchProcessor for GpuInferenceBatchProcessor<M> {
    async fn process_batch(&self, mut batch: Vec<Job>) -> Result<Vec<Job>> {
        let batch_size = batch.len();

        // ジョブタイプの検証
        for job in &batch {
            if job.job_type != JobType::Inference {
                return Err(AnimeSegError::BatchProcessing {
                    batch_size,
                    operation: "ジョブタイプ検証".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Expected Inference jobs, got {:?}", job.job_type),
                    )),
                });
            }
        }

        // バッチで画像を読み込み
        let mut images = Vec::with_capacity(batch_size);
        let mut load_errors = Vec::new();

        for (idx, job) in batch.iter().enumerate() {
            match image::open(&job.payload.input_path) {
                Ok(img) => images.push(img),
                Err(e) => {
                    load_errors.push((idx, e));
                    // エラーが発生した画像は空の画像で置き換え
                    images.push(image::DynamicImage::new_rgb8(1, 1));
                }
            }
        }

        // 真のバッチ推論を実行
        let segmented_images = self.model.segment_images_batch(&images)?;

        // 結果をジョブに反映
        for (i, (job, segmented)) in batch.iter_mut().zip(segmented_images.iter()).enumerate() {
            // 画像読み込みでエラーが発生していた場合はスキップ
            if load_errors.iter().any(|(idx, _)| *idx == i) {
                job.payload
                    .metadata
                    .insert("error".to_string(), "画像読み込みエラー".to_string());
                continue;
            }

            // 一時ファイルとして保存（後処理ワーカーで最終保存）
            // .tmp.png という拡張子にして画像フォーマットを保持
            let temp_path = job.payload.output_path.with_file_name(format!(
                "{}.tmp.png",
                job.payload
                    .output_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));
            if let Some(parent) = temp_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| AnimeSegError::FileSystem {
                    path: parent.to_path_buf(),
                    operation: "ディレクトリ作成".to_string(),
                    source: e,
                })?;
            }

            segmented
                .save(&temp_path)
                .map_err(|e| AnimeSegError::ImageProcessing {
                    path: temp_path.display().to_string(),
                    operation: "一時ファイル保存".to_string(),
                    source: Box::new(e),
                })?;

            // メタデータを更新
            job.payload
                .metadata
                .insert("segmentation_complete".to_string(), "true".to_string());
            job.payload
                .metadata
                .insert("batch_size".to_string(), batch_size.to_string());
            job.payload
                .metadata
                .insert("temp_path".to_string(), temp_path.display().to_string());

            // ジョブタイプを更新
            job.job_type = JobType::Postprocessing;
        }

        Ok(batch)
    }

    fn batch_type(&self) -> JobType {
        JobType::Inference
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocks::MockSegmentationModel;
    use crate::queue::InMemoryQueueProvider;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_distributed_processor_creation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = Config {
            input_dir: temp_dir.path().join("input"),
            output_dir: temp_dir.path().join("output"),
            model_path: "model.onnx".into(),
            format: "png".to_string(),
            device_id: 0,
            batch_size: 4,
            batch_timeout_ms: 1000,
            preprocessing_workers: 2,
            postprocessing_workers: 2,
            max_inference_queue_size: 10,
            worker_timeout_secs: 30,
            inference_timeout_per_batch_item_secs: 5,
        };

        let model = MockSegmentationModel::new(768);
        let queue_provider = Arc::new(InMemoryQueueProvider::new());

        let _processor = DistributedImageProcessor::new(model, queue_provider, config);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_inference_batch_processor() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let model = Arc::new(MockSegmentationModel::new(768));
        let processor = GpuInferenceBatchProcessor::new(model);

        // テスト用の画像ファイルを作成
        let mut jobs = vec![];
        for i in 1..=2 {
            let input_path = temp_dir.path().join(format!("{}.jpg", i));
            let output_path = temp_dir.path().join(format!("{}.png", i));

            // 小さなテスト画像を作成
            let test_img = image::DynamicImage::new_rgb8(100, 100);
            test_img.save(&input_path).unwrap();

            jobs.push(Job::new(JobType::Inference, input_path, output_path));
        }

        let processed = processor.process_batch(jobs).await?;

        assert_eq!(processed.len(), 2);
        for job in &processed {
            assert_eq!(job.job_type, JobType::Postprocessing);
            assert_eq!(
                job.payload.metadata.get("segmentation_complete"),
                Some(&"true".to_string())
            );
        }

        Ok(())
    }
}
