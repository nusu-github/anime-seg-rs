use crate::batch::{BatchConfiguration, BatchProcessor, Batcher};
use crate::config::Config;
use crate::errors::{AnimeSegError, Result};
use crate::queue::{queue_names, Job, JobType, QueueProvider};
use crate::traits::BatchImageSegmentationModel;
use crate::worker::{CpuWorkerPool, WorkerPool, WorkerType};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

/// Three-stage pipeline (preprocessing → inference → postprocessing) with queue-based
/// communication to enable distributed processing and maximize GPU utilization
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

        let preprocessing_pool = CpuWorkerPool::new(
            Arc::clone(&queue_provider),
            queue_names::PREPROCESSING.to_string(),
            queue_names::INFERENCE.to_string(),
            WorkerType::Preprocessing,
            config.preprocessing_workers,
        )
        .with_output_queue_limit(config.max_inference_queue_size)
        .with_timeout(config.standard_worker_timeout());

        let postprocessing_pool = CpuWorkerPool::new(
            Arc::clone(&queue_provider),
            queue_names::POSTPROCESSING.to_string(),
            "completed".to_string(),
            WorkerType::Postprocessing,
            config.postprocessing_workers,
        )
        .with_timeout(config.standard_worker_timeout());

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

    pub async fn start(&self) -> Result<()> {
        self.preprocessing_pool.start().await?;
        self.gpu_batcher.start().await?;
        self.postprocessing_pool.start().await?;
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        self.postprocessing_pool.stop().await?;
        self.gpu_batcher.stop().await?;
        self.preprocessing_pool.stop().await?;
        Ok(())
    }

    pub async fn process_directory(&self, input_dir: &PathBuf, output_dir: &PathBuf) -> Result<()> {
        if !input_dir.exists() {
            return Err(AnimeSegError::FileSystem {
                path: input_dir.clone(),
                operation: "directory existence check".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Input directory does not exist",
                ),
            });
        }

        std::fs::create_dir_all(output_dir).map_err(|e| AnimeSegError::FileSystem {
            path: output_dir.clone(),
            operation: "directory creation".to_string(),
            source: e,
        })?;

        let image_files = self.collect_image_files(input_dir)?;

        if image_files.is_empty() {
            println!("No image files found to process");
            return Ok(());
        }

        println!("Processing {} image files", image_files.len());

        self.start().await?;

        for input_file in &image_files {
            let relative_path =
                input_file
                    .strip_prefix(input_dir)
                    .map_err(|_| AnimeSegError::FileSystem {
                        path: input_file.clone(),
                        operation: "relative path extraction".to_string(),
                        source: std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Input file is not within input directory",
                        ),
                    })?;

            let output_file = output_dir
                .join(relative_path)
                .with_extension(&self.config.format);

            let job = Job::new(JobType::Preprocessing, input_file.clone(), output_file);

            self.queue_provider
                .enqueue(queue_names::PREPROCESSING, job)
                .await?;
        }

        self.wait_for_completion(&image_files.len()).await?;

        self.stop().await?;

        println!("All image processing completed");
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
        let mut completed = 0;
        let mut failed = 0;
        let check_interval = tokio::time::Duration::from_millis(500);

        while (completed + failed) < *total_jobs {
            tokio::time::sleep(check_interval).await;

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

            let queue_status = if inference_size >= self.config.max_inference_queue_size {
                " | Inference queue full"
            } else {
                ""
            };

            let progress_message = if failed > 0 {
                format!(
                    "Progress: preproc={}, inference={}, postproc={}, completed={}, failed={}, total={}/{}{}",
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
                    "Progress: preproc={}, inference={}, postproc={}, completed={}/{}{}",
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
            println!("Warning: {} jobs failed", failed);
        }

        Ok(())
    }
}

/// Processes batches of images using GPU inference to maximize throughput
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

        let timeout_duration = std::time::Duration::from_secs(5 * batch_size as u64);

        let result = tokio::time::timeout(timeout_duration, async {
            for job in &batch {
                if job.job_type != JobType::Inference {
                    return Err(AnimeSegError::BatchProcessing {
                        batch_size,
                        operation: "job type validation".to_string(),
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("Expected Inference jobs, got {:?}", job.job_type),
                        )),
                    });
                }
            }

            let batch_for_load = batch.clone();
            let load_results: Vec<_> = tokio::task::spawn_blocking(move || {
                use rayon::prelude::*;
                batch_for_load
                    .par_iter()
                    .map(|job| {
                        image::open(&job.payload.input_path).map_err(|e| (job.id.clone(), e))
                    })
                    .collect()
            })
            .await
            .unwrap();

            let mut images = Vec::with_capacity(batch_size);
            let mut error_indices = Vec::new();

            for (idx, result) in load_results.into_iter().enumerate() {
                match result {
                    Ok(img) => images.push(img),
                    Err((job_id, e)) => {
                        error_indices.push(idx);
                        eprintln!("Failed to load image for job {}: {}", job_id, e);
                    }
                }
            }

            for idx in error_indices.iter().rev() {
                let mut failed_job = batch.remove(*idx);
                failed_job
                    .payload
                    .metadata
                    .insert("error".to_string(), "Image loading error".to_string());
                failed_job.job_type = JobType::Postprocessing;
            }

            if images.is_empty() {
                return Ok(batch);
            }

            let segmented_images = self.model.segment_images_batch(&images)?;

            let batch = tokio::task::spawn_blocking(move || {
                use rayon::prelude::*;
                batch
                    .par_iter_mut()
                    .zip(segmented_images.par_iter())
                    .for_each(|(job, segmented)| {
                        let temp_path = job.payload.output_path.with_file_name(format!(
                            "{}.tmp.png",
                            job.payload
                                .output_path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                        ));

                        if let Some(parent) = temp_path.parent() {
                            if let Err(e) = std::fs::create_dir_all(parent) {
                                eprintln!("Failed to create directory {:?}: {}", parent, e);
                                return;
                            }
                        }

                        if let Err(e) = segmented.save(&temp_path) {
                            eprintln!("Failed to save temporary file {:?}: {}", temp_path, e);
                            return;
                        }

                        job.payload
                            .metadata
                            .insert("segmentation_complete".to_string(), "true".to_string());
                        job.payload
                            .metadata
                            .insert("batch_size".to_string(), batch_size.to_string());
                        job.payload
                            .metadata
                            .insert("temp_path".to_string(), temp_path.display().to_string());

                        job.job_type = JobType::Postprocessing;
                    });
                batch
            })
            .await
            .unwrap();

            Ok(batch)
        })
        .await;

        match result {
            Ok(Ok(processed_batch)) => Ok(processed_batch),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(AnimeSegError::BatchProcessing {
                batch_size,
                operation: "batch inference".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("Batch processing timed out after {:?}", timeout_duration),
                )),
            }),
        }
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
