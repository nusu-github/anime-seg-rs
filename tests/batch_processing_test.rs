use anime_seg_rs::{
    batch::{BatchConfiguration, BatchProcessor, Batcher},
    distributed::GpuInferenceBatchProcessor,
    mocks::MockSegmentationModel,
    queue::{InMemoryQueueProvider, Job, JobType, QueueProvider},
};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

#[tokio::test]
async fn test_true_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model = Arc::new(MockSegmentationModel::new(768));
    let processor = Arc::new(GpuInferenceBatchProcessor::new(model));

    // 複数のジョブを作成
    let batch_size = 4;
    let mut jobs = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let input_path = temp_dir.path().join(format!("input_{}.jpg", i));
        let output_path = temp_dir.path().join(format!("output_{}.png", i));

        // テスト画像を作成
        let img = image::DynamicImage::new_rgb8(100, 100);
        img.save(&input_path)?;

        jobs.push(Job::new(JobType::Inference, input_path, output_path));
    }

    // バッチ処理を実行
    let processed_jobs = processor.process_batch(jobs).await?;

    // 結果の検証
    assert_eq!(processed_jobs.len(), batch_size);

    for job in &processed_jobs {
        // バッチサイズがメタデータに記録されていることを確認
        assert_eq!(
            job.payload.metadata.get("batch_size"),
            Some(&batch_size.to_string())
        );

        // 一時ファイルが作成されていることを確認
        if let Some(temp_path_str) = job.payload.metadata.get("temp_path") {
            let temp_path = std::path::PathBuf::from(temp_path_str);
            assert!(
                temp_path.exists(),
                "一時ファイルが存在しません: {:?}",
                temp_path
            );
        }

        // ジョブタイプが更新されていることを確認
        assert_eq!(job.job_type, JobType::Postprocessing);
    }

    Ok(())
}

#[tokio::test]
async fn test_batch_processing_with_batcher() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model = Arc::new(MockSegmentationModel::new(768));
    let processor = Arc::new(GpuInferenceBatchProcessor::new(model));
    let queue_provider = Arc::new(InMemoryQueueProvider::new());

    // バッチャーの設定（最大3個または500msでバッチ処理）
    let batch_config = BatchConfiguration::new(3, 500);
    let batcher = Batcher::new(
        batch_config,
        Arc::clone(&queue_provider),
        processor,
        "inference_queue".to_string(),
        "postprocessing_queue".to_string(),
    );

    // バッチャーを開始
    batcher.start().await?;

    // 5つのジョブを投入（3個と2個のバッチに分かれるはず）
    for i in 0..5 {
        let input_path = temp_dir.path().join(format!("batch_{}.jpg", i));
        let output_path = temp_dir.path().join(format!("batch_{}.png", i));

        let img = image::DynamicImage::new_rgb8(50, 50);
        img.save(&input_path)?;

        let job = Job::new(JobType::Inference, input_path, output_path);

        queue_provider.enqueue("inference_queue", job).await?;
    }

    // バッチ処理が完了するまで待つ
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // 後処理キューに5つのジョブがあることを確認
    assert_eq!(queue_provider.queue_size("postprocessing_queue").await?, 5);

    // バッチャーを停止
    batcher.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_batch_processing_performance() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model = Arc::new(MockSegmentationModel::new(768));
    let processor = Arc::new(GpuInferenceBatchProcessor::new(model));

    // 大きなバッチサイズでのテスト
    let batch_size = 16; // MockSegmentationModelのget_optimal_batch_size()が返す値
    let mut jobs = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let input_path = temp_dir.path().join(format!("perf_{}.jpg", i));
        let output_path = temp_dir.path().join(format!("perf_{}.png", i));

        let img = image::DynamicImage::new_rgb8(100, 100);
        img.save(&input_path)?;

        jobs.push(Job::new(JobType::Inference, input_path, output_path));
    }

    let start = std::time::Instant::now();
    let processed_jobs = processor.process_batch(jobs).await?;
    let duration = start.elapsed();

    println!("バッチサイズ {} の処理時間: {:?}", batch_size, duration);

    assert_eq!(processed_jobs.len(), batch_size);

    // 各ジョブがバッチサイズ16で処理されたことを確認
    for job in &processed_jobs {
        assert_eq!(
            job.payload.metadata.get("batch_size"),
            Some(&"16".to_string())
        );
    }

    Ok(())
}
