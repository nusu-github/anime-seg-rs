use anime_seg_rs::{
    batch::BatchConfiguration, config::Config, distributed::DistributedImageProcessor,
    mocks::MockSegmentationModel, queue::InMemoryQueueProvider,
};
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn test_distributed_processing_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // テスト環境のセットアップ
    let temp_dir = TempDir::new()?;
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    let subdir = input_dir.join("subdir");

    fs::create_dir_all(&subdir)?;

    // テスト画像を作成
    let test_images = vec![
        ("test1.jpg", input_dir.clone()),
        ("test2.png", input_dir.clone()),
        ("test3.jpg", subdir.clone()),
    ];

    for (filename, dir) in &test_images {
        let path = dir.join(filename);
        let img = image::DynamicImage::new_rgb8(100, 100);
        img.save(&path)?;
    }

    // 設定
    let config = Config {
        input_dir: input_dir.clone(),
        output_dir: output_dir.clone(),
        model_path: "model.onnx".into(),
        format: "png".to_string(),
        device_id: 0,
        batch_size: 2,
        batch_timeout_ms: 1000,
        preprocessing_workers: 2,
        postprocessing_workers: 2,
        max_inference_queue_size: 10,
        worker_timeout_secs: 30,
        inference_timeout_per_batch_item_secs: 5,
    };

    // コンポーネントの初期化
    let model = MockSegmentationModel::new(768);
    let queue_provider = Arc::new(InMemoryQueueProvider::new());
    let processor = DistributedImageProcessor::new(model, queue_provider, config);

    // パイプラインを実行
    processor.process_directory(&input_dir, &output_dir).await?;

    // 結果を検証
    let output_files = vec![
        output_dir.join("test1.png"),
        output_dir.join("test2.png"),
        output_dir.join("subdir/test3.png"),
    ];

    for output_file in &output_files {
        assert!(
            output_file.exists(),
            "出力ファイルが存在しません: {:?}",
            output_file
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_distributed_processing_empty_directory() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let input_dir = temp_dir.path().join("empty_input");
    let output_dir = temp_dir.path().join("output");

    fs::create_dir_all(&input_dir)?;

    let config = Config {
        input_dir: input_dir.clone(),
        output_dir: output_dir.clone(),
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
    let processor = DistributedImageProcessor::new(model, queue_provider, config);

    // 空のディレクトリでもエラーにならないことを確認
    processor.process_directory(&input_dir, &output_dir).await?;

    assert!(output_dir.exists());

    Ok(())
}

#[tokio::test]
async fn test_batch_configuration_from_config() {
    let config = Config {
        input_dir: "input".into(),
        output_dir: "output".into(),
        model_path: "model.onnx".into(),
        format: "png".to_string(),
        device_id: 0,
        batch_size: 16,
        batch_timeout_ms: 3000,
        preprocessing_workers: 4,
        postprocessing_workers: 4,
        max_inference_queue_size: 100,
        worker_timeout_secs: 30,
        inference_timeout_per_batch_item_secs: 5,
    };

    let batch_config = BatchConfiguration::new(config.batch_size as usize, config.batch_timeout_ms);

    assert_eq!(batch_config.max_batch_size, 16);
    assert_eq!(batch_config.timeout, std::time::Duration::from_millis(3000));
}
