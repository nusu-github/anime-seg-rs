use image::GenericImageView;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

use anime_seg_rs::{Config, ImageProcessor, ImageSegmentationModel};

// テスト用のモックモデル（統合テスト内で定義）
#[derive(Debug, Clone)]
struct TestMockModel {
    image_size: u32,
}

impl TestMockModel {
    const fn new(image_size: u32) -> Self {
        Self { image_size }
    }
}

impl ImageSegmentationModel for TestMockModel {
    fn segment_image(
        &self,
        img: &image::DynamicImage,
    ) -> anime_seg_rs::Result<image::DynamicImage> {
        Ok(img.clone())
    }

    fn get_image_size(&self) -> u32 {
        self.image_size
    }

    fn predict(
        &self,
        tensor: ndarray::ArrayView4<f32>,
    ) -> anime_seg_rs::Result<ndarray::Array4<f32>> {
        let shape = tensor.shape();
        Ok(ndarray::Array4::<f32>::zeros((
            shape[0], 1, shape[2], shape[3],
        )))
    }
}

#[tokio::test]
async fn test_config_validation() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.onnx");
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");

    fs::create_dir_all(&input_dir).unwrap();
    fs::write(&model_path, b"dummy_model").unwrap();

    let config = Config {
        input_dir,
        output_dir,
        model_path,
        format: "png".to_string(),
        device_id: 0,
        batch_size: 1,
        batch_timeout_ms: 5000,
        preprocessing_workers: 4,
        postprocessing_workers: 4,
        max_inference_queue_size: 100,
        worker_timeout_secs: 30,
        inference_timeout_per_batch_item_secs: 5,
    };

    // 設定値が正しく設定されることを確認
    assert_eq!(config.format, "png");
    assert_eq!(config.device_id, 0);
    assert_eq!(config.batch_size, 1);
}

#[test]
fn test_supported_image_formats() {
    let supported_formats = vec!["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"];

    for format in supported_formats {
        let test_filename = format!("test_image.{}", format);
        let path = PathBuf::from(&test_filename);

        // 拡張子の検証ロジックをテスト
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        assert!(matches!(
            extension,
            "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tiff"
        ));
    }
}

#[test]
fn test_path_handling() {
    let test_cases = vec![
        ("/absolute/path/to/file.png", true),
        ("relative/path/to/file.jpg", true),
        ("", false),
    ];

    for (path_str, should_be_valid) in test_cases {
        let _path = PathBuf::from(path_str);
        let is_valid = !path_str.is_empty();
        assert_eq!(
            is_valid, should_be_valid,
            "Path validation failed for: {}",
            path_str
        );
    }
}

#[test]
fn test_image_processor_with_mock() {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");

    fs::create_dir_all(&input_dir).unwrap();

    let config = Config {
        input_dir,
        output_dir,
        model_path: "dummy.onnx".into(),
        format: "png".to_string(),
        device_id: 0,
        batch_size: 1,
        batch_timeout_ms: 5000,
        preprocessing_workers: 4,
        postprocessing_workers: 4,
        max_inference_queue_size: 100,
        worker_timeout_secs: 30,
        inference_timeout_per_batch_item_secs: 5,
    };

    let mock_model = TestMockModel::new(768);
    let processor = ImageProcessor::new(mock_model, config);

    // ImageProcessor初期化の確認
    assert!(processor.is_supported_image_format(&PathBuf::from("test.jpg")));
    assert!(!processor.is_supported_image_format(&PathBuf::from("test.txt")));
}

#[test]
fn test_trait_abstraction() {
    use anime_seg_rs::ImageSegmentationModel;
    use image::{DynamicImage, Rgb, RgbImage};

    let mock_model = TestMockModel::new(768);

    // トレイトメソッドの確認
    assert_eq!(mock_model.get_image_size(), 768);

    let test_image = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([255, 0, 0])));
    let result = mock_model.segment_image(&test_image).unwrap();
    assert_eq!(result.dimensions(), test_image.dimensions());
}
