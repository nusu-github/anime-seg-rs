use crate::errors::Result;
use crate::traits::ImageSegmentationModel;
use image::DynamicImage;
use ndarray::prelude::*;

#[cfg(test)]
use image::GenericImageView;

/// テスト用のモックセグメンテーションモデル
#[derive(Debug, Clone)]
pub struct MockSegmentationModel {
    pub image_size: u32,
}

impl MockSegmentationModel {
    pub const fn new(image_size: u32) -> Self {
        Self { image_size }
    }
}

impl ImageSegmentationModel for MockSegmentationModel {
    fn segment_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        // テスト用の簡易実装：入力画像をそのまま返す
        Ok(img.clone())
    }

    fn get_image_size(&self) -> u32 {
        self.image_size
    }

    fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>> {
        // テスト用の簡易実装：ゼロマスクを返す
        let shape = tensor.shape();
        Ok(Array4::<f32>::zeros((shape[0], 1, shape[2], shape[3])))
    }
}

impl crate::traits::BatchImageSegmentationModel for MockSegmentationModel {
    fn segment_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<DynamicImage>> {
        // テスト用の簡易実装：全ての画像をそのまま返す
        Ok(images.to_vec())
    }

    fn get_optimal_batch_size(&self) -> usize {
        // テスト用のバッチサイズ
        16
    }

    fn predict_batch(&self, tensors: ArrayView4<f32>) -> Result<Array4<f32>> {
        // predict メソッドに委譲
        self.predict(tensors)
    }
}

/// テスト用のファクトリー関数
pub const fn create_mock_model() -> MockSegmentationModel {
    MockSegmentationModel::new(768)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_mock_model_creation() {
        let mock = create_mock_model();
        assert_eq!(mock.get_image_size(), 768);
    }

    #[test]
    fn test_mock_model_segment_image() -> Result<()> {
        let mock = create_mock_model();
        let test_image = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([255, 0, 0])));

        let result = mock.segment_image(&test_image)?;
        assert_eq!(result.dimensions(), test_image.dimensions());
        Ok(())
    }

    #[test]
    fn test_mock_model_predict() -> Result<()> {
        let mock = create_mock_model();
        let input_tensor = Array4::<f32>::zeros((1, 3, 768, 768));

        let result = mock.predict(input_tensor.view())?;
        assert_eq!(result.shape(), &[1, 1, 768, 768]);
        Ok(())
    }
}
