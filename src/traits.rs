use crate::errors::Result;
use image::DynamicImage;
use ndarray::prelude::*;

/// Abstraction for image segmentation models.
///
/// # Why this trait exists
///
/// This trait decouples the image processing pipeline from specific inference backends,
/// allowing the same processing code to work with different implementations (ONNX Runtime,
/// TensorRT, etc.) without modification. This separation adheres to the Dependency Inversion
/// Principle, where high-level modules depend on abstractions rather than concrete implementations.
///
/// # Design rationale
///
/// Three methods provide clean separation of concerns:
///
/// - `segment_image`: High-level interface for end-to-end segmentation. Users call this
///   without needing to understand tensor operations or preprocessing details.
///
/// - `predict`: Low-level tensor inference. Separated from `segment_image` to allow
///   implementations to customize preprocessing/postprocessing while sharing inference logic,
///   or to enable direct tensor-level access for advanced use cases.
///
/// - `get_image_size`: Exposes model input requirements. This information determines
///   preprocessing parameters and is needed by the processing pipeline, so it's part
///   of the trait contract rather than being hardcoded.
pub trait ImageSegmentationModel {
    fn segment_image(&mut self, img: &DynamicImage) -> Result<DynamicImage>;

    fn get_image_size(&self) -> u32;

    fn predict(&mut self, tensor: ArrayView4<f32>) -> Result<Array4<f32>>;
}
