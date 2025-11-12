use crate::errors::Result;
use image::DynamicImage;
use ndarray::prelude::*;

/// Core trait for image segmentation models
pub trait ImageSegmentationModel {
    fn segment_image(&mut self, img: &DynamicImage) -> Result<DynamicImage>;

    fn get_image_size(&self) -> u32;

    fn predict(&mut self, tensor: ArrayView4<f32>) -> Result<Array4<f32>>;
}
