use crate::errors::Result;
use image::DynamicImage;
use ndarray::prelude::*;

/// Abstracts segmentation models to enable testing, mocking, and swapping implementations
/// without changing business logic (dependency inversion principle)
pub trait ImageSegmentationModel: Send + Sync {
    fn segment_image(&self, img: &DynamicImage) -> Result<DynamicImage>;

    fn get_image_size(&self) -> u32;

    fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>>;
}

/// Extends base model with batch processing to amortize GPU overhead across multiple images
pub trait BatchImageSegmentationModel: ImageSegmentationModel {
    fn segment_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<DynamicImage>>;

    fn get_optimal_batch_size(&self) -> usize;

    fn predict_batch(&self, tensors: ArrayView4<f32>) -> Result<Array4<f32>>;
}

/// Separates pipeline orchestration from model inference to enable different
/// processing modes (simple parallel, distributed, streaming)
pub trait ImageProcessingPipeline: Send + Sync {
    fn process_directory(&self) -> Result<()>;

    fn process_single_image(
        &self,
        input_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<()>;

    fn is_supported_format(&self, path: &std::path::Path) -> bool;
}

/// Abstracts worker pool implementation to enable testing and alternative concurrency strategies
pub trait WorkerPool<T, R>: Send + Sync {
    fn submit_task(&self, task: T) -> Result<()>;

    fn get_results(&self) -> Result<Vec<R>>;

    fn worker_count(&self) -> usize;

    fn shutdown(&self) -> Result<()>;
}

/// Abstracts queue provider to enable switching between in-memory and Redis
/// without changing business logic
pub trait MessageQueue<T>: Send + Sync {
    fn send(&self, message: T) -> Result<()>;

    fn receive(&self) -> Result<Option<T>>;

    fn send_batch(&self, messages: Vec<T>) -> Result<()>;

    fn receive_batch(&self, max_size: usize) -> Result<Vec<T>>;

    fn queue_size(&self) -> Result<usize>;
}

/// Aggregates items into batches with timeout to balance throughput vs latency
pub trait Batcher<T>: Send + Sync {
    fn add_item(&self, item: T) -> Result<()>;

    fn is_batch_ready(&self) -> bool;

    fn get_batch(&self) -> Result<Vec<T>>;

    fn set_max_batch_size(&mut self, size: usize);

    fn set_timeout(&mut self, timeout_ms: u64);
}

/// Configuration abstraction following "convention over configuration" to minimize
/// boilerplate while maintaining flexibility
pub trait ConfigurationProvider: Send + Sync {
    fn get_string(&self, key: &str) -> Result<Option<String>>;
    fn get_int(&self, key: &str) -> Result<Option<i32>>;
    fn get_bool(&self, key: &str) -> Result<Option<bool>>;

    fn get_string_or_default(&self, key: &str, default: &str) -> String;
    fn get_int_or_default(&self, key: &str, default: i32) -> i32;
    fn get_bool_or_default(&self, key: &str, default: bool) -> bool;
}
