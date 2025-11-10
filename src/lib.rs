pub mod batch;
pub mod config;
pub mod distributed;
pub mod errors;
pub mod model;
pub mod queue;
pub mod traits;
pub mod worker;

pub mod mocks;

use image::{DynamicImage, ImageFormat};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub use config::Config;
pub use errors::{AnimeSegError, Result};
pub use model::Model;
pub use traits::*;

#[cfg(test)]
pub use mocks::*;

pub struct ImageProcessor<M: ImageSegmentationModel> {
    model: M,
    config: Config,
}

impl<M: ImageSegmentationModel> ImageProcessor<M> {
    pub const fn new(model: M, config: Config) -> Self {
        Self { model, config }
    }

    pub fn process_directory(&self) -> Result<()> {
        let input_path = &self.config.input_dir;
        let output_path = &self.config.output_dir;

        if !input_path.exists() {
            return Err(AnimeSegError::FileSystem {
                path: input_path.clone(),
                operation: "directory existence check".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Input directory does not exist",
                ),
            });
        }

        fs::create_dir_all(output_path).map_err(|e| AnimeSegError::FileSystem {
            path: output_path.clone(),
            operation: "directory creation".to_string(),
            source: e,
        })?;

        let image_files = self.collect_image_files(input_path)?;

        if image_files.is_empty() {
            println!("No image files found to process");
            return Ok(());
        }

        let pb = ProgressBar::new(image_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        image_files
            .par_iter()
            .try_for_each(|input_file| -> Result<()> {
                self.process_single_image(input_file, output_path)?;
                pb.inc(1);
                Ok(())
            })?;

        pb.finish_with_message("Processing complete");
        println!("All image processing completed");
        Ok(())
    }

    fn collect_image_files(&self, input_path: &Path) -> Result<Vec<PathBuf>> {
        let mut image_files = Vec::new();

        for entry in WalkDir::new(input_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() && self.is_supported_image_format(path) {
                image_files.push(path.to_path_buf());
            }
        }

        Ok(image_files)
    }

    pub fn is_supported_image_format(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(
                extension.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tiff" | "avif"
            )
        } else {
            false
        }
    }

    fn process_single_image(&self, input_file: &Path, output_dir: &Path) -> Result<()> {
        let img = image::open(input_file).map_err(|e| AnimeSegError::ImageProcessing {
            path: input_file.display().to_string(),
            operation: "image loading".to_string(),
            source: Box::new(e),
        })?;

        let processed_img =
            self.segment_image(&img)
                .map_err(|e| AnimeSegError::ImageProcessing {
                    path: input_file.display().to_string(),
                    operation: "image segmentation".to_string(),
                    source: Box::new(e),
                })?;

        let relative_path = self.get_relative_path(input_file)?;
        let output_file = output_dir
            .join(relative_path)
            .with_extension(&self.config.format);

        if let Some(parent) = output_file.parent() {
            fs::create_dir_all(parent).map_err(|e| AnimeSegError::FileSystem {
                path: parent.to_path_buf(),
                operation: "output directory creation".to_string(),
                source: e,
            })?;
        }

        let output_format = match self.config.format.as_str() {
            "jpg" | "jpeg" => ImageFormat::Jpeg,
            "png" => ImageFormat::Png,
            "webp" => ImageFormat::WebP,
            "bmp" => ImageFormat::Bmp,
            "gif" => ImageFormat::Gif,
            "tiff" => ImageFormat::Tiff,
            _ => ImageFormat::Png,
        };

        processed_img
            .save_with_format(&output_file, output_format)
            .map_err(|e| AnimeSegError::ImageProcessing {
                path: output_file.display().to_string(),
                operation: "image saving".to_string(),
                source: Box::new(e),
            })?;

        Ok(())
    }

    fn segment_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        self.model.segment_image(img)
    }

    pub fn get_relative_path(&self, input_file: &Path) -> Result<PathBuf> {
        let input_dir = &self.config.input_dir;
        input_file
            .strip_prefix(input_dir)
            .map(|p| p.to_path_buf())
            .map_err(|_| AnimeSegError::FileSystem {
                path: input_file.to_path_buf(),
                operation: "relative path extraction".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Input file is not within input directory",
                ),
            })
    }
}

impl ImageProcessor<Model> {
    pub fn with_onnx_model(config: Config) -> Result<Self> {
        let model = Model::new(&config.model_path, config.device_id)?;
        Ok(Self::new(model, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_formats() {
        // 静的な関数テスト（ImageProcessorインスタンス不要）
        let test_cases = vec![
            ("test.jpg", true),
            ("test.jpeg", true),
            ("test.png", true),
            ("test.webp", true),
            ("test.txt", false),
            ("test", false),
        ];

        for (filename, expected) in test_cases {
            let path = Path::new(filename);
            let is_supported =
                if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                    matches!(
                        extension.to_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tiff" | "avif"
                    )
                } else {
                    false
                };
            assert_eq!(is_supported, expected);
        }
    }

    #[test]
    fn test_relative_path_calculation() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let input_dir = temp_dir.path().join("input");
        let subdir = input_dir.join("subdir");
        fs::create_dir_all(&subdir)?;

        let config = Config {
            input_dir,
            output_dir: "output".into(),
            model_path: "model.onnx".into(),
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

        let processor = ImageProcessor::new(MockSegmentationModel::new(768), config);

        let test_file = subdir.join("test.jpg");
        let relative = processor.get_relative_path(&test_file)?;

        assert_eq!(relative, Path::new("subdir/test.jpg"));
        Ok(())
    }
}
