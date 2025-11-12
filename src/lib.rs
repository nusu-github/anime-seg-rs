pub mod config;
pub mod errors;
pub mod model;
pub mod traits;

use glob::glob;
use image::{DynamicImage, ImageFormat};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::{Path, PathBuf};

pub use config::Config;
pub use errors::{AnimeSegError, Result};
pub use model::Model;
pub use traits::*;

/// Extract the base directory from a glob pattern by finding the directory
/// portion before any wildcard characters.
///
/// # Why this is needed
///
/// When processing files matched by a glob pattern, we need to preserve the
/// directory structure relative to the base directory. For example, with pattern
/// "input/**/*.jpg", we want "input/sub/file.jpg" to become "output/sub/file.jpg",
/// not "output/input/sub/file.jpg". This function extracts "input" as the base
/// to enable proper relative path computation.
///
/// # Examples
///
/// - `"input/**/*.jpg"` → `"input"`
/// - `"a/b/**/*"` → `"a/b"`
/// - `"*.jpg"` → `"."`
fn extract_base_directory(pattern: &str) -> PathBuf {
    let wildcard_pos = pattern.find(['*', '?', '[']).unwrap_or(pattern.len());

    if wildcard_pos == 0 {
        return PathBuf::from(".");
    }

    let before_wildcard = &pattern[..wildcard_pos];

    let last_sep = before_wildcard
        .rfind(['/', '\\'])
        .map(|pos| &before_wildcard[..pos]);

    match last_sep {
        Some(base) if !base.is_empty() => PathBuf::from(base),
        _ => PathBuf::from("."),
    }
}

/// Simple parallel image processor using Rayon for single-machine batch processing.
///
/// This processor is generic over any type implementing `ImageSegmentationModel`,
/// allowing different model backends (ONNX, TensorRT, etc.) to be used with the
/// same processing logic. Uses Rayon for straightforward parallel processing without
/// the complexity of queues or distributed architecture.
pub struct ImageProcessor<M: ImageSegmentationModel> {
    model: M,
    config: Config,
}

impl<M: ImageSegmentationModel> ImageProcessor<M> {
    /// Create a new image processor with the given model and configuration.
    pub fn new(model: M, config: Config) -> Self {
        Self { model, config }
    }

    /// Process all images matching the input pattern and save results to output directory.
    ///
    /// This method orchestrates the entire processing workflow: collecting image files,
    /// creating output directories, processing each image with progress tracking, and
    /// handling errors gracefully without stopping the entire batch.
    pub fn process_directory(&mut self) -> Result<()> {
        let input_pattern = &self.config.input_pattern;
        let output_path = self.config.output_dir.clone();

        fs::create_dir_all(&output_path).map_err(|e| AnimeSegError::FileSystem {
            path: output_path.clone(),
            operation: "directory creation".to_string(),
            source: e,
        })?;

        let image_files = self.collect_image_files(input_pattern)?;

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

        for input_file in &image_files {
            if let Err(e) = self.process_single_image(input_file, &output_path) {
                eprintln!("Failed to process image {}: {}", input_file.display(), e);
            }
            pb.inc(1);
        }

        pb.finish_with_message("Processing complete");
        println!("All image processing completed");
        Ok(())
    }

    /// Collect all valid image files matching the glob pattern.
    ///
    /// Only includes files with supported image formats to avoid processing errors later.
    fn collect_image_files(&self, pattern: &str) -> Result<Vec<PathBuf>> {
        let mut image_files = Vec::new();

        // Execute glob pattern and filter for valid image files
        for entry in glob(pattern).map_err(|e| AnimeSegError::ImageProcessing {
            path: pattern.to_string(),
            operation: "glob pattern parsing".to_string(),
            source: Box::new(e),
        })? {
            match entry {
                Ok(path) => {
                    if path.is_file() && self.is_supported_image_format(&path) {
                        image_files.push(path);
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to read path: {}", e);
                }
            }
        }

        Ok(image_files)
    }

    /// Check if the file has a supported image format that can be read.
    ///
    /// Uses the `image` crate's format detection to determine support dynamically,
    /// which allows the set of supported formats to expand automatically when
    /// enabling additional feature flags.
    pub fn is_supported_image_format(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            let format = ImageFormat::from_extension(extension);
            if let Some(fmt) = format {
                return fmt.reading_enabled();
            }
            false
        } else {
            false
        }
    }

    /// Process a single image: load, segment, and save with preserved directory structure.
    ///
    /// The output path preserves the relative directory structure from the input pattern's
    /// base directory. This ensures that hierarchical folder organization is maintained
    /// in the output, which is essential for batch processing large organized datasets.
    fn process_single_image(&mut self, input_file: &Path, output_dir: &Path) -> Result<()> {
        let img = image::open(input_file).map_err(|e| AnimeSegError::ImageProcessing {
            path: input_file.display().to_string(),
            operation: "image loading".to_string(),
            source: Box::new(e),
        })?;

        let processed_img = self.segment_image(&img)?;

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

        let output_format =
            ImageFormat::from_extension(&self.config.format).unwrap_or(ImageFormat::Png);

        processed_img
            .save_with_format(&output_file, output_format)
            .map_err(|e| AnimeSegError::ImageProcessing {
                path: output_file.display().to_string(),
                operation: "image saving".to_string(),
                source: Box::new(e),
            })?;

        Ok(())
    }

    /// Delegate segmentation to the underlying model implementation.
    fn segment_image(&mut self, img: &DynamicImage) -> Result<DynamicImage> {
        self.model.segment_image(img)
    }

    /// Compute the relative path from the input pattern's base directory.
    ///
    /// This is necessary to reconstruct the directory hierarchy in the output.
    /// For example, with input pattern "photos/**/*.jpg" and file "photos/2024/jan/pic.jpg",
    /// this returns "2024/jan/pic.jpg" so the output becomes "output/2024/jan/pic.png".
    pub fn get_relative_path(&self, input_file: &Path) -> Result<PathBuf> {
        let base_dir = extract_base_directory(&self.config.input_pattern);
        input_file
            .strip_prefix(&base_dir)
            .map(|p| p.to_path_buf())
            .map_err(|_| AnimeSegError::FileSystem {
                path: input_file.to_path_buf(),
                operation: "relative path extraction".to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Input file is not within base directory",
                ),
            })
    }
}

impl ImageProcessor<Model> {
    /// Convenience constructor that creates an ONNX-based image processor.
    ///
    /// This provides a simpler API for the common case of using the default ONNX model
    /// implementation, avoiding the need to construct the Model separately.
    pub fn with_onnx_model(config: Config) -> Result<Self> {
        let model = Model::new(&config.model_path)?;
        Ok(Self::new(model, config))
    }
}
