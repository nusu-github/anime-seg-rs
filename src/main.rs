use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{ensure, Result};
use image::{buffer::ConvertBuffer, DynamicImage, ImageFormat, RgbaImage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::config::Config;
use crate::imageops_ai::{clip_minimum_border, mask::apply};
use crate::model::{postprocess_mask, preprocess, Model};

mod config;
mod imageops_ai;
mod model;
mod semaphore;

struct ImageProcessor {
    config: Config,
    model: Arc<Model>,
    alpha_channel: bool,
}

impl ImageProcessor {
    fn new(config: Config) -> Result<Self> {
        ensure!(config.model_path.exists(), "Model path does not exist");
        ensure!(config.input_dir.exists(), "Input directory does not exist");

        let format = ImageFormat::from_extension(&config.format).unwrap();
        let mut v = Cursor::new(Vec::new());
        let alpha_channel = match RgbaImage::new(1, 1).write_to(&mut v, format) {
            Ok(_) => true,
            Err(_) => false,
        };

        let model = Arc::new(Model::new(
            &config.model_path,
            config.device_id,
            config.batch_size as _,
        )?);

        Ok(Self {
            config,
            model,
            alpha_channel,
        })
    }

    fn process_images(&self) -> Result<()> {
        let paths = self.collect_image_paths();
        let progress = self.create_progress_bar(paths.len());

        paths.par_iter().try_for_each(|path| {
            let result = self.process_single_image(path);
            progress.inc(1);
            result
        })?;

        progress.finish_with_message("Processing completed");
        Ok(())
    }

    fn process_single_image(&self, path: &Path) -> Result<()> {
        let image = image::open(path)?;
        let processed_image = self.process_image(image)?;
        self.save_image(path, &processed_image)
    }

    fn process_image(&self, image: DynamicImage) -> Result<DynamicImage> {
        let image = image.into_rgb8();
        let (tensor, crop) = preprocess(&image, self.model.image_size)?;
        let mask = self.model.predict(tensor.view())?;
        let (width, height) = image.dimensions();
        let mask = postprocess_mask(mask, self.model.image_size, crop, width, height);
        let image = apply(&image, &mask, true)?;
        let image = clip_minimum_border(image, 1, 32);
        if self.alpha_channel {
            Ok(DynamicImage::ImageRgba8(image))
        } else {
            Ok(DynamicImage::ImageRgb8(image.convert()))
        }
    }

    fn save_image(&self, input_path: &Path, image: &DynamicImage) -> Result<()> {
        let relative_path = input_path.strip_prefix(&self.config.input_dir)?;
        let output_path = self
            .config
            .output_dir
            .join(relative_path)
            .with_extension(&self.config.format);

        std::fs::create_dir_all(output_path.parent().unwrap())?;
        image
            .save(&output_path)
            .map_err(|e| anyhow::anyhow!("Failed to save image: {}", e))
    }

    fn collect_image_paths(&self) -> Vec<PathBuf> {
        let mut entries: Vec<_> = WalkDir::new(&self.config.input_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| {
                ImageFormat::from_path(e.file_name())
                    .map(|f| f.reading_enabled())
                    .unwrap_or(false)
            })
            .map(walkdir::DirEntry::into_path)
            .collect();
        entries.par_sort_unstable();
        entries
    }

    fn create_progress_bar(&self, total: usize) -> ProgressBar {
        let multi_progress = MultiProgress::new();
        let progress = multi_progress.add(ProgressBar::new(total as u64));
        progress.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"));
        progress
    }
}

fn main() -> Result<()> {
    let config = Config::new();
    let processor = ImageProcessor::new(config)?;
    processor.process_images()?;
    Ok(())
}
