use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{ensure, Result};
use image::{
    buffer::ConvertBuffer, DynamicImage, ImageFormat, Pixel, Primitive, Rgb, RgbImage, Rgba,
    RgbaImage,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::imageops_ai::Image;
use {
    config::Config,
    imageops_ai::{AlphaMaskImage, ClipMinimumBorder, ForegroundEstimator, MargeAlpha},
    model::{postprocess_mask, preprocess, Model},
};

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
        let alpha_channel = RgbaImage::new(1, 1).write_to(&mut v, format).is_ok();

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
        let processed_image = match image {
            DynamicImage::ImageLuma8(img) => {
                let img: RgbImage = img.convert();
                self.process_image(&img)?
            }
            DynamicImage::ImageLumaA8(img) => {
                let img: Image<Rgb<u8>> = img.marge_alpha().convert();
                self.process_image(&img)?
            }
            DynamicImage::ImageRgb8(img) => self.process_image(&img)?,
            DynamicImage::ImageRgba8(img) => {
                let img = img.marge_alpha();
                self.process_image(&img)?
            }
            DynamicImage::ImageLuma16(img) => {
                let img: RgbImage = img.convert();
                self.process_image(&img)?
            }
            DynamicImage::ImageLumaA16(img) => {
                let img: Image<Rgb<u16>> = img.marge_alpha().convert();
                self.process_image(&img)?
            }
            DynamicImage::ImageRgb16(img) => self.process_image(&img)?,
            DynamicImage::ImageRgba16(img) => {
                let img = img.marge_alpha();
                self.process_image(&img)?
            }
            DynamicImage::ImageRgb32F(img) => self.process_image(&img)?,
            DynamicImage::ImageRgba32F(img) => {
                let img = img.marge_alpha();
                self.process_image(&img)?
            }
            _ => return Err(anyhow::anyhow!("Unsupported image format")),
        };
        self.save_image(path, processed_image)
    }

    fn process_image<S>(&self, image: &Image<Rgb<S>>) -> Result<DynamicImage>
    where
        Rgba<S>: Pixel<Subpixel = S>,
        Rgb<S>: Pixel<Subpixel = S>,
        S: Primitive + 'static,
        DynamicImage: From<Image<Rgba<S>>>,
    {
        let (tensor, crop) = preprocess(image, self.model.image_size);
        let mask = self.model.predict(tensor.view())?;
        let (width, height) = image.dimensions();
        let mask = postprocess_mask(mask, self.model.image_size, crop, width, height);
        let image = image
            .clone()
            .estimate_foreground(&mask, 90)
            .add_alpha_mask(&mask)?;
        let image = image.clip_minimum_border(1, 8);
        Ok(DynamicImage::from(image))
    }

    fn save_image(&self, input_path: &Path, image: DynamicImage) -> Result<()> {
        let relative_path = input_path.strip_prefix(&self.config.input_dir)?;
        let output_path = self
            .config
            .output_dir
            .join(relative_path)
            .with_extension(&self.config.format);

        std::fs::create_dir_all(output_path.parent().unwrap())?;

        let image = if self.alpha_channel {
            let image = image.into_rgba8();
            DynamicImage::ImageRgba8(image)
        } else {
            let image = image.into_rgb8();
            DynamicImage::ImageRgb8(image)
        };

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
