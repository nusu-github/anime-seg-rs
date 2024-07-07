use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use image::ImageFormat;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::{prelude::*, ThreadPoolBuilder};
use walkdir::WalkDir;

use crate::model::Model;

mod imageops_ai;
mod model;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Config {
    #[arg(short, long)]
    input_dir: String,

    #[arg(short, long)]
    output_dir: String,

    #[arg(short, long)]
    model_path: String,

    #[arg(short, long, default_value_t = String::from("png"))]
    format: String,

    #[arg(short, long, default_value_t = 0)]
    device_id: i32,

    #[arg(
        short, long, default_value_t = thread::available_parallelism().unwrap().get()
    )]
    num_threads: usize,
}

fn main() -> Result<()> {
    let config = Arc::new(Config::parse());

    ensure!(
        Path::new(&config.model_path).exists(),
        "Model path does not exist"
    );
    ensure!(
        Path::new(&config.input_dir).exists(),
        "Input directory does not exist"
    );
    ensure!(
        ImageFormat::from_extension(&config.format).is_some(),
        "Invalid format"
    );

    let model = Model::new(&config.model_path, config.num_threads, config.device_id)?;
    ThreadPoolBuilder::new()
        .num_threads(config.num_threads)
        .build_global()?;

    let image_paths = WalkDir::new(&config.input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| ImageFormat::from_path(e.path()).is_ok())
        .map(|e| e.into_path())
        .collect::<Vec<_>>();

    let progress_bar = ProgressBar::new(image_paths.len() as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} {eta})",
        )?
        .progress_chars("#>-"),
    );

    image_paths
        .par_iter()
        .progress_with(progress_bar.clone())
        .try_for_each_with(model, |model, path| {
            let image = image::open(&path)
                .with_context(|| format!("Failed to open image: {}", path.display()))?
                .into_rgb8();

            let image = model.predict(&image)?;
            let output_path = construct_output_path(path, &config)?;
            image
                .save(&output_path)
                .with_context(|| format!("Failed to save image: {}", output_path.display()))
        })?;

    progress_bar.finish();

    Ok(())
}

fn relocate<P: AsRef<Path>>(path: P, prefix: P, new_prefix: P) -> PathBuf {
    new_prefix
        .as_ref()
        .join(path.as_ref().strip_prefix(prefix).unwrap())
        .to_path_buf()
}

fn construct_output_path<P: AsRef<Path>>(path: P, config: &Config) -> Result<PathBuf> {
    let output_dir = relocate(
        path.as_ref(),
        (&config.input_dir).as_ref(),
        (&config.output_dir).as_ref(),
    );

    fs::create_dir_all(output_dir.parent().unwrap())?;
    Ok(output_dir.with_extension(&config.format))
}
