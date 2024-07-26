use std::{fs, path::PathBuf};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use image::{ImageFormat, RgbImage, RgbaImage};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use walkdir::WalkDir;

use crate::model::Model;

mod imageops_ai;
mod model;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    #[arg(short, long)]
    model_path: PathBuf,
    #[arg(short, long, default_value = "png")]
    format: String,
    #[arg(short, long, default_value_t = 0)]
    device_id: i32,
    #[arg(short, long, default_value_t = std::thread::available_parallelism().unwrap().get())]
    batch_size: usize,
    #[arg(long, default_value_t = std::thread::available_parallelism().unwrap().get())]
    thread_size: usize,
}

fn main() -> Result<()> {
    let config = Config::parse();

    ensure!(config.model_path.exists(), "Model path does not exist");
    ensure!(config.input_dir.exists(), "Input directory does not exist");

    let model = Model::new(&config.model_path, config.device_id)?;

    let image_paths: Vec<_> = WalkDir::new(&config.input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| ImageFormat::from_path(e.path()).is_ok())
        .map(|e| e.into_path())
        .collect();

    let progress_bar =
        ProgressBar::new(image_paths.len() as u64).with_finish(ProgressFinish::Abandon);
    progress_bar.set_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} {eta})",
    )?);

    let (load_tx, load_rx): (Sender<(RgbImage, PathBuf)>, Receiver<(RgbImage, PathBuf)>) =
        bounded(config.thread_size);
    let (inference_tx, inference_rx): (
        Sender<(RgbaImage, PathBuf)>,
        Receiver<(RgbaImage, PathBuf)>,
    ) = bounded(config.thread_size);

    rayon::scope(|s| {
        s.spawn(move |s| {
            for path in image_paths {
                let load_tx = load_tx.clone();
                s.spawn(move |_| {
                    let image = image::open(&path).unwrap().into_rgb8();
                    load_tx.send((image, path)).unwrap();
                })
            }
        });

        s.spawn(move |_| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.batch_size)
                .build()
                .unwrap();

            while let Ok((image, path)) = load_rx.recv() {
                let inference_tx = inference_tx.clone();
                let model = model.clone();
                pool.spawn(move || {
                    let inferred_image = model.predict(&image).unwrap();
                    inference_tx.send((inferred_image, path)).unwrap();
                })
            }
        });

        s.spawn(move |s| {
            while let Ok((image, path)) = inference_rx.recv() {
                let relative_path = path.strip_prefix(&config.input_dir).unwrap();
                let output_path = config
                    .output_dir
                    .join(relative_path)
                    .with_extension(&config.format);
                s.spawn(move |_| {
                    fs::create_dir_all(output_path.parent().unwrap()).unwrap();
                    image
                        .save(&output_path)
                        .with_context(|| format!("Failed to save image: {}", output_path.display()))
                        .unwrap();
                });
                progress_bar.inc(1);
            }
        });
    });

    Ok(())
}
