use std::{fs, path::PathBuf};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, unbounded};
use image::ImageFormat;
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use walkdir::WalkDir;

use crate::imageops_ai::clip_minimum_border::clip_minimum_border;
use crate::imageops_ai::mask::apply_mask;
use crate::model::{postprocess_mask, preprocess, Model};

mod imageops_ai;
mod model;

#[derive(Parser, Clone)]
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
    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,
}

fn main() -> Result<()> {
    let config = Config::parse();

    ensure!(config.model_path.exists(), "Model path does not exist");
    ensure!(config.input_dir.exists(), "Input directory does not exist");

    let model = Model::new(&config.model_path, config.device_id)?;
    let image_size = model.image_size;

    let image_paths: Vec<_> = WalkDir::new(&config.input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| ImageFormat::from_path(e.path()).is_ok())
        .map(|e| e.into_path())
        .collect();
    let batched_paths: Vec<_> = image_paths.chunks(config.batch_size).collect();

    let (load_bar, inference_bar, output_bar) = progress(image_paths.len() as u64)?;

    let (load_tx, load_rx) = bounded(config.batch_size);
    let (inference_tx, inference_rx) = bounded(config.batch_size);

    rayon::scope(|s| {
        s.spawn(move |s| {
            for paths in batched_paths {
                let load_tx = load_tx.clone();

                let (result_tx, result_rx) = unbounded();
                paths.into_iter().enumerate().for_each(|(i, path)| {
                    s.spawn({
                        let result_tx = result_tx.clone();
                        let load_bar = load_bar.clone();
                        move |_| {
                            load_bar.set_message(path.display().to_string());
                            let image = image::open(&path).unwrap().into_rgb8();
                            let (tensor, crop) = preprocess(&image, image_size);
                            result_tx
                                .send((i, (image, path.clone(), tensor, crop)))
                                .unwrap();

                            load_bar.inc(1);
                        }
                    })
                });
                drop(result_tx);

                let mut batch: Vec<_> = result_rx.iter().collect();
                batch.sort_by(|a, b| a.0.cmp(&b.0));
                let batch = batch.into_iter().map(|(_, x)| x).collect::<Vec<_>>();

                for (image, path, tensor, crop) in batch.into_iter() {
                    load_tx.send((image, path, tensor, crop)).unwrap();
                }
            }
        });

        s.spawn(move |_| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.batch_size)
                .build()
                .unwrap();

            while let Ok((image, path, tensor, crop)) = load_rx.recv() {
                let inference_tx = inference_tx.clone();
                let inference_bar = inference_bar.clone();
                pool.spawn(move || {
                    let mask = model.predict(tensor.view()).unwrap();
                    inference_tx.send((image, path, mask, crop)).unwrap();

                    inference_bar.inc(1);
                })
            }
        });

        s.spawn(move |s| {
            while let Ok((image, path, mask, crop)) = inference_rx.recv() {
                let relative_path = path.strip_prefix(&config.input_dir).unwrap();
                let output_path = config
                    .output_dir
                    .join(relative_path)
                    .with_extension(&config.format);
                s.spawn(move |_| {
                    let (width, height) = image.dimensions();
                    let mask = postprocess_mask(mask, image_size, crop, width, height);
                    let image = apply_mask(&image, &mask, true).unwrap();
                    let image = clip_minimum_border(image, 1, 8);

                    fs::create_dir_all(output_path.parent().unwrap()).unwrap();
                    image
                        .save(&output_path)
                        .with_context(|| format!("Failed to save image: {}", output_path.display()))
                        .unwrap();
                });

                output_bar.inc(1);
            }
        });
    });

    Ok(())
}

fn progress(image_paths_count: u64) -> Result<(ProgressBar, ProgressBar, ProgressBar)> {
    let mp = MultiProgress::new();

    let load_bar = mp
        .add(ProgressBar::new(image_paths_count))
        .with_finish(ProgressFinish::AbandonWithMessage("Completed!".into()));
    load_bar.set_style(ProgressStyle::default_bar()
        .template("Preprocessing: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
        .progress_chars("#>-"));

    let inference_bar = mp
        .add(ProgressBar::new(image_paths_count))
        .with_finish(ProgressFinish::AbandonWithMessage("Completed!".into()));
    inference_bar.set_style(ProgressStyle::default_bar()
        .template("Inference: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    let output_bar = mp
        .add(ProgressBar::new(image_paths_count))
        .with_finish(ProgressFinish::AbandonWithMessage("Completed!".into()));
    output_bar.set_style(ProgressStyle::default_bar()
        .template("Output: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    mp.set_move_cursor(true);
    Ok((load_bar, inference_bar, output_bar))
}
