use std::path::Path;

use anyhow::Result;
use clap::Parser;

mod image_processor;
mod model;
mod progress_tracker;

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
pub(crate) struct Config {
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
}

fn main() -> Result<()> {
    let config = Config::parse();

    let model_path = Path::new(&config.model_path);
    if !model_path.exists() {
        return Err(anyhow::anyhow!("Model path does not exist"));
    }

    let input_dir = Path::new(&config.input_dir);
    if !input_dir.exists() {
        return Err(anyhow::anyhow!("Input directory does not exist"));
    }

    if image::ImageFormat::from_extension(&config.format).is_none() {
        return Err(anyhow::anyhow!("Invalid output format"));
    }

    let model = model::MaskPredictor::new(model_path, config.device_id)?;
    let image_processor = image_processor::ImageProcessor::new(&config);
    let progress_tracker = progress_tracker::ProgressTracker::new(input_dir);

    progress_tracker.process_images(&image_processor, &model);

    Ok(())
}
