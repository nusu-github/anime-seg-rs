use anime_seg_rs::distributed::DistributedImageProcessor;
use anime_seg_rs::queue::InMemoryQueueProvider;
use anime_seg_rs::{Config, Model};
use anyhow::Result;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing ONNX Runtime environment...");
    ort::init()
        .with_name("anime-seg-rs")
        .with_telemetry(false)
        .commit()?;
    println!("ONNX Runtime initialization complete");

    let config = Config::new();

    println!("Loading model...");
    let model = Model::new(&config.model_path, config.device_id)?;
    println!("Model loading complete");

    let queue_provider = Arc::new(InMemoryQueueProvider::new());

    let processor = DistributedImageProcessor::new(model, queue_provider, config.clone());
    processor
        .process_directory(&config.input_dir, &config.output_dir)
        .await?;
    Ok(())
}
