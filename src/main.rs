use anime_seg_rs::distributed::DistributedImageProcessor;
use anime_seg_rs::queue::InMemoryQueueProvider;
use anime_seg_rs::{Config, Model};
use anyhow::Result;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::new();
    let model = Model::new(&config.model_path, config.device_id)?;
    let queue_provider = Arc::new(InMemoryQueueProvider::new());

    let processor = DistributedImageProcessor::new(model, queue_provider, config.clone());
    processor
        .process_directory(&config.input_dir, &config.output_dir)
        .await?;
    Ok(())
}
