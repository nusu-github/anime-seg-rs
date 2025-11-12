use anime_seg_rs::{Config, ImageProcessor, Result};

fn main() -> Result<()> {
    let config = Config::new();

    println!("Loading model...");
    let mut processor = ImageProcessor::with_onnx_model(config)?;
    println!("Model loading complete");

    processor.process_directory()?;

    Ok(())
}
