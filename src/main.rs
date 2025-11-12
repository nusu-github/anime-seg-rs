use anime_seg_rs::{Config, ImageProcessor, Result};

/// Entry point for the anime segmentation CLI tool.
///
/// # Why this is minimal
///
/// Following the thin main principle: main.rs only handles program initialization
/// and delegates all business logic to the library. This design keeps the binary
/// target separate from the library crate, enabling the library to be used as a
/// dependency by other projects without pulling in CLI-specific concerns.
fn main() -> Result<()> {
    let config = Config::new();

    println!("Loading model...");
    let mut processor = ImageProcessor::with_onnx_model(config)?;
    println!("Model loading complete");

    processor.process_directory()?;

    Ok(())
}
