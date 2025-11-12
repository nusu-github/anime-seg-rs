use clap::Parser;
use image::ImageFormat;
use std::path::PathBuf;

/// Command-line configuration for the anime segmentation tool.
///
/// # Why clap's derive macro
///
/// Using clap's derive API provides automatic help generation, type validation,
/// and ergonomic argument parsing without manual string manipulation. The derive
/// approach is preferred over the builder API for its compile-time safety and
/// reduced boilerplate.
#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Glob pattern for input images (e.g., "input/**/*.jpg" or "input/**/*").
    pub input_pattern: String,

    /// Path to the directory where output images will be saved.
    #[arg(default_value = "output")]
    pub output_dir: PathBuf,

    /// Path to the ONNX model file.
    #[arg(short, long)]
    pub model_path: PathBuf,

    /// Output image format.
    #[arg(short, long, default_value = "png", value_parser = check_format)]
    pub format: String,
}

impl Config {
    pub fn new() -> Self {
        Self::parse()
    }
}

impl Default for Config {
    /// Provide a default configuration for testing purposes.
    ///
    /// # Why this exists
    ///
    /// Tests need valid Config instances without requiring command-line arguments.
    /// This default implementation provides placeholder values that satisfy the
    /// type system. Production code uses `Config::new()` which parses real arguments.
    fn default() -> Self {
        Config::parse_from(["test", "input/**/*", "--model-path", "model.onnx"])
    }
}

/// Validate that the requested format is supported for writing.
///
/// # Why validation at parse time
///
/// Failing early during argument parsing provides immediate feedback to users
/// rather than discovering unsupported formats after potentially expensive processing.
/// The validation uses the image crate's runtime capability detection, ensuring
/// we only accept formats that are actually enabled through feature flags.
fn check_format(s: &str) -> Result<String, String> {
    let supported: Vec<_> = ImageFormat::all()
        .filter(|f| f.writing_enabled())
        .flat_map(|f| f.extensions_str())
        .map(|s| format!("`{}`", s))
        .collect();
    let supported_message = format!("Supported formats: {}", supported.join(", "));

    let format = ImageFormat::from_extension(s)
        .ok_or_else(|| format!("{} is not supported. {}", s, supported_message))?;
    if !format.writing_enabled() {
        return Err(format!("{} is not supported. {}", s, supported_message));
    }

    Ok(s.to_string())
}
