use clap::Parser;
use image::ImageFormat;
use std::path::PathBuf;

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
    fn default() -> Self {
        // This is mainly for tests, parse() is the main way to get a config.
        // We need to provide dummy patterns that are valid for parsing.
        Config::parse_from(["test", "input/**/*", "--model-path", "model.onnx"])
    }
}

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
