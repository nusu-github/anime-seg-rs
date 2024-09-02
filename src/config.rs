use clap::Parser;
use image::ImageFormat;
use std::path::PathBuf;

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Config {
    pub input_dir: PathBuf,

    #[arg(default_value = "output")]
    pub output_dir: PathBuf,

    #[arg(short, long)]
    pub model_path: PathBuf,

    #[arg(short, long, default_value = "png", value_parser = check_format)]
    pub format: String,

    #[arg(short, long, default_value_t = 0)]
    pub device_id: i32,

    #[arg(short, long, default_value_t = 1)]
    pub batch_size: u32,
}

impl Config {
    pub fn new() -> Self {
        Self::parse()
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
        .ok_or(format!("{} is not supported. {}", s, supported_message))?;
    if !format.writing_enabled() {
        return Err(format!("{} is not supported. {}", s, supported_message));
    }

    Ok(s.to_string())
}
