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

    #[arg(long, default_value_t = 5000)]
    pub batch_timeout_ms: u64,

    #[arg(long, default_value_t = 4)]
    pub preprocessing_workers: usize,

    #[arg(long, default_value_t = 4)]
    pub postprocessing_workers: usize,

    #[arg(long, default_value_t = 100)]
    pub max_inference_queue_size: usize,

    #[arg(long, default_value_t = 30)]
    pub worker_timeout_secs: u64,

    #[arg(long, default_value_t = 5)]
    pub inference_timeout_per_batch_item_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        Self::parse()
    }

    /// バッチサイズに基づいて推論ワーカーのタイムアウトを計算
    pub const fn inference_worker_timeout(&self) -> std::time::Duration {
        let base_timeout = self.worker_timeout_secs;
        let batch_size_multiplier =
            self.batch_size as u64 * self.inference_timeout_per_batch_item_secs;
        std::time::Duration::from_secs(base_timeout + batch_size_multiplier)
    }

    /// 前処理・後処理ワーカーのタイムアウト
    pub const fn standard_worker_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.worker_timeout_secs)
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
