use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnimeSegError {
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Filesystem error: {operation} failed for {path:?}")]
    FileSystem {
        path: PathBuf,
        operation: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Image processing error: {operation} failed (file: {path})")]
    ImageProcessing {
        path: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error>,
    },

    #[error("Model error: {operation} failed")]
    Model {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error>,
    },

    #[error("Validation error: {field} {reason}")]
    Validation { field: String, reason: String },
}

pub type Result<T> = std::result::Result<T, AnimeSegError>;

impl From<anyhow::Error> for AnimeSegError {
    fn from(err: anyhow::Error) -> Self {
        // This is a catch-all. Not ideal, but for simplicity.
        AnimeSegError::Configuration {
            message: err.to_string(),
        }
    }
}

impl From<std::io::Error> for AnimeSegError {
    fn from(err: std::io::Error) -> Self {
        Self::FileSystem {
            path: PathBuf::from("unknown"),
            operation: "unknown".to_string(),
            source: err,
        }
    }
}

impl From<image::ImageError> for AnimeSegError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageProcessing {
            path: "unknown".to_string(),
            operation: "image processing".to_string(),
            source: Box::new(err),
        }
    }
}

impl From<ort::Error> for AnimeSegError {
    fn from(err: ort::Error) -> Self {
        Self::Model {
            operation: "ort operation".to_string(),
            source: Box::new(err),
        }
    }
}

impl From<ndarray::ShapeError> for AnimeSegError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::Model {
            operation: "tensor shape conversion".to_string(),
            source: Box::new(err),
        }
    }
}
