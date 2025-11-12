use std::path::PathBuf;
use thiserror::Error;

/// Structured error types for the anime segmentation application.
///
/// # Why structured errors
///
/// Each variant captures context specific to its error domain (filesystem, image processing,
/// model operations, etc.), providing detailed diagnostic information without requiring
/// callers to parse error strings. The thiserror crate generates Display implementations
/// automatically from format strings, reducing boilerplate while maintaining type safety.
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

/// Convert anyhow errors to configuration errors.
///
/// # Why this conversion exists
///
/// Some dependencies return anyhow::Error which lacks structured error information.
/// Rather than propagating the generic error type throughout the codebase, we convert
/// to our domain-specific error type at boundaries. This trade-off prioritizes API
/// consistency over preserving fine-grained error details for these specific cases.
impl From<anyhow::Error> for AnimeSegError {
    fn from(err: anyhow::Error) -> Self {
        AnimeSegError::Configuration {
            message: err.to_string(),
        }
    }
}

/// Convert I/O errors to filesystem errors.
///
/// # Why default values for context
///
/// Some I/O errors occur without specific path/operation context. Rather than
/// requiring all callsites to wrap errors manually, this conversion provides
/// a fallback. Code that has context should construct AnimeSegError::FileSystem
/// directly with the specific path and operation.
impl From<std::io::Error> for AnimeSegError {
    fn from(err: std::io::Error) -> Self {
        Self::FileSystem {
            path: PathBuf::from("unknown"),
            operation: "unknown".to_string(),
            source: err,
        }
    }
}

/// Convert image crate errors to image processing errors.
impl From<image::ImageError> for AnimeSegError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageProcessing {
            path: "unknown".to_string(),
            operation: "image processing".to_string(),
            source: Box::new(err),
        }
    }
}

/// Convert ONNX Runtime errors to model errors.
impl From<ort::Error> for AnimeSegError {
    fn from(err: ort::Error) -> Self {
        Self::Model {
            operation: "ort operation".to_string(),
            source: Box::new(err),
        }
    }
}

/// Convert ndarray shape errors to model errors.
///
/// # Why model error category
///
/// Shape errors occur during tensor operations which are part of model inference,
/// so they're categorized as model errors rather than a separate tensor error type.
/// This keeps the error hierarchy flat and focused on user-facing error domains.
impl From<ndarray::ShapeError> for AnimeSegError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::Model {
            operation: "tensor shape conversion".to_string(),
            source: Box::new(err),
        }
    }
}
