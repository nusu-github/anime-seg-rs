use std::path::PathBuf;
use thiserror::Error;

/// Error types designed for distributed processing with structured classification
/// to enable intelligent retry strategies and proper error recovery
#[derive(Error, Debug)]
pub enum AnimeSegError {
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Filesystem error: {operation} failed for {path}")]
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
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Model error: {operation} failed")]
    Model {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Batch processing error: {operation} failed for batch size {batch_size}")]
    BatchProcessing {
        batch_size: usize,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Worker pool error: {operation} failed for pool {pool_id}")]
    WorkerPool {
        pool_id: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Queue error: {operation} failed for queue {queue_name}")]
    Queue {
        queue_name: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Network error: {operation} failed for {endpoint}")]
    Network {
        endpoint: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Timeout error: {operation} did not complete within {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },

    #[error(
        "Resource exhausted: {resource_type} insufficient (required: {required}, available: {available})"
    )]
    ResourceExhausted {
        resource_type: String,
        required: u64,
        available: u64,
    },

    #[error("Validation error: {field} {reason}")]
    Validation { field: String, reason: String },

    #[error("System error: {operation} failed")]
    System {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Application error: {message}")]
    Application {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

pub type Result<T> = std::result::Result<T, AnimeSegError>;

impl AnimeSegError {
    /// Determines if error warrants automatic retry based on transient failure patterns
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::Timeout { .. } => true,
            Self::ResourceExhausted { .. } => true,
            Self::Queue { .. } => true,
            Self::WorkerPool { .. } => false,
            Self::Model { .. } => false,
            Self::Configuration { .. } => false,
            Self::Validation { .. } => false,
            Self::FileSystem { .. } => false,
            Self::ImageProcessing { .. } => false,
            Self::BatchProcessing { .. } => true,
            Self::System { .. } => false,
            Self::Application { .. } => false,
        }
    }

    /// Returns severity level to prioritize error handling and logging
    pub const fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Configuration { .. } => ErrorSeverity::Critical,
            Self::Model { .. } => ErrorSeverity::Critical,
            Self::Validation { .. } => ErrorSeverity::High,
            Self::FileSystem { .. } => ErrorSeverity::High,
            Self::ImageProcessing { .. } => ErrorSeverity::Medium,
            Self::Network { .. } => ErrorSeverity::Medium,
            Self::Queue { .. } => ErrorSeverity::Medium,
            Self::WorkerPool { .. } => ErrorSeverity::High,
            Self::BatchProcessing { .. } => ErrorSeverity::Medium,
            Self::Timeout { .. } => ErrorSeverity::Low,
            Self::ResourceExhausted { .. } => ErrorSeverity::High,
            Self::System { .. } => ErrorSeverity::High,
            Self::Application { .. } => ErrorSeverity::Medium,
        }
    }

    /// Suggests recovery strategy optimized for each error type's failure characteristics
    pub const fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::Network { .. } => RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff_ms: 1000,
            },
            Self::Timeout { .. } => RecoveryStrategy::Retry {
                max_attempts: 2,
                backoff_ms: 500,
            },
            Self::ResourceExhausted { .. } => RecoveryStrategy::WaitAndRetry { wait_ms: 5000 },
            Self::Queue { .. } => RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff_ms: 2000,
            },
            Self::BatchProcessing { .. } => RecoveryStrategy::ReduceBatchSizeAndRetry,
            Self::System { .. } => RecoveryStrategy::Fail,
            _ => RecoveryStrategy::Fail,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Recovery strategies tailored to different failure modes in distributed processing
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    Fail,
    Retry { max_attempts: u32, backoff_ms: u64 },
    WaitAndRetry { wait_ms: u64 },
    ReduceBatchSizeAndRetry,
    Fallback,
}

impl From<anyhow::Error> for AnimeSegError {
    fn from(err: anyhow::Error) -> Self {
        Self::Application {
            message: err.to_string(),
            source: Some(err.into()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryability() {
        let network_error = AnimeSegError::Network {
            endpoint: "redis://localhost:6379".to_string(),
            operation: "connect".to_string(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "test",
            )),
        };
        assert!(network_error.is_retryable());

        let config_error = AnimeSegError::Configuration {
            message: "Invalid configuration".to_string(),
        };
        assert!(!config_error.is_retryable());
    }

    #[test]
    fn test_error_severity() {
        let config_error = AnimeSegError::Configuration {
            message: "Test".to_string(),
        };
        assert_eq!(config_error.severity(), ErrorSeverity::Critical);

        let timeout_error = AnimeSegError::Timeout {
            operation: "test".to_string(),
            timeout_ms: 1000,
        };
        assert_eq!(timeout_error.severity(), ErrorSeverity::Low);
    }

    #[test]
    fn test_recovery_strategy() {
        let network_error = AnimeSegError::Network {
            endpoint: "test".to_string(),
            operation: "test".to_string(),
            source: Box::new(std::io::Error::other("test")),
        };

        match network_error.recovery_strategy() {
            RecoveryStrategy::Retry { max_attempts, .. } => {
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected Retry strategy"),
        }
    }
}
