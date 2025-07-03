use std::path::PathBuf;
use thiserror::Error;

/// アプリケーション固有のエラータイプ
///
/// 分散処理環境でのエラーハンドリングを強化し、
/// 適切なエラー分類とリカバリー戦略を提供する
#[derive(Error, Debug)]
pub enum AnimeSegError {
    /// 設定関連のエラー
    #[error("設定エラー: {message}")]
    Configuration { message: String },

    /// ファイルシステム関連のエラー
    #[error("ファイルシステムエラー: {path}で{operation}に失敗")]
    FileSystem {
        path: PathBuf,
        operation: String,
        #[source]
        source: std::io::Error,
    },

    /// 画像処理関連のエラー
    #[error("画像処理エラー: {operation}に失敗 (ファイル: {path})")]
    ImageProcessing {
        path: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// モデル関連のエラー
    #[error("モデルエラー: {operation}に失敗")]
    Model {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// バッチ処理関連のエラー
    #[error("バッチ処理エラー: バッチサイズ{batch_size}で{operation}に失敗")]
    BatchProcessing {
        batch_size: usize,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// ワーカープール関連のエラー
    #[error("ワーカープールエラー: プール{pool_id}で{operation}に失敗")]
    WorkerPool {
        pool_id: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// キューイングシステム関連のエラー
    #[error("キューエラー: キュー{queue_name}で{operation}に失敗")]
    Queue {
        queue_name: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// ネットワーク関連のエラー（Redis等）
    #[error("ネットワークエラー: {endpoint}への{operation}に失敗")]
    Network {
        endpoint: String,
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// タイムアウトエラー
    #[error("タイムアウトエラー: {operation}が{timeout_ms}ms以内に完了しませんでした")]
    Timeout { operation: String, timeout_ms: u64 },

    /// リソース不足エラー
    #[error(
        "リソース不足: {resource_type}が不足しています（要求: {required}, 利用可能: {available}）"
    )]
    ResourceExhausted {
        resource_type: String,
        required: u64,
        available: u64,
    },

    /// 検証エラー
    #[error("検証エラー: {field}は{reason}")]
    Validation { field: String, reason: String },

    /// システム関連のエラー
    #[error("システムエラー: {operation}に失敗")]
    System {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// 一般的なアプリケーションエラー
    #[error("アプリケーションエラー: {message}")]
    Application {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

/// 結果型のエイリアス
pub type Result<T> = std::result::Result<T, AnimeSegError>;

impl AnimeSegError {
    /// エラーが再試行可能かどうかを判定
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::Timeout { .. } => true,
            Self::ResourceExhausted { .. } => true,
            Self::Queue { .. } => true,
            Self::WorkerPool { .. } => false, // プールエラーは通常再試行不可
            Self::Model { .. } => false,      // モデルエラーは通常再試行不可
            Self::Configuration { .. } => false,
            Self::Validation { .. } => false,
            Self::FileSystem { .. } => false,
            Self::ImageProcessing { .. } => false,
            Self::BatchProcessing { .. } => true, // バッチサイズ調整で再試行可能
            Self::System { .. } => false,
            Self::Application { .. } => false,
        }
    }

    /// エラーの重要度を取得
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

    /// 推奨されるリカバリー戦略を取得
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

/// エラーの重要度
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// リカバリー戦略
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// 失敗として処理
    Fail,
    /// 指定回数再試行
    Retry { max_attempts: u32, backoff_ms: u64 },
    /// 待機後再試行
    WaitAndRetry { wait_ms: u64 },
    /// バッチサイズを減らして再試行
    ReduceBatchSizeAndRetry,
    /// 代替処理に切り替え
    Fallback,
}

/// anyhow::Errorからの変換
impl From<anyhow::Error> for AnimeSegError {
    fn from(err: anyhow::Error) -> Self {
        Self::Application {
            message: err.to_string(),
            source: Some(err.into()),
        }
    }
}

/// std::io::Errorからの変換
impl From<std::io::Error> for AnimeSegError {
    fn from(err: std::io::Error) -> Self {
        Self::FileSystem {
            path: PathBuf::from("unknown"),
            operation: "unknown".to_string(),
            source: err,
        }
    }
}

/// image::ImageErrorからの変換
impl From<image::ImageError> for AnimeSegError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageProcessing {
            path: "unknown".to_string(),
            operation: "image processing".to_string(),
            source: Box::new(err),
        }
    }
}

/// ort::Errorからの変換
impl From<ort::Error> for AnimeSegError {
    fn from(err: ort::Error) -> Self {
        Self::Model {
            operation: "ort operation".to_string(),
            source: Box::new(err),
        }
    }
}

/// ndarray::ShapeErrorからの変換
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
