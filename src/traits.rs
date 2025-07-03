use crate::errors::Result;
use image::DynamicImage;
use ndarray::prelude::*;

/// 画像セグメンテーションモデルの抽象化
///
/// 依存関係逆転原則（DIP）に従い、具象クラスではなく抽象に依存する
pub trait ImageSegmentationModel: Send + Sync {
    /// 画像のセグメンテーション処理を実行
    fn segment_image(&self, img: &DynamicImage) -> Result<DynamicImage>;

    /// モデルの入力画像サイズを取得
    fn get_image_size(&self) -> u32;

    /// テンソル予測（低レベルAPI）
    fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>>;
}

/// バッチ処理対応のセグメンテーションモデル
///
/// simple_ai_pipeline.mermaidアーキテクチャでのバッチ処理に対応
pub trait BatchImageSegmentationModel: ImageSegmentationModel {
    /// バッチ画像のセグメンテーション処理
    fn segment_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<DynamicImage>>;

    /// 最適なバッチサイズを取得
    fn get_optimal_batch_size(&self) -> usize;

    /// バッチテンソル予測
    fn predict_batch(&self, tensors: ArrayView4<f32>) -> Result<Array4<f32>>;
}

/// 画像処理パイプラインの抽象化
///
/// Clean Architectureの原則に従い、ビジネスロジックとインフラを分離
pub trait ImageProcessingPipeline: Send + Sync {
    /// ディレクトリ全体の処理
    fn process_directory(&self) -> Result<()>;

    /// 単一画像の処理
    fn process_single_image(
        &self,
        input_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<()>;

    /// サポートされている画像フォーマットの確認
    fn is_supported_format(&self, path: &std::path::Path) -> bool;
}

/// ワーカープール処理の抽象化
///
/// simple_ai_pipeline.mermaidの並列処理層に対応
pub trait WorkerPool<T, R>: Send + Sync {
    /// タスクをワーカープールに投入
    fn submit_task(&self, task: T) -> Result<()>;

    /// 結果を取得
    fn get_results(&self) -> Result<Vec<R>>;

    /// ワーカー数を取得
    fn worker_count(&self) -> usize;

    /// プールの停止
    fn shutdown(&self) -> Result<()>;
}

/// キューイングシステムの抽象化
///
/// simple_ai_pipeline.mermaidのRedis Queueに対応
pub trait MessageQueue<T>: Send + Sync {
    /// メッセージをキューに送信
    fn send(&self, message: T) -> Result<()>;

    /// メッセージをキューから受信
    fn receive(&self) -> Result<Option<T>>;

    /// バッチメッセージの送信
    fn send_batch(&self, messages: Vec<T>) -> Result<()>;

    /// バッチメッセージの受信
    fn receive_batch(&self, max_size: usize) -> Result<Vec<T>>;

    /// キューサイズの取得
    fn queue_size(&self) -> Result<usize>;
}

/// バッチング処理の抽象化
///
/// simple_ai_pipeline.mermaidのバッチャーに対応
pub trait Batcher<T>: Send + Sync {
    /// アイテムをバッチに追加
    fn add_item(&self, item: T) -> Result<()>;

    /// バッチが準備完了かチェック
    fn is_batch_ready(&self) -> bool;

    /// バッチを取得してクリア
    fn get_batch(&self) -> Result<Vec<T>>;

    /// 最大バッチサイズを設定
    fn set_max_batch_size(&mut self, size: usize);

    /// タイムアウト時間を設定
    fn set_timeout(&mut self, timeout_ms: u64);
}

/// 設定管理の抽象化
///
/// Convention over Configurationの原則に従った設定管理
pub trait ConfigurationProvider: Send + Sync {
    /// 設定値を取得
    fn get_string(&self, key: &str) -> Result<Option<String>>;
    fn get_int(&self, key: &str) -> Result<Option<i32>>;
    fn get_bool(&self, key: &str) -> Result<Option<bool>>;

    /// デフォルト値付きで設定値を取得
    fn get_string_or_default(&self, key: &str, default: &str) -> String;
    fn get_int_or_default(&self, key: &str, default: i32) -> i32;
    fn get_bool_or_default(&self, key: &str, default: bool) -> bool;
}
