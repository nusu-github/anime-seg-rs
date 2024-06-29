# anime-seg-rs

anime-seg-rsは、[anime-segmentation](https://github.com/SkyTNT/anime-segmentation)
の推論部分をRustで実装したものです。このツールは、画像からアニメキャラクターをセグメンテーション（分割抽出）するために使用されます。

## 重要な注意点

- ONNXモデルを用いた推論のみ

## 機能

- ディレクトリ内の画像のバッチ処理
- 様々な画像フォーマット（PNG、JPEG、WebP）のサポート
- CUDAによる高速化（オプション）
- マルチスレッド処理
- 処理状況を追跡するプログレスバー
- 高速推論のためのFP16（半精度）モデルのサポート

## 前提条件

- Rust 1.68以降
- CUDAツールキット（オプション）

## インストール

1. リポジトリをクローンします：
   ```shell
   git clone https://github.com/yourusername/anime-seg-rs.git
   cd anime-seg-rs
   ```

2. プロジェクトをビルドします：
   ```shell
   cargo build --release
   ```

   CUDAサポートを有効にする場合は、`cuda`を追加します：
   ```shell
   cargo build --release --features cuda
   ```

## 使用方法

以下のコマンドでツールを実行します：

```shell
anime-seg-rs --input-dir <入力ディレクトリ> --output-dir <出力ディレクトリ> --model-path <モデルパス> [オプション]
```

### 引数：

- `--input-dir, -i`: 入力画像を含むディレクトリ
- `--output-dir, -o`: 処理済み画像を保存するディレクトリ
- `--model-path, -m`: ISNetモデルファイルのパス（ONNXフォーマット）
- `--format, -f`: 出力画像フォーマット（デフォルト: "png"）
- `--device-id, -d`: CUDA デバイスID（デフォルト: 0）
- `--num-threads, -n`: 処理スレッド数（デフォルト: 論理CPUの数）
    - バッチサイズも兼任しているのでGPUのメモリ次第で調整してください

### 使用例：

```shell
anime-seg-rs --input-dir ./input_images --output-dir ./output_images --model-path ./isnet_fp16.onnx --format png
```

## ソースからのビルド

ソースからプロジェクトをビルドするには、RustとCargoがインストールされていることを確認してから、以下を実行します:

```shell
cargo build --release
```

## モデルの互換性と最適化

このRust実装は、元のanime-segmentationプロジェクトのISNet ONNXモデルで動作するように
設計されています。事前学習済みのISNetモデルは [Hugging Face](https://huggingface.co/skytnt/anime-seg)
からダウンロードできます。

### FP16モデル変換

より高速な推論のために、元のFP32モデルをFP16（半精度）に変換することをおすすめします。
これにより、特にGPU上でのパフォーマンスが大幅に向上する可能性があります。
モデルを変換するには、onnxconverter-commonをインストールして、
元のanime-segmentationプロジェクトの `export.py`スクリプトに以下のようなコードを追加してください。

```shell
pip install onnxconverter-common
```

```python
from onnxconverter_common import float16

# 元のモデルを読み込む
model = onnx.load("元のモデルへのパス.onnx")

# FP16に変換
model_fp16 = float16.convert_float_to_float16(model, min_positive_val=2e-11, max_finite_val=6)
```

## ライセンス

- [Apache License 2.0](LICENSE)

## 謝辞

- [SkyTNT](https://github.com/SkyTNT)による元の[anime-segmentation](https://github.com/SkyTNT/anime-segmentation)プロジェクト
