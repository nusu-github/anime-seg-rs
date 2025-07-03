use std::{ops::Div, path::Path};

use crate::{
    errors::{AnimeSegError, Result},
    traits::ImageSegmentationModel,
};
use image::{
    imageops, imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel,
    Primitive, Rgb,
};
use imageops_ai::{AlphaMaskError, Image, ModifyAlpha, NormalizedFrom, Padding};
use imageproc::map::map_colors;
use ndarray::prelude::*;
use nshare::AsNdarray3;
use ort::value::TensorRef;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::SessionBuilder, Session},
};
use parking_lot::Mutex;

pub struct Model {
    pub image_size: u32,
    session: Mutex<Session>,
}

impl Model {
    pub fn new(model_path: &Path, device_id: i32) -> Result<Self> {
        let mut session = SessionBuilder::new()
            .map_err(|e| AnimeSegError::Model {
                operation: "セッションビルダー初期化".to_string(),
                source: Box::new(e),
            })?
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
            ])
            .map_err(|e| AnimeSegError::Model {
                operation: "実行プロバイダー設定".to_string(),
                source: Box::new(e),
            })?
            .with_memory_pattern(true)
            .map_err(|e| AnimeSegError::Model {
                operation: "メモリパターン設定".to_string(),
                source: Box::new(e),
            })?
            .commit_from_file(model_path)
            .map_err(|e| AnimeSegError::Model {
                operation: format!("モデルファイル読み込み: {}", model_path.display()),
                source: Box::new(e),
            })?;

        let image_size =
            session.inputs[0]
                .input_type
                .tensor_shape()
                .ok_or_else(|| AnimeSegError::Model {
                    operation: "モデル入力形状取得".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "テンソル形状が取得できません",
                    )),
                })?[2] as u32;

        // initialize model
        let data = Array4::<f32>::zeros((1, 3, image_size as usize, image_size as usize));
        session.run(ort::inputs!["img" => TensorRef::from_array_view(&data).map_err(|e| AnimeSegError::Model {
            operation: "初期化テンソル作成".to_string(),
            source: Box::new(e),
        })?]).map_err(|e| AnimeSegError::Model {
            operation: "モデル初期化実行".to_string(),
            source: Box::new(e),
        })?;

        Ok(Self {
            image_size,
            session: Mutex::new(session),
        })
    }

    pub fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>> {
        let mut binding = self.session.lock();
        let outputs = binding.run(
            ort::inputs!["img" => TensorRef::from_array_view(&tensor.as_standard_layout())?],
        )?;
        Ok(outputs["mask"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned())
    }

    #[cfg(test)]
    pub fn new_dummy() -> Self {
        // テスト用の簡易実装
        // 実際にはモックやトレイトベースの設計を使用すべき
        panic!("new_dummy はトレイトベースのリファクタリング後に実装予定")
    }
}

impl ImageSegmentationModel for Model {
    fn segment_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        let rgb_img = img.to_rgb8();
        let (tensor, crop) = preprocess(&rgb_img, self.image_size)?;
        let mask = self.predict(tensor.view())?;
        let (width, height) = img.dimensions();

        let processed_mask = postprocess_mask(mask, self.image_size, crop, width, height);

        // マスクを適用して前景を抽出
        let result = apply_mask_to_image(img, &processed_mask)?;
        Ok(result)
    }

    fn get_image_size(&self) -> u32 {
        self.image_size
    }

    fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>> {
        let mut binding = self.session.lock();
        let outputs = binding.run(
            ort::inputs!["img" => TensorRef::from_array_view(&tensor.as_standard_layout())?],
        )?;
        Ok(outputs["mask"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned())
    }
}

impl crate::traits::BatchImageSegmentationModel for Model {
    fn segment_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<DynamicImage>> {
        if images.is_empty() {
            return Ok(vec![]);
        }

        // 各画像の前処理とメタデータを収集
        let mut batch_tensors = Vec::with_capacity(images.len());
        let mut crop_infos = Vec::with_capacity(images.len());
        let mut dimensions = Vec::with_capacity(images.len());

        for img in images {
            let rgb_img = img.to_rgb8();
            let (tensor, crop) = preprocess(&rgb_img, self.image_size)?;
            batch_tensors.push(tensor);
            crop_infos.push(crop);
            dimensions.push(img.dimensions());
        }

        // バッチテンソルを作成（バッチ次元で結合）
        let batch_shape = (
            images.len(),
            3,
            self.image_size as usize,
            self.image_size as usize,
        );
        let mut batch_tensor = Array4::<f32>::zeros(batch_shape);

        for (i, tensor) in batch_tensors.iter().enumerate() {
            batch_tensor
                .slice_mut(s![i..i + 1, .., .., ..])
                .assign(tensor);
        }

        // バッチ推論
        let batch_masks = self.predict(batch_tensor.view())?;

        // 各マスクを後処理して結果を生成
        let mut results = Vec::with_capacity(images.len());
        for i in 0..images.len() {
            let mask = batch_masks.slice(s![i, .., .., ..]).to_owned();
            let mask_3d = mask.into_dimensionality::<Ix3>()?;
            let mask_4d = mask_3d.insert_axis(Axis(0));

            let (width, height) = dimensions[i];
            let processed_mask =
                postprocess_mask(mask_4d, self.image_size, crop_infos[i], width, height);

            let result = apply_mask_to_image(&images[i], &processed_mask)?;
            results.push(result);
        }

        Ok(results)
    }

    fn get_optimal_batch_size(&self) -> usize {
        // GPUメモリに基づいて最適なバッチサイズを決定
        // TODO: 実際のGPUメモリから計算
        32
    }

    fn predict_batch(&self, tensors: ArrayView4<f32>) -> Result<Array4<f32>> {
        // predict メソッドは既にバッチ対応
        self.predict(tensors)
    }
}

pub fn preprocess<S>(image: &Image<Rgb<S>>, image_size: u32) -> Result<(Array4<f32>, [u32; 4])>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Primitive + 'static,
{
    let image = imageops::resize(image, image_size, image_size, FilterType::Lanczos3);
    let (w, h) = image.dimensions();
    let zero = S::zero();
    let (image, (x, y)) = image
        .add_padding_square(Rgb([zero, zero, zero]))
        .map_err(|e| AnimeSegError::ImageProcessing {
            path: "unknown".to_string(),
            operation: "パディング追加".to_string(),
            source: Box::new(e),
        })?;

    let tensor = image.as_ndarray3().slice_move(s![NewAxis, ..;-1, .., ..]);
    let max = S::DEFAULT_MAX_VALUE.into();
    let tensor = if max == (<f32 as Primitive>::DEFAULT_MAX_VALUE) {
        tensor.map(|v| (*v).into())
    } else {
        tensor.map(|v| <S as Into<f32>>::into(*v).div(max))
    };

    Ok((tensor, [x as u32, y as u32, w, h]))
}

pub fn postprocess_mask<S: Primitive + 'static>(
    mask: Array4<S>,
    image_size: u32,
    crop: [u32; 4],
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<S>, Vec<S>> {
    let [x, y, w, h] = crop;
    let mask =
        ImageBuffer::from_raw(image_size, image_size, mask.into_raw_vec_and_offset().0).unwrap();
    let mask = mask.view(x, y, w, h).inner().to_owned();
    imageops::resize(&mask, width, height, FilterType::Lanczos3)
}

fn apply_mask_to_image(
    img: &DynamicImage,
    mask: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> Result<DynamicImage> {
    let mut rgba_img = img.to_rgba8();
    let mask = map_colors(mask, |Luma([alpha])| {
        Luma([NormalizedFrom::normalized_from(alpha)])
    });
    rgba_img.replace_alpha_mut(&mask).map_err(|err| match err {
        AlphaMaskError::DimensionMismatch { expected, actual } => AnimeSegError::ImageProcessing {
            path: "unknown".to_string(),
            operation: "マスク適用".to_string(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "画像とマスクのサイズが一致しません: 画像{}x{}, マスク{}x{}",
                    expected.0, expected.1, actual.0, actual.1
                ),
            )),
        },
        _ => AnimeSegError::ImageProcessing {
            path: "unknown".to_string(),
            operation: "マスク適用".to_string(),
            source: Box::new(err),
        },
    })?;
    Ok(DynamicImage::ImageRgba8(rgba_img))
}
