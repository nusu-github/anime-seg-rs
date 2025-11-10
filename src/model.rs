use std::{ops::Div, path::Path};

use crate::{
    errors::{AnimeSegError, Result},
    traits::ImageSegmentationModel,
};
use image::{
    imageops, imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel,
    Primitive, Rgb,
};
use imageops_kit::{AlphaMaskError, Image, ModifyAlphaExt, NormalizedFrom, PaddingExt};
use imageproc::map::map_colors;
use ndarray::prelude::*;
use ort::value::TensorRef;
use ort::{
    execution_providers::{
        cuda::CuDNNConvAlgorithmSearch, ArenaExtendStrategy, CUDAExecutionProvider,
        TensorRTExecutionProvider,
    },
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session},
};
use parking_lot::Mutex;

/// ONNX model wrapper optimized with TensorRT + CUDA execution providers
/// to maximize GPU inference throughput
pub struct Model {
    pub image_size: u32,
    session: Mutex<Session>,
    #[allow(dead_code)]
    device_id: i32,
}

impl Model {
    pub fn new(model_path: &Path, device_id: i32) -> Result<Self> {
        let cache_dir = model_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("cache");
        std::fs::create_dir_all(&cache_dir).map_err(|e| AnimeSegError::FileSystem {
            path: cache_dir.clone(),
            operation: "cache directory creation".to_string(),
            source: e,
        })?;

        let trt_cache_dir = cache_dir.join("trt");
        std::fs::create_dir_all(&trt_cache_dir).map_err(|e| AnimeSegError::FileSystem {
            path: trt_cache_dir.clone(),
            operation: "TensorRT cache directory creation".to_string(),
            source: e,
        })?;

        let mut session = SessionBuilder::new()
            .map_err(|e| AnimeSegError::Model {
                operation: "session builder initialization".to_string(),
                source: Box::new(e),
            })?
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
                    .with_fp16(true)
                    .with_max_workspace_size(2_147_483_648)
                    .with_engine_cache(true)
                    .with_engine_cache_path(trt_cache_dir.display().to_string())
                    .with_timing_cache(true)
                    .with_timing_cache_path(cache_dir.join("timing.cache").display().to_string())
                    .with_min_subgraph_size(3)
                    .build(),
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .with_memory_limit(4_294_967_296)
                    .with_arena_extend_strategy(ArenaExtendStrategy::NextPowerOfTwo)
                    .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Exhaustive)
                    .with_cuda_graph(true)
                    .with_tf32(true)
                    .with_prefer_nhwc(true)
                    .build(),
            ])
            .map_err(|e| AnimeSegError::Model {
                operation: "execution provider configuration".to_string(),
                source: Box::new(e),
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AnimeSegError::Model {
                operation: "optimization level configuration".to_string(),
                source: Box::new(e),
            })?
            .with_intra_threads(4)
            .map_err(|e| AnimeSegError::Model {
                operation: "intra thread configuration".to_string(),
                source: Box::new(e),
            })?
            .with_parallel_execution(false)
            .map_err(|e| AnimeSegError::Model {
                operation: "parallel execution mode configuration".to_string(),
                source: Box::new(e),
            })?
            .with_memory_pattern(true)
            .map_err(|e| AnimeSegError::Model {
                operation: "memory pattern configuration".to_string(),
                source: Box::new(e),
            })?
            .with_optimized_model_path(cache_dir.join(format!(
                "{}.optimized.onnx",
                model_path.file_stem().unwrap_or_default().to_string_lossy()
            )))
            .map_err(|e| AnimeSegError::Model {
                operation: "optimized model path configuration".to_string(),
                source: Box::new(e),
            })?
            .commit_from_file(model_path)
            .map_err(|e| AnimeSegError::Model {
                operation: format!("model file loading: {}", model_path.display()),
                source: Box::new(e),
            })?;

        let image_size =
            session.inputs[0]
                .input_type
                .tensor_shape()
                .ok_or_else(|| AnimeSegError::Model {
                    operation: "model input shape extraction".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Cannot extract tensor shape",
                    )),
                })?[2] as u32;

        println!("Warming up model...");
        let warmup_data = Array4::<f32>::zeros((1, 3, image_size as usize, image_size as usize));
        session
            .run(ort::inputs!["img" => TensorRef::from_array_view(&warmup_data).map_err(|e| AnimeSegError::Model {
                operation: "warmup tensor creation".to_string(),
                source: Box::new(e),
            })?])
            .map_err(|e| AnimeSegError::Model {
                operation: "model warmup execution".to_string(),
                source: Box::new(e),
            })?;
        println!("Warmup complete");

        Ok(Self {
            image_size,
            session: Mutex::new(session),
            device_id,
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
        panic!("new_dummy should be replaced with trait-based design using MockSegmentationModel")
    }
}

impl ImageSegmentationModel for Model {
    fn segment_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        let rgb_img = img.to_rgb8();
        let (tensor, crop) = preprocess(&rgb_img, self.image_size)?;
        let mask = self.predict(tensor.view())?;
        let (width, height) = img.dimensions();

        let processed_mask = postprocess_mask(mask, self.image_size, crop, width, height);

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

        let batch_size = images.len();

        let mut batch_tensors = Vec::with_capacity(batch_size);
        let mut crop_infos = Vec::with_capacity(batch_size);
        let mut dimensions = Vec::with_capacity(batch_size);

        for img in images {
            let rgb_img = img.to_rgb8();
            let (tensor, crop) = preprocess(&rgb_img, self.image_size)?;
            batch_tensors.push(tensor);
            crop_infos.push(crop);
            dimensions.push(img.dimensions());
        }

        let batch_shape = (
            batch_size,
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

        let batch_masks = self.predict(batch_tensor.view())?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
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
        32
    }

    fn predict_batch(&self, tensors: ArrayView4<f32>) -> Result<Array4<f32>> {
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
    let (image, (x, y)) =
        image
            .to_square(Rgb([zero, zero, zero]))
            .map_err(|e| AnimeSegError::ImageProcessing {
                path: "unknown".to_string(),
                operation: "パディング追加".to_string(),
                source: Box::new(e),
            })?;

    let (width, height) = image.dimensions();
    let channels = 3;
    let raw = image.into_raw();

    let array_hwc = Array3::from_shape_vec((height as usize, width as usize, channels), raw)
        .map_err(|e| AnimeSegError::ImageProcessing {
            path: "unknown".to_string(),
            operation: "Converting an image to an ndarray".to_string(),
            source: Box::new(e),
        })?;
    let tensor = array_hwc.permuted_axes([2, 0, 1]).insert_axis(Axis(0));

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
