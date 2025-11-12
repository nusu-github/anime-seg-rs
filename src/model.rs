use std::path::Path;

use crate::{
    errors::{AnimeSegError, Result},
    traits::ImageSegmentationModel,
};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgba};
use imageops_kit::{
    estimate_foreground_colors, AlphaMaskError, ApplyAlphaMaskExt, NormalizedFrom, PaddingExt,
};
use imageproc::map::map_colors;
use ndarray::prelude::*;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

pub struct Model {
    pub image_size: u32,
    session: Session,
}

impl Model {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| AnimeSegError::Model {
                operation: "session builder initialization".to_string(),
                source: Box::new(e),
            })?
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| AnimeSegError::Model {
                operation: "execution provider configuration".to_string(),
                source: Box::new(e),
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AnimeSegError::Model {
                operation: "optimization level configuration".to_string(),
                source: Box::new(e),
            })?
            .with_intra_threads(1)
            .map_err(|e| AnimeSegError::Model {
                operation: "intra thread configuration".to_string(),
                source: Box::new(e),
            })?
            .with_inter_threads(1)
            .map_err(|e| AnimeSegError::Model {
                operation: "inter thread configuration".to_string(),
                source: Box::new(e),
            })?
            .with_memory_pattern(false)
            .map_err(|e| AnimeSegError::Model {
                operation: "memory pattern configuration".to_string(),
                source: Box::new(e),
            })?
            .with_intra_op_spinning(false)
            .map_err(|e| AnimeSegError::Model {
                operation: "intra-op spinning configuration".to_string(),
                source: Box::new(e),
            })?
            .with_inter_op_spinning(false)
            .map_err(|e| AnimeSegError::Model {
                operation: "inter-op spinning configuration".to_string(),
                source: Box::new(e),
            })?
            .commit_from_file(model_path)
            .map_err(|e| AnimeSegError::Model {
                operation: format!("model file loading: {}", model_path.display()),
                source: Box::new(e),
            })?;

        let input_shape =
            session.inputs[0]
                .input_type
                .tensor_shape()
                .ok_or_else(|| AnimeSegError::Model {
                    operation: "model input type check".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Model input is not a tensor",
                    )),
                })?;

        let image_size = *input_shape.get(2).ok_or_else(|| AnimeSegError::Model {
            operation: "model input shape extraction".to_string(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Cannot extract image size from model input shape's 3rd dimension",
            )),
        })? as u32;

        Ok(Self {
            image_size,
            session,
        })
    }
}

impl ImageSegmentationModel for Model {
    fn segment_image(&mut self, img: &DynamicImage) -> Result<DynamicImage> {
        let (tensor, crop) = preprocess(img, self.image_size)?;
        let mask = self.predict(tensor.view())?;
        let (width, height) = img.dimensions();

        let processed_mask = postprocess_mask(mask, self.image_size, crop, width, height)?;

        let result = apply_mask_to_image(img, &processed_mask)?;
        Ok(result)
    }

    fn get_image_size(&self) -> u32 {
        self.image_size
    }

    fn predict(&mut self, tensor: ArrayView4<f32>) -> Result<Array4<f32>> {
        let outputs = self.session.run(
            ort::inputs!["img" => TensorRef::from_array_view(&tensor.as_standard_layout())?],
        )?;
        Ok(outputs["mask"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned())
    }
}

pub fn preprocess(image: &DynamicImage, image_size: u32) -> Result<(Array4<f32>, [u32; 4])> {
    let (w, h) = image.dimensions();
    let color = image.color();
    let mut dst_img = DynamicImage::new(image_size, image_size, color);

    let mut resizer = Resizer::new();
    let resize_options =
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer
        .resize(image, &mut dst_img, Some(&resize_options))
        .map_err(|e| AnimeSegError::ImageProcessing {
            path: "unknown".to_string(),
            operation: "image resizing".to_string(),
            source: Box::new(e),
        })?;

    let rgb_image = dst_img.to_rgb8();

    // Apply square padding
    let (padded_image, (x, y)) =
        rgb_image
            .to_square(Rgb([0u8, 0u8, 0u8]))
            .map_err(|e| AnimeSegError::ImageProcessing {
                path: "unknown".to_string(),
                operation: "padding".to_string(),
                source: Box::new(e),
            })?;

    let (width, height) = padded_image.dimensions();
    let raw = padded_image.into_raw();

    let array_hwc =
        Array3::from_shape_vec((height as usize, width as usize, 3), raw).map_err(|e| {
            AnimeSegError::ImageProcessing {
                path: "unknown".to_string(),
                operation: "Converting an image to an ndarray".to_string(),
                source: Box::new(e),
            }
        })?;
    let tensor = array_hwc.permuted_axes([2, 0, 1]).insert_axis(Axis(0));

    let tensor = tensor.map(|v| (*v as f32) / 255.0);

    Ok((tensor, [x as u32, y as u32, w, h]))
}

pub fn postprocess_mask(
    mask: Array4<f32>,
    image_size: u32,
    crop: [u32; 4],
    width: u32,
    height: u32,
) -> Result<ImageBuffer<Luma<f32>, Vec<f32>>> {
    let [x, y, w, h] = crop;
    let mask: ImageBuffer<Luma<f32>, Vec<f32>> =
        ImageBuffer::from_raw(image_size, image_size, mask.into_raw_vec_and_offset().0)
            .expect("Failed to create mask buffer from raw data");

    let src_img = DynamicImage::from(mask);
    let mut dst_img = DynamicImage::from(ImageBuffer::<Luma<f32>, Vec<f32>>::new(width, height));
    let mut resizer = Resizer::new();
    let resize_options =
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer
        .resize(&src_img, &mut dst_img, Some(&resize_options))
        .map_err(|e| AnimeSegError::ImageProcessing {
            path: "unknown".to_string(),
            operation: "image resizing".to_string(),
            source: Box::new(e),
        })?;

    let dst_img = dst_img.view(x, y, w, h).inner().to_owned();

    Ok(dst_img.to_luma32f())
}

fn apply_mask_to_image(
    img: &DynamicImage,
    mask: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> Result<DynamicImage> {
    let rgba_img = img.to_rgb8();
    let mask = map_colors(mask, |Luma([v])| Luma([NormalizedFrom::normalized_from(v)]));
    let rgba_img = estimate_foreground_colors(&rgba_img, &mask, 91, 1)
        .map_err(|err| match err {
            AlphaMaskError::BlurFusionError(ref source) => AnimeSegError::ImageProcessing {
                path: "unknown".to_string(),
                operation: source.to_string(),
                source: Box::new(err),
            },
            _ => AnimeSegError::ImageProcessing {
                path: "unknown".to_string(),
                operation: "foreground color estimation".to_string(),
                source: Box::new(err),
            },
        })?
        .apply_alpha_mask(&mask)
        .map_err(|err| match err {
            AlphaMaskError::DimensionMismatch { expected, actual } => {
                AnimeSegError::ImageProcessing {
                    path: "unknown".to_string(),
                    operation: "mask application".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Image and mask dimensions do not match: image {}x{}, mask {}x{}",
                            expected.0, expected.1, actual.0, actual.1
                        ),
                    )),
                }
            }
            _ => AnimeSegError::ImageProcessing {
                path: "unknown".to_string(),
                operation: "mask application".to_string(),
                source: Box::new(err),
            },
        })?;
    let rgba_img = map_colors(&rgba_img, |Rgba([r, g, b, a])| {
        Rgba([
            NormalizedFrom::normalized_from(r),
            NormalizedFrom::normalized_from(g),
            NormalizedFrom::normalized_from(b),
            NormalizedFrom::normalized_from(a),
        ])
    });
    Ok(DynamicImage::ImageRgba8(rgba_img))
}
