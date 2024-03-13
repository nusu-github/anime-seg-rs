use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use image::{
    imageops, DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel, RgbImage, RgbaImage,
};
use ndarray::prelude::*;
use ort::{CUDAExecutionProvider, DirectMLExecutionProvider, Session, Tensor};

struct PreprocessedImage {
    tensor: Array4<f32>,
    crop: Crop,
}

struct Crop {
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
}

pub(crate) struct MaskPredictor {
    session: Mutex<Session>,
}

impl PreprocessedImage {
    fn new(image: &DynamicImage) -> Result<Self> {
        let img = image
            .resize(1024, 1024, imageops::FilterType::Lanczos3)
            .to_rgb8();
        let (width, height) = img.dimensions();
        let mut img_buf = RgbImage::new(1024, 1024);
        let x1 = (1024 - width) / 2;
        let y1 = (1024 - height) / 2;
        let x2 = x1 + width;
        let y2 = y1 + height;
        imageops::overlay(&mut img_buf, &img, x1 as i64, y1 as i64);

        let tensor = Array3::from_shape_vec((1024, 1024, 3), img_buf.into_raw())?;
        let tensor = tensor
            .slice(s![NewAxis, .., .., ..;-1])
            .permuted_axes([0, 3, 1, 2])
            .mapv(|x| f32::from(x) / 255.0);

        Ok(Self {
            tensor,
            crop: Crop { x1, y1, x2, y2 },
        })
    }
}

impl MaskPredictor {
    pub(crate) fn new(model_path: &Path, device_id: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .with_conv_max_workspace(true)
                    .with_copy_in_default_stream(true)
                    .build(),
                DirectMLExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
            ])?
            .with_memory_pattern(true)?
            .with_model_from_file(model_path)?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    pub(crate) fn predict(&self, image_path: &Path) -> Result<RgbaImage> {
        let (original_image, preprocessed_image) = MaskPredictor::preprocess_image(image_path)?;
        let (width, height) = original_image.dimensions();
        let mask = self.predict_mask(&preprocessed_image, width, height)?;

        let (width, height) = original_image.dimensions();
        let masked_image =
            MaskPredictor::apply_mask(&original_image.to_rgb8(), &mask, width, height);

        Ok(masked_image)
    }

    fn preprocess_image(image_path: &Path) -> Result<(DynamicImage, PreprocessedImage)> {
        let original_image = image::open(image_path)?;
        let preprocessed_image = PreprocessedImage::new(&original_image)?;
        Ok((original_image, preprocessed_image))
    }

    fn apply_mask(
        image: &RgbImage,
        mask: &ImageBuffer<Luma<f32>, Vec<f32>>,
        width: u32,
        height: u32,
    ) -> RgbaImage {
        let mut masked_image = RgbaImage::new(width, height);
        for ((mask_pixel, image_pixel), out_pixel) in mask
            .pixels()
            .zip(image.pixels())
            .zip(masked_image.pixels_mut())
        {
            let mask_value = mask_pixel.channels()[0];
            let channels = image_pixel
                .channels()
                .iter()
                .zip(out_pixel.channels_mut().iter_mut());
            for (in_channel, out_channel) in channels {
                *out_channel =
                    ((mask_value * f32::from(*in_channel)) + 255.0 * (1.0 - mask_value)) as u8;
            }
            out_pixel.channels_mut()[3] = (mask_value * 255.0) as u8;
        }
        masked_image
    }

    fn extract_mask(mask_tensor: &Tensor<f32>, crop: &Crop) -> Array3<f32> {
        mask_tensor
            .view()
            .index_axis(Axis(0), 0)
            .slice(s![
                ..,
                crop.y1 as usize..crop.y2 as usize,
                crop.x1 as usize..crop.x2 as usize,
            ])
            .permuted_axes([1, 2, 0])
            .to_owned()
    }

    fn predict_mask(
        &self,
        preprocessed_image: &PreprocessedImage,
        original_width: u32,
        original_height: u32,
    ) -> Result<ImageBuffer<Luma<f32>, Vec<f32>>> {
        let mask = {
            let session = self.session.lock().unwrap();
            let outputs = session.run(ort::inputs!["img" => preprocessed_image.tensor.view()]?)?;
            let outputs = outputs.get("mask").unwrap().extract_tensor::<f32>()?;
            MaskPredictor::extract_mask(&outputs, &preprocessed_image.crop)
        };

        let (height, width, _) = mask.dim();
        let mask = ImageBuffer::from_raw(width as u32, height as u32, mask.into_raw_vec()).unwrap();
        let mask = imageops::resize(
            &mask,
            original_width,
            original_height,
            imageops::FilterType::Lanczos3,
        );

        Ok(mask)
    }
}
