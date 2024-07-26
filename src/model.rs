use std::{path::Path, sync::OnceLock};

use anyhow::Result;
use image::{
    imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, RgbImage,
    RgbaImage,
};
use ndarray::prelude::*;
use ort::Session;

use crate::imageops_ai::{
    clip_minimum_border::clip_minimum_border,
    mask::apply_mask,
    padding::{get_position, padding_square, Position},
};

static SESSION: OnceLock<Session> = OnceLock::new();

#[derive(Copy, Clone)]
pub struct Model {
    image_size: u32,
    session: &'static Session,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, device_id: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .with_memory_pattern(true)?
            .commit_from_file(model_path)?;

        let image_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[2] as u32;

        Ok(Self {
            image_size,
            session: SESSION.get_or_init(|| session),
        })
    }

    pub fn predict(&self, image: &RgbImage) -> Result<RgbaImage> {
        let (width, height) = image.dimensions();
        let (tensor, crop) = self.preprocess(image);

        let outputs = self.session.run(ort::inputs!["img" => tensor.view()]?)?;
        let mask = outputs["mask"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix4>()?;
        let mask = self.postprocess_mask(mask, crop, width, height);

        let image = apply_mask(image, &mask, true).unwrap();
        Ok(clip_minimum_border(image, 1, 8))
    }

    fn preprocess(&self, image: &RgbImage) -> (Array4<f32>, [u32; 4]) {
        let image = image::imageops::resize(
            image,
            self.image_size,
            self.image_size,
            FilterType::Lanczos3,
        );
        let (w, h) = image.dimensions();
        let (x, y) = get_position(w, h, self.image_size, self.image_size, Position::Center)
            .unwrap_or_default();
        let image = padding_square(&image, Position::Center, Rgb([0, 0, 0])).unwrap();
        let image = DynamicImage::from(image).into_rgb32f();

        let tensor = unsafe {
            ArrayView3::from_shape_ptr(
                (self.image_size as usize, self.image_size as usize, 3),
                image.as_ptr(),
            )
        }
        .permuted_axes([2, 0, 1])
        .slice(s![NewAxis, .., .., ..])
        .into_owned();

        (tensor, [x as u32, y as u32, w, h])
    }

    fn postprocess_mask(
        &self,
        mask: ArrayView4<f32>,
        crop: [u32; 4],
        width: u32,
        height: u32,
    ) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let mask = mask.slice(s![0, 0, .., ..]).to_owned();
        let [x, y, w, h] = crop;
        let mask =
            ImageBuffer::from_raw(self.image_size, self.image_size, mask.into_raw_vec()).unwrap();
        let mask = mask.view(x, y, w, h).to_image();
        image::imageops::resize(&mask, width, height, FilterType::Lanczos3)
    }
}
