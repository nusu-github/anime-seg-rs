use std::{path::Path, sync::OnceLock};

use anyhow::Result;
use image::{
    imageops, imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb,
    RgbImage,
};
use ndarray::prelude::*;
use ort::{CUDAExecutionProvider, Session};

use crate::imageops_ai::padding::{get_position, padding_square, Position};

static SESSION: OnceLock<Session> = OnceLock::new();

#[derive(Copy, Clone)]
pub struct Model {
    pub image_size: u32,

    session: &'static Session,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, device_id: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default()
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

    pub fn predict(&self, tensor: ArrayView4<f32>) -> Result<Vec<f32>> {
        let outputs = self.session.run(ort::inputs!["img" => tensor]?)?;
        Ok(outputs["mask"].try_extract_raw_tensor::<f32>()?.1.to_vec())
    }
}

pub fn preprocess(image: &RgbImage, image_size: u32) -> (Array4<f32>, [u32; 4]) {
    let image = imageops::resize(image, image_size, image_size, FilterType::Lanczos3);
    let (w, h) = image.dimensions();
    let (x, y) = get_position(w, h, image_size, image_size, Position::Center).unwrap_or_default();
    let image = padding_square(&image, Position::Center, Rgb([0, 0, 0])).unwrap();
    let image = DynamicImage::from(image).into_rgb32f();

    let tensor = Array3::from_shape_vec(
        (image_size as usize, image_size as usize, 3),
        image.into_raw(),
    )
    .unwrap()
    .permuted_axes([2, 0, 1])
    .slice(s![NewAxis, .., .., ..])
    .into_owned();

    (tensor, [x as u32, y as u32, w, h])
}

pub fn postprocess_mask(
    mask: Vec<f32>,
    image_size: u32,
    crop: [u32; 4],
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let [x, y, w, h] = crop;
    let mask = ImageBuffer::from_raw(image_size, image_size, mask).unwrap();
    let mask = mask.view(x, y, w, h).to_image();
    imageops::resize(&mask, width, height, FilterType::Lanczos3)
}
