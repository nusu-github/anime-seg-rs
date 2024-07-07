use std::{path::Path, sync::Arc};

use anyhow::Result;
use image::{
    imageops::{self, FilterType},
    DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage, RgbaImage,
};
use ndarray::prelude::*;
use num_traits::AsPrimitive;
use ort::Session;

use crate::imageops_ai::{
    clip_minimum_border::{clip_minimum_border, Crop},
    mask::{apply_mask, Gray32FImage},
    padding::{get_position, padding_square, Position},
};

#[derive(Clone)]
pub struct Model {
    image_size: u32,
    session: Arc<Session>,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, num_threads: usize, device_id: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .with_intra_threads(num_threads)?
            .with_memory_pattern(true)?
            .commit_from_file(model_path)?;

        let image_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[2];

        let tensor = Array4::<f32>::zeros([1, 3, image_size.as_(), image_size.as_()]);
        session.run(ort::inputs!["img" => tensor.view()]?)?;

        Ok(Self {
            image_size: image_size.as_(),
            session: Arc::new(session),
        })
    }

    pub fn predict(&self, image: &RgbImage) -> Result<RgbaImage> {
        let (width, height) = image.dimensions();
        let (tensor, crop) = self.preprocess(image)?;

        let mask = self.predict_mask(&tensor, crop)?;
        let mask = imageops::resize(&mask, width, height, FilterType::Lanczos3);

        let image = apply_mask(image, &mask, true).unwrap();
        let image = clip_minimum_border(image, 1, 8);

        Ok(image)
    }

    fn predict_mask(&self, tensor: &Array4<f32>, crop: Crop) -> Result<Gray32FImage> {
        let outputs = self.session.run(ort::inputs!["img" => tensor.view()]?)?;
        let outputs = outputs["mask"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix4>()?;

        let mask = extract_mask(&outputs.view(), 0.75);

        let mask =
            ImageBuffer::from_raw(self.image_size, self.image_size, mask.into_raw_vec()).unwrap();
        let [x, y, w, h] = crop;
        let mask = mask.view(x, y, w, h).to_image();

        Ok(mask)
    }

    fn preprocess(&self, image: &RgbImage) -> Result<(Array4<f32>, Crop)> {
        let image = imageops::resize(
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
                (self.image_size.as_(), self.image_size.as_(), 3),
                image.as_ptr(),
            )
        }
        .permuted_axes([2, 0, 1])
        .slice(s![NewAxis, .., .., ..])
        .into_owned();

        Ok((tensor, [x.as_(), y.as_(), w, h]))
    }
}

fn extract_mask(tensor: &ArrayView4<f32>, threshold: f32) -> Array3<f32> {
    tensor
        .index_axis(Axis(0), 0)
        .permuted_axes([1, 2, 0])
        .mapv(|x| if x > threshold { x } else { 0.0 })
}
