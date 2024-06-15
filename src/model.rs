use std::{ops::{Div, Mul}, path::Path, sync::Arc};

use anyhow::Result;
use image::{
    EncodableLayout,
    GenericImageView, ImageBuffer, imageops::{self, FilterType}, Rgb, RgbaImage, RgbImage,
};
use ndarray::prelude::*;
use num_traits::AsPrimitive;
use ort::{Session, TensorElementType};

use crate::imageops_ai::{
    clip_minimum_border::{clip_minimum_border, Crop},
    mask::{apply_mask, Gray32FImage},
    padding::{get_position, padding_square, Position},
};

#[derive(Clone)]
enum Precision {
    Float32,
    Float16,
}

#[derive(Clone)]
pub struct Model {
    image_size: u32,
    session: Arc<Session>,

    precision: Precision,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, num_threads: usize, device_id: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                #[cfg(feature = "cuda")]
                ort::CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
            ])?
            .with_intra_threads(num_threads)?
            .with_memory_pattern(true)?
            .commit_from_file(model_path)?;

        let input_type = session.inputs[0].input_type.tensor_type().unwrap();
        let image_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[2];

        let tensor = Array4::<f32>::zeros([1, 3, image_size.as_(), image_size.as_()]);
        let precision = match input_type {
            TensorElementType::Float32 => {
                let tensor: Array4<f32> = tensor.mapv(AsPrimitive::as_);
                session.run(ort::inputs!["img" => tensor.view()]?)?;

                Precision::Float32
            }
            TensorElementType::Float16 => {
                let tensor: Array4<half::f16> = tensor.mapv(AsPrimitive::as_);
                session.run(ort::inputs!["img" => tensor.view()]?)?;

                Precision::Float16
            }
            _ => unimplemented!(),
        };

        Ok(Self {
            image_size: image_size.as_(),
            session: Arc::new(session),

            precision,
        })
    }

    pub fn predict(&self, image: &RgbImage) -> Result<RgbaImage> {
        let (width, height) = image.dimensions();
        let (tensor, crop) = self.preprocess(image)?;

        let mask = self.predict_mask(tensor.view(), crop)?;
        let mask = imageops::resize(&mask, width, height, FilterType::Lanczos3);

        let image = apply_mask(image, &mask, true).unwrap();
        let image = clip_minimum_border(image, 1, 8);

        Ok(image)
    }

    fn predict_mask(&self, tensor: ArrayView4<f32>, crop: Crop) -> Result<Gray32FImage> {
        let outputs: Array4<f32> = match self.precision {
            Precision::Float32 => {
                let outputs = self.session.run(ort::inputs!["img" => tensor.view()]?)?;
                outputs["mask"]
                    .try_extract_tensor::<f32>()?
                    .into_dimensionality::<Ix4>()?
                    .into_owned()
            }
            Precision::Float16 => {
                let tensor: Array4<half::f16> = tensor.mapv(AsPrimitive::as_);
                let outputs = self.session.run(ort::inputs!["img" => tensor.view()]?)?;
                outputs["mask"]
                    .try_extract_tensor::<half::f16>()?
                    .into_dimensionality::<Ix4>()?
                    .mapv(AsPrimitive::as_)
            }
        };

        let mask = extract_mask(outputs, 0.5);

        let mask =
            ImageBuffer::from_vec(self.image_size, self.image_size, mask.into_raw_vec()).unwrap();
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

        let tensor = ArrayView3::from_shape(
            (self.image_size.as_(), self.image_size.as_(), 3),
            image.as_bytes(),
        )?
            .permuted_axes([2, 0, 1])
            .slice(s![NewAxis, .., .., ..])
            .mapv(AsPrimitive::as_)
            .div(255.0);

        Ok((tensor, [x.as_(), y.as_(), w, h]))
    }
}

fn extract_mask(tensor: Array4<f32>, threshold: f32) -> Array3<f32> {
    let threshold = threshold * 255.0;
    tensor
        .index_axis(Axis(0), 0)
        .permuted_axes([1, 2, 0])
        .mul(255.0)
        .mapv(|x| {
            let x = x.floor();
            if x > threshold { x } else { 0.0 }
        })
        .div(255.0)
}
