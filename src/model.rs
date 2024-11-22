use std::{ops::Div, path::Path};

use anyhow::Result;
use image::{
    imageops, imageops::FilterType, GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb,
};
use ndarray::prelude::*;
use nshare::AsNdarray3;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::SessionBuilder, Session},
};

use super::{imageops_ai::Padding, semaphore::Semaphore};

pub(super) struct Model {
    pub(super) image_size: u32,
    session: Session,
    semaphore: Semaphore,
}

impl Model {
    pub(super) fn new(model_path: &Path, device_id: i32, batch_size: usize) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
            ])?
            .with_memory_pattern(true)?
            .commit_from_file(model_path)?;

        let image_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[2] as u32;

        // initialize model
        let data = Array4::<f32>::zeros((1, 3, image_size as usize, image_size as usize));
        session.run(ort::inputs!["img" => data.view()]?)?;

        Ok(Self {
            image_size,
            session,
            semaphore: Semaphore::new(batch_size),
        })
    }

    pub(super) fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>> {
        let _guard = self.semaphore.acquire();
        let outputs = self.session.run(ort::inputs!["img" => tensor]?)?;
        Ok(outputs["mask"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned())
    }
}

pub(super) fn preprocess<I, S>(image: &I, image_size: u32) -> (Array4<f32>, [u32; 4])
where
    I: GenericImageView<Pixel = Rgb<S>>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    let image = imageops::resize(image, image_size, image_size, FilterType::Lanczos3);
    let (w, h) = image.dimensions();
    let zero = S::zero();
    let (image, (x, y)) = image.padding_square(Rgb([zero, zero, zero]));

    let tensor = image.as_ndarray3().slice_move(s![NewAxis, ..;-1, .., ..]);
    let max = S::DEFAULT_MAX_VALUE.to_f32().unwrap();
    let tensor = if max == 1.0 {
        tensor.map(|v| v.to_f32().unwrap())
    } else {
        tensor.map(|v| v.to_f32().unwrap().div(max))
    };

    (tensor, [x, y, w, h])
}

pub(super) fn postprocess_mask<S: Primitive + 'static>(
    mask: Array4<S>,
    image_size: u32,
    crop: [u32; 4],
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<S>, Vec<S>> {
    let [x, y, w, h] = crop;
    let mask =
        ImageBuffer::from_raw(image_size, image_size, mask.into_raw_vec_and_offset().0).unwrap();
    let mask = mask.view(x, y, w, h).to_image();
    imageops::resize(&mask, width, height, FilterType::Lanczos3)
}
