use std::{ops::Div, path::Path};

use anyhow::Result;
use image::{
    imageops, imageops::FilterType, GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb,
};
use ndarray::prelude::*;
use nshare::AsNdarray3;
use num_traits::AsPrimitive;
use ort::{CUDAExecutionProvider, Session, SessionBuilder};

use crate::imageops_ai::{get_max_value, is_floating_point, Padding};
use crate::semaphore::Semaphore;

pub struct Model {
    pub image_size: u32,
    session: Session,
    semaphore: Semaphore,
}

impl Model {
    pub fn new(model_path: &Path, device_id: i32, batch_size: usize) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
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

    pub fn predict(&self, tensor: ArrayView4<f32>) -> Result<Array4<f32>> {
        let _guard = self.semaphore.acquire();
        let outputs = self.session.run(ort::inputs!["img" => tensor]?)?;
        Ok(outputs["mask"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned())
    }
}

pub fn preprocess<I, S>(image: &I, image_size: u32) -> Result<(Array4<f32>, [u32; 4])>
where
    I: GenericImageView<Pixel = Rgb<S>>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32> + 'static,
{
    let image = imageops::resize(image, image_size, image_size, FilterType::Lanczos3);
    let (w, h) = image.dimensions();
    let zero = S::zero();
    let (image, (x, y)) = image.padding_square(Rgb([zero, zero, zero]));

    let tensor = image.as_ndarray3().slice_move(s![NewAxis, ..;-1, .., ..]);
    let tensor = if is_floating_point::<S>() {
        tensor.map(|v| v.as_())
    } else {
        tensor.map(|v| v.as_().div(get_max_value::<S>().as_()))
    };

    Ok((tensor, [x, y, w, h]))
}

pub fn postprocess_mask<S>(
    mask: Array4<S>,
    image_size: u32,
    crop: [u32; 4],
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<S>, Vec<S>>
where
    S: Primitive + 'static,
{
    let [x, y, w, h] = crop;
    let mask =
        ImageBuffer::from_raw(image_size, image_size, mask.into_raw_vec_and_offset().0).unwrap();
    let mask = mask.view(x, y, w, h).to_image();
    imageops::resize(&mask, width, height, FilterType::Lanczos3)
}
