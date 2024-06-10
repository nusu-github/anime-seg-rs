use std::path::Path;

use anyhow::Result;
use image::{
    EncodableLayout,
    GenericImageView, ImageBuffer, imageops::{self, FilterType}, Rgb, RgbaImage, RgbImage,
};
use ndarray::prelude::*;
use num_traits::AsPrimitive;
use ort::Session;

use crate::imageops_ai::{
    clip_minimum_border::{clip_minimum_border, Crop},
    mask::{apply_mask, Gray32FImage},
    padding::{get_position, padding_square, Position},
};

#[cfg(feature = "fp16")]
type Precision = half::f16;
#[cfg(not(feature = "fp16"))]
type Precision = f32;

pub struct Model {
    image_size: u32,
    session: Session,
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

        let image_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[2];
        let tensor: Array4<Precision> =
            Array4::<f32>::zeros([1, 3, image_size.as_(), image_size.as_()]).mapv(AsPrimitive::as_);
        session.run(ort::inputs!["img" => tensor]?)?;

        Ok(Self {
            image_size: image_size.as_(),
            session,
        })
    }

    pub fn predict(&self, image: RgbImage) -> Result<RgbaImage> {
        let (width, height) = image.dimensions();
        let (tensor, crop) = self.preprocess(&image)?;

        let mask = self.predict_mask(tensor, crop, width, height)?;
        let image = apply_mask(&image, &mask, true).unwrap();

        let image = clip_minimum_border(image, 1, 8);

        Ok(image)
    }

    fn predict_mask(
        &self,
        tensor: Array4<Precision>,
        crop: Crop,
        original_width: u32,
        original_height: u32,
    ) -> Result<Gray32FImage> {
        let outputs = self.session.run(ort::inputs!["img" => tensor]?)?;
        let outputs = outputs["mask"]
            .try_extract_tensor::<Precision>()?
            .into_dimensionality::<Ix4>()?;
        let mask = extract_mask(outputs);

        let mask =
            ImageBuffer::from_vec(self.image_size, self.image_size, mask.into_raw_vec()).unwrap();
        let [x, y, w, h] = crop;
        let mask = mask.view(x, y, w, h).to_image();
        let mask = imageops::resize(&mask, original_width, original_height, FilterType::Triangle);

        Ok(mask)
    }

    fn preprocess(&self, image: &RgbImage) -> Result<(Array4<Precision>, Crop)> {
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
            .slice(s![NewAxis, ..;-1, .., ..])
            .mapv(|x| {
                <u8 as AsPrimitive<Precision>>::as_(x) / <f32 as AsPrimitive<Precision>>::as_(255.0)
            });

        Ok((tensor, [x.as_(), y.as_(), w, h]))
    }
}

fn extract_mask(tensor: ArrayView4<Precision>) -> Array3<f32> {
    tensor
        .index_axis(Axis(0), 0)
        .permuted_axes([1, 2, 0])
        .mapv(AsPrimitive::as_)
}
