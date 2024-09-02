use anyhow::{anyhow, ensure, Result};
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};
use num_traits::AsPrimitive;

use crate::imageops_ai::get_max_value;

pub fn apply<SI, SM>(
    image: &ImageBuffer<Rgb<SI>, Vec<SI>>,
    mask: &ImageBuffer<Luma<SM>, Vec<SM>>,
) -> Result<ImageBuffer<Rgba<SI>, Vec<SI>>>
where
    Rgb<SI>: Pixel<Subpixel = SI>,
    Rgba<SI>: Pixel<Subpixel = SI>,
    Luma<SM>: Pixel<Subpixel = SM>,
    SI: Primitive + AsPrimitive<f32> + 'static,
    SM: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<SI> + AsPrimitive<SM>,
{
    ensure!(
        image.dimensions() == mask.dimensions(),
        "Image and mask dimensions do not match"
    );

    let si_max = get_max_value::<SI>().as_();
    let sm_max = get_max_value::<SM>().as_();

    let processed_pixels = image
        .pixels()
        .zip(mask.pixels())
        .flat_map(|(&image_pixel, &mask_pixel)| {
            let Rgb([red, green, blue]) = image_pixel;
            let Luma([alpha]) = mask_pixel;
            let alpha = (alpha.as_() / sm_max) * si_max;
            let alpha_scaled = alpha.as_();
            vec![red, green, blue, alpha_scaled]
        })
        .collect::<Vec<SI>>();

    ImageBuffer::from_raw(image.width(), image.height(), processed_pixels)
        .ok_or_else(|| anyhow!("Failed to create ImageBuffer from processed pixels"))
}
