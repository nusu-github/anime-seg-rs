use anyhow::{anyhow, ensure, Result};
use image::{GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};
use num_traits::AsPrimitive;

use crate::imageops_ai::get_max_value;

pub fn apply<I, M, SI, SM>(
    image: &I,
    mask: &M,
    apply_edit: bool,
) -> Result<ImageBuffer<Rgba<SI>, Vec<SI>>>
where
    I: GenericImageView<Pixel = Rgb<SI>>,
    M: GenericImageView<Pixel = Luma<SM>>,
    Rgba<SI>: Pixel<Subpixel = SI>,
    SI: Primitive + 'static + AsPrimitive<f32>,
    SM: Primitive + 'static + AsPrimitive<f32>,
    f32: AsPrimitive<SI>,
    f32: AsPrimitive<SM>,
{
    ensure!(
        image.dimensions() == mask.dimensions(),
        "Image and mask dimensions do not match"
    );

    let sm_max: f32 = get_max_value::<SM>();
    let si_max: f32 = get_max_value::<SI>();

    let processed_pixels = image
        .pixels()
        .zip(mask.pixels())
        .flat_map(|(image_pixel, mask_pixel)| {
            let Rgb([red, green, blue]) = image_pixel.2;
            let alpha = (mask_pixel.2 .0[0].as_() / sm_max) * si_max;
            let alpha_scaled: SI = alpha.as_();

            if apply_edit {
                [red, green, blue]
                    .iter()
                    .map(|&c| ((c.as_() / si_max) * alpha).as_())
                    .chain(std::iter::once(alpha_scaled))
                    .collect::<Vec<SI>>()
            } else {
                vec![red, green, blue, alpha_scaled]
            }
        })
        .collect::<Vec<SI>>();

    ImageBuffer::from_raw(image.width(), image.height(), processed_pixels)
        .ok_or_else(|| anyhow!("Failed to create ImageBuffer from processed pixels"))
}
