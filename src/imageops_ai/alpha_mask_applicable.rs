use anyhow::{anyhow, ensure, Result};
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};
use num_traits::AsPrimitive;

use crate::imageops_ai::get_max_value;

pub trait AlphaMaskApplicable<SI>
where
    SI: Primitive + AsPrimitive<f32> + 'static,
{
    fn apply_alpha_mask<SM>(
        self,
        mask: &ImageBuffer<Luma<SM>, Vec<SM>>,
    ) -> Result<ImageBuffer<Rgba<SI>, Vec<SI>>>
    where
        Rgba<SI>: Pixel<Subpixel = SI>,
        SM: Primitive + AsPrimitive<f32> + 'static,
        f32: AsPrimitive<SM>;
}

impl<SI> AlphaMaskApplicable<SI> for ImageBuffer<Rgb<SI>, Vec<SI>>
where
    Rgb<SI>: Pixel<Subpixel = SI>,
    SI: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<SI>,
{
    fn apply_alpha_mask<SM>(
        self,
        mask: &ImageBuffer<Luma<SM>, Vec<SM>>,
    ) -> Result<ImageBuffer<Rgba<SI>, Vec<SI>>>
    where
        Rgba<SI>: Pixel<Subpixel = SI>,
        SM: Primitive + AsPrimitive<f32> + 'static,
        f32: AsPrimitive<SM>,
    {
        ensure!(
            self.dimensions() == mask.dimensions(),
            "Image and mask dimensions do not match"
        );

        let si_max = get_max_value::<SI>().as_();
        let sm_max = get_max_value::<SM>().as_();

        let processed_pixels = self
            .pixels()
            .zip(mask.pixels())
            .flat_map(|(&image_pixel, &mask_pixel)| {
                let Rgb([red, green, blue]) = image_pixel;
                let Luma([alpha]) = mask_pixel;
                let alpha = (alpha.as_() / sm_max * si_max).as_();
                vec![red, green, blue, alpha]
            })
            .collect::<Vec<SI>>();

        ImageBuffer::from_raw(self.width(), self.height(), processed_pixels)
            .ok_or_else(|| anyhow!("Failed to create ImageBuffer from processed pixels"))
    }
}
