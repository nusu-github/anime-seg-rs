use crate::imageops_ai::get_max_value;
use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Primitive};
use num_traits::{AsPrimitive, ToPrimitive};
use std::ops::{Div, Mul};

pub trait ClipMinimumBorder {
    fn clip_minimum_border(self, iterations: usize, threshold: u8) -> Self;
}

impl<P, S> ClipMinimumBorder for ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S> + 'static,
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    fn clip_minimum_border(mut self, iterations: usize, threshold: u8) -> Self {
        for i in 0..iterations {
            let corners = self.extract_corners();
            let background = &corners[i % 4];
            let [x, y, w, h] = self.find_content_bounds(background, threshold);

            if w == 0 || h == 0 {
                break;
            }

            self = self.view(x, y, w, h).to_image();
        }
        self
    }
}

trait ImageProcessing<P, S>
where
    P: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    fn extract_corners(&self) -> Vec<Luma<S>>;
    fn find_content_bounds(&self, background: &Luma<S>, threshold: u8) -> [u32; 4];
    fn calculate_pixel_difference(&self, pixel: &P, background: &Luma<S>, max: f32) -> u8;
}

impl<P, S> ImageProcessing<P, S> for ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    fn extract_corners(&self) -> Vec<Luma<S>> {
        let (width, height) = self.dimensions();
        vec![
            merge_alpha(self.get_pixel(0, 0).to_luma_alpha()),
            merge_alpha(self.get_pixel(width.saturating_sub(1), 0).to_luma_alpha()),
            merge_alpha(self.get_pixel(0, height.saturating_sub(1)).to_luma_alpha()),
            merge_alpha(
                self.get_pixel(width.saturating_sub(1), height.saturating_sub(1))
                    .to_luma_alpha(),
            ),
        ]
    }

    fn find_content_bounds(&self, background: &Luma<S>, threshold: u8) -> [u32; 4] {
        let max = get_max_value::<S>().as_();
        let (width, height) = self.dimensions();
        let mut bounds = [width, height, 0, 0]; // [x1, y1, x2, y2]

        for (x, y, pixel) in self.enumerate_pixels() {
            let diff = self.calculate_pixel_difference(pixel, background, max);
            if diff > threshold {
                update_bounds(&mut bounds, x, y);
            }
        }

        [
            bounds[0],
            bounds[1],
            bounds[2].saturating_sub(bounds[0]),
            bounds[3].saturating_sub(bounds[1]),
        ]
    }

    fn calculate_pixel_difference(&self, pixel: &P, background: &Luma<S>, max: f32) -> u8 {
        let pixel_value = merge_alpha(pixel.to_luma_alpha())[0]
            .as_()
            .div(max)
            .mul(255.0);
        let background_value = background[0].as_().div(max).mul(255.0);
        pixel_value
            .to_u8()
            .unwrap()
            .abs_diff(background_value.to_u8().unwrap())
    }
}

fn merge_alpha<S>(image: LumaA<S>) -> Luma<S>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    let max = get_max_value::<S>().as_();
    let LumaA([l, a]) = image;
    let l = l.as_();
    let a = a.as_() / max;
    let l = (l * a).as_();
    Luma([l])
}

fn update_bounds(bounds: &mut [u32; 4], x: u32, y: u32) {
    bounds[0] = bounds[0].min(x);
    bounds[1] = bounds[1].min(y);
    bounds[2] = bounds[2].max(x);
    bounds[3] = bounds[3].max(y);
}
