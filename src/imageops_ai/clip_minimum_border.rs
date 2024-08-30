use crate::imageops_ai::get_max_value;
use image::{GenericImageView, ImageBuffer, Luma, Pixel, Primitive};
use num_traits::{AsPrimitive, ToPrimitive};
use std::ops::{Div, Mul};

pub fn clip_minimum_border<P, S>(
    mut image: ImageBuffer<P, Vec<S>>,
    iterations: usize,
    threshold: u8,
) -> ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S> + 'static,
    S: Primitive + 'static + AsPrimitive<f32>,
    f32: AsPrimitive<S>,
{
    for i in 0..iterations {
        let corners = extract_corners(&image);
        let background = corners[i % 4];
        let [x, y, w, h] = clip_image(&image, &background, threshold);

        if w == 0 || h == 0 {
            break;
        }

        image = image.view(x, y, w, h).to_image();
    }
    image
}

fn extract_corners<I, P, S>(image: &I) -> Vec<Luma<S>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();
    vec![
        image.get_pixel(0, 0).to_luma(),
        image.get_pixel(width.saturating_sub(1), 0).to_luma(),
        image.get_pixel(0, height.saturating_sub(1)).to_luma(),
        image
            .get_pixel(width.saturating_sub(1), height.saturating_sub(1))
            .to_luma(),
    ]
}

fn clip_image<P, S>(image: &ImageBuffer<P, Vec<S>>, background: &Luma<S>, threshold: u8) -> [u32; 4]
where
    P: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32>,
    f32: AsPrimitive<S>,
{
    let max: f32 = get_max_value::<S>();
    let (width, height) = image.dimensions();
    let mut x1 = width;
    let mut y1 = height;
    let mut x2 = 0;
    let mut y2 = 0;

    for (x, y, pixel) in image.enumerate_pixels() {
        let diff = pixel.to_luma()[0]
            .as_()
            .div(max)
            .mul(255.0)
            .to_u8()
            .unwrap()
            .abs_diff(background[0].as_().div(max).mul(255.0).to_u8().unwrap());
        if diff > threshold {
            x1 = x1.min(x);
            y1 = y1.min(y);
            x2 = x2.max(x);
            y2 = y2.max(y);
        }
    }

    [x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1)]
}
