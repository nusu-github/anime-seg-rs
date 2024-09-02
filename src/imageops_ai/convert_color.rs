use image::{buffer::ConvertBuffer, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use num_traits::AsPrimitive;

use crate::imageops_ai::get_max_value;

trait MargeAlpha {
    type Output;
    fn marge_alpha(&self) -> Self::Output;
}

impl<S> MargeAlpha for ImageBuffer<LumaA<S>, Vec<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    type Output = ImageBuffer<Luma<S>, Vec<S>>;

    fn marge_alpha(&self) -> Self::Output {
        let max = get_max_value::<S>().as_();
        let mut img = ImageBuffer::new(self.width(), self.height());
        for (x, y, p) in self.enumerate_pixels() {
            let LumaA([l, a]) = p;
            let l_f32 = l.as_();
            let a_f32 = a.as_() / max;
            let merged = (l_f32 * a_f32).as_();
            img.put_pixel(x, y, Luma([merged]));
        }
        img
    }
}

impl<S> MargeAlpha for ImageBuffer<Rgba<S>, Vec<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    type Output = ImageBuffer<Rgb<S>, Vec<S>>;

    fn marge_alpha(&self) -> Self::Output {
        let max = get_max_value::<S>().as_();
        let mut img = ImageBuffer::new(self.width(), self.height());
        for (x, y, p) in self.enumerate_pixels() {
            let Rgba([r, g, b, a]) = p;
            let a_f32 = a.as_() / max;
            let merged = |channel: &S| (channel.as_() * a_f32).as_();
            img.put_pixel(x, y, Rgb([merged(r), merged(g), merged(b)]));
        }
        img
    }
}

pub trait ConvertColor {
    type Output;
    fn wrap_convert(self) -> Self::Output;
}

impl<S> ConvertColor for ImageBuffer<Luma<S>, Vec<S>>
where
    S: Primitive + 'static,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
{
    type Output = ImageBuffer<LumaA<S>, Vec<S>>;

    fn wrap_convert(self) -> Self::Output {
        self.convert()
    }
}

impl<S> ConvertColor for ImageBuffer<LumaA<S>, Vec<S>>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
    f32: AsPrimitive<S>,
{
    type Output = ImageBuffer<Luma<S>, Vec<S>>;

    fn wrap_convert(self) -> Self::Output {
        self.marge_alpha()
    }
}

impl<S> ConvertColor for ImageBuffer<Rgb<S>, Vec<S>>
where
    S: Primitive + 'static,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
{
    type Output = ImageBuffer<Rgba<S>, Vec<S>>;

    fn wrap_convert(self) -> Self::Output {
        self.convert()
    }
}

impl<S> ConvertColor for ImageBuffer<Rgba<S>, Vec<S>>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    f32: AsPrimitive<S>,
{
    type Output = ImageBuffer<Rgb<S>, Vec<S>>;

    fn wrap_convert(self) -> Self::Output {
        self.marge_alpha()
    }
}
