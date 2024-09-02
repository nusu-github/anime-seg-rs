use image::{buffer::ConvertBuffer, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use num_traits::AsPrimitive;

use crate::imageops_ai::get_max_value;

trait MargeAlpha<S>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
{
    type Output;
    fn marge_alpha(&self) -> Self::Output;
}

impl<S> MargeAlpha<S> for ImageBuffer<LumaA<S>, Vec<S>>
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
            let l = l.as_();
            let a = a.as_() / max;
            let l = (l * a).as_();
            img.put_pixel(x, y, Luma([l]));
        }
        img
    }
}

impl<S> MargeAlpha<S> for ImageBuffer<Rgba<S>, Vec<S>>
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
            let r = r.as_();
            let g = g.as_();
            let b = b.as_();
            let a = a.as_() / max;
            let r = (r * a).as_();
            let g = (g * a).as_();
            let b = (b * a).as_();
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
        img
    }
}

pub trait ConvertColor<S>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
{
    fn to_luma(self) -> Option<ImageBuffer<Luma<S>, Vec<S>>>;
    fn to_luma_alpha(self) -> Option<ImageBuffer<LumaA<S>, Vec<S>>>;
    fn to_rgb(self) -> Option<ImageBuffer<Rgb<S>, Vec<S>>>;
    fn to_rgba(self) -> Option<ImageBuffer<Rgba<S>, Vec<S>>>;
}

impl<S> ConvertColor<S> for ImageBuffer<Luma<S>, Vec<S>>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
{
    fn to_luma(self) -> Option<ImageBuffer<Luma<S>, Vec<S>>> {
        Some(self)
    }
    fn to_luma_alpha(self) -> Option<ImageBuffer<LumaA<S>, Vec<S>>> {
        Some(self.convert())
    }
    fn to_rgb(self) -> Option<ImageBuffer<Rgb<S>, Vec<S>>> {
        Some(self.convert())
    }
    fn to_rgba(self) -> Option<ImageBuffer<Rgba<S>, Vec<S>>> {
        Some(self.convert())
    }
}

impl<S> ConvertColor<S> for ImageBuffer<LumaA<S>, Vec<S>>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
{
    fn to_luma(self) -> Option<ImageBuffer<Luma<S>, Vec<S>>> {
        Some(self.marge_alpha())
    }
    fn to_luma_alpha(self) -> Option<ImageBuffer<LumaA<S>, Vec<S>>> {
        Some(self)
    }
    fn to_rgb(self) -> Option<ImageBuffer<Rgb<S>, Vec<S>>> {
        Some(self.marge_alpha().convert())
    }
    fn to_rgba(self) -> Option<ImageBuffer<Rgba<S>, Vec<S>>> {
        Some(self.convert())
    }
}

impl<S> ConvertColor<S> for ImageBuffer<Rgb<S>, Vec<S>>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
{
    fn to_luma(self) -> Option<ImageBuffer<Luma<S>, Vec<S>>> {
        None
    }
    fn to_luma_alpha(self) -> Option<ImageBuffer<LumaA<S>, Vec<S>>> {
        None
    }
    fn to_rgb(self) -> Option<ImageBuffer<Rgb<S>, Vec<S>>> {
        Some(self)
    }
    fn to_rgba(self) -> Option<ImageBuffer<Rgba<S>, Vec<S>>> {
        Some(self.convert())
    }
}

impl<S> ConvertColor<S> for ImageBuffer<Rgba<S>, Vec<S>>
where
    S: Primitive + AsPrimitive<f32> + 'static,
    f32: AsPrimitive<S>,
    Luma<S>: Pixel<Subpixel = S>,
    LumaA<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
{
    fn to_luma(self) -> Option<ImageBuffer<Luma<S>, Vec<S>>> {
        None
    }
    fn to_luma_alpha(self) -> Option<ImageBuffer<LumaA<S>, Vec<S>>> {
        None
    }
    fn to_rgb(self) -> Option<ImageBuffer<Rgb<S>, Vec<S>>> {
        Some(self.marge_alpha())
    }
    fn to_rgba(self) -> Option<ImageBuffer<Rgba<S>, Vec<S>>> {
        Some(self)
    }
}
