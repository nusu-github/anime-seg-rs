use image::{GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};
use num_traits::AsPrimitive;

pub type Gray32FImage = ImageBuffer<Luma<f32>, Vec<f32>>;

pub fn apply_mask<I, M, S>(image: &I, mask: &M, edit: bool) -> Option<ImageBuffer<Rgba<S>, Vec<S>>>
where
    I: GenericImageView<Pixel = Rgb<S>>,
    M: GenericImageView<Pixel = Luma<f32>>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: AsPrimitive<S>,
{
    if image.width() != mask.width() || image.height() != mask.height() {
        return None;
    }

    let new = match edit {
        true => image
            .pixels()
            .zip(mask.pixels())
            .map(|(image, mask)| {
                let [r, g, b] = image.2 .0;
                let a = mask.2 .0[0];
                let r: S = (a * r.to_f32().unwrap()).as_();
                let g: S = (a * g.to_f32().unwrap()).as_();
                let b: S = (a * b.to_f32().unwrap()).as_();
                let a: S = (a * S::DEFAULT_MAX_VALUE.to_f32().unwrap()).as_();
                [r, g, b, a]
            })
            .flatten()
            .collect::<Vec<S>>(),
        false => image
            .pixels()
            .zip(mask.pixels())
            .map(|(image, mask)| {
                let [r, g, b] = image.2 .0;
                let a = mask.2 .0[0];
                let a: S = (a * S::DEFAULT_MAX_VALUE.to_f32().unwrap()).as_();
                [r, g, b, a]
            })
            .flatten()
            .collect::<Vec<S>>(),
    };

    ImageBuffer::from_raw(image.width(), image.height(), new)
}
