use image::{imageops, GenericImageView, ImageBuffer, Pixel, Primitive};
use num_traits::AsPrimitive;

#[allow(dead_code)]
pub enum Position {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

pub fn to_position(
    width: u32,
    height: u32,
    pad_width: u32,
    pad_height: u32,
    position: &Position,
) -> Option<(i64, i64)> {
    if width > pad_width || height > pad_height {
        return None;
    }

    let (x, y) = match position {
        Position::Top => ((pad_width - width) / 2, 0),
        Position::Bottom => ((pad_width - width) / 2, pad_height - height),
        Position::Left => (0, (pad_height - height) / 2),
        Position::Right => (pad_width - width, (pad_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (pad_width - width, 0),
        Position::BottomLeft => (0, pad_height - height),
        Position::BottomRight => (pad_width - width, pad_height - height),
        Position::Center => ((pad_width - width) / 2, (pad_height - height) / 2),
    };

    Some((x.as_(), y.as_()))
}

pub fn padding<I, P, S>(
    image: &I,
    pad_width: u32,
    pad_height: u32,
    position: &Position,
    color: P,
) -> Option<ImageBuffer<P, Vec<S>>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();

    to_position(width, height, pad_width, pad_height, position).map(|(x, y)| {
        let mut canvas = ImageBuffer::from_pixel(pad_width, pad_height, color);
        imageops::overlay(&mut canvas, image, x, y);
        canvas
    })
}

pub fn square<I, P, S>(image: &I, position: &Position, color: P) -> Option<ImageBuffer<P, Vec<S>>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();

    let (pad_width, pad_height) = if width > height {
        (width, width)
    } else {
        (height, height)
    };

    padding(image, pad_width, pad_height, position, color)
}
