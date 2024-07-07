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

pub fn get_position(
    width: u32,
    height: u32,
    pad_width: u32,
    pad_height: u32,
    position: Position,
) -> Option<(i64, i64)> {
    if width > pad_width || height > pad_height {
        return None;
    }

    let (x, y) = match position {
        Position::Top => {
            let x = (pad_width - width) / 2;
            let y = 0;
            (x, y)
        }
        Position::Bottom => {
            let x = (pad_width - width) / 2;
            let y = pad_height - height;
            (x, y)
        }
        Position::Left => {
            let x = 0;
            let y = (pad_height - height) / 2;
            (x, y)
        }
        Position::Right => {
            let x = pad_width - width;
            let y = (pad_height - height) / 2;
            (x, y)
        }
        Position::TopLeft => (0, 0),
        Position::TopRight => {
            let x = pad_width - width;
            let y = 0;
            (x, y)
        }
        Position::BottomLeft => {
            let x = 0;
            let y = pad_height - height;
            (x, y)
        }
        Position::BottomRight => {
            let x = pad_width - width;
            let y = pad_height - height;
            (x, y)
        }
        Position::Center => {
            let x = (pad_width - width) / 2;
            let y = (pad_height - height) / 2;
            (x, y)
        }
    };

    Some((x.as_(), y.as_()))
}

pub fn padding<I, P, S>(
    image: &I,
    pad_width: u32,
    pad_height: u32,
    position: Position,
    color: P,
) -> Option<ImageBuffer<P, Vec<S>>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();

    match get_position(width, height, pad_width, pad_height, position) {
        None => None,
        Some((x, y)) => {
            let mut canvas = ImageBuffer::from_pixel(pad_width, pad_height, color);
            imageops::overlay(&mut canvas, image, x, y);

            Some(canvas)
        }
    }
}

pub fn padding_square<I, P, S>(
    image: &I,
    position: Position,
    color: P,
) -> Option<ImageBuffer<P, Vec<S>>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();

    let (pad_width, pad_height) = match (width, height) {
        (x, y) if x > y => (width, width),
        (x, y) if x < y => (height, height),
        _ => (width, height),
    };

    padding(image, pad_width, pad_height, position, color)
}
