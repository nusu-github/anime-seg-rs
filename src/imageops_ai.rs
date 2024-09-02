mod alpha_mask_applicable;
mod clip_minimum_border;
mod convert_color;
mod padding;

pub use alpha_mask_applicable::AlphaMaskApplicable;
pub use clip_minimum_border::ClipMinimumBorder;
pub use convert_color::ConvertColor;
pub use padding::Padding;

use num_traits::{Bounded, NumCast};
use std::any::TypeId;

pub fn is_floating_point<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

pub fn get_max_value<T: Bounded + NumCast + 'static>() -> T {
    if is_floating_point::<T>() {
        T::from(1.0).unwrap()
    } else {
        T::max_value()
    }
}
