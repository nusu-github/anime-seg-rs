mod clip_minimum_border;
pub mod convert_color;
pub mod mask;
pub mod padding;

pub use clip_minimum_border::clip_minimum_border;

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
