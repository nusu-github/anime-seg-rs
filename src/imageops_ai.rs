mod clip_minimum_border;
pub mod mask;
pub mod padding;

pub use clip_minimum_border::clip_minimum_border;

use num_traits::{AsPrimitive, Bounded, Num};
use std::any::TypeId;

pub fn is_floating_point<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>()
}

pub fn get_max_value<T: Num + Bounded + AsPrimitive<f32>>() -> f32 {
    if is_floating_point::<T>() {
        1.0
    } else {
        T::max_value().as_()
    }
}
