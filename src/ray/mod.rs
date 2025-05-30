//! This module holds the [`Ray`] definition, and `RayIntersection` functions.
mod intersect_default;
mod ray_impl;

#[cfg(feature = "simd")]
mod intersect_x86_64;

pub use self::ray_impl::*;
