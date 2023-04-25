//! This module holds the ray definition, and overloads for x86_64 simd operations

mod ray_impl;
mod intersect_default;

#[cfg(all(feature = "full_simd", target_arch = "x86_64"))]
mod intersect_x86_64;

pub use self::ray_impl::*;