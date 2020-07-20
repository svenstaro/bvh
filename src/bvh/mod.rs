//! This module defines a [`BVH`].
//!
//! [`BVH`]: struct.BVH.html
//!

mod bvh_impl;
mod optimization;

pub use self::bvh_impl::*;
pub use self::optimization::*;
