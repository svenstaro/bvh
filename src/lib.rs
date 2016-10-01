//! A crate which exports rays, axis-aligned bounding boxes, and binary bounding
//! volume hierarchies.

#![deny(missing_docs)]

#![feature(plugin)]
#![feature(test)]

#![plugin(clippy)]

#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate rand;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub extern crate nalgebra;

/// A minimal floating value used as a lower bound.
/// TODO: replace by/add ULPS/relative float comparison methods.
pub const EPSILON: f32 = 0.00001;

pub mod aabb;
pub mod bvh;
pub mod ray;
