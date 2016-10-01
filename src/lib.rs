//! A crate which exports rays, axis-aligned bounding boxes, and binary bounding
//! volume hierarchies.
//!
//! ## About
//!
//! This crate can be used for application which contain intersection computations of rays
//! with primitives. For this purpose a binary BVH (Bounding Volume Hierarchy) is of great use
//! if the scene which the ray traverses contains a huge number of primitives. With a BVH the
//! intersection test complexity is reduced from O(n) to O(log2(n)) at the cost of building
//! the BVH once in advance. This technique is especially useful in Ray/Path tracers. For
//! use in a shader this module also exports a flattening procedure, which allows for
//! iterative traversal of the BVH.
//! The library is built on top of `nalgebra`.
//!
//! ## Example
//!
//! ```
//! use bvh::aabb::{AABB, Bounded};
//! use bvh::bvh::BVH;
//! use bvh::nalgebra::{Point3, Vector3};
//! use bvh::ray::Ray;
//!
//! let origin = Point3::new(0.0,0.0,0.0);
//! let direction = Vector3::new(1.0,0.0,0.0);
//! let ray = Ray::new(origin, direction);
//!
//! struct Sphere {
//!     position: Point3<f32>,
//!     radius: f32,
//! }
//!
//! impl Bounded for Sphere {
//!     fn aabb(&self) -> AABB {
//!         let half_size = Vector3::new(self.radius, self.radius, self.radius);
//!         let min = self.position - half_size;
//!         let max = self.position + half_size;
//!         AABB::with_bounds(min, max)
//!     }
//! }
//!
//! let mut spheres = Vec::new();
//! for i in 0..1000u32 {
//!     let position = Point3::new(i as f32, i as f32, i as f32);
//!     let radius = (i % 10) as f32 + 1.0;
//!     spheres.push(Sphere {
//!         position: position,
//!         radius: radius,
//!     });
//! }
//!
//! let bvh = BVH::build(&spheres);
//! let hit_sphere_aabbs = bvh.traverse_recursive(&ray, &spheres);
//! ```

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
