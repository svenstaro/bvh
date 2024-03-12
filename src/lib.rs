//! A crate which exports rays, axis-aligned bounding boxes, and binary bounding
//! volume hierarchies.
//!
//! ## About
//!
//! This crate can be used for applications which contain intersection computations of rays
//! with primitives. For this purpose a binary tree [`Bvh`](bvh::Bvh) (Bounding Volume Hierarchy) is of great
//! use if the scene which the ray traverses contains a huge number of primitives. With a [`Bvh`](bvh::Bvh) the
//! intersection test complexity is reduced from O(n) to O(log2(n)) at the cost of building
//! the [`Bvh`](bvh::Bvh) once in advance. This technique is especially useful in ray/path tracers. For
//! use in a shader this module also exports a flattening procedure, which allows for
//! iterative traversal of the [`Bvh`](bvh::Bvh).
//!
//! ## Note
//!
//! If you are concerned about performance and do not mind using nightly, it is recommended to
//! use the `simd` feature as it introduces explicitly written simd to optimize certain areas
//! of the BVH.
//!
//! ## Example
//!
//! ```
//! use bvh::aabb::{Aabb, Bounded};
//! use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
//! use bvh::bvh::Bvh;
//! use nalgebra::{Point3, Vector3};
//! use bvh::ray::Ray;
//!
//! let origin = Point3::new(0.0,0.0,0.0);
//! let direction = Vector3::new(1.0,0.0,0.0);
//! let ray = Ray::new(origin, direction);
//!
//! struct Sphere {
//!     position: Point3<f32>,
//!     radius: f32,
//!     node_index: usize,
//! }
//!
//! impl Bounded<f32,3> for Sphere {
//!     fn aabb(&self) -> Aabb<f32,3> {
//!         let half_size = Vector3::new(self.radius, self.radius, self.radius);
//!         let min = self.position - half_size;
//!         let max = self.position + half_size;
//!         Aabb::with_bounds(min, max)
//!     }
//! }
//!
//! impl BHShape<f32,3> for Sphere {
//!     fn set_bh_node_index(&mut self, index: usize) {
//!         self.node_index = index;
//!     }
//!
//!     fn bh_node_index(&self) -> usize {
//!         self.node_index
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
//!         node_index: 0,
//!     });
//! }
//!
//! let bvh = Bvh::build_par(&mut spheres);
//! let hit_sphere_aabbs = bvh.traverse(&ray, &spheres);
//! ```
//!
//! ## Features
//!
//! - `serde` (default **disabled**) - adds `Serialize` and `Deserialize` implementations for some types
//! - `simd` (default **disabled**) - adds explicitly written SIMD instructions for certain architectures (requires nightly)
//!

#![deny(missing_docs)]
#![cfg_attr(feature = "bench", feature(test))]
#![cfg_attr(feature = "simd", feature(min_specialization))]

#[cfg(all(feature = "bench", test))]
extern crate test;

pub mod aabb;
pub mod bounding_hierarchy;
pub mod bvh;
pub mod flat_bvh;
pub mod ray;
mod utils;

#[cfg(test)]
mod testbase;

#[cfg(doctest)]
doc_comment::doctest!("../README.md", readme);
