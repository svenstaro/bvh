#![feature(test)]
//! A crate which exports rays, axis-aligned bounding boxes, and binary bounding
//! volume hierarchies.
//!
//! ## About
//!
//! This crate can be used for applications which contain intersection computations of rays
//! with primitives. For this purpose a binary tree BVH (Bounding Volume Hierarchy) is of great
//! use if the scene which the ray traverses contains a huge number of primitives. With a BVH the
//! intersection test complexity is reduced from O(n) to O(log2(n)) at the cost of building
//! the BVH once in advance. This technique is especially useful in ray/path tracers. For
//! use in a shader this module also exports a flattening procedure, which allows for
//! iterative traversal of the BVH.
//!
//! ## Example
//!
//! ```
//! use dynbvh_f32::aabb::{AABB, Bounded};
//! use dynbvh_f32::bounding_hierarchy::{BoundingHierarchy, BHShape};
//! use dynbvh_f32::bvh::BVH;
//! use dynbvh_f32::{Point3, Vector3};
//! use dynbvh_f32::ray::Ray;
//! use dynbvh_f32::Real;
//!
//! let origin = Point3::new(0.0,0.0,0.0);
//! let direction = Vector3::new(1.0,0.0,0.0);
//! let ray = Ray::new(origin, direction);
//!
//! struct Sphere {
//!     position: Point3,
//!     radius: Real,
//!     node_index: usize,
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
//! impl BHShape for Sphere {
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
//!     let position = Point3::new(i as Real, i as Real, i as Real);
//!     let radius = (i % 10) as Real + 1.0;
//!     spheres.push(Sphere {
//!         position: position,
//!         radius: radius,
//!         node_index: 0,
//!     });
//! }
//!
//! let bvh = BVH::build(&mut spheres);
//! let hit_sphere_aabbs = bvh.traverse(&ray, &spheres);
//! ```
//!
//! ## Features
//!
//! - `serde_impls` (default **disabled**) - adds `Serialize` and `Deserialize` implementations for some types
//!

#[cfg(all(feature = "bench", test))]
extern crate test;

/// Point math type used by this crate. Type alias for [`glam::DVec3`].
#[cfg(feature = "f64")]
pub type Point3 = glam::DVec3;

/// Vector math type used by this crate. Type alias for [`glam::DVec3`].
#[cfg(feature = "f64")]
pub type Vector3 = glam::DVec3;

/// Matrix math type used by this crate. Type alias for [`glam::DMat4`].
#[cfg(feature = "f64")]
pub type Mat4 = glam::DMat4;

/// Matrix math type used by this crate. Type alias for [`glam::DQuat`].
#[cfg(feature = "f64")]
pub type Quat = glam::DQuat;

#[cfg(feature = "f64")]
/// Float type used by this crate
pub type Real = f64;

/// Point math type used by this crate. Type alias for [`glam::Vec3`].
#[cfg(not(feature = "f64"))]
pub type Point3 = glam::Vec3;

/// Vector math type used by this crate. Type alias for [`glam::Vec3`].
#[cfg(not(feature = "f64"))]
pub type Vector3 = glam::Vec3;

/// Matrix math type used by this crate. Type alias for [`glam::Mat4`].
#[cfg(not(feature = "f64"))]
pub type Mat4 = glam::Mat4;

/// Quat math type used by this crate. Type alias for [`glam::Quat`].
#[cfg(not(feature = "f64"))]
pub type Quat = glam::Quat;

#[cfg(not(feature = "f64"))]
/// Float type used by this crate
pub type Real = f32;

/// A minimal floating value used as a lower bound.
/// TODO: replace by/add ULPS/relative float comparison methods.
pub const EPSILON: Real = 0.00001;

pub mod aabb;
pub mod axis;
pub mod bounding_hierarchy;
pub mod bvh;
pub mod flat_bvh;
pub mod ray;
pub mod shapes;
mod utils;

#[cfg(test)]
mod testbase;


use aabb::{Bounded, AABB};
use bounding_hierarchy::BHShape;


#[derive(Debug)]
struct Sphere {
    position: Point3,
    radius: Real,
    node_index: usize,
}

impl Bounded for Sphere {
    fn aabb(&self) -> AABB {
        let half_size = Vector3::new(self.radius, self.radius, self.radius);
        let min = self.position - half_size;
        let max = self.position + half_size;
        AABB::with_bounds(min, max)
    }
}

impl BHShape for Sphere {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// A triangle struct. Instance of a more complex `Bounded` primitive.
#[derive(Debug)]
pub struct Triangle {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
    aabb: AABB,
    node_index: usize,
}

impl Triangle {
    pub fn new(a: Point3, b: Point3, c: Point3) -> Triangle {
        Triangle {
            a,
            b,
            c,
            aabb: AABB::empty().grow(&a).grow(&b).grow(&c),
            node_index: 0,
        }
    }
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        self.aabb
    }
}

impl BHShape for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

// impl<I: FromPrimitive + Integer> FromRawVertex<I> for Triangle {
//     fn process(
//         vertices: Vec<(f32, f32, f32, f32)>,
//         _: Vec<(f32, f32, f32)>,
//         _: Vec<(f32, f32, f32)>,
//         polygons: Vec<Polygon>,
//     ) -> ObjResult<(Vec<Self>, Vec<I>)> {
//         // Convert the vertices to `Point3`s.
//         let points = vertices
//             .into_iter()
//             .map(|v| Point3::new(v.0.into(), v.1.into(), v.2.into()))
//             .collect::<Vec<_>>();

//         // Estimate for the number of triangles, assuming that each polygon is a triangle.
//         let mut triangles = Vec::with_capacity(polygons.len());
//         {
//             let mut push_triangle = |indices: &Vec<usize>| {
//                 let mut indices_iter = indices.iter();
//                 let anchor = points[*indices_iter.next().unwrap()];
//                 let mut second = points[*indices_iter.next().unwrap()];
//                 for third_index in indices_iter {
//                     let third = points[*third_index];
//                     triangles.push(Triangle::new(anchor, second, third));
//                     second = third;
//                 }
//             };

//             // Iterate over the polygons and populate the `Triangle`s vector.
//             for polygon in polygons.into_iter() {
//                 match polygon {
//                     Polygon::P(ref vec) => push_triangle(vec),
//                     Polygon::PT(ref vec) | Polygon::PN(ref vec) => {
//                         push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
//                     }
//                     Polygon::PTN(ref vec) => {
//                         push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
//                     }
//                 }
//             }
//         }
//         Ok((triangles, Vec::new()))
//     }
// }

// pub fn load_sponza_scene() -> (Vec<Triangle>, AABB) {
//     use std::fs::File;
//     use std::io::BufReader;

//     let file_input =
//         BufReader::new(File::open("media/sponza.obj").expect("Failed to open .obj file."));
//     let sponza_obj: Obj<Triangle> = load_obj(file_input).expect("Failed to decode .obj file data.");
//     let triangles = sponza_obj.vertices;

//     let mut bounds = AABB::empty();
//     for triangle in &triangles {
//         bounds.join_mut(&triangle.aabb());
//     }

//     (triangles, bounds)
// }

// pub fn main() {
//     let (mut triangles, _bounds) = load_sponza_scene();
//     let mut bvh = BVH::build(triangles.as_mut_slice());

//     for _i in 0..10 {
//         bvh.rebuild(triangles.as_mut_slice());
//     }
// }
