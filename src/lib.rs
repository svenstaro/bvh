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
//! use bvh::aabb::{AABB, Bounded};
//! use bvh::bounding_hierarchy::{BoundingHierarchy, BHShape};
//! use bvh::bvh::BVH;
//! use bvh::{Point3, Vector3};
//! use bvh::ray::Ray;
//!
//! let origin = Point3::new(0.0,0.0,0.0);
//! let direction = Vector3::new(1.0,0.0,0.0);
//! let ray = Ray::new(origin, direction);
//!
//! struct Sphere {
//!     position: Point3,
//!     radius: f64,
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
//!     let position = Point3::new(i as f64, i as f64, i as f64);
//!     let radius = (i % 10) as f64 + 1.0;
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

//#![deny(missing_docs)]
#![cfg_attr(feature = "bench", feature(test))]

#[cfg(all(feature = "bench", test))]
extern crate test;

/// A minimal floating value used as a lower bound.
/// TODO: replace by/add ULPS/relative float comparison methods.
pub const EPSILON: f64 = 0.00001;

/// Point math type used by this crate. Type alias for [`glam::Vec3`].
pub type Point3 = glam::DVec3;

/// Vector math type used by this crate. Type alias for [`glam::Vec3`].
pub type Vector3 = glam::DVec3;

pub mod aabb;
pub mod axis;
pub mod bounding_hierarchy;
pub mod bvh;
pub mod flat_bvh;
pub mod ray;
pub mod shapes;
mod utils;

use aabb::{Bounded, AABB};
use bounding_hierarchy::BHShape;
use bvh::BVHNode;
use ray::Ray;
use shapes::{Capsule, Sphere, OBB};
use glam::DQuat;
use bvh::BUILD_THREAD_COUNT;
use std::sync::atomic::{AtomicUsize, Ordering};
use bvh::qbvh::BoundableQ;
use parry3d_f64::partitioning::{QBVH, QBVHDataGenerator, IndexedData};
use parry3d_f64::bounding_volume::aabb::AABB as QAABB;
use parry3d_f64::math::{Point, Vector};
use parry3d_f64::query::Ray as RayQ;
use std::boxed;
use parry3d_f64::query::visitors::RayIntersectionsVisitor;

#[macro_use]
extern crate lazy_static;


#[cfg(test)]
mod testbase;


#[no_mangle]
pub extern fn add_numbers(number1: i32, number2: i32) -> i32 {
    println!("Hello from rust!");
    number1 + number2
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vector3D {    
    pub x: f64,
    pub y: f64,
    pub z: f64
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Float3 {    
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BoundsD {
    pub min: Vector3D,
    pub max: Vector3D
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BVHBounds {
    pub bounds: BoundsD,
    pub index: i32,
    pub array_index: i32
}

#[repr(C)]
pub struct BVHRef {
    pub ptr: *mut BVHNode,
    pub len: i32,
    pub cap: i32
}

#[repr(C)]
pub struct QBVHRef {
    pub ptr: *mut QBVH<RefNode>
}

impl QBVHRef {
    fn from_box(boxed: Box<QBVH<RefNode>>) -> Self {
        let ptr = Box::into_raw(boxed);
        QBVHRef {
            ptr
        }
    }
    fn to_box(&self) -> Box<QBVH<RefNode>> {
        unsafe { Box::from_raw(self.ptr) }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QuaternionD {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point32 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AABB32 {
    pub min: Point32,
    pub max: Point32
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FlatNode32 {
    pub aabb: AABB32,
    pub entry_index: u32,
    pub exit_index: u32,
    pub shape_index: u32
}

impl Bounded for BVHBounds {
    fn aabb(&self) -> AABB {
        let min = to_vec(&self.bounds.min);
        let max = to_vec(&self.bounds.max);
        AABB::with_bounds(min, max)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RefNode {
    pub index: usize
}

impl IndexedData for RefNode {
    fn index(&self) -> usize {
        self.index
    }
    fn default() -> Self {
        RefNode {
            index: usize::MAX
        }
    }
}

impl BVHBounds {
    fn qaabb(&self) -> QAABB {
        let min = Point::new(self.bounds.min.x, self.bounds.min.y, self.bounds.min.z);
        let max = Point::new(self.bounds.max.x, self.bounds.max.y, self.bounds.max.z);
        QAABB::new(min, max)
    }
}

impl BHShape for BVHBounds {
    fn set_bh_node_index(&mut self, index: usize) {
        self.index = index as i32;
    }

    fn bh_node_index(&self) -> usize {
        self.index as usize
    }
}

pub fn to_vec(a: &Vector3D) -> Vector3 {
    Vector3::new(a.x, a.y, a.z)
}

pub fn to_vecd(a: &Vector3) -> Vector3D {
    Vector3D {
        x: a.x,
        y: a.y,
        z: a.z
    }
}

pub fn to_quat(a: &QuaternionD) -> DQuat {
    DQuat::from_xyzw(a.x, a.y, a.z, a.w)
}

struct BoundsData<'a> {
    data: &'a mut [BVHBounds]
}

impl <'a> BoundsData<'a> {
    fn new(data: &'a mut [BVHBounds]) -> BoundsData {
        BoundsData {
            data
        }
    }
}

impl <'a> QBVHDataGenerator<RefNode> for BoundsData<'a> {
    fn size_hint(&self) -> usize {
        self.data.len()
    }

    fn for_each(&mut self, mut f: impl FnMut(RefNode, QAABB)) {
        for i in 0..self.data.len()
        {
            let bounds = self.data[i];
            f(RefNode {
                index: i
            }, bounds.qaabb());
        }
            
    }
}

#[no_mangle]
pub extern fn set_build_thread_count(count: i32)
{
    BUILD_THREAD_COUNT.store(count as usize, Ordering::Relaxed);
}


#[no_mangle]
pub extern fn add_vecs(a_ptr: *mut Float3, b_ptr: *mut Float3, out_ptr: *mut Float3)
{
    let a = unsafe {*a_ptr};

    let a = glam::Vec3::new(a.x, a.y, a.z);
    let b = unsafe {*b_ptr};
    let b = glam::Vec3::new(b.x, b.y, b.z);
    let mut c = glam::Vec3::new(0.0, 0.0, 0.0);

    for i in 0 .. 100000 {
        c = a + b + c;
    }

    unsafe {
        *out_ptr = Float3 {
            x: c.x,
            y: c.y,
            z: c.z
        };
    }
}

#[no_mangle]
pub extern fn build_bvh(a: *mut BVHBounds, count: i32) -> BVHRef 
{
    let mut s = unsafe { std::slice::from_raw_parts_mut(a, count as usize) };

    let mut bvh = bvh::BVH::build(&mut s);
    let len = bvh.nodes.len();
    let cap = bvh.nodes.capacity();
    let p = bvh.nodes.as_mut_ptr();
    std::mem::forget(bvh.nodes);

    BVHRef { ptr: p, len: len as i32, cap: cap as i32 }
}


#[no_mangle]
pub extern fn build_qbvh(a: *mut BVHBounds, count: i32) -> QBVHRef 
{
    let mut s = unsafe { std::slice::from_raw_parts_mut(a, count as usize) };

    let data = BoundsData::new(s);
    let mut bvh = Box::new(QBVH::new());

    bvh.clear_and_rebuild(data, 0.0);

    QBVHRef::from_box(bvh)
}

#[no_mangle]
pub extern fn query_ray_q(bvh_ref: *const QBVHRef, origin_vec: *const Vector3D, dir_vec: *const Vector3D, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };


    let bvh = unsafe {(*bvh_ref).to_box()};
    let o_vec = unsafe {*origin_vec};
    let d_vec = unsafe {*dir_vec};
    let ray = RayQ::new(Point::new(o_vec.x, o_vec.y, o_vec.z), Vector::new(d_vec.x, d_vec.y, d_vec.z));
    let mut i = 0;

    let mut stack_arr: [u32; 32] = [0; 32];

    let mut stack = unsafe { Vec::from_raw_parts(&mut stack_arr as *mut u32, 0, 32)};
    
    let mut visit = |node: &RefNode| {
        if i < max_res {
            buffer[i as usize] = shapes[node.index];
            i += 1;
            return false;
        }
        i += 1;
        true
    };

    let mut visitor = RayIntersectionsVisitor::new(&ray, 1000000000000.0, &mut visit);

    bvh.traverse_depth_first_with_stack(&mut visitor, &mut stack);

    std::mem::forget(stack);
    std::mem::forget(bvh);

    i as i32
}



#[no_mangle]
pub extern fn rebuild_bvh(bvh_ref: *const BVHRef, a: *mut BVHBounds, count: i32) -> BVHRef
{
    let mut s = unsafe { std::slice::from_raw_parts_mut(a, count as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let mut bvh = bvh::BVH {
        nodes: v
    };

    bvh.rebuild(s);

    let len = bvh.nodes.len();
    let cap = bvh.nodes.capacity();
    let p = bvh.nodes.as_mut_ptr();
    std::mem::forget(bvh.nodes);

    BVHRef { ptr: p, len: len as i32, cap: cap as i32 }
}

#[no_mangle]
pub extern fn query_ray(bvh_ref: *const BVHRef, origin_vec: *const Vector3D, dir_vec: *const Vector3D, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let bvh = bvh::BVH {
        nodes: v
    };

    let ray = Ray::new(to_vec(& unsafe{*origin_vec}), to_vec(& unsafe{*dir_vec}));
    let mut i = 0;

    for x in bvh.traverse_iterator(&ray, &shapes) {
        if i < max_res {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    std::mem::forget(bvh.nodes);

    i as i32
}

#[no_mangle]
pub extern fn batch_query_rays(bvh_ref: *const BVHRef, origin_vecs: *mut Vector3D, dir_vecs: *mut Vector3D, hits: *mut i32, ray_count: i32, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32)
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };
    let origins = unsafe { std::slice::from_raw_parts_mut(origin_vecs, ray_count as usize) };
    let dirs = unsafe { std::slice::from_raw_parts_mut(dir_vecs, ray_count as usize) };
    let hits = unsafe { std::slice::from_raw_parts_mut(hits, ray_count as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let bvh = bvh::BVH {
        nodes: v
    };
    let mut i = 0;
    for r in 0..ray_count as usize {
        let ray = Ray::new(to_vec(&origins[r]), to_vec(&dirs[r]));
        let mut res = 0;
        for x in bvh.traverse_iterator(&ray, &shapes) {
            if i < max_res {
                buffer[i as usize] = *x;
            }
            i += 1;
            res += 1;
        }
        hits[r] = res;
    }

    std::mem::forget(bvh.nodes);
}


#[no_mangle]
pub extern fn query_sphere(bvh_ref: *const BVHRef, center: *const Vector3D, radius: f64, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let bvh = bvh::BVH {
        nodes: v
    };

    let test_shape = Sphere::new(to_vec(&unsafe { *center }), radius);
    let mut i = 0;

    for x in bvh.traverse_iterator(&test_shape, &shapes) {
        if i < max_res {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    std::mem::forget(bvh.nodes);

    i as i32
}

#[no_mangle]
pub extern fn query_capsule(bvh_ref: *const BVHRef, start: *const Vector3D, end: *const Vector3D, radius: f64, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let bvh = bvh::BVH {
        nodes: v
    };

    let test_shape = Capsule::new(to_vec(&unsafe { *start }), to_vec(&unsafe { *end }), radius);
    let mut i = 0;

    for x in bvh.traverse_iterator(&test_shape, &shapes) {
        if i < max_res {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    std::mem::forget(bvh.nodes);

    i as i32
}

#[no_mangle]
pub extern fn query_aabb(bvh_ref: *const BVHRef, bounds: *const BoundsD, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let bvh = bvh::BVH {
        nodes: v
    };

    let min = to_vec(&unsafe { *bounds }.min);
    let max = to_vec(&unsafe { *bounds }.max);
    let test_shape = AABB::with_bounds(min, max);
    let mut i = 0;

    for x in bvh.traverse_iterator(&test_shape, &shapes) {
        if i < max_res {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    std::mem::forget(bvh.nodes);

    i as i32
}

#[no_mangle]
pub extern fn query_obb(bvh_ref: *const BVHRef, ori: *const QuaternionD, extents: *const Vector3D, center: *const Vector3D, boxes: *mut BVHBounds, count: i32, res: *mut BVHBounds, max_res: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(res, max_res as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};

    let bvh = bvh::BVH {
        nodes: v
    };
    let obb = OBB {
        orientation: to_quat(&unsafe { *ori }),
        extents: to_vec(&unsafe { *extents }),
        center: to_vec(&unsafe { *center })
    };

    let mut i = 0;

    for x in bvh.traverse_iterator(&obb, &shapes) {
        if i < max_res {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    std::mem::forget(bvh.nodes);

    i as i32
}

#[no_mangle]
pub extern fn free_bvh(bvh_ref: *const BVHRef)
{
    let _v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
}


#[no_mangle]
pub extern fn add_node(bvh_ref: *const BVHRef, new_shape: i32, boxes: *mut BVHBounds, count: i32) -> BVHRef
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let mut bvh = bvh::BVH {
        nodes: v
    };

    bvh.add_node(shapes, new_shape as usize);
    let len = bvh.nodes.len();
    let cap = bvh.nodes.capacity();
    let p = bvh.nodes.as_mut_ptr();
    std::mem::forget(bvh.nodes);
    
    BVHRef { ptr: p, len: len as i32, cap: cap as i32 }
}

#[no_mangle]
pub extern fn remove_node(bvh_ref: *const BVHRef, remove_shape: i32, boxes: *mut BVHBounds, count: i32) -> BVHRef
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let mut bvh = bvh::BVH {
        nodes: v
    };

    bvh.remove_node(shapes, remove_shape as usize, true);
    let len = bvh.nodes.len();
    let cap = bvh.nodes.capacity();
    let p = bvh.nodes.as_mut_ptr();
    std::mem::forget(bvh.nodes);
    
    BVHRef { ptr: p, len: len as i32, cap: cap as i32 }
}

#[no_mangle]
pub extern fn flatten_bvh(bvh_ref: *const BVHRef, boxes: *mut BVHBounds, count: i32, res: *mut FlatNode32, res_count: i32) -> i32
{
    let shapes = unsafe { std::slice::from_raw_parts_mut(boxes, count as usize) };
    let results = unsafe { std::slice::from_raw_parts_mut(res, res_count as usize) };

    let v = unsafe { Vec::from_raw_parts((*bvh_ref).ptr, (*bvh_ref).len as usize, (*bvh_ref).cap as usize)};
    
    let bvh = bvh::BVH {
        nodes: v
    };

    let flattened = bvh.flatten_custom(shapes, &node_32_constructor);

    for i in 0..flattened.len() {
        results[i] = flattened[i];
    }

    std::mem::forget(bvh.nodes);

    flattened.len() as i32
}

pub fn node_32_constructor(aabb: &AABB, entry_index: u32, exit_index: u32, shape_index: u32) -> FlatNode32
{
    let min = Point32 {
        x: aabb.min.x as f32,
        y: aabb.min.y as f32,
        z: aabb.min.z as f32
    };
    let max = Point32 {
        x: aabb.max.x as f32,
        y: aabb.max.y as f32,
        z: aabb.max.z as f32
    };
    let b = AABB32 {
        min,
        max
    };
    FlatNode32 {
        aabb: b,
        entry_index,
        exit_index,
        shape_index
    }
}


#[test]
fn test_building_and_querying() {
    let min = Vector3D {
        x: -1.0,
        y: -1.0,
        z: -1.0
    };
    let max = Vector3D {
        x: 1.0,
        y: 1.0,
        z: 1.0
    };
    let bounds = BoundsD {
        min,
        max
    };
    let mut b = BVHBounds {
        bounds,
        array_index: 0,
        index: 0
    };
    let ptr = unsafe {
        &mut b as *mut BVHBounds
    };


    let mut out = BVHBounds {
        bounds,
        array_index: 0,
        index: 0
    };
    let out_ptr = unsafe {
        &mut out as *mut BVHBounds
    };

    let origin = Vector3D {
        x: 0.0,
        y: -5.0,
        z: 0.0
    };
    let dir = Vector3D {
        x: 0.0,
        y: 1.0,
        z: 0.0
    };
    let o_ptr = unsafe {
        &origin as * const Vector3D
    };
    let d_ptr = unsafe {
        &dir as * const Vector3D
    };

    let mut bvhRef = build_qbvh(ptr, 1);

    let bvh_ptr = unsafe {
        &bvhRef as * const QBVHRef
    };

    let x = query_ray_q(bvh_ptr, o_ptr, d_ptr, ptr, 1, out_ptr, 1);
    assert_eq!(x, 1);
}




