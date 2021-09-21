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
use bvh::{BVH, BVHNode};
use ray::Ray;
use shapes::{Capsule, Sphere, OBB};
use glam::DQuat;
use bvh::BUILD_THREAD_COUNT;
use std::sync::atomic::{AtomicUsize, Ordering};
use parry3d_f64::partitioning::{QBVH, QBVHDataGenerator, IndexedData};
use parry3d_f64::bounding_volume::aabb::AABB as QAABB;
use parry3d_f64::math::{Point, Vector};
use parry3d_f64::query::Ray as RayQ;
use parry3d_f64::query::visitors::RayIntersectionsVisitor;
use interoptopus::{ffi_function, ffi_type};
use interoptopus::patterns::slice::{FFISliceMut};
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::lang::rust::CTypeInfo;
use interoptopus::lang::c::{CType, CompositeType, Documentation, Field, OpaqueType, PrimitiveType, Visibility, Meta};
use flexi_logger::{FileSpec, Logger, detailed_format};
use log::{info, warn, error};

#[macro_use]
extern crate lazy_static;


#[cfg(test)]
mod testbase;


#[no_mangle]
pub extern "C" fn add_numbers(number1: i32, number2: i32) -> i32 {
    println!("Hello from rust!");
    number1 + number2
}

#[repr(C)]
#[ffi_type]
#[derive(Copy, Clone, Debug)]
pub struct Double3 {    
    pub x: f64,
    pub y: f64,
    pub z: f64
}

#[repr(C)]
#[ffi_type]
#[derive(Copy, Clone, Debug)]
pub struct Float3 {    
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[repr(C)]
#[ffi_type(name="BoundingBoxD")]
#[derive(Copy, Clone, Debug)]
pub struct BoundsD {
    pub min: Double3,
    pub max: Double3
}

#[repr(C)]
#[ffi_type(name="BvhNode")]
#[derive(Copy, Clone, Debug)]
pub struct BVHBounds {
    pub bounds: BoundsD,
    pub internal_bvh_index: i32,
    pub index: i32
}

#[repr(C)]
pub struct BvhRef {
    bvh: Box<bvh::BVH>
}

unsafe impl CTypeInfo for BvhRef {
    fn type_info() -> CType {
        let fields: Vec<Field> = vec![
            Field::with_documentation("bvh".to_string(), CType::ReadPointer(Box::new(CType::Opaque(OpaqueType::new("BvhPtr".to_string(), Meta::new())))), Visibility::Private, Documentation::new()),
        ];
        let composite = CompositeType::new("BvhRef".to_string(), fields);
        CType::Composite(composite)
    }
}

#[ffi_type(opaque)]
#[repr(C)]
pub struct QBVHRef {
    bvh: Box<QBVH<RefNode>>
}

#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QuatD {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64
}

#[ffi_type(name="Float3")]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point32 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[ffi_type(name="BoundingBox")]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AABB32 {
    pub min: Point32,
    pub max: Point32
}

#[ffi_type(name="FlatNode")]
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

#[ffi_type(opaque)]
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
        self.internal_bvh_index = index as i32;
    }

    fn bh_node_index(&self) -> usize {
        self.internal_bvh_index as usize
    }
}

pub fn to_vec(a: &Double3) -> Vector3 {
    Vector3::new(a.x, a.y, a.z)
}

pub fn to_vecd(a: &Vector3) -> Double3 {
    Double3 {
        x: a.x,
        y: a.y,
        z: a.z
    }
}

pub fn to_quat(a: &QuatD) -> DQuat {
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
pub extern "C" fn set_build_thread_count(count: i32)
{
    BUILD_THREAD_COUNT.store(count as usize, Ordering::Relaxed);
}

static LOGGER_INITIALIZED: AtomicUsize = AtomicUsize::new(0);

#[ffi_function]
#[no_mangle]
pub extern "C" fn init_logger(log_path: AsciiPointer)
{
    let init_count = LOGGER_INITIALIZED.fetch_add(1, Ordering::SeqCst);
    if init_count == 0 {
        let path = log_path.as_str().unwrap();
        let file = FileSpec::default()
        .directory(path)
        .basename("bvh_f64")
        .suffix("log");
        Logger::try_with_str("info").unwrap()
        .log_to_file(file)
        .format_for_files(detailed_format)
        .start().unwrap();
        log_panics::init();

        info!("Log initialized in folder {}", path);
    }
}


#[no_mangle]
pub extern "C" fn add_vecs(a_ptr: *mut Float3, b_ptr: *mut Float3, out_ptr: *mut Float3)
{
    let a = unsafe {*a_ptr};

    let a = glam::Vec3::new(a.x, a.y, a.z);
    let b = unsafe {*b_ptr};
    let b = glam::Vec3::new(b.x, b.y, b.z);
    let mut c = glam::Vec3::new(0.0, 0.0, 0.0);

    for _i in 0 .. 100000 {
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

#[ffi_function]
#[no_mangle]
pub extern "C" fn build_bvh(shapes: &mut FFISliceMut<BVHBounds>) -> BvhRef 
{
    let bvh = Box::new(bvh::BVH::build(shapes.as_slice_mut()));
    info!("Building bvh");

    BvhRef { bvh }
}


#[ffi_function]
#[no_mangle]
pub extern "C" fn build_qbvh(shapes: &mut FFISliceMut<BVHBounds>) -> QBVHRef 
{

    let data = BoundsData::new(shapes.as_slice_mut());
    let mut bvh = Box::new(QBVH::new());

    bvh.clear_and_rebuild(data, 0.0);

    QBVHRef { bvh }
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn query_ray_q(bvh_ref: &QBVHRef, o_vec: &Double3, d_vec: &Double3, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>) -> i32
{
    let bvh = &bvh_ref.bvh;
    let ray = RayQ::new(Point::new(o_vec.x, o_vec.y, o_vec.z), Vector::new(d_vec.x, d_vec.y, d_vec.z));
    let mut i = 0;

    let mut stack_arr: [u32; 32] = [0; 32];

    let mut stack = unsafe { Vec::from_raw_parts(&mut stack_arr as *mut u32, 0, 32)};
    
    let mut visit = |node: &RefNode| {
        if i < buffer.len() {
            buffer[i as usize] = shapes[node.index];
            i += 1;
            return false;
        }
        i += 1;
        true
    };

    let mut visitor = RayIntersectionsVisitor::new(&ray, 1000000000000.0, &mut visit);

    bvh.traverse_depth_first_with_stack(&mut visitor, &mut stack);

    // stack is pretending to be a heap allocated vector
    std::mem::forget(stack);

    i as i32
}



#[ffi_function]
#[no_mangle]
pub extern "C" fn rebuild_bvh(bvh_ref: &mut BvhRef, shapes: &mut FFISliceMut<BVHBounds>)
{
    let bvh = &mut bvh_ref.bvh;
    bvh.rebuild(shapes.as_slice_mut());
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn query_ray(bvh_ref: &BvhRef, origin_vec: &Double3, dir_vec: &Double3, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>) -> i32
{
    let bvh = &bvh_ref.bvh;

    let ray = Ray::new(to_vec(origin_vec), to_vec(dir_vec));
    let mut i = 0;

    for x in bvh.traverse_iterator(&ray, &shapes) {
        if i < buffer.len() {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    i as i32
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn batch_query_rays(bvh_ref: &BvhRef, origins: &FFISliceMut<Double3>, dirs: &FFISliceMut<Double3>, hits: &mut FFISliceMut<i32>, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>)
{
    let bvh = &bvh_ref.bvh;
    let mut i = 0;
    let ray_count = origins.len();
    for r in 0..ray_count as usize {
        let ray = Ray::new(to_vec(&origins[r]), to_vec(&dirs[r]));
        let mut res = 0;
        for x in bvh.traverse_iterator(&ray, &shapes) {
            if i < buffer.len() {
                buffer[i as usize] = *x;
            }
            i += 1;
            res += 1;
        }
        hits[r] = res;
    }
}


#[ffi_function]
#[no_mangle]
pub extern "C" fn query_sphere(bvh_ref: &BvhRef, center: &Double3, radius: f64, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>) -> i32
{

    let bvh = &bvh_ref.bvh;

    let test_shape = Sphere::new(to_vec(center), radius);
    let mut i = 0;

    for x in bvh.traverse_iterator(&test_shape, &shapes) {
        if i < buffer.len() {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    i as i32
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn query_capsule(bvh_ref: &BvhRef, start: &Double3, end: &Double3, radius: f64, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>) -> i32
{
    let bvh = &bvh_ref.bvh;

    let test_shape = Capsule::new(to_vec(start), to_vec(end), radius);
    let mut i = 0;

    for x in bvh.traverse_iterator(&test_shape, &shapes) {
        if i < buffer.len() {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    i as i32
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn query_aabb(bvh_ref: &BvhRef, bounds: &BoundsD, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>) -> i32
{
    let bvh = &bvh_ref.bvh;

    let min = to_vec(&bounds.min);
    let max = to_vec(&bounds.max);
    let test_shape = AABB::with_bounds(min, max);
    let mut i = 0;

    for x in bvh.traverse_iterator(&test_shape, &shapes) {
        if i < buffer.len() {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    i as i32
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn query_obb(bvh_ref: &BvhRef, ori: &QuatD, extents: &Double3, center: &Double3, shapes: &mut FFISliceMut<BVHBounds>, buffer: &mut FFISliceMut<BVHBounds>) -> i32
{
    let bvh = &bvh_ref.bvh;
    let obb = OBB {
        orientation: to_quat(ori),
        extents: to_vec(extents),
        center: to_vec(center)
    };

    let mut i = 0;

    for x in bvh.traverse_iterator(&obb, &shapes) {
        if i < buffer.len() {
            buffer[i as usize] = *x;
        }
        i += 1;
    }
    i as i32
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn free_bvh(_bvh_ref: BvhRef)
{
}


#[ffi_function]
#[no_mangle]
pub extern "C" fn add_node(bvh_ref: &mut BvhRef, new_shape: i32, shapes: &mut FFISliceMut<BVHBounds>)
{
    let bvh = &mut bvh_ref.bvh;
    bvh.add_node(shapes.as_slice_mut(), new_shape as usize);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn remove_node(bvh_ref: &mut BvhRef, remove_shape: i32, shapes: &mut FFISliceMut<BVHBounds>)
{
    let bvh = &mut bvh_ref.bvh;
    bvh.remove_node(shapes.as_slice_mut(), remove_shape as usize, true);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn update_node(bvh_ref: &mut BvhRef, update_shape: i32, shapes: &mut FFISliceMut<BVHBounds>)
{
    let bvh = &mut bvh_ref.bvh;
    bvh.remove_node(shapes.as_slice_mut(), update_shape as usize, false);
    bvh.add_node(shapes, update_shape as usize);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn flatten_bvh(bvh_ref: &mut BvhRef, shapes: &mut FFISliceMut<BVHBounds>, results: &mut FFISliceMut<FlatNode32>) -> i32
{
    let bvh = &bvh_ref.bvh;

    let flattened = bvh.flatten_custom(shapes.as_slice_mut(), &node_32_constructor);

    for i in 0..flattened.len() {
        results[i] = flattened[i];
    }

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

interoptopus::inventory!(my_inventory, [], [
    init_logger,
    build_bvh,
    build_qbvh,
    query_ray_q,
    rebuild_bvh,
    query_ray,
    batch_query_rays,
    query_sphere,
    query_capsule,
    query_aabb,
    query_obb,
    free_bvh,
    add_node,
    remove_node,
    update_node,
    flatten_bvh,
    ], [], []);

use interoptopus::util::NamespaceMappings;
use interoptopus::{Error, Interop};

#[test]
fn bindings_csharp() -> Result<(), Error> {
    use interoptopus_backend_csharp::{Config, Generator, Unsafe, overloads::{DotNet, Unity}};

    Generator::new(
        Config {
            class: "NativeBvhInterop".to_string(),
            dll_name: "bvh_f64".to_string(),
            namespace_mappings: NamespaceMappings::new("Assets.Scripts.Native"),
            use_unsafe: Unsafe::UnsafePlatformMemCpy,
            ..Config::default()
        },
        my_inventory(),
    )
    .add_overload_writer(Unity::new())
    .add_overload_writer(DotNet::new())
    .write_file("bindings/csharp/Interop.cs")?;

    Ok(())
}


#[test]
fn test_building_and_querying() {
    let min = Double3 {
        x: -1.0,
        y: -1.0,
        z: -1.0
    };
    let max = Double3 {
        x: 1.0,
        y: 1.0,
        z: 1.0
    };
    let bounds = BoundsD {
        min,
        max
    };
    let b = BVHBounds {
        bounds,
        index: 0,
        internal_bvh_index: 0
    };

    let out = BVHBounds {
        bounds,
        index: 0,
        internal_bvh_index: 0
    };

    let origin = Double3 {
        x: 0.0,
        y: -5.0,
        z: 0.0
    };
    let dir = Double3 {
        x: 0.0,
        y: 1.0,
        z: 0.0
    };
    let in_slice = &mut [b];
    let mut input = FFISliceMut::<BVHBounds>::from_slice(in_slice);
    let out_slice = &mut [b];
    let mut out = FFISliceMut::<BVHBounds>::from_slice(out_slice);

    let bvh_ref = build_qbvh(&mut input);

    let x = query_ray_q(&bvh_ref, &origin, &dir, &mut input, &mut out);
    assert_eq!(x, 1);
}




