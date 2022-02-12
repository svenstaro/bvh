use std::sync::atomic::{AtomicUsize, Ordering};

use bvh_f64::aabb::{Bounded, AABB};
use bvh_f64::bounding_hierarchy::BHShape;
use bvh_f64::bvh::BVH;
use bvh_f64::capsule::Capsule;
use bvh_f64::obb::OBB;
use bvh_f64::ray::Ray;
use bvh_f64::sphere::Sphere;
use bvh_f64::Vector3;
use flexi_logger::{detailed_format, FileSpec, Logger};
use glam::DQuat;
use interoptopus::lang::c::{
    CType, CompositeType, Documentation, Field, Meta, OpaqueType, Visibility,
};
use interoptopus::lang::rust::CTypeInfo;
use interoptopus::patterns::slice::FFISliceMut;
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::util::NamespaceMappings;
use interoptopus::{ffi_function, ffi_type, Error, Interop};
use log::info;

#[repr(C)]
#[ffi_type]
#[derive(Copy, Clone, Debug)]
pub struct Double3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(C)]
#[ffi_type]
#[derive(Copy, Clone, Debug)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[ffi_type(name = "BoundingBoxD")]
#[derive(Copy, Clone, Debug)]
pub struct BoundsD {
    pub min: Double3,
    pub max: Double3,
}

#[repr(C)]
#[ffi_type(name = "BvhNode")]
#[derive(Copy, Clone, Debug)]
pub struct BVHBounds {
    pub bounds: BoundsD,
    pub internal_bvh_index: i32,
    pub index: i32,
}

#[repr(C)]
pub struct BvhRef {
    bvh: Box<BVH>,
}

unsafe impl CTypeInfo for BvhRef {
    fn type_info() -> CType {
        let fields: Vec<Field> = vec![Field::with_documentation(
            "bvh".to_string(),
            CType::ReadPointer(Box::new(CType::Opaque(OpaqueType::new(
                "BvhPtr".to_string(),
                Meta::new(),
            )))),
            Visibility::Private,
            Documentation::new(),
        )];
        let composite = CompositeType::new("BvhRef".to_string(), fields);
        CType::Composite(composite)
    }
}

#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QuatD {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

#[ffi_type(name = "Float3")]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point32 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[ffi_type(name = "BoundingBox")]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AABB32 {
    pub min: Point32,
    pub max: Point32,
}

#[ffi_type(name = "FlatNode")]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FlatNode32 {
    pub aabb: AABB32,
    pub entry_index: u32,
    pub exit_index: u32,
    pub shape_index: u32,
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
    pub index: usize,
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
        z: a.z,
    }
}

pub fn to_quat(a: &QuatD) -> DQuat {
    DQuat::from_xyzw(a.x, a.y, a.z, a.w)
}

static LOGGER_INITIALIZED: AtomicUsize = AtomicUsize::new(0);

#[ffi_function]
#[no_mangle]
pub extern "C" fn init_logger(log_path: AsciiPointer) {
    let init_count = LOGGER_INITIALIZED.fetch_add(1, Ordering::SeqCst);
    if init_count == 0 {
        let path = log_path.as_str().unwrap();
        let file = FileSpec::default()
            .directory(path)
            .basename("bvh_lib")
            .suffix("log");
        Logger::try_with_str("info")
            .unwrap()
            .log_to_file(file)
            .format_for_files(detailed_format)
            .start()
            .unwrap();
        log_panics::init();

        info!("Log initialized in folder {}", path);
    }
}

#[no_mangle]
pub unsafe extern "C" fn add_vecs(a_ptr: *mut Float3, b_ptr: *mut Float3, out_ptr: *mut Float3) {
    let a = *a_ptr;

    let a = glam::Vec3::new(a.x, a.y, a.z);
    let b = *b_ptr;
    let b = glam::Vec3::new(b.x, b.y, b.z);
    let mut c = glam::Vec3::new(0.0, 0.0, 0.0);

    for _i in 0..100000 {
        c = a + b + c;
    }

    *out_ptr = Float3 {
        x: c.x,
        y: c.y,
        z: c.z,
    };
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn build_bvh(shapes: &mut FFISliceMut<BVHBounds>) -> BvhRef {
    let bvh = Box::new(BVH::build(shapes.as_slice_mut()));
    info!("Building bvh");

    BvhRef { bvh }
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn rebuild_bvh(bvh_ref: &mut BvhRef, shapes: &mut FFISliceMut<BVHBounds>) {
    let bvh = &mut bvh_ref.bvh;
    bvh.rebuild(shapes.as_slice_mut());
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn query_ray(
    bvh_ref: &BvhRef,
    origin_vec: &Double3,
    dir_vec: &Double3,
    shapes: &mut FFISliceMut<BVHBounds>,
    buffer: &mut FFISliceMut<BVHBounds>,
) -> i32 {
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
pub extern "C" fn batch_query_rays(
    bvh_ref: &BvhRef,
    origins: &FFISliceMut<Double3>,
    dirs: &FFISliceMut<Double3>,
    hits: &mut FFISliceMut<i32>,
    shapes: &mut FFISliceMut<BVHBounds>,
    buffer: &mut FFISliceMut<BVHBounds>,
) {
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
pub extern "C" fn query_sphere(
    bvh_ref: &BvhRef,
    center: &Double3,
    radius: f64,
    shapes: &mut FFISliceMut<BVHBounds>,
    buffer: &mut FFISliceMut<BVHBounds>,
) -> i32 {
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
pub extern "C" fn query_capsule(
    bvh_ref: &BvhRef,
    start: &Double3,
    end: &Double3,
    radius: f64,
    shapes: &mut FFISliceMut<BVHBounds>,
    buffer: &mut FFISliceMut<BVHBounds>,
) -> i32 {
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
pub extern "C" fn query_aabb(
    bvh_ref: &BvhRef,
    bounds: &BoundsD,
    shapes: &mut FFISliceMut<BVHBounds>,
    buffer: &mut FFISliceMut<BVHBounds>,
) -> i32 {
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
pub extern "C" fn query_obb(
    bvh_ref: &BvhRef,
    ori: &QuatD,
    extents: &Double3,
    center: &Double3,
    shapes: &mut FFISliceMut<BVHBounds>,
    buffer: &mut FFISliceMut<BVHBounds>,
) -> i32 {
    let bvh = &bvh_ref.bvh;
    let obb = OBB {
        orientation: to_quat(ori),
        extents: to_vec(extents),
        center: to_vec(center),
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
pub extern "C" fn free_bvh(_bvh_ref: BvhRef) {}

#[ffi_function]
#[no_mangle]
pub extern "C" fn add_node(
    bvh_ref: &mut BvhRef,
    new_shape: i32,
    shapes: &mut FFISliceMut<BVHBounds>,
) {
    let bvh = &mut bvh_ref.bvh;
    bvh.add_node(shapes.as_slice_mut(), new_shape as usize);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn remove_node(
    bvh_ref: &mut BvhRef,
    remove_shape: i32,
    shapes: &mut FFISliceMut<BVHBounds>,
) {
    let bvh = &mut bvh_ref.bvh;
    bvh.remove_node(shapes.as_slice_mut(), remove_shape as usize, true);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn update_node(
    bvh_ref: &mut BvhRef,
    update_shape: i32,
    shapes: &mut FFISliceMut<BVHBounds>,
) {
    let bvh = &mut bvh_ref.bvh;
    bvh.remove_node(shapes.as_slice_mut(), update_shape as usize, false);
    bvh.add_node(shapes, update_shape as usize);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn flatten_bvh(
    bvh_ref: &mut BvhRef,
    shapes: &mut FFISliceMut<BVHBounds>,
    results: &mut FFISliceMut<FlatNode32>,
) -> i32 {
    let bvh = &bvh_ref.bvh;

    let flattened = bvh.flatten_custom(shapes.as_slice_mut(), &node_32_constructor);

    for i in 0..flattened.len() {
        results[i] = flattened[i];
    }

    flattened.len() as i32
}

pub fn node_32_constructor(
    aabb: &AABB,
    entry_index: u32,
    exit_index: u32,
    shape_index: u32,
) -> FlatNode32 {
    let min = Point32 {
        x: aabb.min.x as f32,
        y: aabb.min.y as f32,
        z: aabb.min.z as f32,
    };
    let max = Point32 {
        x: aabb.max.x as f32,
        y: aabb.max.y as f32,
        z: aabb.max.z as f32,
    };
    let b = AABB32 { min, max };
    FlatNode32 {
        aabb: b,
        entry_index,
        exit_index,
        shape_index,
    }
}

interoptopus::inventory!(
    my_inventory,
    [],
    [
        init_logger,
        build_bvh,
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
    ],
    [],
    []
);

fn bindings_csharp() -> Result<(), Error> {
    use interoptopus_backend_csharp::{
        overloads::{DotNet, Unity},
        Config, Generator, Unsafe,
    };

    Generator::new(
        Config {
            class: "NativeBvhInterop".to_string(),
            dll_name: "bvh_lib".to_string(),
            namespace_mappings: NamespaceMappings::new("Assets.Scripts.Native"),
            use_unsafe: Unsafe::UnsafePlatformMemCpy,
            ..Config::default()
        },
        my_inventory(),
    )
    .add_overload_writer(Unity::new())
    .add_overload_writer(DotNet::new())
    .write_file("../bindings/csharp/Interop.cs")?;

    Ok(())
}

#[test]
fn gen_bindings() {
    bindings_csharp().unwrap();
}

#[test]
fn test_building_and_querying() {
    let min = Double3 {
        x: -1.0,
        y: -1.0,
        z: -1.0,
    };
    let max = Double3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    let bounds = BoundsD { min, max };
    let b = BVHBounds {
        bounds,
        index: 0,
        internal_bvh_index: 0,
    };

    let _out = BVHBounds {
        bounds,
        index: 0,
        internal_bvh_index: 0,
    };

    let origin = Double3 {
        x: 0.0,
        y: -5.0,
        z: 0.0,
    };
    let dir = Double3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let in_slice = &mut [b];
    let mut input = FFISliceMut::<BVHBounds>::from_slice(in_slice);
    let out_slice = &mut [b];
    let mut out = FFISliceMut::<BVHBounds>::from_slice(out_slice);
    let bvh_ref = build_bvh(&mut input);

    let x = query_ray(&bvh_ref, &origin, &dir, &mut input, &mut out);
    assert_eq!(x, 1);
}
