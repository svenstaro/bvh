//! This file contains packed SIMD kernels for ray-AABB intersection,
//! selected at runtime from the value type. They are used by the
//! [`RayIntersection`] implementation in `intersect_default`.

use core::any::Any;

use nalgebra::SVector;
use wide::*;

use crate::{
    aabb::Aabb,
    bounding_hierarchy::BHValue,
    utils::{fast_max, fast_min, has_nan},
};

use super::Ray;

/// Copy `D` coordinates into `N` lanes, duplicating the last one to fill the
/// register. This is sound because the slab test reduces all lanes with
/// `min`/`max`, so duplicated lanes cannot change the result.
///
/// Requires `1 <= D <= N`, which [`ray_intersects_aabb_fast`] checks.
fn padded_lanes<T: Copy, const N: usize, const D: usize>(coords: &[T]) -> [T; N] {
    core::array::from_fn(|i| coords[i.min(D - 1)])
}

/// Converts a vector of coordinates into its packed SIMD register.
trait ToRegisterType {
    type Register: SlabTest;

    fn to_register(&self) -> Self::Register;
}

impl<const D: usize> ToRegisterType for SVector<f32, D> {
    type Register = f32x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f32x4::new(padded_lanes::<_, 4, D>(self.as_slice()))
    }
}

impl<const D: usize> ToRegisterType for SVector<f64, D> {
    type Register = f64x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f64x4::new(padded_lanes::<_, 4, D>(self.as_slice()))
    }
}

/// The slab test for one register type. Implemented per register so the
/// generic kernel stays a single body.
trait SlabTest: Copy {
    fn ray_intersects_aabb(self, inv_dir: Self, aabb_0: Self, aabb_1: Self) -> bool;
}

impl SlabTest for f32x4 {
    #[inline(always)]
    fn ray_intersects_aabb(self, inv_dir: Self, aabb_0: Self, aabb_1: Self) -> bool {
        let v1 = (aabb_0 - self) * inv_dir;
        let v2 = (aabb_1 - self) * inv_dir;

        if has_nan(&v1.to_array()) | has_nan(&v2.to_array()) {
            return false;
        }

        // `fast_min`/`fast_max` skip NaN checks; NaN inputs were rejected above.
        let inf = v1.fast_min(v2);
        let sup = v1.fast_max(v2);

        let a = inf.to_array();
        let tmin = fast_max(fast_max(a[0], a[1]), fast_max(a[2], a[3]));
        let a = sup.to_array();
        let tmax = fast_min(fast_min(a[0], a[1]), fast_min(a[2], a[3]));

        tmax >= fast_max(tmin, 0.0)
    }
}

impl SlabTest for f64x4 {
    #[inline(always)]
    fn ray_intersects_aabb(self, inv_dir: Self, aabb_0: Self, aabb_1: Self) -> bool {
        let v1 = (aabb_0 - self) * inv_dir;
        let v2 = (aabb_1 - self) * inv_dir;

        if has_nan(&v1.to_array()) | has_nan(&v2.to_array()) {
            return false;
        }

        let inf = v1.min(v2);
        let sup = v1.max(v2);

        let a = inf.to_array();
        let tmin = fast_max(fast_max(a[0], a[1]), fast_max(a[2], a[3]));
        let a = sup.to_array();
        let tmax = fast_min(fast_min(a[0], a[1]), fast_min(a[2], a[3]));

        tmax >= fast_max(tmin, 0.0)
    }
}

/// Pack the ray and [`Aabb`] into registers and run the slab test.
fn packed_kernel<T: BHValue, const D: usize>(ray: &Ray<T, D>, aabb: &Aabb<T, D>) -> bool
where
    SVector<T, D>: ToRegisterType,
{
    let ro = ray.origin.coords.to_register();
    let ri = ray.inv_direction.to_register();
    let aabb_0 = aabb[0].coords.to_register();
    let aabb_1 = aabb[1].coords.to_register();

    ro.ray_intersects_aabb(ri, aabb_0, aabb_1)
}

/// SIMD fast path for the ray-AABB intersection test.
///
/// Returns `None` if no kernel exists for this `(T, D)` combination and the
/// caller must fall back to the scalar implementation.
pub(super) fn ray_intersects_aabb_fast<T: BHValue, const D: usize>(
    ray: &Ray<T, D>,
    aabb: &Aabb<T, D>,
) -> Option<bool> {
    // The registers hold at most 4 lanes.
    if !(2..=4).contains(&D) {
        return None;
    }

    // Downcasting succeeds only if `T` is the concrete type. After
    // monomorphization, this folds to a constant.
    let (ray, aabb): (&dyn Any, &dyn Any) = (ray, aabb);

    if let (Some(ray), Some(aabb)) = (ray.downcast_ref::<Ray<f32, D>>(), aabb.downcast_ref()) {
        return Some(packed_kernel(ray, aabb));
    }

    if let (Some(ray), Some(aabb)) = (ray.downcast_ref::<Ray<f64, D>>(), aabb.downcast_ref()) {
        return Some(packed_kernel(ray, aabb));
    }

    None
}
