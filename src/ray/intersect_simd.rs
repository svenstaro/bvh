//! This file contains overrides for SIMD implementations of [`RayIntersection`]
//! for any architectures supported by the `wide` crate.

use nalgebra::SVector;
use wide::*;

use crate::{
    aabb::Aabb,
    utils::{fast_max, fast_min, has_nan},
};

use super::{Ray, intersect_default::RayIntersection};

trait ToRegisterType {
    type Register;

    fn to_register(&self) -> Self::Register;
}

impl ToRegisterType for SVector<f32, 2> {
    type Register = f32x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f32x4::new([self.y, self.y, self.y, self.x])
    }
}

impl ToRegisterType for SVector<f32, 3> {
    type Register = f32x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f32x4::new([self.z, self.z, self.y, self.x])
    }
}

impl ToRegisterType for SVector<f32, 4> {
    type Register = f32x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f32x4::new([self.w, self.z, self.y, self.x])
    }
}

/// Compute the horizontal maximum of the SIMD vector
#[inline(always)]
fn max_elem_f32x4(v: f32x4) -> f32 {
    let a = v.to_array();
    fast_max(fast_max(a[0], a[1]), fast_max(a[2], a[3]))
}

/// Compute the horizontal minimum of the SIMD vector
#[inline(always)]
fn min_elem_f32x4(v: f32x4) -> f32 {
    let a = v.to_array();
    fast_min(fast_min(a[0], a[1]), fast_min(a[2], a[3]))
}

#[inline(always)]
fn has_nan_f32x4(v: f32x4) -> bool {
    has_nan(&v.to_array())
}

#[inline(always)]
fn has_nan_f64x2(v: f64x2) -> bool {
    has_nan(&v.to_array())
}

#[inline(always)]
fn has_nan_f64x4(v: f64x4) -> bool {
    has_nan(&v.to_array())
}

#[inline(always)]
fn ray_intersects_aabb_f32x4(
    ray_origin: f32x4,
    ray_inv_dir: f32x4,
    aabb_0: f32x4,
    aabb_1: f32x4,
) -> bool {
    let v1 = (aabb_0 - ray_origin) * ray_inv_dir;
    let v2 = (aabb_1 - ray_origin) * ray_inv_dir;

    if has_nan_f32x4(v1) | has_nan_f32x4(v2) {
        return false;
    }

    let inf = v1.fast_min(v2);
    let sup = v1.fast_max(v2);

    let tmin = max_elem_f32x4(inf);
    let tmax = min_elem_f32x4(sup);

    tmax >= fast_max(tmin, 0.0)
}

impl RayIntersection<f32, 2> for Ray<f32, 2> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f32, 2>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_f32x4(ro, ri, aabb_0, aabb_1)
    }
}

impl RayIntersection<f32, 3> for Ray<f32, 3> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f32, 3>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_f32x4(ro, ri, aabb_0, aabb_1)
    }
}

impl RayIntersection<f32, 4> for Ray<f32, 4> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f32, 4>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_f32x4(ro, ri, aabb_0, aabb_1)
    }
}

impl ToRegisterType for SVector<f64, 2> {
    type Register = f64x2;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f64x2::new([self.y, self.x])
    }
}

/// Compute the horizontal maximum of the SIMD vector
#[inline(always)]
fn max_elem_f64x2(v: f64x2) -> f64 {
    let a = v.to_array();
    fast_max(a[0], a[1])
}

/// Compute the horizontal minimum of the SIMD vector
#[inline(always)]
fn min_elem_f64x2(v: f64x2) -> f64 {
    let a = v.to_array();
    fast_min(a[0], a[1])
}

#[inline(always)]
fn ray_intersects_aabb_f64x2(
    ray_origin: f64x2,
    ray_inv_dir: f64x2,
    aabb_0: f64x2,
    aabb_1: f64x2,
) -> bool {
    let v1 = (aabb_0 - ray_origin) * ray_inv_dir;
    let v2 = (aabb_1 - ray_origin) * ray_inv_dir;

    if has_nan_f64x2(v1) | has_nan_f64x2(v2) {
        return false;
    }

    let inf = v1.min(v2);
    let sup = v1.max(v2);

    let tmin = max_elem_f64x2(inf);
    let tmax = min_elem_f64x2(sup);

    tmax >= fast_max(tmin, 0.0)
}

impl RayIntersection<f64, 2> for Ray<f64, 2> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f64, 2>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_f64x2(ro, ri, aabb_0, aabb_1)
    }
}

impl ToRegisterType for SVector<f64, 3> {
    type Register = f64x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f64x4::new([self.z, self.z, self.y, self.x])
    }
}

impl ToRegisterType for SVector<f64, 4> {
    type Register = f64x4;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        f64x4::new([self.w, self.z, self.y, self.x])
    }
}

/// Compute the horizontal maximum of the SIMD vector
#[inline(always)]
fn max_elem_f64x4(v: f64x4) -> f64 {
    let a = v.to_array();
    fast_max(fast_max(a[0], a[1]), fast_max(a[2], a[3]))
}

/// Compute the horizontal minimum of the SIMD vector
#[inline(always)]
fn min_elem_f64x4(v: f64x4) -> f64 {
    let a = v.to_array();
    fast_min(fast_min(a[0], a[1]), fast_min(a[2], a[3]))
}

#[inline(always)]
fn ray_intersects_aabb_f64x4(
    ray_origin: f64x4,
    ray_inv_dir: f64x4,
    aabb_0: f64x4,
    aabb_1: f64x4,
) -> bool {
    let v1 = (aabb_0 - ray_origin) * ray_inv_dir;
    let v2 = (aabb_1 - ray_origin) * ray_inv_dir;

    if has_nan_f64x4(v1) | has_nan_f64x4(v2) {
        return false;
    }

    let inf = v1.min(v2);
    let sup = v1.max(v2);

    let tmin = max_elem_f64x4(inf);
    let tmax = min_elem_f64x4(sup);

    tmax >= fast_max(tmin, 0.0)
}

impl RayIntersection<f64, 3> for Ray<f64, 3> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f64, 3>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_f64x4(ro, ri, aabb_0, aabb_1)
    }
}

impl RayIntersection<f64, 4> for Ray<f64, 4> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f64, 4>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_f64x4(ro, ri, aabb_0, aabb_1)
    }
}
