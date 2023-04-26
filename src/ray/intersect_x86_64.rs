use std::arch::x86_64::*;

use nalgebra::SVector;

use crate::{aabb::Aabb, utils::fast_max};

use super::{intersect_default::RayIntersection, Ray};

trait ToRegisterType {
    type Register;

    fn to_register(&self) -> Self::Register;
}

impl ToRegisterType for SVector<f32, 2> {
    type Register = __m128;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        unsafe { _mm_set_ps(self.y, self.y, self.y, self.x) }
    }
}

impl ToRegisterType for SVector<f32, 3> {
    type Register = __m128;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        unsafe { _mm_set_ps(self.z, self.z, self.y, self.x) }
    }
}

impl ToRegisterType for SVector<f32, 4> {
    type Register = __m128;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        unsafe { _mm_set_ps(self.w, self.z, self.y, self.x) }
    }
}

#[inline(always)]
fn max_elem_m128(mm: __m128) -> f32 {
    unsafe {
        let a = _mm_unpacklo_ps(mm, mm); // x x y y
        let b = _mm_unpackhi_ps(mm, mm); // z z w w
        let c = _mm_max_ps(a, b); // ..., max(x, z), ..., ...
        let res = _mm_max_ps(mm, c); // ..., max(y, max(x, z)), ..., ...
        let mut data = std::mem::MaybeUninit::<[f32; 4]>::uninit();
        _mm_store_ps(data.as_mut_ptr() as *mut f32, res);
        let data = data.assume_init();
        data[1]
    }
}

#[inline(always)]
fn min_elem_m128(mm: __m128) -> f32 {
    unsafe {
        let a = _mm_unpacklo_ps(mm, mm); // x x y y
        let b = _mm_unpackhi_ps(mm, mm); // z z w w
        let c = _mm_min_ps(a, b); // ..., min(x, z), ..., ...
        let res = _mm_min_ps(mm, c); // ..., min(y, min(x, z)), ..., ...
        let mut data = std::mem::MaybeUninit::<[f32; 4]>::uninit();
        _mm_store_ps(data.as_mut_ptr() as *mut f32, res);
        let data = data.assume_init();
        data[1]
    }
}

#[inline(always)]
fn ray_intersects_aabb_m128(
    ray_origin: __m128,
    ray_inv_dir: __m128,
    aabb_0: __m128,
    aabb_1: __m128,
) -> bool {
    unsafe {
        let v1 = _mm_mul_ps(_mm_sub_ps(aabb_0, ray_origin), ray_inv_dir);
        let v2 = _mm_mul_ps(_mm_sub_ps(aabb_1, ray_origin), ray_inv_dir);

        let inf = _mm_min_ps(v1, v2);
        let sup = _mm_max_ps(v1, v2);

        let tmin = max_elem_m128(inf);
        let tmax = min_elem_m128(sup);

        tmax > fast_max(tmin, 0.0)
    }
}

impl RayIntersection<f32, 2> for Ray<f32, 2> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f32, 2>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_m128(ro, ri, aabb_0, aabb_1)
    }
}

impl RayIntersection<f32, 3> for Ray<f32, 3> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f32, 3>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_m128(ro, ri, aabb_0, aabb_1)
    }
}

impl RayIntersection<f32, 4> for Ray<f32, 4> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f32, 4>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_m128(ro, ri, aabb_0, aabb_1)
    }
}

impl ToRegisterType for SVector<f64, 2> {
    type Register = __m128d;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        unsafe { _mm_set_pd(self.y, self.x) }
    }
}

#[inline(always)]
fn max_elem_m128d(mm: __m128d) -> f64 {
    unsafe {
        let a = _mm_unpacklo_pd(mm, mm); // x x
        let b = _mm_max_pd(mm, a); // max(x, x), max(x, y)
        let mut data = std::mem::MaybeUninit::<[f64; 2]>::uninit();
        _mm_store_pd(data.as_mut_ptr() as *mut f64, b);
        let data = data.assume_init();
        data[1]
    }
}

#[inline(always)]
fn min_elem_m128d(mm: __m128d) -> f64 {
    unsafe {
        let a = _mm_unpacklo_pd(mm, mm); // x x
        let b = _mm_unpackhi_pd(mm, mm); // y y
        let c = _mm_min_pd(a, b); // min(x, y), min(x, y)
        let mut data = std::mem::MaybeUninit::<[f64; 2]>::uninit();
        _mm_store_pd(data.as_mut_ptr() as *mut f64, c);
        let data = data.assume_init();
        data[0]
    }
}

#[inline(always)]
fn ray_intersects_aabb_m128d(
    ray_origin: __m128d,
    ray_inv_dir: __m128d,
    aabb_0: __m128d,
    aabb_1: __m128d,
) -> bool {
    unsafe {
        let v1 = _mm_mul_pd(_mm_sub_pd(aabb_0, ray_origin), ray_inv_dir);
        let v2 = _mm_mul_pd(_mm_sub_pd(aabb_1, ray_origin), ray_inv_dir);

        let inf = _mm_min_pd(v1, v2);
        let sup = _mm_max_pd(v1, v2);

        let tmin = max_elem_m128d(inf);
        let tmax = min_elem_m128d(sup);

        tmax > fast_max(tmin, 0.0)
    }
}

impl RayIntersection<f64, 2> for Ray<f64, 2> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f64, 2>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_m128d(ro, ri, aabb_0, aabb_1)
    }
}

impl ToRegisterType for SVector<f64, 3> {
    type Register = __m256d;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        unsafe { _mm256_set_pd(self.z, self.z, self.y, self.x) }
    }
}

impl ToRegisterType for SVector<f64, 4> {
    type Register = __m256d;

    #[inline(always)]
    fn to_register(&self) -> Self::Register {
        unsafe { _mm256_set_pd(self.w, self.z, self.y, self.x) }
    }
}

#[inline(always)]
fn max_elem_m256d(mm: __m256d) -> f64 {
    unsafe {
        let a = _mm256_unpacklo_pd(mm, mm); // x x y y
        let b = _mm256_unpackhi_pd(mm, mm); // z z w w
        let c = _mm256_max_pd(a, b); // ..., max(x, z), ..., ...
        let res = _mm256_max_pd(mm, c); // ..., max(y, max(x, z)), ..., ...
        let mut data = std::mem::MaybeUninit::<[f64; 4]>::uninit();
        _mm256_store_pd(data.as_mut_ptr() as *mut f64, res);
        let data = data.assume_init();
        data[1]
    }
}

#[inline(always)]
fn min_elem_m256d(mm: __m256d) -> f64 {
    unsafe {
        let a = _mm256_unpacklo_pd(mm, mm); // x x y y
        let b = _mm256_unpackhi_pd(mm, mm); // z z w w
        let c = _mm256_min_pd(a, b); // ..., min(x, z), ..., ...
        let res = _mm256_min_pd(mm, c); // ..., min(y, min(x, z)), ..., ...
        let mut data = std::mem::MaybeUninit::<[f64; 4]>::uninit();
        _mm256_store_pd(data.as_mut_ptr() as *mut f64, res);
        let data = data.assume_init();
        data[1]
    }
}

#[inline(always)]
fn ray_intersects_aabb_m256d(
    ray_origin: __m256d,
    ray_inv_dir: __m256d,
    aabb_0: __m256d,
    aabb_1: __m256d,
) -> bool {
    unsafe {
        let v1 = _mm256_mul_pd(_mm256_sub_pd(aabb_0, ray_origin), ray_inv_dir);
        let v2 = _mm256_mul_pd(_mm256_sub_pd(aabb_1, ray_origin), ray_inv_dir);

        let inf = _mm256_min_pd(v1, v2);
        let sup = _mm256_max_pd(v1, v2);

        let tmin = max_elem_m256d(inf);
        let tmax = min_elem_m256d(sup);

        tmax > fast_max(tmin, 0.0)
    }
}

impl RayIntersection<f64, 3> for Ray<f64, 3> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f64, 3>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_m256d(ro, ri, aabb_0, aabb_1)
    }
}

impl RayIntersection<f64, 4> for Ray<f64, 4> {
    #[inline(always)]
    fn ray_intersects_aabb(&self, aabb: &Aabb<f64, 4>) -> bool {
        let ro = self.origin.coords.to_register();
        let ri = self.inv_direction.to_register();
        let aabb_0 = aabb[0].coords.to_register();
        let aabb_1 = aabb[1].coords.to_register();

        ray_intersects_aabb_m256d(ro, ri, aabb_0, aabb_1)
    }
}
