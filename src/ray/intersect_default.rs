//! This file contains the generic implementation of [`RayIntersection`]

use super::Ray;
use crate::{
    aabb::Aabb,
    bounding_hierarchy::BHValue,
    utils::{fast_max, has_nan},
};

/// The [`RayIntersection`] trait allows for generic implementation of ray intersection
/// useful for our SIMD optimizations.
pub(crate) trait RayIntersection<T: BHValue, const D: usize> {
    fn ray_intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool;
}

impl<T: BHValue, const D: usize> RayIntersection<T, D> for Ray<T, D> {
    #[inline]
    fn ray_intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        // Use the packed SIMD kernel if one exists for this (T, D) combination.
        #[cfg(feature = "simd")]
        if let Some(intersects) = super::intersect_simd::ray_intersects_aabb_fast(self, aabb) {
            return intersects;
        }

        let lbr = (aabb[0].coords - self.origin.coords).component_mul(&self.inv_direction);
        let rtr = (aabb[1].coords - self.origin.coords).component_mul(&self.inv_direction);

        if has_nan(&lbr) | has_nan(&rtr) {
            // Assumption: the ray is in the plane of an AABB face. Be consistent and
            // consider this a non-intersection. This avoids making the result depend
            // on which axis/axes have NaN (min/max in the code that follows are not
            // commutative).
            return false;
        }

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax >= fast_max(tmin, T::zero())
    }
}
