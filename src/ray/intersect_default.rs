//! This file contains the generic implementation of [`RayIntersection`]

use super::Ray;
use crate::{
    aabb::Aabb,
    bounding_hierarchy::BHValue,
    utils::{fast_max, has_nan},
};

/// The [`RayIntersection`] trait allows for generic implementation of ray intersection
/// useful for our SIMD optimizations.
pub trait RayIntersection<T: BHValue, const D: usize> {
    fn ray_intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool;
}

#[cfg(not(feature = "simd"))]
impl<T: BHValue, const D: usize> RayIntersection<T, D> for Ray<T, D> {
    fn ray_intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<T: BHValue, const D: usize> RayIntersection<T, D> for Ray<T, D> {
    default fn ray_intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        let lbr = (aabb[0].coords - self.origin.coords).component_mul(&self.inv_direction);
        let rtr = (aabb[1].coords - self.origin.coords).component_mul(&self.inv_direction);

        if has_nan(&lbr) | has_nan(&rtr) {
            return false;
        }

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax >= fast_max(tmin, T::zero())
    }
}
