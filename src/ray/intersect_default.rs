//! This file contains the generic implementation of [`RayIntersection`]

use super::Ray;
use crate::{aabb::Aabb, bounding_hierarchy::BHValue, utils::fast_max};

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

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax > fast_max(tmin, T::zero())
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<T: BHValue, const D: usize> RayIntersection<T, D> for Ray<T, D> {
    default fn ray_intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        let lbr = (aabb[0].coords - self.origin.coords).component_mul(&self.inv_direction);
        let rtr = (aabb[1].coords - self.origin.coords).component_mul(&self.inv_direction);

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax > fast_max(tmin, T::zero())
    }
}
