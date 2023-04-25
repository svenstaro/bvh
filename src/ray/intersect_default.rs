use super::Ray;
use crate::{aabb::AABB, utils::fast_max};
use nalgebra::{ClosedMul, ClosedSub, Scalar, SimdPartialOrd};
use num::Zero;

pub trait RayIntersection<T, const D: usize>
where
    T: Copy + Scalar,
{
    fn ray_intersects_aabb(&self, aabb: &AABB<T, D>) -> bool;
}

#[cfg(not(feature = "simd"))]
impl<T, const D: usize> RayIntersection<T, D> for Ray<T, D>
where
    T: Scalar + Copy + ClosedSub + ClosedMul + Zero + PartialOrd + SimdPartialOrd,
{
    fn ray_intersects_aabb(&self, aabb: &AABB<T, D>) -> bool {
        let lbr = (aabb[0].coords - self.origin.coords).component_mul(&self.inv_direction);
        let rtr = (aabb[1].coords - self.origin.coords).component_mul(&self.inv_direction);

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax > fast_max(tmin, T::zero())
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<T, const D: usize> RayIntersection<T, D> for Ray<T, D>
where
    T: Scalar + Copy + ClosedSub + ClosedMul + Zero + PartialOrd + SimdPartialOrd,
{
    default fn ray_intersects_aabb(&self, aabb: &AABB<T, D>) -> bool {
        let lbr = (aabb[0].coords - self.origin.coords).component_mul(&self.inv_direction);
        let rtr = (aabb[1].coords - self.origin.coords).component_mul(&self.inv_direction);

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax > max(tmin, T::zero())
    }
}
