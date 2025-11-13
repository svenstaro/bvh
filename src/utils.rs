//! Utilities module.

use crate::bounding_hierarchy::{BHShape, BHValue};
use crate::bvh::ShapeIndex;
use crate::{aabb::Aabb, bvh::Shapes};

use nalgebra::Scalar;
use num_traits::Float;

/// Fast floating point minimum.  This function matches the semantics of
///
/// ```no_compile
/// if x < y { x } else { y }
/// ```
///
/// which has efficient instruction sequences on many platforms (1 instruction on x86).  For most
/// values, it matches the semantics of `x.min(y)`; the special cases are:
///
/// ```text
/// min(-0.0, +0.0); +0.0
/// min(+0.0, -0.0): -0.0
/// min( NaN,  1.0):  1.0
/// min( 1.0,  NaN):  NaN
/// ```
///
/// Note: This exists because [`std::cmp::min`] requires Ord which floating point types do not satisfy
#[inline(always)]
#[allow(dead_code)]
pub fn fast_min<T: Scalar + Copy + PartialOrd>(x: T, y: T) -> T {
    if x < y { x } else { y }
}

/// Fast floating point maximum.  This function matches the semantics of
///
/// ```no_compile
/// if x > y { x } else { y }
/// ```
///
/// which has efficient instruction sequences on many platforms (1 instruction on x86).  For most
/// values, it matches the semantics of `x.max(y)`; the special cases are:
///
/// ```text
/// max(-0.0, +0.0); +0.0
/// max(+0.0, -0.0): -0.0
/// max( NaN,  1.0):  1.0
/// max( 1.0,  NaN):  NaN
/// ```
///
/// Note: This exists because [`std::cmp::max`] requires Ord which floating point types do not satisfy
#[inline(always)]
#[allow(dead_code)]
pub fn fast_max<T: Scalar + Copy + PartialOrd>(x: T, y: T) -> T {
    if x > y { x } else { y }
}

/// Defines a Bucket utility object. Used to store the properties of shape-partitions
/// in the [`Bvh`] build procedure using SAH.
#[derive(Clone, Copy)]
pub struct Bucket<T: BHValue, const D: usize> {
    /// The number of shapes in this [`Bucket`].
    pub size: usize,

    /// The joint [`Aabb`] of the shapes in this [`Bucket`].
    pub aabb: Aabb<T, D>,

    /// The [`Aabb`] of the centers of the shapes in this [`Bucket`]
    pub centroid: Aabb<T, D>,
}

impl<T: BHValue, const D: usize> Bucket<T, D> {
    /// Returns an empty bucket.
    pub fn empty() -> Bucket<T, D> {
        Bucket {
            size: 0,
            aabb: Aabb::empty(),
            centroid: Aabb::empty(),
        }
    }

    /// Extend this [`Bucket`] by a shape with the given [`Aabb`].
    pub fn add_aabb(&mut self, aabb: &Aabb<T, D>) {
        self.size += 1;
        self.aabb = self.aabb.join(aabb);
        self.centroid.grow_mut(&aabb.center());
    }

    /// Join the contents of two [`Bucket`]'s.
    pub fn join_bucket(a: Bucket<T, D>, b: &Bucket<T, D>) -> Bucket<T, D> {
        Bucket {
            size: a.size + b.size,
            aabb: a.aabb.join(&b.aabb),
            centroid: a.centroid.join(&b.centroid),
        }
    }
}

pub(crate) fn joint_aabb_of_shapes<T: BHValue, const D: usize, Shape: BHShape<T, D>>(
    indices: &[ShapeIndex],
    shapes: &Shapes<Shape>,
) -> (Aabb<T, D>, Aabb<T, D>) {
    let mut aabb = Aabb::empty();
    let mut centroid = Aabb::empty();
    for index in indices {
        let shape = shapes.get(*index);
        aabb.join_mut(&shape.aabb());
        centroid.grow_mut(&shape.aabb().center());
    }
    (aabb, centroid)
}

/// Returns `true` if and only if any of the floats returned by `iter` are NaN.
#[inline(always)]
pub(crate) fn has_nan<'a, T: Float + 'a>(iter: impl IntoIterator<Item = &'a T>) -> bool {
    iter.into_iter().any(|f| f.is_nan())
}
