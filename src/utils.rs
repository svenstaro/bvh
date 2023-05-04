//! Utilities module.

use crate::aabb::Aabb;
use crate::bounding_hierarchy::BHShape;

use nalgebra::{Scalar, SimdPartialOrd};
use num::Float;

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
    if x < y {
        x
    } else {
        y
    }
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
    if x > y {
        x
    } else {
        y
    }
}

/// Concatenates the list of vectors into a single vector.
/// Drains the elements from the source `vectors`.
pub fn concatenate_vectors<T: Sized>(vectors: &mut [Vec<T>]) -> Vec<T> {
    let mut result = Vec::new();
    for vector in vectors.iter_mut() {
        result.append(vector);
    }
    result
}

/// Defines a Bucket utility object. Used to store the properties of shape-partitions
/// in the [`Bvh`] build procedure using SAH.
#[derive(Clone, Copy)]
pub struct Bucket<T: Scalar + Copy, const D: usize> {
    /// The number of shapes in this `Bucket`.
    pub size: usize,

    /// The joint [`Aabb`] of the shapes in this [`Bucket`].
    pub aabb: Aabb<T, D>,
}

impl<T: Scalar + Copy + Float + SimdPartialOrd, const D: usize> Bucket<T, D> {
    /// Returns an empty bucket.
    pub fn empty() -> Bucket<T, D> {
        Bucket {
            size: 0,
            aabb: Aabb::empty(),
        }
    }

    /// Extend this [`Bucket`] by a shape with the given [`Aabb`].
    pub fn add_aabb(&mut self, aabb: &Aabb<T, D>) {
        self.size += 1;
        self.aabb = self.aabb.join(aabb);
    }

    /// Join the contents of two [`Bucket`]'s.
    pub fn join_bucket(a: Bucket<T, D>, b: &Bucket<T, D>) -> Bucket<T, D> {
        Bucket {
            size: a.size + b.size,
            aabb: a.aabb.join(&b.aabb),
        }
    }
}

pub fn joint_aabb_of_shapes<
    T: Scalar + Copy + Float + SimdPartialOrd,
    const D: usize,
    Shape: BHShape<T, D>,
>(
    indices: &[usize],
    shapes: &[Shape],
) -> Aabb<T, D> {
    let mut aabb = Aabb::empty();
    for index in indices {
        let shape = &shapes[*index];
        aabb.join_mut(&shape.aabb());
    }
    aabb
}

#[cfg(test)]
mod tests {
    use crate::utils::concatenate_vectors;

    #[test]
    /// Test if concatenating no [`Vec`]s yields an empty [`Vec`].
    fn test_concatenate_empty() {
        let mut vectors: Vec<Vec<usize>> = vec![];
        let expected = vec![];
        assert_eq!(concatenate_vectors(vectors.as_mut_slice()), expected);
        let expected_remainder: Vec<Vec<usize>> = vec![];
        assert_eq!(vectors, expected_remainder);
    }

    #[test]
    /// Test if concatenating some [`Vec`]s yields the concatenation of the vectors.
    fn test_concatenate_vectors() {
        let mut vectors = vec![vec![1, 2, 3], vec![], vec![4, 5, 6], vec![7, 8], vec![9]];
        let result = concatenate_vectors(vectors.as_mut_slice());
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(result, expected);
        assert_eq!(vectors, vec![vec![], vec![], vec![], vec![], vec![]]);
    }
}
