//! Utilities module.

use crate::aabb::AABB;
use crate::bounding_hierarchy::BHShape;
use crate::{Point3, Real, Vector3};

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
/// in the BVH build procedure using SAH.
#[derive(Copy, Clone)]
pub struct Bucket {
    /// The number of shapes in this `Bucket`.
    pub size: usize,

    /// The joint `AABB` of the shapes in this `Bucket`.
    pub aabb: AABB,

    /// the `AABB` of the centroid of the centers of the shapes in this `Bucket`
    pub centroid: AABB,
}

impl Bucket {
    /// Returns an empty bucket.
    pub fn empty() -> Bucket {
        Bucket {
            size: 0,
            aabb: AABB::empty(),
            centroid: AABB::empty(),
        }
    }

    /// Extend this `Bucket` by a shape with the given `AABB`.
    pub fn add_aabb(&mut self, aabb: &AABB) {
        self.size += 1;
        self.aabb.join_mut(aabb);
        self.centroid.grow_mut(&aabb.center());
    }

    /// Join the contents of two `Bucket`s.
    pub fn join_bucket(a: Bucket, b: &Bucket) -> Bucket {
        Bucket {
            size: a.size + b.size,
            aabb: a.aabb.join(&b.aabb),
            centroid: a.centroid.join(&b.centroid),
        }
    }
}

pub fn joint_aabb_of_shapes<Shape: BHShape>(indices: &[usize], shapes: &[Shape]) -> (AABB, AABB) {
    let mut aabb = AABB::empty();
    let mut centroid = AABB::empty();
    for index in indices {
        let shape = &shapes[*index];
        aabb.join_mut(&shape.aabb());
        centroid.grow_mut(&shape.aabb().center());
    }
    (aabb, centroid)
}

/// Helper function that given a line segment and a target point, finds the closest point on the line segment to target
pub fn nearest_point_on_line(start: &Point3, dir: &Vector3, len: Real, target: &Point3) -> Point3 {
    let v = *target - *start;
    let d = v.dot(*dir);
    *start + (*dir * d.clamp(0.0, len))
}

#[cfg(test)]
mod tests {
    use crate::utils::concatenate_vectors;

    #[test]
    /// Test if concatenating no `Vec`s yields an empty `Vec`.
    fn test_concatenate_empty() {
        let mut vectors: Vec<Vec<usize>> = vec![];
        let expected: Vec<usize> = vec![];
        assert_eq!(concatenate_vectors(vectors.as_mut_slice()), expected);
        let expected_remainder: Vec<Vec<usize>> = vec![];
        assert_eq!(vectors, expected_remainder);
    }

    #[test]
    /// Test if concatenating some `Vec`s yields the concatenation of the vectors.
    fn test_concatenate_vectors() {
        let mut vectors = vec![vec![1, 2, 3], vec![], vec![4, 5, 6], vec![7, 8], vec![9]];
        let result = concatenate_vectors(vectors.as_mut_slice());
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(result, expected);
        let expected_vecs: Vec<Vec<i32>> = vec![vec![], vec![], vec![], vec![], vec![]];
        assert_eq!(vectors, expected_vecs);
    }
}
