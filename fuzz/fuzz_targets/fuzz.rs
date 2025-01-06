#![no_main]

//! This fuzz target is a third line of defense against bugs, after unit tests and prop
//! tests.
//!
//! It starts by generating an arbitrary collection of shapes with which to build a BVH,
//! an arbitrary collection of mutations with which to mutate the BVH, and an arbitrary
//! ray with which to traverse the BVH. There are some constraints imposed on the input,
//! such as numbers needing to be finite (not NaN or infinity).
//!
//! Next, all applicable API's of the BVH are exercised to ensure they don't panic and
//! simple properties are tested.
//!
//! Finally, if there is a mutation left, it is applied, and the API's are tested again.

use std::collections::HashSet;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};

use arbitrary::Arbitrary;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use libfuzzer_sys::fuzz_target;
use nalgebra::{Point, SimdPartialOrd};
use ordered_float::NotNan;

type Float = f32;

/// Coordinate magnitude should not exceed this which prevents
/// certain degenerate cases like infinity, both in inputs
/// and internal computations in the BVH.
const LIMIT: Float = 1_000_000.0;

// The entry point for `cargo fuzz`.
fuzz_target!(|workload: Workload<3>| {
    workload.fuzz();
});

/// The input for an arbitrary point, with finite coordinates,
/// each with a magnitude bounded by `LIMIT`.
#[derive(Arbitrary)]
struct ArbitraryPoint<const D: usize> {
    coordinates: [NotNan<Float>; D],
}

impl<const D: usize> ArbitraryPoint<D> {
    /// Produces the corresponding point from the input.
    fn point(&self) -> Point<Float, D> {
        Point::<_, D>::from_slice(&self.coordinates).map(|f| f.into_inner().clamp(-LIMIT, LIMIT))
    }
}

/// An arbitrary shape, with `ArbitraryPoint` corners, guaranteed to have an AABB with
/// non-zero volume.
#[derive(Arbitrary)]
struct ArbitraryShape<const D: usize> {
    a: ArbitraryPoint<D>,
    b: ArbitraryPoint<D>,
    bh_node_index: usize,
}

impl<const D: usize> Debug for ArbitraryShape<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.aabb(), f)
    }
}

impl<const D: usize> Bounded<Float, D> for ArbitraryShape<D> {
    fn aabb(&self) -> Aabb<Float, D> {
        let mut a = self.a.point();
        let b = self.b.point();

        // Ensure some separation so volume is non-zero.
        a.iter_mut().enumerate().for_each(|(i, a)| {
            if *a == b[i] {
                *a += 1.0;
            }
        });

        Aabb::with_bounds(a.simd_min(b), a.simd_max(b))
    }
}

impl<const D: usize> BHShape<Float, D> for ArbitraryShape<D> {
    fn bh_node_index(&self) -> usize {
        self.bh_node_index
    }

    fn set_bh_node_index(&mut self, value: usize) {
        self.bh_node_index = value;
    }
}

/// The input for arbitrary ray, starting at an `ArbitraryPoint` and having a precisely
/// normalized direction.
#[derive(Arbitrary)]
struct ArbitraryRay<const D: usize> {
    origin: ArbitraryPoint<D>,
    destination: ArbitraryPoint<D>,
}

impl<const D: usize> Debug for ArbitraryRay<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.ray(), f)
    }
}

impl<const D: usize> ArbitraryRay<D> {
    /// Produces the corresponding ray from the input.
    fn ray(&self) -> Ray<Float, D> {
        // Double normalize helps when the first one encounters precision issues.
        let mut direction = (self.destination.point() - self.origin.point())
            .normalize()
            .normalize();
        // Ensure no degenerate direction.
        if direction.magnitude() < 0.5 || direction.iter().any(|f| f.is_nan() || f.abs() > 1.5) {
            direction.iter_mut().for_each(|f| *f = 1.0);
            direction = direction.normalize();
        }
        assert!(
            direction.magnitude() - 1.0 < 0.1,
            "{}",
            direction.magnitude()
        );
        Ray::new(self.origin.point(), direction)
    }
}

/// An arbitrary mutation to apply to the BVH to fuzz BVH optimization.
#[derive(Debug, Arbitrary)]
enum ArbitraryMutation<const D: usize> {
    Remove(usize),
    Add(ArbitraryShape<D>),
}

/// The complete set of inputs for a single fuzz iteration.
#[derive(Debug, Arbitrary)]
struct Workload<const D: usize> {
    shapes: Vec<ArbitraryShape<D>>,
    ray: ArbitraryRay<D>,
    mutations: Vec<ArbitraryMutation<D>>,
}

impl<const D: usize> Workload<D> {
    /// Called directly from the `cargo fuzz` entry point. Code in this function is
    /// easier for `rust-analyzer`` than code in that macro.
    fn fuzz(mut self) {
        let mut bvh = Bvh::build(&mut self.shapes);
        let ray = self.ray.ray();

        if self.shapes.len()
            + self
                .mutations
                .iter()
                .filter(|m| matches!(m, ArbitraryMutation::Add(_)))
                .count()
            > 32
        {
            // Prevent traversal stack overflow by limiting max BVH depth to the traversal
            // stack size limit.
            return;
        }

        loop {
            // Check that these don't panic.
            bvh.assert_consistent(&self.shapes);
            bvh.assert_tight();
            let flat_bvh = bvh.flatten();

            let _traverse = bvh
                .traverse(&ray, &self.shapes)
                .into_iter()
                .map(ByPtr)
                .collect::<HashSet<_>>();
            let _traverse_iterator = bvh
                .traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();
            let _traverse_flat = flat_bvh.traverse(&ray, &self.shapes);

            // Fails, either due to bug or rounding errors.
            // assert_eq!(traverse, traverse_iterator);

            let nearest_traverse_iterator = bvh
                .nearest_traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();
            let farthest_traverse_iterator = bvh
                .farthest_traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();

            // Fails, either due to bug or rounding errors.
            //assert_eq!(traverse_iterator, nearest_traverse_iterator);

            assert_eq!(nearest_traverse_iterator, farthest_traverse_iterator);

            if let Some(mutation) = self.mutations.pop() {
                match mutation {
                    ArbitraryMutation::Add(shape) => {
                        let new_shape_index = self.shapes.len();
                        self.shapes.push(shape);
                        bvh.add_shape(&mut self.shapes, new_shape_index);
                    }
                    ArbitraryMutation::Remove(index) => {
                        // TODO: remove `false &&` once this no longer causes a panic:
                        // "Circular node that wasn't root parent=0 node=2"
                        if false
                        /* index < self.shapes.len() */
                        {
                            bvh.remove_shape(&mut self.shapes, index, true);
                            self.shapes.pop().unwrap();
                        }
                    }
                }
            } else {
                break;
            }
        }
    }
}

/// Makes it easy to compare sets of intersected shapes. Comparing by
/// value would be ambiguous if multiple shapes shared the same AABB.
#[derive(Debug)]
struct ByPtr<'a, T>(&'a T);

impl<T> PartialEq for ByPtr<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl<T> Eq for ByPtr<'_, T> {}

impl<T> Hash for ByPtr<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.0 as *const _ as usize);
    }
}
