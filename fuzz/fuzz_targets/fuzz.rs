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
//! Finally, if there are any mutations left, one is applied, and the API's are tested
//! again.

use std::cmp::Ordering;
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
const LIMIT: Float = 5_000.0;

// The entry point for `cargo fuzz`.
fuzz_target!(|workload: Workload<3>| {
    workload.fuzz();
});

/// The input for an arbitrary point, with finite coordinates,
/// each with a magnitude bounded by `LIMIT`.
#[derive(Clone, Arbitrary)]
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
#[derive(Clone, Arbitrary)]
struct ArbitraryShape<const D: usize> {
    a: ArbitraryPoint<D>,
    b: ArbitraryPoint<D>,
    mode: Mode,
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

        let mut aabb = Aabb::with_bounds(a.simd_min(b), a.simd_max(b));

        if self.mode.is_grid() {
            let mut center = aabb.center();
            center.iter_mut().for_each(|f| *f = f.round());
            // Unit AABB around center.
            aabb.min.iter_mut().enumerate().for_each(|(i, f)| {
                *f = center[i] - 0.5;
            });
            aabb.max.iter_mut().enumerate().for_each(|(i, f)| {
                *f = center[i] + 0.5;
            });
        }

        aabb
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
#[derive(Clone, Arbitrary)]
struct ArbitraryRay<const D: usize> {
    origin: ArbitraryPoint<D>,
    destination: ArbitraryPoint<D>,
    mode: Mode,
}

impl<const D: usize> Debug for ArbitraryRay<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.ray(), f)
    }
}

impl<const D: usize> ArbitraryRay<D> {
    /// Produces the corresponding ray from the input.
    fn ray(&self) -> Ray<Float, D> {
        // Note that this eventually gets normalized in `Ray::new`. We don't expect precision issues
        // becaues `LIMIT` limits the magnitude of each point's coordinates, so the length of `direction`
        // is bounded.
        let mut direction = self.destination.point() - self.origin.point();

        // All components are zero or close to zero, resulting in
        // either NaN or a near-zero normalized vector. Replace with a
        // different vector so that `Ray::new` is able to normalize.
        if (direction.normalize().magnitude() - 1.0).partial_cmp(&0.1) != Some(Ordering::Less) {
            direction[0] = 1.0;
        }

        let mut ray = Ray::new(self.origin.point(), direction);

        if self.mode.is_grid() {
            ray.origin.iter_mut().for_each(|f| *f = f.round());

            // Algorithm to find the closest unit-vector parallel to one of the axes. For fuzzing purposes,
            // we just want all 6 unit-vectors parallel to an axis (in the 3D case) to be *possible*.
            //
            // See: https://stackoverflow.com/a/25825980/3064544
            let (axis_number, axis_direction) = ray
                .direction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(i, f)| (i, 1f32.copysign(*f)))
                .unwrap();

            // The resulting ray will be parallel to the axis numbered `axis_number`.
            //
            // `axis_direction` will be -1 or 1, indicating whether the vector points
            // in the negative or positive direction on that axis.
            ray.direction.iter_mut().enumerate().for_each(|(i, f)| {
                *f = if i == axis_number {
                    axis_direction
                } else {
                    0.0
                }
            });
        }

        ray
    }
}

/// An arbitrary mutation to apply to the BVH to fuzz BVH optimization.
#[derive(Debug, Arbitrary)]
enum ArbitraryMutation<const D: usize> {
    Remove(usize),
    Add(ArbitraryShape<D>),
}

#[derive(Copy, Clone, Debug, Arbitrary)]
/// Controls whether the input is modified to help test certain properties.
enum Mode {
    /// AABB's may have mostly arbitrary bounds, and ray may have mostly arbitrary
    /// origin and direction.
    Chaos,
    /// AABB's are unit cubes, and must have integer coordinates. Ray must have an
    /// origin consisting of integer coordinates and a direction that is parallel to
    /// one of the axes.
    ///
    /// In this mode, all types of traversal are expected to yield the same results,
    /// except when bugs exist that have yet to be fixed.
    Grid,
}

impl Mode {
    fn is_grid(self) -> bool {
        matches!(self, Self::Grid)
    }
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
    ///
    /// The contents are explained in the module-level comment up above.
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
            // Under these circumstances, the ray either definitively hits an AABB or it definitively
            // doesn't. The lack of near hits and near misses prevents rounding errors that could cause
            // different traversal algorithms to disagree.
            //
            // This relates to the current state of the BVH. It may change after each mutation is applied
            // e.g. we could add the first non-grid shape or remove the last non-grid shape.
            let assert_traversal_agreement =
                self.ray.mode.is_grid() && self.shapes.iter().all(|s| s.mode.is_grid());

            // Check that these don't panic.
            bvh.assert_consistent(&self.shapes);
            bvh.assert_tight();
            let flat_bvh = bvh.flatten();

            let traverse = bvh
                .traverse(&ray, &self.shapes)
                .into_iter()
                .map(ByPtr)
                .collect::<HashSet<_>>();
            let traverse_iterator = bvh
                .traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();
            let traverse_flat = flat_bvh
                .traverse(&ray, &self.shapes)
                .into_iter()
                .map(ByPtr)
                .collect::<HashSet<_>>();

            if assert_traversal_agreement {
                assert_eq!(traverse, traverse_iterator);
                assert_eq!(traverse, traverse_flat);
            } else {
                // Fails, probably due to normal rounding errors.
            }

            let nearest_traverse_iterator = bvh
                .nearest_traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();
            let farthest_traverse_iterator = bvh
                .farthest_traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();

            if assert_traversal_agreement {
                // TODO: Fails, due to bug(s) e.g. https://github.com/svenstaro/bvh/issues/119
                //assert_eq!(traverse_iterator, nearest_traverse_iterator);
            } else {
                // Fails, probably due to normal rounding errors.
            }

            // Since the algorithm is similar, these should agree regardless of mode.
            assert_eq!(nearest_traverse_iterator, farthest_traverse_iterator);

            if let Some(mutation) = self.mutations.pop() {
                match mutation {
                    ArbitraryMutation::Add(shape) => {
                        let new_shape_index = self.shapes.len();
                        self.shapes.push(shape);
                        bvh.add_shape(&mut self.shapes, new_shape_index);
                    }
                    ArbitraryMutation::Remove(index) => {
                        if index < self.shapes.len() {
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
