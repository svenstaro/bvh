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
use std::env;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use arbitrary::Arbitrary;
use bvh::aabb::{Aabb, Bounded, IntersectsAabb};
use bvh::ball::Ball;
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use bvh::flat_bvh::FlatBvh;
use bvh::point_query::PointDistance;
use bvh::ray::Ray;
use libfuzzer_sys::fuzz_target;
use nalgebra::{Point, SimdPartialOrd};
use ordered_float::NotNan;

type Float = f32;

/// Coordinate magnitude should not exceed this which prevents
/// certain degenerate cases like infinity, both in inputs
/// and internal computations in the BVH. For `Mode::Grid`,
/// offsets of 1/3 should be representable.
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
    mode: Mode,
}

impl<const D: usize> Debug for ArbitraryPoint<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.point(), f)
    }
}

impl<const D: usize> ArbitraryPoint<D> {
    /// Produces the corresponding point from the input.
    fn point(&self) -> Point<Float, D> {
        Point::<_, D>::from_slice(&self.coordinates).map(|f| {
            // Float may be large or infinite, but is guaranteed to not be NaN due
            // to the `NotNan` wrapper.
            //
            // Clamp it to a smaller range so that offsets of 1/3 are easily representable,
            // which helps `Mode::Grid`.
            let ret = f.into_inner().clamp(-LIMIT, LIMIT);

            if self.mode.is_grid() {
                // Round each coordinate to an integer, as per `Mode::Grid` docs.
                ret.round()
            } else {
                ret
            }
        })
    }
}

/// An arbitrary shape, with `ArbitraryPoint` corners, guaranteed to have an AABB with
/// non-zero volume.
#[derive(Clone, Arbitrary)]
struct ArbitraryShape<const D: usize> {
    a: ArbitraryPoint<D>,
    b: ArbitraryPoint<D>,
    /// This will end up being mutated, but initializing it arbitrarily could catch bugs.
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

        if self.mode_is_grid() {
            let min = aabb.min;
            aabb.max.iter_mut().enumerate().for_each(|(i, f)| {
                // Coordinate should already be an integer, because `max` is in grid mode.
                //
                // Use `max` to ensure the AABB has volume, and add a margin described by `Mode::Grid`.
                *f = f.max(min[i]) + 1.0 / 3.0;
            });
            aabb.min.iter_mut().for_each(|f| {
                // Coordinate should already be an integer, because `min` is in grid mode.
                //
                // Add a margin described by `Mode::Grid`.
                *f -= 1.0 / 3.0;
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

impl<const D: usize> ArbitraryShape<D> {
    fn mode_is_grid(&self) -> bool {
        self.a.mode.is_grid() && self.b.mode.is_grid()
    }
}

impl<const D: usize> PointDistance<Float, D> for ArbitraryShape<D> {
    fn distance_squared(&self, query_point: Point<Float, D>) -> Float {
        let center = self.aabb().center();
        let offset = query_point - center;
        offset.dot(&offset)
    }
}

/// The input for arbitrary ray, starting at an `ArbitraryPoint` and having a precisely
/// normalized direction.
#[derive(Clone, Arbitrary)]
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
    fn mode_is_grid(&self) -> bool {
        self.origin.mode.is_grid() && self.destination.mode.is_grid()
    }

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

        if self.mode_is_grid() {
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

/// The input for arbitrary ray, starting at an `ArbitraryPoint` and having a precisely
/// normalized direction.
#[derive(Clone, Arbitrary)]
struct ArbitraryBall<const D: usize> {
    center: ArbitraryPoint<D>,
    radius: NotNan<f32>,
}

impl<const D: usize> Debug for ArbitraryBall<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.ball(), f)
    }
}

impl<const D: usize> ArbitraryBall<D> {
    fn ball(&self) -> Ball<Float, D> {
        Ball {
            center: self.center.point(),
            radius: self.radius.into_inner().max(0.0),
        }
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
    /// AABB's bound integer coordinates with a margin of 1/3. Twice the margin, 2/3,
    /// is less than 1, so AABB's can either deeply intersect or will have a gap at
    /// least 1/3 wide. Ray must have an origin consisting of integer coordinates and
    /// a direction that is parallel to one of the axes. Point must have integer
    /// coordinates.
    ///
    /// In this mode, all types of ray, AABB, and point traversal are expected to
    /// yield the same results, except when bugs exist that have yet to be fixed.
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
    /// Traverse by ray.
    ray: ArbitraryRay<D>,
    /// Traverse by point and for nearest_to query.
    point: ArbitraryPoint<D>,
    /// Traverse by AABB.
    aabb: ArbitraryShape<D>,
    /// Traverse by ball.
    ball: ArbitraryBall<D>,
    mutations: Vec<ArbitraryMutation<D>>,
}

impl<const D: usize> Workload<D> {
    /// Compares normal, iterative, and flat-BVH traversal of the same query.
    ///
    /// Returns the result of normal traversal.
    ///
    /// # Panics
    /// If `assert_agreement` is true, panics if the results differ in an
    /// unexpected way.
    fn fuzz_traversal<'a>(
        &'a self,
        bvh: &'a Bvh<Float, D>,
        flat_bvh: &'a FlatBvh<Float, D>,
        query: &impl IntersectsAabb<Float, D>,
        assert_agreement: bool,
    ) -> HashSet<ByPtr<'a, ArbitraryShape<D>>> {
        let traverse = bvh
            .traverse(query, &self.shapes)
            .into_iter()
            .map(ByPtr)
            .collect::<HashSet<_>>();
        let traverse_iterator = bvh
            .traverse_iterator(query, &self.shapes)
            .map(ByPtr)
            .collect::<HashSet<_>>();
        let traverse_flat = flat_bvh
            .traverse(query, &self.shapes)
            .into_iter()
            .map(ByPtr)
            .collect::<HashSet<_>>();

        if assert_agreement {
            assert_eq!(traverse, traverse_iterator);
            assert_eq!(traverse, traverse_flat);
        } else {
            // Fails, probably due to normal rounding errors.
        }

        traverse
    }

    /// Compares normal and flat-BVH nearest_to query.
    ///
    /// # Panics
    /// Panics if the results differ in an unexpected way.
    fn fuzz_nearest_to<'a>(
        &'a self,
        bvh: &'a Bvh<Float, D>,
        flat_bvh: &'a FlatBvh<Float, D>,
        query_point: Point<Float, D>,
    ) {
        let best = bvh.nearest_to(query_point, &self.shapes);
        let flat_best = flat_bvh.nearest_to(query_point, &self.shapes);

        // Assert we get None if and only if the bvh is empty.
        assert_eq!(best.is_none(), self.shapes.is_empty());
        assert_eq!(flat_best.is_none(), self.shapes.is_empty());

        // Asserting equality between the two results return many false positives
        // where two differents shapes give the same distance due to numerical approximations
        // e.g.
        // left: Some((ByPtr((Aabb { min: [-5000.0, 0.0, 0.0], max: [5000.0, 1.0, 1.0] }, [5000.0, 0.0, 0.0])), 5000.0))
        // right: Some((ByPtr((Aabb { min: [5.418294e-39, -5000.0, 9.1834e-41], max: [5.418294e-39, -5000.0, 9.1834e-41] }, [-5000.0, 9.1834e-41, 0.0])), 5000.0))
        // assert_eq!(best, flat_best);
        // as such we only assert for distance equivalence if they are not None
        if let Some((best, flat_best)) = best.zip(flat_best) {
            const EPSILON: f32 = 0.001;

            approx::assert_abs_diff_eq!(best.1, flat_best.1, epsilon = EPSILON);

            // Brute force search for the ground truth, shapes is not empty since the results are not None.
            let mut bruteforce = (
                &self.shapes[0],
                self.shapes[0].distance_squared(query_point),
            );
            for shape in &self.shapes[1..] {
                let distance = shape.distance_squared(query_point);
                if distance < bruteforce.1 {
                    bruteforce = (shape, distance);
                }
            }
            bruteforce = (bruteforce.0, bruteforce.1.sqrt());
            // Again we only test for distances.
            approx::assert_abs_diff_eq!(best.1, bruteforce.1, epsilon = EPSILON);
            approx::assert_abs_diff_eq!(flat_best.1, bruteforce.1, epsilon = EPSILON);
        }
    }

    /// Called directly from the `cargo fuzz` entry point. Code in this function is
    /// easier for `rust-analyzer`` than code in that macro.
    ///
    /// The contents are explained in the module-level comment up above.
    fn fuzz(mut self) {
        let ray = self.ray.ray();
        let aabb = self.aabb.aabb();

        // Make sure these functions agree and that the latter doesn't return NaN/infinity.
        let intersects = ray.intersects_aabb(&aabb);
        if let Some((entry, exit)) = ray.intersection_slice_for_aabb(&aabb) {
            assert!(
                intersects,
                "intersection_slice_for_aabb returned Some for non-intersection"
            );
            assert!(entry >= 0.0, "entry ({entry}) must be non-negative");
            assert!(
                entry <= exit,
                "entry ({entry}) must not exceed exit ({exit})"
            );
            assert!(
                entry.is_finite() && exit.is_finite(),
                "neither entry ({entry}) nor exit ({exit}) may be infinite or NaN for AABB's and rays with finite coordinates"
            );
        } else {
            assert!(
                !intersects,
                "intersection_slice_for_aabb returned None for intersection"
            );
        }

        // If this environment variable is present, only test limited-size BVH's without mutations.
        //
        // This way, problems detected with more complex BVH's might be able to be minimized
        // to simple BVH's.
        //
        // This is accessible via commands in the project's `justfile`.
        if let Some(max_shapes) = env::var("MAX_SHAPES")
            .ok()
            .map(|s| usize::from_str(&s).expect("MAX_SHAPES"))
        {
            #[allow(clippy::collapsible_if)]
            if !self.mutations.is_empty() || self.shapes.len() > max_shapes {
                return;
            }
        }

        let mut bvh = Bvh::build(&mut self.shapes);

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
            // `self.shapes` are all in grid mode.
            let all_shapes_grid = self.shapes.iter().all(|s| s.mode_is_grid());

            // Under these circumstances, the ray either definitively hits an AABB or it definitively
            // doesn't. The lack of near hits and near misses prevents rounding errors that could cause
            // different traversal algorithms to disagree.
            //
            // This relates to the current state of the BVH. It may change after each mutation is applied
            // e.g. we could add the first non-grid shape or remove the last non-grid shape.
            let assert_ray_traversal_agreement = self.ray.mode_is_grid() && all_shapes_grid;

            // Under these circumstances, the `self.aabb` either definitively intersects with an AABB or
            // it definitively doesn't.
            //
            // Similar meaning to `assert_ray_traversal_agreement`.
            let assert_aabb_traversal_agreement = self.aabb.mode_is_grid() && all_shapes_grid;

            // Under these circumstances, the `self.point` is either definitively contained by an AABB or
            // definitively not contained.
            //
            // Similar meaning to `assert_ray_traversal_agreement`.
            let assert_point_traversal_agreement = self.point.mode.is_grid() && all_shapes_grid;

            // Check that these don't panic.
            bvh.assert_consistent(&self.shapes);
            bvh.assert_tight();
            let flat_bvh = bvh.flatten();

            let traverse_ray =
                self.fuzz_traversal(&bvh, &flat_bvh, &ray, assert_ray_traversal_agreement);
            self.fuzz_traversal(&bvh, &flat_bvh, &aabb, assert_aabb_traversal_agreement);
            self.fuzz_traversal(
                &bvh,
                &flat_bvh,
                &self.point.point(),
                assert_point_traversal_agreement,
            );
            // Due to sphere geometry, `Mode::Grid` doesn't imply traversals will agree.
            self.fuzz_traversal(&bvh, &flat_bvh, &self.ball.ball(), false);

            // In addition to collecting the output into a `HashSet`, make sure each subsequent
            // item is further (or equally close) with respect to the ray, thus testing the
            // correctness of the iterator.
            let mut last_distance = f32::NEG_INFINITY;
            let nearest_traverse_iterator = bvh
                .nearest_traverse_iterator(&ray, &self.shapes)
                .inspect(|shape| {
                    // Use `.0` to get the entry distance, which the iterator is based on.
                    let distance = ray.intersection_slice_for_aabb(&shape.aabb())
                        .expect("nearest_traverse_iterator returned aabb for which no intersection slice exists")
                        .0;
                    // Until https://github.com/svenstaro/bvh/issues/140 is completely fixed, the `distance`
                    // and the relationship between `distance` and `last_distance` only necessarily holds if
                    // ray-AABB intersection is definitive, so don't panic if `!assert_ray_traversal_agreement`.
                    assert!(
                        !assert_ray_traversal_agreement || distance >= last_distance,
                        "nearest_traverse_iterator returned shapes out of order: {distance} {last_distance}"
                    );
                    last_distance = distance;
                })
                .map(ByPtr)
                .collect::<HashSet<_>>();

            // Same as the above, but in the opposite direction.
            let mut last_distance = f32::INFINITY;
            let farthest_traverse_iterator = bvh
                .farthest_traverse_iterator(&ray, &self.shapes)
                .inspect(|shape| {
                    // Use `.1` to get the exit distance, which the iterator is based on.
                    let distance = ray.intersection_slice_for_aabb(&shape.aabb())
                        .expect("farthest_traverse_iterator returned aabb for which no intersection slice exists")
                        .1;
                    assert!(
                        !assert_ray_traversal_agreement || distance <= last_distance,
                        "farthest_traverse_iterator returned shapes out of order: {distance} {last_distance}"
                    );
                    last_distance = distance;
                })
                .map(ByPtr)
                .collect::<HashSet<_>>();

            // We do not assert the order, because we didn't check if any children overlap, the
            // condition under which order isn't guaranteed. TODO: check this.
            let nearest_child_traverse_iterator = bvh
                .nearest_child_traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();

            // Same as the above, but in the opposite direction.
            let farthest_child_traverse_iterator = bvh
                .farthest_child_traverse_iterator(&ray, &self.shapes)
                .map(ByPtr)
                .collect::<HashSet<_>>();

            if assert_ray_traversal_agreement {
                assert_eq!(traverse_ray, nearest_traverse_iterator);
                assert_eq!(traverse_ray, nearest_child_traverse_iterator);
            } else {
                // Fails, probably due to normal rounding errors.
            }

            // Since the algorithm is similar, these should agree regardless of mode.
            assert_eq!(nearest_traverse_iterator, farthest_traverse_iterator);
            assert_eq!(
                nearest_child_traverse_iterator,
                farthest_child_traverse_iterator
            );

            // Check that these don't panic and return the same value.
            self.fuzz_nearest_to(&bvh, &flat_bvh, self.point.point());

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
