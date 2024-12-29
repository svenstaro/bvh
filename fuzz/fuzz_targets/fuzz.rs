#![no_main]
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
const LIMIT: Float = 1_000_000.0;

fuzz_target!(|workload: Workload<3>| {
    workload.fuzz();
});

#[derive(Arbitrary)]
struct ArbitraryPoint<const D: usize> {
    coordinates: [NotNan<Float>; D],
}

impl<const D: usize> ArbitraryPoint<D> {
    fn point(&self) -> Point<Float, D> {
        Point::<_, D>::from_slice(&self.coordinates).map(|f| f.into_inner().clamp(-LIMIT, LIMIT))
    }
}

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

        // Ensure some separation.
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

#[derive(Debug, Arbitrary)]
enum ArbitraryMutation<const D: usize> {
    Remove(usize),
    Add(ArbitraryShape<D>),
}

#[derive(Debug, Arbitrary)]
struct Workload<const D: usize> {
    shapes: Vec<ArbitraryShape<D>>,
    ray: ArbitraryRay<D>,
    mutations: Vec<ArbitraryMutation<D>>,
}

impl<const D: usize> Workload<D> {
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
            // Prevent traversal stack overflow.
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

            // Remove condition once https://github.com/svenstaro/bvh/pull/112 merges.
            if !self.shapes.is_empty() {
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
            }

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
                        if false && index < self.shapes.len() {
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

#[derive(Debug)]
struct ByPtr<'a, T>(&'a T);

impl<'a, T> PartialEq for ByPtr<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl<'a, T> Eq for ByPtr<'a, T> {}

impl<'a, T> Hash for ByPtr<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.0 as *const _ as usize);
    }
}
