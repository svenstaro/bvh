use parry3d_f64::partitioning::{QBVH, QBVHDataGenerator, IndexedData};
use parry3d_f64::bounding_volume::aabb::{AABB};
use parry3d_f64::math::{Real, SimdBool, SimdReal, SIMD_WIDTH};
use parry3d_f64::query::{Ray, RayCast, RayIntersection, SimdRay};
use parry3d_f64::partitioning::{SimdBestFirstVisitStatus, SimdBestFirstVisitor};
use parry3d_f64::bounding_volume::SimdAABB;
use parry3d_f64::simba::simd::{SimdBool as _, SimdPartialOrd, SimdValue};

pub trait BoundableQ {
    fn aabb(&self) -> AABB;
}

/// A visitor for casting a ray on a composite shape.
pub struct RayBoundableToiBestFirstVisitor<'a> {
    ray: &'a Ray,
    simd_ray: SimdRay,
    max_toi: Real,
    solid: bool,
}

impl<'a> RayBoundableToiBestFirstVisitor<'a> {
    /// Initialize a visitor for casting a ray on a composite shape.
    pub fn new(ray: &'a Ray, max_toi: Real, solid: bool) -> Self {
        Self {
            ray,
            simd_ray: SimdRay::splat(*ray),
            max_toi,
            solid,
        }
    }
}

impl<'a, T: BoundableQ + IndexedData> SimdBestFirstVisitor<T, SimdAABB>
    for RayBoundableToiBestFirstVisitor<'a>
{
    type Result = (usize, Real);

    #[inline]
    fn visit(
        &mut self,
        best: Real,
        aabb: &SimdAABB,
        data: Option<[Option<&T>; SIMD_WIDTH]>,
    ) -> SimdBestFirstVisitStatus<Self::Result> {
        let (hit, toi) = aabb.cast_local_ray(&self.simd_ray, SimdReal::splat(self.max_toi));

        if let Some(data) = data {
            let mut weights = [0.0; SIMD_WIDTH];
            let mut mask = [false; SIMD_WIDTH];
            let mut results = [None; SIMD_WIDTH];

            let better_toi = toi.simd_lt(SimdReal::splat(best));
            let bitmask = (hit & better_toi).bitmask();

            for ii in 0..SIMD_WIDTH {
                if (bitmask & (1 << ii)) != 0 && data[ii].is_some() {
                    let shape = data[ii].unwrap();
                    let toi = shape.aabb().cast_local_ray(&self.ray, self.max_toi, self.solid);
                    if let Some(toi) = toi {
                        results[ii] = Some((shape.index(), toi));
                        mask[ii] = true;
                        weights[ii] = toi;
                    }
                }
            }

            SimdBestFirstVisitStatus::MaybeContinue {
                weights: SimdReal::from(weights),
                mask: SimdBool::from(mask),
                results,
            }
        } else {
            SimdBestFirstVisitStatus::MaybeContinue {
                weights: toi,
                mask: hit,
                results: [None; SIMD_WIDTH],
            }
        }
    }
}









//#![cfg(test)]
#[cfg(all(feature = "bench", test))]
mod bench {
    use parry3d_f64::partitioning::{QBVH, QBVHDataGenerator, IndexedData};
    use parry3d_f64::bounding_volume::aabb::AABB;
    use parry3d_f64::math::{Point, Vector};
    use std::rc::{Rc};
    use crate::testbase::{
        create_n_cubes, default_bounds,
        Triangle, create_ray
    };
    use crate::bvh::qbvh::{RayBoundableToiBestFirstVisitor, BoundableQ};
    use parry3d_f64::query::{Ray, RayCast, RayIntersection, SimdRay};
    use parry3d_f64::query::visitors::RayIntersectionsVisitor;

    #[derive(Clone, Debug)]
    pub struct Triangles {
        Tris: Rc<Vec<Triangle>>
    }

    #[derive(Copy, Clone, Debug)]
    pub struct IndexedTri {
        tri: Triangle,
        i: usize
    }

    impl IndexedData for IndexedTri {
        fn default() -> Self {
            IndexedTri {
                tri: Triangle::new(Default::default(), Default::default(), Default::default()),
                i: 0
            }
        }

        fn index(&self) -> usize {
            self.i
        }
    }

    impl BoundableQ for IndexedTri {
        fn aabb(&self) -> AABB {
            let tri = self.tri;
            let mut min = unsafe { Point::new_uninitialized() };
            let mut max = unsafe { Point::new_uninitialized() };
            let a = tri.a;
            let b = tri.b;
            let c = tri.c;
            for d in 0..3 {
                min.coords[d] = a[d].min(b[d]).min(c[d]);
                max.coords[d] = a[d].max(b[d]).max(c[d]);
            }
            AABB::new(min, max)
        }
    }

    impl QBVHDataGenerator<IndexedTri> for Triangles {
        fn size_hint(&self) -> usize {
            return self.Tris.len();
        }

        fn for_each(&mut self, mut f: impl FnMut(IndexedTri, AABB)) {
            for i in 0..self.Tris.len()
            {
                let tri = self.Tris[i];
                let mut new_tri = IndexedTri {
                    tri: tri,
                    i: i
                };
                f(new_tri, new_tri.aabb());
            }
                
        }
    }

    #[bench]
    /// Benchmark building a qbvh
    fn bench_build_1200_qbvh(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(100, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        b.iter(|| {
            bvh.clear_and_rebuild(generator.clone(), 0.0);
        });
    }

    #[bench]
    /// Benchmark building a qbvh
    fn bench_build_12000_qbvh(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(1000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        b.iter(|| {
            bvh.clear_and_rebuild(generator.clone(), 0.0);
        });
    }

    #[bench]
    /// Benchmark building a qbvh
    fn bench_build_120k_qbvh(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        b.iter(|| {
            bvh.clear_and_rebuild(generator.clone(), 0.0);
        });
    }

    #[bench]
    /// Benchmark building a qbvh
    fn bench_1200_qbvh_ray_intersection(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(100, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let r = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(1.0, 0.0, 0.0));
        let mut visitor = RayBoundableToiBestFirstVisitor::new(&r, 1000000000000.0, true);
        b.iter(|| {
            bvh.traverse_best_first(&mut visitor);
        });
    }

    #[bench]
    /// Benchmark building a qbvh
    fn bench_12000_qbvh_ray_intersection(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(1000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let r = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(1.0, 0.0, 0.0));
        let mut visitor = RayBoundableToiBestFirstVisitor::new(&r, 1000000000000.0, true);
        b.iter(|| {
            bvh.traverse_best_first(&mut visitor);
        });
    }

    #[bench]
    /// Benchmark building a qbvh
    fn bench_120k_qbvh_ray_intersection(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let r = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(1.0, 0.0, 0.0));
        let mut visitor = RayBoundableToiBestFirstVisitor::new(&r, 1000000000000.0, true);
        b.iter(|| {
            bvh.traverse_best_first(&mut visitor);
        });
    }

/*
    fn visit_triangle(tri: &IndexedTri) -> bool {
        true
    }
*/
    #[bench]
    /// Benchmark building a qbvh
    fn bench_1200_qbvh_ray_intersection_stack(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(100, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let mut visit_triangle = |tri: &IndexedTri| {
            true
        };
        let mut stack = Vec::new();
        let mut seed = 0;
        b.iter(|| {
            stack.clear();
            let ray = create_ray(&mut seed, &bounds);
            let o = Point::new(ray.origin.x, ray.origin.y, ray.origin.z);
            let d = Vector::new(ray.direction.x, ray.direction.y, ray.direction.z);
            let r = Ray::new(o, d);
            let mut visitor = RayIntersectionsVisitor::new(&r, 1000000000000.0, &mut visit_triangle);
            bvh.traverse_depth_first_with_stack(&mut visitor, &mut stack);
        });
    }

    
    #[bench]
    /// Benchmark building a qbvh
    fn bench_12000_qbvh_ray_intersection_stack(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(1000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let mut visit_triangle = |tri: &IndexedTri| {
            true
        };
        let mut stack = Vec::new();
        let mut seed = 0;
        b.iter(|| {
            stack.clear();
            let ray = create_ray(&mut seed, &bounds);
            let o = Point::new(ray.origin.x, ray.origin.y, ray.origin.z);
            let d = Vector::new(ray.direction.x, ray.direction.y, ray.direction.z);
            let r = Ray::new(o, d);
            let mut visitor = RayIntersectionsVisitor::new(&r, 1000000000000.0, &mut visit_triangle);
            bvh.traverse_depth_first_with_stack(&mut visitor, &mut stack);
        });
    }
    
    #[bench]
    /// Benchmark building a qbvh
    fn bench_120k_qbvh_ray_intersection_stack(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let mut visit_triangle = |tri: &IndexedTri| {
            true
        };
        let mut stack = Vec::new();
        let mut seed = 0;
        b.iter(|| {
            stack.clear();
            let ray = create_ray(&mut seed, &bounds);
            let o = Point::new(ray.origin.x, ray.origin.y, ray.origin.z);
            let d = Vector::new(ray.direction.x, ray.direction.y, ray.direction.z);
            let r = Ray::new(o, d);
            let mut visitor = RayIntersectionsVisitor::new(&r, 1000000000000.0, &mut visit_triangle);
            bvh.traverse_depth_first_with_stack(&mut visitor, &mut stack);
        });
    }

    
    #[bench]
    /// Benchmark building a qbvh
    fn bench_120k_qbvh_ray_intersection_stackalloc(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10000, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        let mut bvh = QBVH::new();
        bvh.clear_and_rebuild(generator.clone(), 0.0);
        let mut visit_triangle = |tri: &IndexedTri| {
            true
        };
        let mut seed = 0;
        b.iter(|| {
            let ray = create_ray(&mut seed, &bounds);
            let o = Point::new(ray.origin.x, ray.origin.y, ray.origin.z);
            let d = Vector::new(ray.direction.x, ray.direction.y, ray.direction.z);
            let r = Ray::new(o, d);
            let mut visitor = RayIntersectionsVisitor::new(&r, 1000000000000.0, &mut visit_triangle);
            bvh.traverse_depth_first(&mut visitor);
        });
    }

}