//#![cfg(test)]
#[cfg(all(feature = "bench", test))]
mod bench {
    use parry3d_f64::partitioning::{QBVH, QBVHDataGenerator, IndexedData};
    use parry3d_f64::bounding_volume::aabb::AABB;
    use parry3d_f64::math::Point;
    use std::rc::{Rc};
    use crate::testbase::{
        create_n_cubes, default_bounds,
        Triangle,
    };

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

    impl QBVHDataGenerator<IndexedTri> for Triangles {
        fn size_hint(&self) -> usize {
            return self.Tris.len();
        }

        fn for_each(&mut self, mut f: impl FnMut(IndexedTri, AABB)) {
            for i in 0..self.Tris.len()
            {
                let tri = self.Tris[i];
                let mut min = unsafe { Point::new_uninitialized() };
                let mut max = unsafe { Point::new_uninitialized() };
                let a = tri.a;
                let b = tri.b;
                let c = tri.c;
                for d in 0..3 {
                    min.coords[d] = a[d].min(b[d]).min(c[d]);
                    max.coords[d] = a[d].max(b[d]).max(c[d]);
                }
                let mut aabb = AABB::new(min, max);
                let mut new_tri = IndexedTri {
                    tri: tri,
                    i: i
                };
                f(new_tri, aabb);
            }
                
        }
    }

    //fn test() {
    #[bench]
    /// Benchmark building a qbvh
    fn bench_build_1200_qbvh(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(100, &bounds);
        let rc = Rc::new(triangles);
        let mut generator = Triangles {
            Tris: rc
        };


        b.iter(|| {
            let mut bvh = QBVH::new();
            bvh.clear_and_rebuild(generator.clone(), 1.0);
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


        b.iter(|| {
            let mut bvh = QBVH::new();
            bvh.clear_and_rebuild(generator.clone(), 1.0);
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
        bvh.clear_and_rebuild(generator.clone(), 1.0);
        b.iter(|| {
            bvh.clear_and_rebuild(generator.clone(), 1.0);
        });
    }


}