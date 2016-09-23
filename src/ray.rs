use aabb::AABB;
use nalgebra::{Vector3, Point3, Norm};

/// A struct which defines a ray and some of its cached values.
#[derive(Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    inv_direction: Vector3<f32>,
    sign: Vector3<usize>,
}

impl Ray {
    /// Creates a new `Ray`.
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Ray {
        let direction = direction.normalize();
        Ray {
            origin: origin,
            direction: direction,
            inv_direction: Vector3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z),
            sign: Vector3::new((direction.x < 0.0) as usize,
                               (direction.y < 0.0) as usize,
                               (direction.z < 0.0) as usize),
        }
    }

    /// Tests the intersection of a `Ray` with an `AABB` using the optimized algorithm
    /// from this paper: http://www.cs.utah.edu/~awilliam/box/box.pdf
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let mut ray_min = (aabb[self.sign.x].x - self.origin.x) * self.inv_direction.x;
        let mut ray_max = (aabb[1 - self.sign.x].x - self.origin.x) * self.inv_direction.x;

        let y_min = (aabb[self.sign.y].y - self.origin.y) * self.inv_direction.y;
        let y_max = (aabb[1 - self.sign.y].y - self.origin.y) * self.inv_direction.y;

        if (ray_min > y_max) || (y_min > ray_max) {
            return false;
        }

        if y_min > ray_min {
            ray_min = y_min;
        }
        if y_max < ray_max {
            ray_max = y_max;
        }

        let z_min = (aabb[self.sign.z].z - self.origin.z) * self.inv_direction.z;
        let z_max = (aabb[1 - self.sign.z].z - self.origin.z) * self.inv_direction.z;

        if (ray_min > z_max) || (z_min > ray_max) {
            return false;
        }
        // if z_min > ray_min {
        // ray_min = z_min;
        // }
        if z_max < ray_max {
            ray_max = z_max;
        }

        ray_max > 0.0
    }

    /// Naive implementation of a `Ray`/`AABB` intersection algorithm.
    pub fn intersects_aabb_2(&self, aabb: &AABB) -> bool {
        let hit_min_x = (aabb.min.x - self.origin.x) * self.inv_direction.x;
        let hit_max_x = (aabb.max.x - self.origin.x) * self.inv_direction.x;

        let hit_min_y = (aabb.min.y - self.origin.y) * self.inv_direction.y;
        let hit_max_y = (aabb.max.y - self.origin.y) * self.inv_direction.y;

        let hit_min_z = (aabb.min.z - self.origin.z) * self.inv_direction.z;
        let hit_max_z = (aabb.max.z - self.origin.z) * self.inv_direction.z;

        let x_entry = hit_min_x.min(hit_max_x);
        let y_entry = hit_min_y.min(hit_max_y);
        let z_entry = hit_min_z.min(hit_max_z);
        let x_exit = hit_min_x.max(hit_max_x);
        let y_exit = hit_min_y.max(hit_max_y);
        let z_exit = hit_min_z.max(hit_max_z);

        let latest_entry = x_entry.max(y_entry).max(z_entry);
        let earliest_exit = x_exit.min(y_exit).min(z_exit);

        latest_entry < earliest_exit && earliest_exit > 0.0
    }
}

#[cfg(test)]
mod tests {
    use ray::Ray;
    use aabb::AABB;
    use nalgebra::{Point3, Vector3};
    use rand::{Rng, StdRng, SeedableRng};

    const EPSILON: f32 = 0.00001;

    type TupleVec = (f32, f32, f32);

    fn tuple_to_point(tpl: &TupleVec) -> Point3<f32> {
        Point3::new(tpl.0, tpl.1, tpl.2)
    }

    fn tuple_to_vector(tpl: &TupleVec) -> Vector3<f32> {
        Vector3::new(tpl.0, tpl.1, tpl.2)
    }

    /// Test whether a `Ray` which points at the center of an `AABB` intersects it.
    quickcheck!{
        fn test_ray_points_at_aabb_center(pos: TupleVec,
                                          point_a: TupleVec,
                                          point_b: TupleVec)
                                          -> bool {
            // Generate a random AABB
            let aabb = AABB::empty()
                .union_point(&tuple_to_point(&point_a))
                .union_point(&tuple_to_point(&point_b));

            // Get its center
            let center = aabb.center();

            // Generate random ray pointing at the center
            let pos = tuple_to_point(&pos);
            let ray = Ray::new(pos, center - pos);

            // It must always intersect the AABB
            ray.intersects_aabb(&aabb)
        }
    }

    /// Test whether a `Ray` which points at the center of an `AABB` intersects it.
    /// Uses the naive algorithm.
    quickcheck!{
        fn test_ray_points_at_aabb_center_2(pos: TupleVec,
                                            point_a: TupleVec,
                                            point_b: TupleVec)
                                            -> bool {
            // Generate a random AABB
            let aabb = AABB::empty()
                .union_point(&tuple_to_point(&point_a))
                .union_point(&tuple_to_point(&point_b));

            // Get its center
            let center = aabb.center();

            // Generate random ray pointing at the center
            let pos = tuple_to_point(&pos);
            let ray = Ray::new(pos, center - pos);

            // It must always intersect the AABB
            ray.intersects_aabb_2(&aabb)
        }
    }

    /// Test whether a `Ray` which points away from the center of an `AABB`
    /// does not intersect it, unless its origin is inside the `AABB`.
    quickcheck!{
        fn test_ray_points_from_aabb_center(pos: TupleVec,
                                            point_a: TupleVec,
                                            point_b: TupleVec)
                                            -> bool {
            let aabb = AABB::empty()
                .union_point(&tuple_to_point(&point_a))
                .union_point(&tuple_to_point(&point_b));
            let center = aabb.center();
            let pos = tuple_to_point(&pos);
            let ray = Ray::new(pos, -(center - pos));
            !ray.intersects_aabb(&aabb) || aabb.contains(&ray.origin)
        }
    }

    /// Test whether a `Ray` which points away from the center of an `AABB`
    /// does not intersect it, unless its origin is inside the `AABB`.
    /// Uses the naive algorithm.
    quickcheck!{
        fn test_ray_points_from_aabb_center_2(pos: TupleVec,
                                              point_a: TupleVec,
                                              point_b: TupleVec)
                                              -> bool {
            let aabb = AABB::empty()
                .union_point(&tuple_to_point(&point_a))
                .union_point(&tuple_to_point(&point_b));
            let center = aabb.center();
            let pos = tuple_to_point(&pos);
            let ray = Ray::new(pos, -(center - pos));
            !ray.intersects_aabb_2(&aabb) || aabb.contains(&ray.origin)
        }
    }

    #[bench]
    /// Benchmark for the optimized intersection algorithm.
    fn bench_intersects_aabb(b: &mut ::test::Bencher) {
        let seed = [0];
        let mut rng = StdRng::from_seed(&seed);

        b.iter(|| {
            let one_million = ::test::black_box(1000);
            for _ in 0..one_million {
                let a = tuple_to_point(&rng.gen::<TupleVec>());
                let b = tuple_to_point(&rng.gen::<TupleVec>());
                let c = tuple_to_point(&rng.gen::<TupleVec>());
                let d = tuple_to_vector(&rng.gen::<TupleVec>());

                let aabb = AABB::empty().union_point(&a).union_point(&b);
                let ray = Ray::new(c, d);
                ray.intersects_aabb(&aabb);
            }
        });
    }

    #[bench]
    /// Benchmark for the naive intersection algorithm.
    fn bench_intersects_aabb_2(b: &mut ::test::Bencher) {
        let seed = [0];
        let mut rng = StdRng::from_seed(&seed);

        b.iter(|| {
            let one_million = ::test::black_box(1000);
            for _ in 0..one_million {
                let a = tuple_to_point(&rng.gen::<TupleVec>());
                let b = tuple_to_point(&rng.gen::<TupleVec>());
                let c = tuple_to_point(&rng.gen::<TupleVec>());
                let d = tuple_to_vector(&rng.gen::<TupleVec>());

                let aabb = AABB::empty().union_point(&a).union_point(&b);
                let ray = Ray::new(c, d);
                ray.intersects_aabb_2(&aabb);
            }
        });
    }
}
