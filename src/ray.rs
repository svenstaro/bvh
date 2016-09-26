use aabb::AABB;
use nalgebra::{Vector3, Point3, Norm};

/// A struct which defines a ray and some of its cached values.
#[derive(Debug)]
pub struct Ray {
    /// The ray origin.
    pub origin: Point3<f32>,

    /// The ray direction.
    pub direction: Vector3<f32>,

    /// Inverse (1/x) ray direction. Cached for use in [`AABB`] intersections.
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    inv_direction: Vector3<f32>,

    /// Sign of the direction. 0 means positive, 1 means negative.
    /// Cached for use in [`AABB`] intersections.
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    sign: Vector3<usize>,
}

impl Ray {
    /// Creates a new [`Ray`] from an `origin` and a `direction`.
    /// `direction` will be normalized.
    ///
    /// # Examples
    /// ```
    /// use bvh::ray::Ray;
    /// use bvh::nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// assert_eq!(ray.origin, origin);
    /// assert_eq!(ray.direction, direction);
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    ///
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

    /// Tests the intersection of a [`Ray`] with an [`AABB`] using the optimized algorithm
    /// from [this paper](http://www.cs.utah.edu/~awilliam/box/box.pdf).
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::ray::Ray;
    /// use bvh::nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// let point1 = Point3::new(99.9,-1.0,-1.0);
    /// let point2 = Point3::new(100.1,1.0,1.0);
    /// let aabb = AABB::with_bounds(point1, point2);
    ///
    /// assert!(ray.intersects_aabb(&aabb));
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    /// [`AABB`]: struct.AABB.html
    ///
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
        // Using the following solution significantly decreases the performance
        // ray_min = ray_min.max(y_min);

        if y_max < ray_max {
            ray_max = y_max;
        }
        // Using the following solution significantly decreases the performance
        // ray_max = ray_max.min(y_max);

        let z_min = (aabb[self.sign.z].z - self.origin.z) * self.inv_direction.z;
        let z_max = (aabb[1 - self.sign.z].z - self.origin.z) * self.inv_direction.z;

        if (ray_min > z_max) || (z_min > ray_max) {
            return false;
        }

        // Only required for bounded intersection intervals.
        // if z_min > ray_min {
        // ray_min = z_min;
        // }

        if z_max < ray_max {
            ray_max = z_max;
        }
        // Using the following solution significantly decreases the performance
        // ray_max = ray_max.min(y_max);

        ray_max > 0.0
    }

    /// Naive implementation of a [`Ray`]/[`AABB`] intersection algorithm.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::ray::Ray;
    /// use bvh::nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// let point1 = Point3::new(99.9,-1.0,-1.0);
    /// let point2 = Point3::new(100.1,1.0,1.0);
    /// let aabb = AABB::with_bounds(point1, point2);
    ///
    /// assert!(ray.intersects_aabb_naive(&aabb));
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn intersects_aabb_naive(&self, aabb: &AABB) -> bool {
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

    /// Implementation of the algorithm described [here]
    /// (https://tavianator.com/fast-branchless-raybounding-box-intersections/).
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::ray::Ray;
    /// use bvh::nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// let point1 = Point3::new(99.9,-1.0,-1.0);
    /// let point2 = Point3::new(100.1,1.0,1.0);
    /// let aabb = AABB::with_bounds(point1, point2);
    ///
    /// assert!(ray.intersects_aabb_branchless(&aabb));
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn intersects_aabb_branchless(&self, aabb: &AABB) -> bool {
        let tx1 = (aabb.min.x - self.origin.x) * self.inv_direction.x;
        let tx2 = (aabb.max.x - self.origin.x) * self.inv_direction.x;

        let mut tmin = tx1.min(tx2);
        let mut tmax = tx1.max(tx2);

        let ty1 = (aabb.min.y - self.origin.y) * self.inv_direction.y;
        let ty2 = (aabb.max.y - self.origin.y) * self.inv_direction.y;

        tmin = tmin.max(ty1.min(ty2));
        tmax = tmax.min(ty1.max(ty2));

        let tz1 = (aabb.min.z - self.origin.z) * self.inv_direction.z;
        let tz2 = (aabb.max.z - self.origin.z) * self.inv_direction.z;

        tmin = tmin.max(tz1.min(tz2));
        tmax = tmax.min(tz1.max(tz2));

        tmax >= tmin && tmax >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use ray::Ray;
    use aabb::AABB;
    use nalgebra::{Point3, Vector3};
    use rand::{Rng, StdRng, SeedableRng};

    type TupleVec = (f32, f32, f32);

    fn tuple_to_point(tpl: &TupleVec) -> Point3<f32> {
        Point3::new(tpl.0, tpl.1, tpl.2)
    }

    fn tuple_to_vector(tpl: &TupleVec) -> Vector3<f32> {
        Vector3::new(tpl.0, tpl.1, tpl.2)
    }

    /// Generates a random `Ray` which points at at a random `AABB`.
    fn gen_ray_to_aabb(data: (TupleVec, TupleVec, TupleVec)) -> (Ray, AABB) {
        // Generate a random AABB
        let aabb = AABB::empty()
            .grow(&tuple_to_point(&data.0))
            .grow(&tuple_to_point(&data.1));

        // Get its center
        let center = aabb.center();

        // Generate random ray pointing at the center
        let pos = tuple_to_point(&data.2);
        let ray = Ray::new(pos, center - pos);
        (ray, aabb)
    }

    /// Test whether a `Ray` which points at the center of an `AABB` intersects it.
    /// Uses the optimized algorithm.
    quickcheck!{
        fn test_ray_points_at_aabb_center(data: (TupleVec, TupleVec, TupleVec)) -> bool {
            let (ray, aabb) = gen_ray_to_aabb(data);
            ray.intersects_aabb(&aabb)
        }
    }

    /// Test whether a `Ray` which points at the center of an `AABB` intersects it.
    /// Uses the naive algorithm.
    quickcheck!{
        fn test_ray_points_at_aabb_center_naive(data: (TupleVec, TupleVec, TupleVec)) -> bool {
            let (ray, aabb) = gen_ray_to_aabb(data);
            ray.intersects_aabb_naive(&aabb)
        }
    }

    /// Test whether a `Ray` which points at the center of an `AABB` intersects it.
    /// Uses the branchless algorithm.
    quickcheck!{
        fn test_ray_points_at_aabb_center_branchless(data: (TupleVec, TupleVec, TupleVec)) -> bool {
            let (ray, aabb) = gen_ray_to_aabb(data);
            ray.intersects_aabb_branchless(&aabb)
        }
    }

    /// Test whether a `Ray` which points away from the center of an `AABB`
    /// does not intersect it, unless its origin is inside the `AABB`.
    /// Uses the optimized algorithm.
    quickcheck!{
        fn test_ray_points_from_aabb_center(data: (TupleVec, TupleVec, TupleVec)) -> bool {
            let (mut ray, aabb) = gen_ray_to_aabb(data);

            // Invert the direction of the ray
            ray.direction = -ray.direction;
            ray.inv_direction = -ray.inv_direction;
            !ray.intersects_aabb(&aabb) || aabb.contains(&ray.origin)
        }
    }

    /// Test whether a `Ray` which points away from the center of an `AABB`
    /// does not intersect it, unless its origin is inside the `AABB`.
    /// Uses the naive algorithm.
    quickcheck!{
        fn test_ray_points_from_aabb_center_naive(data: (TupleVec, TupleVec, TupleVec)) -> bool {
            let (mut ray, aabb) = gen_ray_to_aabb(data);

             // Invert the ray direction
            ray.direction = -ray.direction;
            ray.inv_direction = -ray.inv_direction;
            !ray.intersects_aabb_naive(&aabb) || aabb.contains(&ray.origin)
        }
    }

    /// Test whether a `Ray` which points away from the center of an `AABB`
    /// does not intersect it, unless its origin is inside the `AABB`.
    /// Uses the branchless algorithm.
    quickcheck!{
        fn test_ray_points_from_aabb_center_branchless(data: (TupleVec, TupleVec, TupleVec))
                                                       -> bool {
            let (mut ray, aabb) = gen_ray_to_aabb(data);
            // Invert the ray direction
            ray.direction = -ray.direction;
            ray.inv_direction = -ray.inv_direction;
            !ray.intersects_aabb_branchless(&aabb) || aabb.contains(&ray.origin)
        }
    }

    /// Generates some random deterministic `Ray`/`AABB` pairs.
    fn gen_random_ray_aabb(rng: &mut StdRng) -> (Ray, AABB) {
        let a = tuple_to_point(&rng.gen::<TupleVec>());
        let b = tuple_to_point(&rng.gen::<TupleVec>());
        let c = tuple_to_point(&rng.gen::<TupleVec>());
        let d = tuple_to_vector(&rng.gen::<TupleVec>());

        let aabb = AABB::empty().grow(&a).grow(&b);
        let ray = Ray::new(c, d);
        (ray, aabb)
    }

    #[bench]
    /// Benchmark for the optimized intersection algorithm.
    fn bench_intersects_aabb(b: &mut ::test::Bencher) {
        let seed = [0];
        let mut rng = StdRng::from_seed(&seed);

        b.iter(|| {
            let one_thousand = ::test::black_box(1000);
            for _ in 0..one_thousand {
                let (ray, aabb) = gen_random_ray_aabb(&mut rng);
                ray.intersects_aabb(&aabb);
            }
        });
    }

    #[bench]
    /// Benchmark for the naive intersection algorithm.
    fn bench_intersects_aabb_naive(b: &mut ::test::Bencher) {
        let seed = [0];
        let mut rng = StdRng::from_seed(&seed);

        b.iter(|| {
            let one_thousand = ::test::black_box(1000);
            for _ in 0..one_thousand {
                let (ray, aabb) = gen_random_ray_aabb(&mut rng);
                ray.intersects_aabb_naive(&aabb);
            }
        });
    }

    #[bench]
    /// Benchmark for the branchless intersection algorithm.
    fn bench_intersects_aabb_branchless(b: &mut ::test::Bencher) {
        let seed = [0];
        let mut rng = StdRng::from_seed(&seed);

        b.iter(|| {
            let one_thousand = ::test::black_box(1000);
            for _ in 0..one_thousand {
                let (ray, aabb) = gen_random_ray_aabb(&mut rng);
                ray.intersects_aabb_branchless(&aabb);
            }
        });
    }
}
