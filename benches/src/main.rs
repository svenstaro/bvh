use std::{hint::black_box, time::Instant};

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::Bvh,
    ray::Ray,
};
use clap::Parser;
use nalgebra::{Point3, Vector3};
use rand::{rng, Rng};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    rays: usize,
    #[arg(long)]
    triangles: usize,
    #[arg(long)]
    samples: usize,
}

fn main() {
    let cli = Cli::parse();
    let mut rng = rng();

    let mut samples = Vec::new();
    let mut rays = Vec::new();
    let mut triangles = Vec::new();

    for i in 0..cli.samples {
        rays.clear();
        triangles.clear();

        for _ in 0..cli.rays {
            rays.push(Ray::<f32, 3>::new(
                Point3::new(
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                ),
                Vector3::new(
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                ),
            ));
        }

        for _ in 0..cli.triangles {
            let center = Point3::new(
                rng.random_range(-1000.0..=1000.0),
                rng.random_range(-1000.0..=1000.0),
                rng.random_range(-1000.0..=1000.0),
            );
            let a = center
                + Vector3::new(
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                );
            let b = center
                + Vector3::new(
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                );
            let c = center
                + Vector3::new(
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                    rng.random_range(-1.0..=1.0),
                );
            triangles.push(Triangle {
                a,
                b,
                c,
                aabb: Aabb::empty().grow(&a).grow(&b).grow(&c),
                bh_node_index: usize::MAX,
            });
        }

        let mut brute_force_duration = f64::NAN;
        let mut bvh_duration = f64::NAN;

        let mut measure_brute_force = |triangles: &[Triangle]| {
            let start_brute_force = Instant::now();
            for ray in &rays {
                black_box(
                    black_box(&triangles)
                        .iter()
                        .filter(|triangle| triangle.intersect(&ray))
                        .count(),
                );
            }
            brute_force_duration = start_brute_force.elapsed().as_secs_f64();
        };

        let mut measure_bvh = |triangles: &mut Vec<Triangle>| {
            let start_bvh = Instant::now();
            let bvh = Bvh::build(black_box(triangles));
            for ray in &rays {
                black_box(
                    bvh.traverse_iterator(black_box(ray), black_box(&triangles))
                        .filter(|triangle| triangle.intersect(&ray))
                        .count(),
                );
            }
            bvh_duration = start_bvh.elapsed().as_secs_f64();
        };

        // Flip order to minimize bias due to caching.
        if i % 2 == 0 {
            measure_bvh(&mut triangles);
            measure_brute_force(&triangles);
        } else {
            measure_brute_force(&triangles);
            measure_bvh(&mut triangles);
        }

        let bvh_speedup = brute_force_duration / bvh_duration;
        samples.push(bvh_speedup);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Median.
    println!("{}", samples[cli.samples / 2]);
}

struct Triangle {
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
    aabb: Aabb<f32, 3>,
    bh_node_index: usize,
}

impl Triangle {
    fn intersect(&self, ray: &Ray<f32, 3>) -> bool {
        ray.intersects_triangle(&self.a, &self.b, &self.c)
            .distance
            .is_finite()
    }
}

impl Bounded<f32, 3> for Triangle {
    fn aabb(&self) -> Aabb<f32, 3> {
        self.aabb
    }
}

impl BHShape<f32, 3> for Triangle {
    fn bh_node_index(&self) -> usize {
        self.bh_node_index
    }

    fn set_bh_node_index(&mut self, val: usize) {
        self.bh_node_index = val;
    }
}
