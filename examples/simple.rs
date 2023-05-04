use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use nalgebra::{Point, SVector};

#[derive(Debug)]
struct Sphere {
    position: Point<f32, 3>,
    radius: f32,
    node_index: usize,
}

impl Bounded<f32, 3> for Sphere {
    fn aabb(&self) -> Aabb<f32, 3> {
        let half_size = SVector::<f32, 3>::new(self.radius, self.radius, self.radius);
        let min = self.position - half_size;
        let max = self.position + half_size;
        Aabb::with_bounds(min, max)
    }
}

impl BHShape<f32, 3> for Sphere {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub fn main() {
    let mut spheres = Vec::new();
    for i in 0..1000000u32 {
        let position = Point::<f32, 3>::new(i as f32, i as f32, i as f32);
        let radius = (i % 10) as f32 + 1.0;
        spheres.push(Sphere {
            position,
            radius,
            node_index: 0,
        });
    }
    let bvh = Bvh::build(&mut spheres);

    let origin = Point::<f32, 3>::new(0.0, 0.0, 0.0);
    let direction = SVector::<f32, 3>::new(1.0, 0.0, 0.0);
    let ray = Ray::new(origin, direction);
    let hit_sphere_aabbs = bvh.traverse(&ray, &spheres);
    dbg!(hit_sphere_aabbs);
}
