use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::BVH;
use bvh::ray::Ray;
use bvh::{Point3, Vector3};

#[derive(Debug)]
struct Sphere {
    position: Point3,
    radius: f32,
    node_index: usize,
}

impl Bounded for Sphere {
    fn aabb(&self) -> AABB {
        let half_size = Vector3::new(self.radius, self.radius, self.radius);
        let min = self.position - half_size;
        let max = self.position + half_size;
        AABB::with_bounds(min, max)
    }
}

impl BHShape for Sphere {
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
        let position = Point3::new(i as f32, i as f32, i as f32);
        let radius = (i % 10) as f32 + 1.0;
        spheres.push(Sphere {
            position,
            radius,
            node_index: 0,
        });
    }
    let bvh = BVH::build(&mut spheres);

    let origin = Point3::new(0.0, 0.0, 0.0);
    let direction = Vector3::new(1.0, 0.0, 0.0);
    let ray = Ray::new(origin, direction);
    let hit_sphere_aabbs = bvh.traverse(&ray, &spheres);
    dbg!(hit_sphere_aabbs);
}
