use bvh_f64::aabb::{Bounded, AABB};
use bvh_f64::bounding_hierarchy::BHShape;
use bvh_f64::bvh::BVH;
use bvh_f64::ray::Ray;
use bvh_f64::{Point3, Vector3};
use obj::*;
use obj::raw::object::Polygon;
use num::{FromPrimitive, Integer};

#[derive(Debug)]
struct Sphere {
    position: Point3,
    radius: f64,
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

/// A triangle struct. Instance of a more complex `Bounded` primitive.
#[derive(Debug)]
pub struct Triangle {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
    aabb: AABB,
    node_index: usize,
}

impl Triangle {
    pub fn new(a: Point3, b: Point3, c: Point3) -> Triangle {
        Triangle {
            a,
            b,
            c,
            aabb: AABB::empty().grow(&a).grow(&b).grow(&c),
            node_index: 0,
        }
    }
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        self.aabb
    }
}

impl BHShape for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl<I: FromPrimitive + Integer> FromRawVertex<I> for Triangle {
    fn process(
        vertices: Vec<(f32, f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)> {
        // Convert the vertices to `Point3`s.
        let points = vertices
            .into_iter()
            .map(|v| Point3::new(v.0.into(), v.1.into(), v.2.into()))
            .collect::<Vec<_>>();

        // Estimate for the number of triangles, assuming that each polygon is a triangle.
        let mut triangles = Vec::with_capacity(polygons.len());
        {
            let mut push_triangle = |indices: &Vec<usize>| {
                let mut indices_iter = indices.iter();
                let anchor = points[*indices_iter.next().unwrap()];
                let mut second = points[*indices_iter.next().unwrap()];
                for third_index in indices_iter {
                    let third = points[*third_index];
                    triangles.push(Triangle::new(anchor, second, third));
                    second = third;
                }
            };

            // Iterate over the polygons and populate the `Triangle`s vector.
            for polygon in polygons.into_iter() {
                match polygon {
                    Polygon::P(ref vec) => push_triangle(vec),
                    Polygon::PT(ref vec) | Polygon::PN(ref vec) => {
                        push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
                    }
                    Polygon::PTN(ref vec) => {
                        push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
                    }
                }
            }
        }
        Ok((triangles, Vec::new()))
    }
}

pub fn load_sponza_scene() -> (Vec<Triangle>, AABB) {
    use std::fs::File;
    use std::io::BufReader;

    let file_input =
        BufReader::new(File::open("media/sponza.obj").expect("Failed to open .obj file."));
    let sponza_obj: Obj<Triangle> = load_obj(file_input).expect("Failed to decode .obj file data.");
    let triangles = sponza_obj.vertices;

    let mut bounds = AABB::empty();
    for triangle in &triangles {
        bounds.join_mut(&triangle.aabb());
    }

    (triangles, bounds)
}




pub fn main() {
   let (mut triangles, bounds) = load_sponza_scene();
   let mut bvh = BVH::build(triangles.as_mut_slice());

   for i in 0..10 {
        bvh.rebuild(triangles.as_mut_slice());
   }
}
