//! Common utilities shared by unit tests.

use alloc::vec;
use alloc::vec::Vec;
use core::f32;
use hashbrown::HashSet;
use num::{FromPrimitive, Integer};
use obj::raw::object::Polygon;
use obj::*;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::aabb::{Aabb, Bounded, IntersectsAabb};
use crate::ball::Sphere;
use crate::bounding_hierarchy::{BHShape, BoundingHierarchy};
use crate::point_query::PointDistance;

// TODO These all need to be realtyped and bounded

/// A vector represented as a tuple
pub type TupleVec = (f32, f32, f32);

pub type TRay3 = crate::ray::Ray<f32, 3>;
pub type TAabb3 = crate::aabb::Aabb<f32, 3>;
pub type TBvh3 = crate::bvh::Bvh<f32, 3>;
pub type TBvhNode3 = crate::bvh::BvhNode<f32, 3>;
pub type TVector3 = nalgebra::SVector<f32, 3>;
pub type TPoint3 = nalgebra::Point<f32, 3>;
pub type TFlatBvh3 = crate::flat_bvh::FlatBvh<f32, 3>;

/// Generate a [`TupleVec`] for [`proptest::strategy::Strategy`] from -10e10 to 10e10
/// A small enough range to prevent most fp32 errors from breaking certain tests
/// Tests which rely on this strategy should probably be rewritten
pub fn tuplevec_small_strategy() -> impl Strategy<Value = TupleVec> {
    (
        -10e10_f32..10e10_f32,
        -10e10_f32..10e10_f32,
        -10e10_f32..10e10_f32,
    )
}

/// Generate a [`TupleVec`] for [`proptest::strategy::Strategy`] from -10e30 to 10e30
/// A small enough range to prevent `f32::MAX` ranges from breaking certain tests
pub fn tuplevec_large_strategy() -> impl Strategy<Value = TupleVec> {
    (
        -10e30_f32..10e30_f32,
        -10e30_f32..10e30_f32,
        -10e30_f32..10e30_f32,
    )
}

/// Convert a [`TupleVec`] to a [`TPoint3`].
pub fn tuple_to_point(tpl: &TupleVec) -> TPoint3 {
    TPoint3::new(tpl.0, tpl.1, tpl.2)
}

/// Convert a [`TupleVec`] to a [`TVector3`].
pub fn tuple_to_vector(tpl: &TupleVec) -> TVector3 {
    TVector3::new(tpl.0, tpl.1, tpl.2)
}

/// Define some [`Bounded`] structure.
#[derive(PartialEq, Debug)]
pub struct UnitBox {
    pub id: i32,
    pub pos: TPoint3,
    node_index: usize,
}

impl UnitBox {
    pub fn new(id: i32, pos: TPoint3) -> UnitBox {
        UnitBox {
            id,
            pos,
            node_index: 0,
        }
    }
}

/// [`UnitBox`]'s [`Aabb`]'s are unit [`Aabb`]'s centered on the box's position.
impl Bounded<f32, 3> for UnitBox {
    fn aabb(&self) -> TAabb3 {
        let min = self.pos + TVector3::new(-0.5, -0.5, -0.5);
        let max = self.pos + TVector3::new(0.5, 0.5, 0.5);
        TAabb3::with_bounds(min, max)
    }
}

impl BHShape<f32, 3> for UnitBox {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl PointDistance<f32, 3> for UnitBox {
    fn distance_squared(&self, query_point: TPoint3) -> f32 {
        self.aabb().min_distance_squared(query_point)
    }
}

/// Generate 21 [`UnitBox`]'s along the X axis centered on whole numbers (-10,9,..,10).
/// The index is set to the rounded x-coordinate of the box center.
pub fn generate_aligned_boxes() -> Vec<UnitBox> {
    // Create 21 boxes along the x-axis
    let mut shapes = Vec::new();
    for x in -10..11 {
        shapes.push(UnitBox::new(x, TPoint3::new(x as f32, 0.0, 0.0)));
    }
    shapes
}

/// Creates a [`BoundingHierarchy`] for a fixed scene structure.
pub fn build_some_bh<BH: BoundingHierarchy<f32, 3>>() -> (Vec<UnitBox>, BH) {
    let mut boxes = generate_aligned_boxes();
    let bh = BH::build(&mut boxes);
    (boxes, bh)
}

/// Creates a [`BoundingHierarchy`] for a fixed scene structure in parallel.
#[cfg(feature = "rayon")]
pub fn build_some_bh_rayon<BH: BoundingHierarchy<f32, 3>>() -> (Vec<UnitBox>, BH) {
    let mut boxes = generate_aligned_boxes();
    let bh = BH::build_par(&mut boxes);
    (boxes, bh)
}

/// Creates a [`BoundingHierarchy`] for an empty scene structure.
pub fn build_empty_bh<BH: BoundingHierarchy<f32, 3>>() -> (Vec<UnitBox>, BH) {
    let mut boxes = Vec::new();
    let bh = BH::build(&mut boxes);
    (boxes, bh)
}

/// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
/// expected hits, verifies, whether the ray hits only the expected shapes.
fn traverse_and_verify<BH: BoundingHierarchy<f32, 3>>(
    query: &impl IntersectsAabb<f32, 3>,
    all_shapes: &[UnitBox],
    bh: &BH,
    expected_shapes: &HashSet<i32>,
) {
    let hit_shapes = bh.traverse(query, all_shapes);

    assert_eq!(expected_shapes.len(), hit_shapes.len());
    for shape in hit_shapes {
        assert!(
            expected_shapes.contains(&shape.id),
            "unexpected shape {}",
            shape.id
        );
    }
}

/// Perform some fixed intersection tests on [`BoundingHierarchy`] structures.
pub fn traverse_some_bh<BH: BoundingHierarchy<f32, 3>>() {
    let (all_shapes, bh) = build_some_bh::<BH>();
    traverse_some_built_bh(&all_shapes, bh);
}

/// Perform some fixed intersection tests on [`BoundingHierarchy`] structures.
#[cfg(feature = "rayon")]
pub fn traverse_some_bh_rayon<BH: BoundingHierarchy<f32, 3>>() {
    let (all_shapes, bh) = build_some_bh_rayon::<BH>();
    traverse_some_built_bh(&all_shapes, bh);
}

/// Perform some fixed intersection tests on [`BoundingHierarchy`] structures.
fn traverse_some_built_bh<BH: BoundingHierarchy<f32, 3>>(all_shapes: &[UnitBox], bh: BH) {
    {
        // Define a ray which traverses the x-axis from afar.
        let origin = TPoint3::new(-1000.0, 0.0, 0.0);
        let direction = TVector3::new(1.0, 0.0, 0.0);
        let mut expected_shapes = HashSet::new();

        // It should hit everything.
        for id in -10..11 {
            expected_shapes.insert(id);
        }
        traverse_and_verify(
            &TRay3::new(origin, direction),
            all_shapes,
            &bh,
            &expected_shapes,
        );
    }

    {
        // Define a ray which traverses the y-axis from afar.
        let origin = TPoint3::new(0.0, -1000.0, 0.0);
        let direction = TVector3::new(0.0, 1.0, 0.0);

        // It should hit only one box.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(0);
        traverse_and_verify(
            &TRay3::new(origin, direction),
            all_shapes,
            &bh,
            &expected_shapes,
        );
    }

    {
        // Define a ray which intersects the x-axis diagonally.
        let origin = TPoint3::new(6.0, 0.5, 0.0);
        let direction = TVector3::new(-2.0, -1.0, 0.0);

        // It should hit exactly three boxes.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(4);
        expected_shapes.insert(5);
        expected_shapes.insert(6);
        traverse_and_verify(
            &TRay3::new(origin, direction),
            all_shapes,
            &bh,
            &expected_shapes,
        );
    }

    {
        // Define a point at the origin.
        let point = TPoint3::new(0.0, 0.0, 0.0);

        // It should be contained by the middle box.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(0);
        traverse_and_verify(&point, all_shapes, &bh, &expected_shapes);
    }

    {
        // Define a point far away.
        let point = TPoint3::new(0.0, 1000.0, 0.0);

        // It shouldn't be contained by any boxes.
        let expected_shapes = HashSet::new();
        traverse_and_verify(&point, all_shapes, &bh, &expected_shapes);
    }

    {
        // Define an AABB intersecting with some boxes.
        let aabb = Aabb::with_bounds(TPoint3::new(5.1, -1.0, -1.0), TPoint3::new(9.9, 1.0, 1.0));

        let mut expected_shapes = HashSet::new();
        for x in 5..=10 {
            expected_shapes.insert(x);
        }
        traverse_and_verify(&aabb, all_shapes, &bh, &expected_shapes);
    }

    {
        // Define a sphere intersecting with some boxes.
        let sphere = Sphere::new(TPoint3::new(5.0, -1.0, -1.0), 1.4);

        let mut expected_shapes = HashSet::new();
        for x in 4..=6 {
            expected_shapes.insert(x);
        }
        traverse_and_verify(&sphere, all_shapes, &bh, &expected_shapes);
    }
}

/// Perform some fixed distance tests on [`BoundingHierarchy`] structures.
pub fn nearest_to_some_bh<BH: BoundingHierarchy<f32, 3>>() {
    let bounds = TAabb3::with_bounds(
        TPoint3::new(-1000.0, -1000.0, -1000.0),
        TPoint3::new(1000.0, 1000.0, 1000.0),
    );

    let mut triangles = create_n_cubes(1000, &default_bounds());
    let bvh = BH::build(&mut triangles);

    let mut query_points = vec![];
    for _ in 0..100 {
        query_points.push(next_point3(&mut 0, &bounds));
    }
    for point in query_points {
        nearest_to_and_verify(point, &bvh, &triangles);
    }
}

/// Given a query point, a bounding hierarchy, the complete list of shapes in the scene and a list of
/// expected hits, verifies that nearest_to returns the correct answer.
fn nearest_to_and_verify<BH: BoundingHierarchy<f32, 3>>(
    query_point: TPoint3,
    bvh: &BH,
    triangles: &[Triangle],
) {
    let result = bvh.nearest_to(query_point, triangles);

    // Bruteforce the nearest triangle.
    let mut best = (&triangles[0], f32::MAX);
    for triangle in triangles {
        // Check if the AABB distance is less than the current best distance
        // for better performance.
        let aabb_min_dist = triangle.aabb().min_distance_squared(query_point);
        if aabb_min_dist.sqrt() < best.1 {
            // Compute the actual distance.
            let distance = triangle.distance_squared(query_point).sqrt();
            if distance < best.1 {
                best = (triangle, distance);
            }
        }
    }
    assert_eq!(result.unwrap(), best);
}

/// A triangle struct. Instance of a more complex [`Bounded`] primitive.
#[derive(Debug, PartialEq)]
pub struct Triangle {
    pub a: TPoint3,
    pub b: TPoint3,
    pub c: TPoint3,
    aabb: TAabb3,
    node_index: usize,
}

impl Triangle {
    pub fn new(a: TPoint3, b: TPoint3, c: TPoint3) -> Triangle {
        Triangle {
            a,
            b,
            c,
            aabb: TAabb3::empty().grow(&a).grow(&b).grow(&c),
            node_index: 0,
        }
    }
}

impl Bounded<f32, 3> for Triangle {
    fn aabb(&self) -> TAabb3 {
        self.aabb
    }
}

impl BHShape<f32, 3> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Project p on \[ab\].
fn closest_point_segment(p: &TPoint3, a: &TPoint3, b: &TPoint3) -> TPoint3 {
    let ab = b - a;
    let m = ab.dot(&ab);

    // find parameter value of closest point on segment
    let ap = p - a;
    let mut s12 = ab.dot(&ap) / m;
    s12 = s12.clamp(0.0, 1.0);

    a + s12 * ab
}
/// Project a point onto a triangle.
/// Adapted from Embree.
/// <https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h#L10>
fn closest_point_triangle(p: &TPoint3, a: &TPoint3, b: &TPoint3, c: &TPoint3) -> TPoint3 {
    // Add safety checks for degenerate triangles
    #[allow(clippy::match_same_arms)]
    match (a.eq(b), b.eq(c), a.eq(c)) {
        (true, true, true) => {
            return *a;
        }
        (true, _, _) => {
            return closest_point_segment(p, a, c);
        }
        (_, true, _) => {
            return closest_point_segment(p, a, b);
        }
        (_, _, true) => {
            return closest_point_segment(p, a, b);
        }
        // they are all different
        _ => {}
    }

    // Actual embree code.
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let v = d2 / (d2 - d6);
        return a + v * ac;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + v * (c - b);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + v * ab + w * ac
}

impl PointDistance<f32, 3> for Triangle {
    fn distance_squared(&self, query_point: nalgebra::Point<f32, 3>) -> f32 {
        // Compute the unsigned distance from the point to the plane of the triangle
        let nearest = closest_point_triangle(&query_point, &self.a, &self.b, &self.c);
        let diff = query_point - nearest;
        diff.dot(&diff)
    }
}

impl<I: FromPrimitive + Integer> FromRawVertex<I> for Triangle {
    fn process(
        vertices: Vec<(f32, f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)> {
        // Convert the vertices to `Point3`'s.
        let points = vertices
            .into_iter()
            .map(|v| TPoint3::new(v.0, v.1, v.2))
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

/// Creates a unit size cube centered at `pos` and pushes the triangles to `shapes`.
fn push_cube(pos: TPoint3, shapes: &mut Vec<Triangle>) {
    let top_front_right = pos + TVector3::new(0.5, 0.5, -0.5);
    let top_back_right = pos + TVector3::new(0.5, 0.5, 0.5);
    let top_back_left = pos + TVector3::new(-0.5, 0.5, 0.5);
    let top_front_left = pos + TVector3::new(-0.5, 0.5, -0.5);
    let bottom_front_right = pos + TVector3::new(0.5, -0.5, -0.5);
    let bottom_back_right = pos + TVector3::new(0.5, -0.5, 0.5);
    let bottom_back_left = pos + TVector3::new(-0.5, -0.5, 0.5);
    let bottom_front_left = pos + TVector3::new(-0.5, -0.5, -0.5);

    shapes.push(Triangle::new(
        top_back_right,
        top_front_right,
        top_front_left,
    ));
    shapes.push(Triangle::new(top_front_left, top_back_left, top_back_right));
    shapes.push(Triangle::new(
        bottom_front_left,
        bottom_front_right,
        bottom_back_right,
    ));
    shapes.push(Triangle::new(
        bottom_back_right,
        bottom_back_left,
        bottom_front_left,
    ));
    shapes.push(Triangle::new(
        top_back_left,
        top_front_left,
        bottom_front_left,
    ));
    shapes.push(Triangle::new(
        bottom_front_left,
        bottom_back_left,
        top_back_left,
    ));
    shapes.push(Triangle::new(
        bottom_front_right,
        top_front_right,
        top_back_right,
    ));
    shapes.push(Triangle::new(
        top_back_right,
        bottom_back_right,
        bottom_front_right,
    ));
    shapes.push(Triangle::new(
        top_front_left,
        top_front_right,
        bottom_front_right,
    ));
    shapes.push(Triangle::new(
        bottom_front_right,
        bottom_front_left,
        top_front_left,
    ));
    shapes.push(Triangle::new(
        bottom_back_right,
        top_back_right,
        top_back_left,
    ));
    shapes.push(Triangle::new(
        top_back_left,
        bottom_back_left,
        bottom_back_right,
    ));
}

/// Implementation of splitmix64.
/// For reference see: http://xoroshiro.di.unimi.it/splitmix64.c
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15u64);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
    z ^ (z >> 31)
}

/// Generates a new [`i32`] triple. Mutates the seed.
pub fn next_point3_raw(seed: &mut u64) -> (i32, i32, i32) {
    let u = splitmix64(seed);
    let a = ((u >> 32) & 0xFFFFFFFF) as i64 - 0x80000000;
    let b = (u & 0xFFFFFFFF) as i64 - 0x80000000;
    let c = a ^ b.rotate_left(6);
    (a as i32, b as i32, c as i32)
}

/// Generates a new [`Point3`], which will lie inside the given [`Aabb`]. Mutates the seed.
pub fn next_point3(seed: &mut u64, aabb: &TAabb3) -> TPoint3 {
    let (a, b, c) = next_point3_raw(seed);
    let float_vector = TVector3::new(
        (a as f32 / i32::MAX as f32) + 1.0,
        (b as f32 / i32::MAX as f32) + 1.0,
        (c as f32 / i32::MAX as f32) + 1.0,
    ) * 0.5;

    assert!(float_vector.x >= 0.0 && float_vector.x <= 1.0);
    assert!(float_vector.y >= 0.0 && float_vector.y <= 1.0);
    assert!(float_vector.z >= 0.0 && float_vector.z <= 1.0);

    let size = aabb.size();
    let offset = TVector3::new(
        float_vector.x * size.x,
        float_vector.y * size.y,
        float_vector.z * size.z,
    );
    aabb.min + offset
}

/// Returns an [`Aabb`] which defines the default testing space bounds.
pub fn default_bounds() -> TAabb3 {
    TAabb3::with_bounds(
        TPoint3::new(-100_000.0, -100_000.0, -100_000.0),
        TPoint3::new(100_000.0, 100_000.0, 100_000.0),
    )
}

/// Creates `n` deterministic random cubes. Returns the [`Vec`] of surface [`Triangle`]'s.
pub fn create_n_cubes(n: usize, bounds: &TAabb3) -> Vec<Triangle> {
    let mut vec = Vec::new();
    let mut seed = 0;
    for _ in 0..n {
        push_cube(next_point3(&mut seed, bounds), &mut vec);
    }
    vec
}

/// Loads the sponza model.
#[cfg(feature = "bench")]
pub fn load_sponza_scene() -> (Vec<Triangle>, TAabb3) {
    use std::fs::File;
    use std::io::BufReader;

    let file_input =
        BufReader::new(File::open("media/sponza.obj").expect("Failed to open .obj file."));
    let sponza_obj: Obj<Triangle> = load_obj(file_input).expect("Failed to decode .obj file data.");
    let triangles = sponza_obj.vertices;

    let mut bounds = TAabb3::empty();
    for triangle in &triangles {
        bounds.join_mut(&triangle.aabb());
    }

    (triangles, bounds)
}

/// This functions moves `amount` shapes in the `triangles` array to a new position inside
/// `bounds`. If `max_offset_option` is not `None` then the wrapped value is used as the maximum
/// offset of a shape. This is used to simulate a realistic scene.
/// Returns a [`HashSet`] of indices of modified triangles.
pub fn randomly_transform_scene(
    triangles: &mut [Triangle],
    amount: usize,
    bounds: &TAabb3,
    max_offset_option: Option<f32>,
    seed: &mut u64,
) -> HashSet<usize> {
    let mut indices: Vec<usize> = (0..triangles.len()).collect();
    let mut seed_array = [0u8; 32];
    for i in 0..seed_array.len() {
        let bytes: [u8; 8] = seed.to_be_bytes();
        seed_array[i] = bytes[i % 8];
    }
    let mut rng: StdRng = SeedableRng::from_seed(seed_array);
    indices.shuffle(&mut rng);
    indices.truncate(amount);

    let max_offset = if let Some(value) = max_offset_option {
        value
    } else {
        f32::INFINITY
    };

    for index in &indices {
        let aabb = triangles[*index].aabb();
        let min_move_bound = bounds.min - aabb.min;
        let max_move_bound = bounds.max - aabb.max;
        let movement_bounds = TAabb3::with_bounds(min_move_bound.into(), max_move_bound.into());

        let mut random_offset = next_point3(seed, &movement_bounds);
        random_offset.x = max_offset.min((-max_offset).max(random_offset.x));
        random_offset.y = max_offset.min((-max_offset).max(random_offset.y));
        random_offset.z = max_offset.min((-max_offset).max(random_offset.z));

        let triangle = &mut triangles[*index];
        let old_index = triangle.bh_node_index();
        *triangle = Triangle::new(
            (triangle.a.coords + random_offset.coords).into(),
            (triangle.b.coords + random_offset.coords).into(),
            (triangle.c.coords + random_offset.coords).into(),
        );
        triangle.set_bh_node_index(old_index);
    }

    indices.into_iter().collect()
}

/// Creates a [`Ray`] from the random `seed`. Mutates the `seed`.
/// The [`Ray`] origin will be inside the `bounds` and point to some other point inside this
/// `bounds`.
#[cfg(feature = "bench")]
pub fn create_ray(seed: &mut u64, bounds: &TAabb3) -> TRay3 {
    let origin = next_point3(seed, bounds);
    let direction = next_point3(seed, bounds);
    TRay3::new(origin, direction.coords)
}

/// Benchmark the construction of a [`BoundingHierarchy`] with `n` triangles.
#[cfg(feature = "bench")]
fn build_n_triangles_bh<T: BoundingHierarchy<f32, 3>>(n: usize, b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    b.iter(|| {
        T::build(&mut triangles);
    });
}

/// Benchmark the construction of a [`BoundingHierarchy`] with 1,200 triangles.
#[cfg(feature = "bench")]
pub fn build_1200_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(100, b);
}

/// Benchmark the construction of a [`BoundingHierarchy`] with 12,000 triangles.
#[cfg(feature = "bench")]
pub fn build_12k_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(1_000, b);
}

/// Benchmark the construction of a [`BoundingHierarchy`] with 120,000 triangles.
#[cfg(feature = "bench")]
pub fn build_120k_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(10_000, b);
}

#[cfg(all(feature = "bench", feature = "rayon"))]
fn build_n_triangles_bh_rayon<T: BoundingHierarchy<f32, 3>>(n: usize, b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    b.iter(|| {
        T::build_par(&mut triangles);
    });
}

/// Benchmark the construction of a [`BoundingHierarchy`] with 1,200 triangles.
#[cfg(all(feature = "bench", feature = "rayon"))]
pub fn build_1200_triangles_bh_rayon<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    build_n_triangles_bh_rayon::<T>(100, b);
}

/// Benchmark the construction of a [`BoundingHierarchy`] with 12,000 triangles.
#[cfg(all(feature = "bench", feature = "rayon"))]
pub fn build_12k_triangles_bh_rayon<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    build_n_triangles_bh_rayon::<T>(1_000, b);
}

/// Benchmark the construction of a [`BoundingHierarchy`] with 120,000 triangles.
#[cfg(all(feature = "bench", feature = "rayon"))]
pub fn build_120k_triangles_bh_rayon<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    build_n_triangles_bh_rayon::<T>(10_000, b);
}

/// Benchmark intersecting the `triangles` list without acceleration structures.
#[cfg(feature = "bench")]
pub fn intersect_list(triangles: &[Triangle], bounds: &TAabb3, b: &mut ::test::Bencher) {
    let mut seed = 0;
    b.iter(|| {
        let ray = create_ray(&mut seed, bounds);

        // Iterate over the list of triangles.
        for triangle in triangles {
            std::hint::black_box(ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c));
        }
    });
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting 120,000 triangles directly.
fn bench_intersect_120k_triangles_list(b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    intersect_list(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting Sponza.
fn bench_intersect_sponza_list(b: &mut ::test::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    intersect_list(&triangles, &bounds, b);
}

/// Benchmark intersecting the `triangles` list with [`Aabb`] checks, but without acceleration
/// structures.
#[cfg(feature = "bench")]
pub fn intersect_list_aabb(triangles: &[Triangle], bounds: &TAabb3, b: &mut ::test::Bencher) {
    let mut seed = 0;
    b.iter(|| {
        let ray = create_ray(&mut seed, bounds);

        // Iterate over the list of triangles.
        for triangle in triangles {
            // First test whether the ray intersects the `Aabb` of the triangle.
            if ray.intersects_aabb(&triangle.aabb()) {
                std::hint::black_box(ray.intersects_triangle(
                    &triangle.a,
                    &triangle.b,
                    &triangle.c,
                ));
            }
        }
    });
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting 120,000 triangles with preceeding [`Aabb`] tests.
fn bench_intersect_120k_triangles_list_aabb(b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    intersect_list_aabb(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting 120,000 triangles with preceeding [`Aabb`] tests.
fn bench_intersect_sponza_list_aabb(b: &mut ::test::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    intersect_list_aabb(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
pub fn intersect_bh<T: BoundingHierarchy<f32, 3>>(
    bh: &T,
    triangles: &[Triangle],
    bounds: &TAabb3,
    b: &mut ::test::Bencher,
) {
    let mut seed = 0;
    b.iter(|| {
        let ray = create_ray(&mut seed, bounds);

        // Traverse the [`BoundingHierarchy`] recursively.
        let hits = bh.traverse(&ray, triangles);

        // Traverse the resulting list of positive `Aabb` tests
        for triangle in &hits {
            ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
        }
    });
}

/// Benchmark the traversal of a [`BoundingHierarchy`] with `n` triangles.
#[cfg(feature = "bench")]
pub fn intersect_n_triangles<T: BoundingHierarchy<f32, 3>>(n: usize, b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    let bh = T::build(&mut triangles);
    intersect_bh(&bh, &triangles, &bounds, b)
}

/// Benchmark the traversal of a [`BoundingHierarchy`] with 1,200 triangles.
#[cfg(feature = "bench")]
pub fn intersect_1200_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(100, b);
}

/// Benchmark the traversal of a [`BoundingHierarchy`] with 12,000 triangles.
#[cfg(feature = "bench")]
pub fn intersect_12k_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(1_000, b);
}

/// Benchmark the traversal of a [`BoundingHierarchy`] with 120,000 triangles.
#[cfg(feature = "bench")]
pub fn intersect_120k_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(10_000, b);
}

/// Benchmark nearest_to on a `triangles` list without acceleration structures.
#[cfg(feature = "bench")]
pub fn nearest_to_list(triangles: &[Triangle], bounds: &TAabb3, b: &mut ::test::Bencher) {
    let mut seed = 0;
    b.iter(|| {
        let point = next_point3(&mut seed, bounds);

        let mut min_dist = f32::MAX;
        // Iterate over the list of triangles.
        for triangle in triangles {
            let dist = std::hint::black_box(triangle.distance_squared(point));
            if dist < min_dist {
                min_dist = dist;
            }
        }

        std::hint::black_box(min_dist)
    });
}

/// Benchmark nearest_to on a `triangles` list with [`Aabb`] checks, but without acceleration
/// structures.
#[cfg(feature = "bench")]
pub fn nearest_to_list_aabb(triangles: &[Triangle], bounds: &TAabb3, b: &mut ::test::Bencher) {
    let mut seed = 0;
    b.iter(|| {
        let point = next_point3(&mut seed, bounds);

        // Note: all distances here are squared.
        let mut min_dist = f32::MAX;
        // Iterate over the list of triangles.
        for triangle in triangles {
            // First test whether the point-aabb distance is within bounds.
            let aabb_min_dist = std::hint::black_box(triangle.aabb().min_distance_squared(point));
            if aabb_min_dist < min_dist {
                let dist = std::hint::black_box(triangle.distance_squared(point));
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        std::hint::black_box(min_dist)
    });
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark nearest_to on 120,000 triangles directly.
fn bench_nearest_to_120k_triangles_list(b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    nearest_to_list(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark nearest_to on Sponza.
fn bench_nearest_to_sponza_list(b: &mut ::test::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    nearest_to_list(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark nearest_to on 120,000 triangles with preceeding [`Aabb`] tests.
fn bench_nearest_to_120k_triangles_list_aabb(b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    nearest_to_list_aabb(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark nearest_to on 120,000 triangles with preceeding [`Aabb`] tests.
fn bench_nearest_to_sponza_list_aabb(b: &mut ::test::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    nearest_to_list_aabb(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
pub fn nearest_to_bh<T: BoundingHierarchy<f32, 3>>(
    bh: &T,
    triangles: &[Triangle],
    bounds: &TAabb3,
    b: &mut ::test::Bencher,
) {
    let mut seed = 0;
    b.iter(|| {
        let point = next_point3(&mut seed, bounds);

        // Traverse the [`BoundingHierarchy`] recursively.
        let _best_result = std::hint::black_box(
            bh.nearest_to(std::hint::black_box(point), std::hint::black_box(triangles)),
        );
    });
}

/// Benchmark the nearest_to traversal of a [`BoundingHierarchy`] with `n` triangles.
#[cfg(feature = "bench")]
pub fn nearest_to_n_triangles<T: BoundingHierarchy<f32, 3>>(n: usize, b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    let bh = T::build(&mut triangles);
    nearest_to_bh(&bh, &triangles, &bounds, b)
}

/// Benchmark the nearest_to traversal of a [`BoundingHierarchy`] with 1,200 triangles.
#[cfg(feature = "bench")]
pub fn nearest_to_1200_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    nearest_to_n_triangles::<T>(100, b);
}

/// Benchmark the nearest_to traversal of a [`BoundingHierarchy`] with 12,000 triangles.
#[cfg(feature = "bench")]
pub fn nearest_to_12k_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    nearest_to_n_triangles::<T>(1_000, b);
}

/// Benchmark the nearest_to traversal of a [`BoundingHierarchy`] with 120,000 triangles.
#[cfg(feature = "bench")]
pub fn nearest_to_120k_triangles_bh<T: BoundingHierarchy<f32, 3>>(b: &mut ::test::Bencher) {
    nearest_to_n_triangles::<T>(10_000, b);
}
