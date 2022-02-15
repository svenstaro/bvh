pub mod aabb;
pub mod capsule;
pub mod obb;
pub mod ray;
pub mod sphere;
pub mod triangle;

#[cfg(test)]
mod tests {
    use crate::aabb::AABB;
    use crate::bounding_hierarchy::IntersectionAABB;
    use crate::capsule::Capsule;
    use crate::obb::OBB;
    use crate::{Point3, Quat, Real, Vector3, PI};

    #[test]
    fn basic_test_capsule() {
        let min = Point3::new(0.0, 0.0, 0.0);
        let max = Point3::new(1.0, 1.0, 1.0);
        let aabb = AABB::empty().grow(&min).grow(&max);
        let start = Point3::new(3.0, 0.0, 0.0);
        let end = Point3::new(1.5, 0.0, 0.0);
        let capsule = Capsule::new(start, end, 0.55);
        assert!(capsule.intersects_aabb(&aabb));
    }

    #[test]
    fn moving_test_capsule() {
        let min = Point3::new(0.0, 0.0, 0.0);
        let max = Point3::new(1.0, 1.0, 1.0);
        let aabb = AABB::empty().grow(&min).grow(&max);
        let start = Point3::new(0.5, 2.0, 2.0);
        let end = Point3::new(0.5, 5.0, -1.0);
        let capsule = Capsule::new(start, end, 1.45);
        assert!(capsule.intersects_aabb(&aabb));

        let dir = (start - end).normalize();
        let offset: Real = 0.005;
        println!("{}", dir);
        for i in 0..800 {
            println!("{}", i);
            let pt = offset * dir * i as Real;
            let cap = Capsule::new(start + pt, end + pt, 1.45);
            assert!(cap.intersects_aabb(&aabb));
        }
    }

    #[test]
    fn basic_obb() {
        let min = Point3::new(0.0, 0.0, 0.0);
        let max = Point3::new(1.0, 1.0, 1.0);
        let aabb = AABB::empty().grow(&min).grow(&max);

        let ori = Quat::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), PI / 4.);
        let extents = Vector3::new(0.5, 0.5, 0.5);
        let pos = Vector3::new(0.5, 2.2, 0.5);

        let obb = OBB {
            orientation: ori,
            extents,
            center: pos,
        };

        assert!(obb.intersects_aabb(&aabb));
    }
}
