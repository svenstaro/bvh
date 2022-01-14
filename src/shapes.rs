use crate::aabb::AABB;
use crate::bounding_hierarchy::IntersectionAABB;
use crate::{Mat4, Point3, Quat, Real, Vector3};

pub struct Sphere {
    center: Point3,
    radius: Real,
}

impl Sphere {
    pub fn new(center: Point3, radius: Real) -> Sphere {
        Sphere { center, radius }
    }
}

impl IntersectionAABB for Sphere {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let vec = aabb.closest_point(self.center);
        vec.distance_squared(self.center) < self.radius * self.radius
    }
}

pub struct Capsule {
    start: Point3,
    radius: Real,

    dir: Vector3,
    len: Real,
}

impl Capsule {
    pub fn new(start: Point3, end: Point3, radius: Real) -> Capsule {
        let line = end - start;
        let dir = line.normalize();
        let len = line.length();

        Capsule {
            start,
            radius,
            dir,
            len,
        }
    }
}

impl IntersectionAABB for Capsule {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        /*
        // Use Distance from closest point
        let mut point: Vector3 = self.start;
        let mut curr_d = 0.0;
        let max_sq = (self.len + self.radius) * (self.len + self.radius) ;
        let r_sq = self.radius * self.radius;

        loop {
            let x = aabb.closest_point(point);
            let d_sq = x.distance_squared(point);
            println!("{:?} closest={:?} d={:?}", point, x, d_sq.sqrt());
            if d_sq <= r_sq
            {
                return true;
            }
            if d_sq > max_sq || curr_d >= self.len
            {
                return false;
            }
            curr_d = (curr_d + d_sq.sqrt()).min(self.len);
            point = self.start + (curr_d * self.dir);
        }
        */
        let mut last = self.start;
        loop {
            let closest = &aabb.closest_point(last);
            let center = nearest_point_on_line(&self.start, &self.dir, self.len, closest);
            let sphere = Sphere {
                center,
                radius: self.radius,
            };
            if sphere.intersects_aabb(aabb) {
                return true;
            }
            if last.distance_squared(center) < 0.0001 {
                return false;
            } else {
                last = center;
            }
        }
    }
}

pub struct OBB {
    pub orientation: Quat,
    pub extents: Vector3,
    pub center: Vector3,
}

impl IntersectionAABB for OBB {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let half_a = self.extents;
        let half_b = (aabb.max - aabb.min) * 0.5;
        let value = (aabb.max + aabb.min) * 0.5;
        let translation = self.orientation * (value - self.center);
        let mat = Mat4::from_rotation_translation(self.orientation, translation.into());

        let vec_1 = Vector3::new(
            translation.x.abs(),
            translation.y.abs(),
            translation.z.abs(),
        );
        let right = right(mat);
        let up = up(mat);
        let backward = back(mat);
        let vec_2 = right * half_b.x;
        let vec_3 = up * half_b.y;
        let vec_4 = backward * half_b.z;
        let num = vec_2.x.abs() + vec_3.x.abs() + vec_4.x.abs();
        let num2 = vec_2.y.abs() + vec_3.y.abs() + vec_4.y.abs();
        let num3 = vec_2.z.abs() + vec_3.z.abs() + vec_4.z.abs();
        if vec_1.x + num <= half_a.x && vec_1.y + num2 <= half_a.y && vec_1.z + num3 <= half_a.z {
            // Contained
            return true;
        }
        if vec_1.x > half_a.x + vec_2.x.abs() + vec_3.x.abs() + vec_4.x.abs() {
            return false;
        }
        if vec_1.y > half_a.y + vec_2.y.abs() + vec_3.y.abs() + vec_4.y.abs() {
            return false;
        }
        if vec_1.z > half_a.z + vec_2.z.abs() + vec_3.z.abs() + vec_4.z.abs() {
            return false;
        }
        if translation.dot(right.abs())
            > half_a.x * right.x.abs()
                + half_a.y * right.y.abs()
                + half_a.z * right.z.abs()
                + half_b.x
        {
            return false;
        }
        if translation.dot(up.abs())
            > half_a.x * up.x.abs() + half_a.y * up.y.abs() + half_a.z * up.z.abs() + half_b.y
        {
            return false;
        }
        if translation.dot(backward.abs())
            > half_a.x * backward.x.abs()
                + half_a.y * backward.y.abs()
                + half_a.z * backward.z.abs()
                + half_b.z
        {
            return false;
        }
        let mut vec_5 = Vector3::new(0.0, -right.z, right.y);
        if translation.dot(vec_5.abs())
            > half_a.y * vec_5.y.abs()
                + half_a.z * vec_5.z.abs()
                + vec_5.dot(vec_3.abs())
                + vec_5.dot(vec_4.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(0.0, -up.z, up.y);
        if translation.dot(vec_5.abs())
            > half_a.y * vec_5.y.abs()
                + half_a.z * vec_5.z.abs()
                + vec_5.dot(vec_4.abs())
                + vec_5.dot(vec_2.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(0.0, -backward.z, backward.y);
        if translation.dot(vec_5.abs())
            > half_a.y * vec_5.y.abs()
                + half_a.z * vec_5.z.abs()
                + vec_5.dot(vec_2.abs())
                + vec_5.dot(vec_3.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(right.z, 0.0, -right.x);
        if translation.dot(vec_5.abs())
            > half_a.z * vec_5.z.abs()
                + half_a.x * vec_5.x.abs()
                + vec_5.dot(vec_3.abs())
                + vec_5.dot(vec_4.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(up.z, 0.0, -up.x);
        if translation.dot(vec_5.abs())
            > half_a.z * vec_5.z.abs()
                + half_a.x * vec_5.x.abs()
                + vec_5.dot(vec_4.abs())
                + vec_5.dot(vec_2.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(backward.z, 0.0, -backward.x);
        if translation.dot(vec_5.abs())
            > half_a.z * vec_5.z.abs()
                + half_a.x * vec_5.x.abs()
                + vec_5.dot(vec_2.abs())
                + vec_5.dot(vec_3.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(-right.y, right.x, 0.0);
        if translation.dot(vec_5.abs())
            > half_a.x * vec_5.x.abs()
                + half_a.y * vec_5.y.abs()
                + vec_5.dot(vec_3.abs())
                + vec_5.dot(vec_4.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(-up.y, up.x, 0.0);
        if translation.dot(vec_5.abs())
            > half_a.x * vec_5.x.abs()
                + half_a.y * vec_5.y.abs()
                + vec_5.dot(vec_4.abs())
                + vec_5.dot(vec_2.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(-backward.y, backward.x, 0.0);
        if translation.dot(vec_5.abs())
            > half_a.x * vec_5.x.abs()
                + half_a.y * vec_5.y.abs()
                + vec_5.dot(vec_2.abs())
                + vec_5.dot(vec_3.abs())
        {
            return false;
        }
        // Intersection
        return true;
    }
}

fn right(matrix: Mat4) -> Vector3 {
    matrix.row(0).truncate().into()
}

fn up(matrix: Mat4) -> Vector3 {
    matrix.row(1).truncate().into()
}

fn back(matrix: Mat4) -> Vector3 {
    matrix.row(2).truncate().into()
}

fn translation(matrix: Mat4) -> Vector3 {
    matrix.row(3).truncate().into()
}

pub fn nearest_point_on_line(p1: &Point3, dir: &Vector3, len: Real, pnt: &Point3) -> Point3 {
    let v = *pnt - *p1;
    let d = v.dot(*dir);
    *p1 + (*dir * d.clamp(0.0, len))
}

#[cfg(test)]
mod tests {
    use crate::aabb::AABB;
    use crate::bounding_hierarchy::IntersectionAABB;
    use crate::shapes::{Capsule, OBB};
    use crate::{Point3, Quat, Real, Vector3};

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

        let ori = Quat::from_axis_angle(Vector3::new(1.0, 0.0, 0.0).into(), 0.785398);
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
