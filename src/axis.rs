//! Axis enum for indexing three-dimensional structures.

#![allow(unused)]
use crate::{Point3, Vector3};
use std::fmt::{Display, Formatter, Result};
use std::ops::{Index, IndexMut};

struct MyType<T>(T);

/// An `Axis` in a three-dimensional coordinate system.
/// Used to access `Vector3`/`Point3` structs via index.
///
/// # Examples
/// ```
/// use bvh::axis::Axis;
///
/// let mut position = [1.0, 0.5, 42.0];
/// position[Axis::Y] *= 4.0;
///
/// assert_eq!(position[Axis::Y], 2.0);
/// ```
///
/// [`Point3`] and [`Vector3`] are also indexable using `Axis`.
///
/// ```
/// extern crate bvh;
///
/// use bvh::axis::Axis;
/// use bvh::Point3;
///
/// # fn main() {
/// let mut position: Point3 = Point3::new(1.0, 2.0, 3.0);
/// position[Axis::X] = 1000.0;
///
/// assert_eq!(position[Axis::X], 1000.0);
/// # }
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Axis {
    /// Index of the X axis.
    X = 0,

    /// Index of the Y axis.
    Y = 1,

    /// Index of the Z axis.
    Z = 2,
}

/// Display implementation for `Axis`.
impl Display for Axis {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match *self {
                Axis::X => "x",
                Axis::Y => "y",
                Axis::Z => "z",
            }
        )
    }
}

/// Make slices indexable by `Axis`.
impl Index<Axis> for [f32] {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        &self[axis as usize]
    }
}

/// Make `Point3` indexable by `Axis`.
impl Index<Axis> for Point3 {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        match axis {
            Axis::X => &self.x,
            Axis::Y => &self.y,
            Axis::Z => &self.z,
        }
    }
}

/// Make `Vector3` indexable by `Axis`.
impl Index<Axis> for MyType<Vector3> {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        match axis {
            Axis::X => &self.0.x,
            Axis::Y => &self.0.y,
            Axis::Z => &self.0.z,
        }
    }
}

/// Make slices mutably accessible by `Axis`.
impl IndexMut<Axis> for [f32] {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        &mut self[axis as usize]
    }
}

/// Make `Point3` mutably accessible by `Axis`.
impl IndexMut<Axis> for Point3 {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        match axis {
            Axis::X => &mut self.x,
            Axis::Y => &mut self.y,
            Axis::Z => &mut self.z,
        }
    }
}

/// Make `Vector3` mutably accessible by `Axis`.
impl IndexMut<Axis> for MyType<Vector3> {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        match axis {
            Axis::X => &mut self.0.x,
            Axis::Y => &mut self.0.y,
            Axis::Z => &mut self.0.z,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::axis::Axis;
    use proptest::prelude::*;

    proptest! {
        // Test whether accessing arrays by index is the same as accessing them by `Axis`.
        #[test]
        fn test_index_by_axis(tpl: (f32, f32, f32)) {
            let a = [tpl.0, tpl.1, tpl.2];

            assert!((a[0] - a[Axis::X]).abs() < f32::EPSILON && (a[1] - a[Axis::Y]).abs() < f32::EPSILON && (a[2] - a[Axis::Z]).abs() < f32::EPSILON);
        }

        // Test whether arrays can be mutably set, by indexing via `Axis`.
        #[test]
        fn test_set_by_axis(tpl: (f32, f32, f32)) {
            let mut a = [0.0, 0.0, 0.0];

            a[Axis::X] = tpl.0;
            a[Axis::Y] = tpl.1;
            a[Axis::Z] = tpl.2;

            assert!((a[0] - tpl.0).abs() < f32::EPSILON && (a[1] - tpl.1).abs() < f32::EPSILON && (a[2] - tpl.2).abs() < f32::EPSILON);
        }
    }
}
