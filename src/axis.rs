//! Axis enum for indexing three-dimensional structures.

use std::ops::{Index, IndexMut};
use std::fmt::{Display, Formatter, Result};

/// An `Axis` in a three-dimensional coordinate system.
/// Used to access `Vector3`/`Point3` structs via index.
///
/// # Example
/// ```
/// use bvh::axis::Axis;
///
/// let mut position = [1.0, 0.5, 42.0];
/// position[Axis::Y] *= 4.0;
///
/// assert_eq!(position[Axis::Y], 2.0);
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

/// Make three-dimensional arrays indexable by `Axis`.
impl Index<Axis> for [f32; 3] {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        &self[axis as usize]
    }
}

/// Make three-dimensional arrays mutably accessible by `Axis`.
impl IndexMut<Axis> for [f32; 3] {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        &mut self[axis as usize]
    }
}

/// Display implementation for `Axis`.
impl Display for Axis {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let c = match *self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        };
        write!(f, "{}", c)
    }
}

#[cfg(test)]
mod test {
    use axis::Axis;

    /// Test whether accessing arrays by index is the same as accessing them by `Axis`.
    quickcheck!{
        fn test_index_by_axis(tpl: (f32, f32, f32)) -> bool {
            let a = [tpl.0, tpl.1, tpl.2];

            a[0] == a[Axis::X] && a[1] == a[Axis::Y] && a[2] == a[Axis::Z]
        }
    }

    quickcheck!{
        fn test_set_by_axis(tpl: (f32, f32, f32)) -> bool {
            let mut a = [0.0, 0.0, 0.0];

            a[Axis::X] = tpl.0;
            a[Axis::Y] = tpl.1;
            a[Axis::Z] = tpl.2;

            a[0] == tpl.0 && a[1] == tpl.1 && a[2] == tpl.2
        }
    }
}
