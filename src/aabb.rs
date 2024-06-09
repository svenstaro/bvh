//! Axis Aligned Bounding Boxes.

use nalgebra::{Point, SVector};
use std::fmt;
use std::ops::Index;

use crate::bounding_hierarchy::BHValue;

/// [`Aabb`] struct.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb<T: BHValue, const D: usize> {
    /// Minimum coordinates
    pub min: Point<T, D>,

    /// Maximum coordinates
    pub max: Point<T, D>,
}

impl<T: BHValue + std::fmt::Display, const D: usize> fmt::Display for Aabb<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Min bound: {}; Max bound: {}", self.min, self.max)
    }
}

/// A trait implemented by things which can be bounded by an [`Aabb`].
///
/// [`Aabb`]: struct.Aabb.html
///
pub trait Bounded<T: BHValue, const D: usize> {
    /// Returns the geometric bounds of this object in the form of an [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
    /// use nalgebra::Point3;
    ///
    /// struct Something;
    ///
    /// impl Bounded<f32,3> for Something {
    ///     fn aabb(&self) -> Aabb<f32,3> {
    ///         let point1 = Point3::new(0.0,0.0,0.0);
    ///         let point2 = Point3::new(1.0,1.0,1.0);
    ///         Aabb::with_bounds(point1, point2)
    ///     }
    /// }
    ///
    /// let something = Something;
    /// let aabb = something.aabb();
    ///
    /// assert!(aabb.contains(&Point3::new(0.0,0.0,0.0)));
    /// assert!(aabb.contains(&Point3::new(1.0,1.0,1.0)));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    fn aabb(&self) -> Aabb<T, D>;
}

impl<T: BHValue, const D: usize, B: Bounded<T, D>> Bounded<T, D> for &B {
    fn aabb(&self) -> Aabb<T, D> {
        B::aabb(self)
    }
}

impl<T: BHValue, const D: usize, B: Bounded<T, D>> Bounded<T, D> for &mut B {
    fn aabb(&self) -> Aabb<T, D> {
        B::aabb(self)
    }
}

impl<T: BHValue, const D: usize, B: Bounded<T, D>> Bounded<T, D> for Box<B> {
    fn aabb(&self) -> Aabb<T, D> {
        B::aabb(self)
    }
}

impl<T: BHValue, const D: usize> Aabb<T, D> {
    /// Creates a new [`Aabb`] with the given bounds.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// assert_eq!(aabb.min.x, -1.0);
    /// assert_eq!(aabb.max.z, 1.0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn with_bounds(min: Point<T, D>, max: Point<T, D>) -> Self {
        Aabb { min, max }
    }

    /// Creates a new empty [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    ///
    /// # fn main() {
    /// let aabb = Aabb::<f32,3>::empty();
    /// let min = &aabb.min;
    /// let max = &aabb.max;
    ///
    /// // For any point
    /// let x = rand::random();
    /// let y = rand::random();
    /// let z = rand::random();
    ///
    /// // An empty `Aabb` should not contain it
    /// assert!(x < min.x && y < min.y && z < min.z);
    /// assert!(max.x < x && max.y < y && max.z < z);
    /// # }
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn empty() -> Self {
        Self {
            min: SVector::<T, D>::from_element(T::infinity()).into(),
            max: SVector::<T, D>::from_element(T::neg_infinity()).into(),
        }
    }

    /// Creates a new infinite [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    ///
    /// # fn main() {
    /// let aabb :Aabb<f32,3> = Aabb::infinite();
    /// let min = &aabb.min;
    /// let max = &aabb.max;
    ///
    /// // For any point
    /// let x = rand::random();
    /// let y = rand::random();
    /// let z = rand::random();
    ///
    /// // An infinite `Aabb` should contain it
    /// assert!(x > min.x && y > min.y && z > min.z);
    /// assert!(max.x > x && max.y > y && max.z > z);
    /// # }
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn infinite() -> Self {
        Self {
            min: SVector::<T, D>::from_element(T::neg_infinity()).into(),
            max: SVector::<T, D>::from_element(T::infinity()).into(),
        }
    }

    /// Returns true if the [`Point`] is inside the [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
    /// let point_inside = Point3::new(0.125, -0.25, 0.5);
    /// let point_outside = Point3::new(1.0, -2.0, 4.0);
    ///
    /// assert!(aabb.contains(&point_inside));
    /// assert!(!aabb.contains(&point_outside));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    /// [`Point`]: nalgebra::Point
    ///
    pub fn contains(&self, p: &Point<T, D>) -> bool {
        p >= &self.min && p <= &self.max
    }

    /// Returns true if the [`Point3`] is approximately inside the [`Aabb`]
    /// with respect to some `epsilon`.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
    /// let point_barely_outside = Point3::new(1.000_000_1, -1.000_000_1, 1.000_000_001);
    /// let point_outside = Point3::new(1.0, -2.0, 4.0);
    ///
    /// assert!(aabb.approx_contains_eps(&point_barely_outside, 0.00001));
    /// assert!(!aabb.approx_contains_eps(&point_outside, 0.00001));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    /// [`Point3`]: nalgebra::Point3
    ///
    pub fn approx_contains_eps(&self, p: &Point<T, D>, epsilon: T) -> bool {
        let ne = -epsilon;
        (p - self.min) > SVector::from_element(ne)
            && (p - self.max) < SVector::from_element(epsilon)
    }

    /// Returns true if the `other` [`Aabb`] is approximately inside this [`Aabb`]
    /// with respect to some `epsilon`.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
    /// let point_barely_outside = Point3::new(1.000_000_1, 1.000_000_1, 1.000_000_1);
    /// let center = aabb.center();
    /// let inner_aabb = Aabb::with_bounds(center, point_barely_outside);
    ///
    /// assert!(aabb.approx_contains_aabb_eps(&inner_aabb, 0.00001));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    pub fn approx_contains_aabb_eps(&self, other: &Aabb<T, D>, epsilon: T) -> bool {
        self.approx_contains_eps(&other.min, epsilon)
            && self.approx_contains_eps(&other.max, epsilon)
    }

    /// Returns true if the `other` [`Aabb`] is approximately equal to this [`Aabb`]
    /// with respect to some `epsilon`.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
    /// let point_barely_outside_min = Point3::new(-1.000_000_1, -1.000_000_1, -1.000_000_1);
    /// let point_barely_outside_max = Point3::new(1.000_000_1, 1.000_000_1, 1.000_000_1);
    /// let other = Aabb::with_bounds(point_barely_outside_min, point_barely_outside_max);
    ///
    /// assert!(aabb.relative_eq(&other, 0.00001));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    pub fn relative_eq(&self, other: &Aabb<T, D>, epsilon: T) -> bool {
        let ep_vec = SVector::from_element(epsilon);
        (self.min - other.min).abs() < ep_vec && (self.max - other.max).abs() < ep_vec
    }

    /// Returns a new minimal [`Aabb`] which contains both this [`Aabb`] and `other`.
    /// The result is the convex hull of the both [`Aabb`]s.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb1 = Aabb::with_bounds(Point3::new(-101.0, 0.0, 0.0), Point3::new(-100.0, 1.0, 1.0));
    /// let aabb2 = Aabb::with_bounds(Point3::new(100.0, 0.0, 0.0), Point3::new(101.0, 1.0, 1.0));
    /// let joint = aabb1.join(&aabb2);
    ///
    /// let point_inside_aabb1 = Point3::new(-100.5, 0.5, 0.5);
    /// let point_inside_aabb2 = Point3::new(100.5, 0.5, 0.5);
    /// let point_inside_joint = Point3::new(0.0, 0.5, 0.5);
    ///
    /// # assert!(aabb1.contains(&point_inside_aabb1));
    /// # assert!(!aabb1.contains(&point_inside_aabb2));
    /// # assert!(!aabb1.contains(&point_inside_joint));
    /// #
    /// # assert!(!aabb2.contains(&point_inside_aabb1));
    /// # assert!(aabb2.contains(&point_inside_aabb2));
    /// # assert!(!aabb2.contains(&point_inside_joint));
    ///
    /// assert!(joint.contains(&point_inside_aabb1));
    /// assert!(joint.contains(&point_inside_aabb2));
    /// assert!(joint.contains(&point_inside_joint));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn join(&self, other: &Aabb<T, D>) -> Aabb<T, D> {
        Aabb::with_bounds(
            self.min.coords.inf(&other.min.coords).into(),
            self.max.coords.sup(&other.max.coords).into(),
        )
    }

    /// Mutable version of [`Aabb::join`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::{Point3, Vector3};
    ///
    /// let size = Vector3::new(1.0, 1.0, 1.0);
    /// let aabb_pos = Point3::new(-101.0, 0.0, 0.0);
    /// let mut aabb = Aabb::with_bounds(aabb_pos, aabb_pos + size);
    ///
    /// let other_pos = Point3::new(100.0, 0.0, 0.0);
    /// let other = Aabb::with_bounds(other_pos, other_pos + size);
    ///
    /// let point_inside_aabb = aabb_pos + size / 2.0;
    /// let point_inside_other = other_pos + size / 2.0;
    /// let point_inside_joint = Point3::new(0.0, 0.0, 0.0) + size / 2.0;
    ///
    /// # assert!(aabb.contains(&point_inside_aabb));
    /// # assert!(!aabb.contains(&point_inside_other));
    /// # assert!(!aabb.contains(&point_inside_joint));
    /// #
    /// # assert!(!other.contains(&point_inside_aabb));
    /// # assert!(other.contains(&point_inside_other));
    /// # assert!(!other.contains(&point_inside_joint));
    ///
    /// aabb.join_mut(&other);
    ///
    /// assert!(aabb.contains(&point_inside_aabb));
    /// assert!(aabb.contains(&point_inside_other));
    /// assert!(aabb.contains(&point_inside_joint));
    /// ```
    ///
    /// [`Aabb::join`]: struct.Aabb.html
    ///
    pub fn join_mut(&mut self, other: &Aabb<T, D>) {
        *self = self.join(other);
    }

    /// Returns a new minimal [`Aabb`] which contains both
    /// this [`Aabb`] and the [`Point3`] `other`.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let point1 = Point3::new(0.0, 0.0, 0.0);
    /// let point2 = Point3::new(1.0, 1.0, 1.0);
    /// let point3 = Point3::new(2.0, 2.0, 2.0);
    ///
    /// let aabb = Aabb::empty();
    /// assert!(!aabb.contains(&point1));
    ///
    /// let aabb1 = aabb.grow(&point1);
    /// assert!(aabb1.contains(&point1));
    ///
    /// let aabb2 = aabb.grow(&point2);
    /// assert!(aabb2.contains(&point2));
    /// assert!(!aabb2.contains(&point3));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    /// [`Point3`]: nalgebra::Point3
    ///
    pub fn grow(&self, other: &Point<T, D>) -> Aabb<T, D> {
        Aabb::with_bounds(
            self.min.coords.inf(&other.coords).into(),
            self.max.coords.sup(&other.coords).into(),
        )
    }

    /// Mutable version of [`Aabb::grow`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let point1 = Point3::new(0.0, 0.0, 0.0);
    /// let point2 = Point3::new(1.0, 1.0, 1.0);
    /// let point3 = Point3::new(2.0, 2.0, 2.0);
    ///
    /// let mut aabb = Aabb::empty();
    /// assert!(!aabb.contains(&point1));
    ///
    /// aabb.grow_mut(&point1);
    /// assert!(aabb.contains(&point1));
    /// assert!(!aabb.contains(&point2));
    ///
    /// aabb.grow_mut(&point2);
    /// assert!(aabb.contains(&point2));
    /// assert!(!aabb.contains(&point3));
    /// ```
    ///
    /// [`Aabb::grow`]: struct.Aabb.html
    /// [`Point3`]: nalgebra::Point3
    ///
    pub fn grow_mut(&mut self, other: &Point<T, D>) {
        *self = self.grow(other);
    }

    /// Returns a new minimal [`Aabb`] which contains both this [`Aabb`] and the [`Bounded`]
    /// `other`.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
    /// use nalgebra::Point3;
    ///
    /// struct Something;
    ///
    /// impl Bounded<f32,3> for Something {
    ///     fn aabb(&self) -> Aabb<f32,3> {
    ///         let point1 = Point3::new(0.0,0.0,0.0);
    ///         let point2 = Point3::new(1.0,1.0,1.0);
    ///         Aabb::with_bounds(point1, point2)
    ///     }
    /// }
    ///
    /// let aabb = Aabb::empty();
    /// let something = Something;
    /// let aabb1 = aabb.join_bounded(&something);
    ///
    /// let center = something.aabb().center();
    /// assert!(aabb1.contains(&center));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    /// [`Bounded`]: trait.Bounded.html
    ///
    pub fn join_bounded<B: Bounded<T, D>>(&self, other: &B) -> Aabb<T, D> {
        self.join(&other.aabb())
    }

    /// Returns the size of this [`Aabb`] in all three dimensions.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// let size = aabb.size();
    /// assert!(size.x == 2.0 && size.y == 2.0 && size.z == 2.0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn size(&self) -> SVector<T, D> {
        self.max - self.min
    }

    /// Returns the half size of this [`Aabb`] in all three dimensions.
    /// This can be interpreted as the distance from the center to the edges
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let aabb = Aabb::with_bounds(Point3::new(0.0,0.0,0.0), Point3::new(2.0,2.0,2.0));
    /// let half_size = aabb.half_size();
    /// assert!(half_size.x == 1.0 && half_size.y == 1.0 && half_size.z == 1.0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    #[inline]
    pub fn half_size(&self) -> SVector<T, D>
    {
        self.size() * T::from_f32(0.5).unwrap()
    }

    /// Returns the center [`Point3`] of the [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = Aabb::with_bounds(min, max);
    /// let center = aabb.center();
    /// assert!(center.x == 42.0 && center.y == 42.0 && center.z == 42.0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    /// [`Point3`]: nalgebra::Point3
    ///
    pub fn center(&self) -> Point<T, D> {
        (self.min.coords + (self.size() * T::from_f32(0.5).unwrap())).into()
    }

    /// An empty [`Aabb`] is an [`Aabb`] where the lower bound is greater than
    /// the upper bound in at least one component
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let empty_aabb: Aabb<f32,3> = Aabb::empty();
    /// assert!(empty_aabb.is_empty());
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = Aabb::with_bounds(min, max);
    /// assert!(!aabb.is_empty());
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn is_empty(&self) -> bool {
        // Special trick here, we use a join/supremum to pick the highest values, and if the highest
        // values are not equal to the max, then obviously a min was higher!
        // This should be two simd instructions (I hope) for vectors/points up to size 4.
        // It might need to be changed to an iterative method later for dimensions above 4
        self.min.coords.sup(&self.max.coords) != self.max.coords
    }

    /// Returns the total surface area of this [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = Aabb::with_bounds(min, max);
    /// let surface_area = aabb.surface_area();
    /// assert!(surface_area == 24.0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn surface_area(&self) -> T {
        let size = self.size();
        T::from_f32(2.0).unwrap() * size.dot(&size)
    }

    /// Returns the volume of this [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = Aabb::with_bounds(min, max);
    /// let volume = aabb.volume();
    /// assert!(volume == 8.0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn volume(&self) -> T {
        self.size().product()
    }

    /// Returns the axis along which the [`Aabb`] is stretched the most.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use nalgebra::Point3;
    ///
    /// let min = Point3::new(-100.0,0.0,0.0);
    /// let max = Point3::new(100.0,0.0,0.0);
    ///
    /// let aabb = Aabb::with_bounds(min, max);
    /// let axis = aabb.largest_axis();
    /// assert!(axis == 0);
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn largest_axis(&self) -> usize {
        self.size().imax()
    }
}

/// Default instance for [`Aabb`]s. Returns an [`Aabb`] which is [`empty()`].
///
/// [`Aabb`]: struct.Aabb.html
/// [`empty()`]: #method.empty
///
impl<T: BHValue, const D: usize> Default for Aabb<T, D> {
    fn default() -> Aabb<T, D> {
        Aabb::empty()
    }
}

/// Make [`Aabb`]s indexable. `aabb[0]` gives a reference to the minimum bound.
/// All other indices return a reference to the maximum bound.
///
/// # Examples
/// ```
/// use bvh::aabb::Aabb;
/// use nalgebra::Point3;
///
/// let min = Point3::new(3.0,4.0,5.0);
/// let max = Point3::new(123.0,123.0,123.0);
///
/// let aabb = Aabb::with_bounds(min, max);
/// assert_eq!(aabb[0], min);
/// assert_eq!(aabb[1], max);
/// ```
///
/// [`Aabb`]: struct.Aabb.html
///
impl<T: BHValue, const D: usize> Index<usize> for Aabb<T, D> {
    type Output = Point<T, D>;

    fn index(&self, index: usize) -> &Point<T, D> {
        [&self.min, &self.max][index]
    }
}

/// Implementation of [`Bounded`] for [`Aabb`].
///
/// # Examples
/// ```
/// use bvh::aabb::{Aabb, Bounded};
/// use nalgebra::Point3;
///
/// let point_a = Point3::new(3.0,4.0,5.0);
/// let point_b = Point3::new(17.0,18.0,19.0);
/// let aabb = Aabb::empty().grow(&point_a).grow(&point_b);
///
/// let aabb_aabb = aabb.aabb();
///
/// assert_eq!(aabb_aabb.min, aabb.min);
/// assert_eq!(aabb_aabb.max, aabb.max);
/// ```
///
/// [`Aabb`]: struct.Aabb.html
/// [`Bounded`]: trait.Bounded.html
///
impl<T: BHValue, const D: usize> Bounded<T, D> for Aabb<T, D> {
    fn aabb(&self) -> Aabb<T, D> {
        *self
    }
}

/// Implementation of [`Bounded`] for [`Point3`].
///
/// # Examples
/// ```
/// use bvh::aabb::{Aabb, Bounded};
/// use nalgebra::Point3;
///
/// let point = Point3::new(3.0,4.0,5.0);
///
/// let aabb = point.aabb();
/// assert!(aabb.contains(&point));
/// ```
///
/// [`Bounded`]: trait.Bounded.html
/// [`Point3`]: nalgebra::Point3
///
impl<T: BHValue, const D: usize> Bounded<T, D> for Point<T, D> {
    fn aabb(&self) -> Aabb<T, D> {
        Aabb::with_bounds(*self, *self)
    }
}

#[cfg(test)]
mod tests {
    use crate::aabb::Bounded;
    use crate::testbase::{
        tuple_to_point, tuple_to_vector, tuplevec_large_strategy, TAabb3, TPoint3, TVector3,
        TupleVec,
    };

    use float_eq::assert_float_eq;
    use proptest::prelude::*;

    proptest! {
        // Test whether an empty `Aabb` does not contains anything.
        #[test]
        fn test_empty_contains_nothing(tpl: TupleVec) {
            // Define a random Point
            let p = tuple_to_point(&tpl);

            // Create an empty Aabb
            let aabb = TAabb3::empty();

            // It should not contain anything
            assert!(!aabb.contains(&p));
        }

        // Test whether a default `Aabb` is empty.
        #[test]
        fn test_default_is_empty(tpl: TupleVec) {
            // Define a random Point
            let p = tuple_to_point(&tpl);

            // Create a default Aabb
            let aabb: TAabb3 = Default::default();

            // It should not contain anything
            assert!(!aabb.contains(&p));
        }

        // Test whether an `Aabb` always contains its center.
        #[test]
        fn test_aabb_contains_center(a: TupleVec, b: TupleVec) {
            // Define two points which will be the corners of the `Aabb`
            let p1 = tuple_to_point(&a);
            let p2 = tuple_to_point(&b);

            // Span the `Aabb`
            let aabb = TAabb3::empty().grow(&p1).join_bounded(&p2);

            // Its center should be inside the `Aabb`
            assert!(aabb.contains(&aabb.center()));
        }

        // Test whether the joint of two point-sets contains all the points.
        #[test]
        fn test_join_two_aabbs(a: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec),
                               b: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec))
                               {
            // Define an array of ten points
            let points = [a.0, a.1, a.2, a.3, a.4, b.0, b.1, b.2, b.3, b.4];

            // Convert these points to `Point3`
            let points = points.iter().map(tuple_to_point).collect::<Vec<TPoint3>>();

            // Create two `Aabb`s. One spanned the first five points,
            // the other by the last five points
            let aabb1 = points.iter().take(5).fold(TAabb3::empty(), |aabb, point| aabb.grow(point));
            let aabb2 = points.iter().skip(5).fold(TAabb3::empty(), |aabb, point| aabb.grow(point));

            // The `Aabb`s should contain the points by which they are spanned
            let aabb1_contains_init_five = points.iter()
                .take(5)
                .all(|point| aabb1.contains(point));
            let aabb2_contains_last_five = points.iter()
                .skip(5)
                .all(|point| aabb2.contains(point));

            // Build the joint of the two `Aabb`s
            let aabbu = aabb1.join(&aabb2);

            // The joint should contain all points
            let aabbu_contains_all = points.iter()
                .all(|point| aabbu.contains(point));

            // Return the three properties
            assert!(aabb1_contains_init_five && aabb2_contains_last_five && aabbu_contains_all);
        }

        // Test whether some points relative to the center of an `Aabb` are classified correctly.
        // Currently doesn't test `approx_contains_eps` or `contains` very well due to scaling by 0.9 and 1.1.
        #[test]
        fn test_points_relative_to_center_and_size(a in tuplevec_large_strategy(), b in tuplevec_large_strategy()) {
            // Generate some nonempty Aabb
            let aabb = TAabb3::empty()
                .grow(&tuple_to_point(&a))
                .grow(&tuple_to_point(&b));

            // Get its size and center
            let size = aabb.size();
            let size_half = size / 2.0;
            let center = aabb.center();

            // Compute the min and the max corners of the `Aabb` by hand
            let inside_ppp = center + size_half * 0.9;
            let inside_mmm = center - size_half * 0.9;

            // Generate two points which are outside the `Aabb`
            let outside_ppp = inside_ppp + size_half * 1.1;
            let outside_mmm = inside_mmm - size_half * 1.1;

            assert!(aabb.approx_contains_eps(&inside_ppp, f32::EPSILON));
            assert!(aabb.approx_contains_eps(&inside_mmm, f32::EPSILON));
            assert!(!aabb.contains(&outside_ppp));
            assert!(!aabb.contains(&outside_mmm));
        }

        // Test whether the surface of a nonempty `Aabb is always positive.
        #[test]
        fn test_surface_always_positive(a: TupleVec, b: TupleVec) {
            let aabb = TAabb3::empty()
                .grow(&tuple_to_point(&a))
                .grow(&tuple_to_point(&b));
            assert!(aabb.surface_area() >= 0.0);
        }

        // Compute and compare the surface area of an `Aabb` by hand.
        #[test]
        fn test_surface_area_cube(pos: TupleVec, size in f32::EPSILON..10e30_f32) {
            // Generate some non-empty Aabb
            let pos = tuple_to_point(&pos);
            let size_vec = TVector3::new(size, size, size);
            let aabb = TAabb3::with_bounds(pos, pos + size_vec);

            // Check its surface area
            let area_a = aabb.surface_area();
            let area_b = 6.0 * size * size;
            assert_float_eq!(area_a, area_b, rmax <= f32::EPSILON);
        }

        // Test whether the volume of a nonempty `Aabb` is always positive.
        #[test]
        fn test_volume_always_positive(a in tuplevec_large_strategy(), b in tuplevec_large_strategy()) {
            let aabb = TAabb3::empty()
                .grow(&tuple_to_point(&a))
                .grow(&tuple_to_point(&b));
            assert!(aabb.volume() >= 0.0);
        }

        // Compute and compare the volume of an `Aabb` by hand.
        #[test]
        fn test_volume_by_hand(pos in tuplevec_large_strategy(), size in tuplevec_large_strategy()) {
            // Generate some non-empty Aabb
            let pos = tuple_to_point(&pos);
            let size = tuple_to_vector(&size);
            let aabb = pos.aabb().grow(&(pos + size));

            // Check its volume
            let volume_a = aabb.volume();
            let volume_b = (size.x * size.y * size.z).abs();
            assert_float_eq!(volume_a, volume_b, rmax <= f32::EPSILON);
        }

        // Test whether generating an `Aabb` from the min and max bounds yields the same `Aabb`.
        #[test]
        fn test_create_aabb_from_indexable(a: TupleVec, b: TupleVec, p: TupleVec) {
            // Create a random point
            let point = tuple_to_point(&p);

            // Create a random `Aabb`
            let aabb = TAabb3::empty()
                .grow(&tuple_to_point(&a))
                .grow(&tuple_to_point(&b));

            // Create an `Aabb` by using the index-access method
            let aabb_by_index = TAabb3::with_bounds(aabb[0], aabb[1]);

            // The `Aabb`s should be the same
            assert!(aabb.contains(&point) == aabb_by_index.contains(&point));
        }
    }
}
