use crate::aabb::Aabb;
use crate::bounding_hierarchy::{BHShape, BHValue};

use crate::ray::Ray;
use crate::utils::{joint_aabb_of_shapes, Bucket};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

const NUM_BUCKETS: usize = 6;

thread_local! {
    /// Thread local for the buckets used while building to reduce allocations during build
    static BUCKETS: RefCell<[Vec<ShapeIndex>; NUM_BUCKETS]> = RefCell::new(Default::default());
}

/// The [`BvhNode`] enum that describes a node in a [`Bvh`].
/// It's either a leaf node and references a shape (by holding its index)
/// or a regular node that has two child nodes.
/// The non-leaf node stores the [`Aabb`]s of its children.
///
/// [`Aabb`]: ../aabb/struct.Aabb.html
/// [`Bvh`]: struct.Bvh.html
/// [`Bvh`]: struct.BvhNode.html
///
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BvhNode<T: BHValue, const D: usize> {
    /// Leaf node.
    Leaf {
        /// The node's parent.
        parent_index: usize,

        /// The shape contained in this leaf.
        shape_index: usize,
    },
    /// Inner node.
    Node {
        /// The node's parent.
        parent_index: usize,

        /// Index of the left subtree's root node.
        child_l_index: usize,

        /// The convex hull of the shapes' [`Aabb`]'s in child_l.
        child_l_aabb: Aabb<T, D>,

        /// Index of the right subtree's root node.
        child_r_index: usize,

        /// The convex hull of the shapes' [`Aabb`]'s in child_r.
        child_r_aabb: Aabb<T, D>,
    },
}

impl<T: BHValue, const D: usize> BvhNode<T, D> {
    /// Builds a [`BvhNode`] recursively using SAH partitioning.
    ///
    /// [`BvhNode`]: enum.BvhNode.html
    ///
    pub fn build<S: BHShape<T, D>>(args: BvhNodeBuildArgs<S, T, D>) {
        if let Some((left, right)) = Self::prep_build(args) {
            Self::build(left);
            Self::build(right);
            // BvhBuildStrategy::<X, Y>::dispatch(|| Self::build(left, strategy), || Self::build(right, strategy), left.indices.len() + right.indices.len(), left.depth as usize)
        }
    }

    /// Builds a [`BvhNode`] recursively in parallel using SAH partitioning.
    ///
    /// [`BvhNode`]: enum.BvhNode.html
    ///
    pub fn build_with_executor<S: BHShape<T, D>>(
        args: BvhNodeBuildArgs<S, T, D>,
        mut executor: impl FnMut(BvhNodeBuildArgs<S, T, D>, BvhNodeBuildArgs<S, T, D>),
    ) {
        if let Some((left, right)) = Self::prep_build(args) {
            // Self::build(left);
            // Self::build(right);
            executor(left, right);
            // RayonBuildStrategy::dispatch(|| Self::build(left), || Self::build(right), left.indices.len() + right.indices.len(), left.depth as usize)
        }
    }

    /// Builds a [`BvhNode`] recursively using SAH partitioning.
    ///
    /// [`BvhNode`]: enum.BvhNode.html
    ///
    pub fn prep_build<S: BHShape<T, D>>(
        args: BvhNodeBuildArgs<S, T, D>,
    ) -> Option<(BvhNodeBuildArgs<S, T, D>, BvhNodeBuildArgs<S, T, D>)> {
        let BvhNodeBuildArgs {
            shapes,
            indices,
            nodes,
            parent_index,
            depth,
            node_index,
            aabb_bounds,
            centroid_bounds,
        } = args;
        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            nodes[0].write(BvhNode::Leaf {
                parent_index,
                shape_index: shape_index.0,
            });
            // Let the shape know the index of the node that represents it.
            shapes.set_node_index(shape_index, node_index);
            return None;
        }

        // Find the axis along which the shapes are spread the most.
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        // The following `if` partitions `indices` for recursively calling `Bvh::build`.
        let (
            (child_l_aabb, child_l_centroid, child_l_indices),
            (child_r_aabb, child_r_centroid, child_r_indices),
        ) = if split_axis_size < T::epsilon() {
            // In this branch the shapes lie too close together so that splitting them in a
            // sensible way is not possible. Instead we just split the list of shapes in half.
            let (child_l_indices, child_r_indices) = indices.split_at_mut(indices.len() / 2);
            let (child_l_aabb, child_l_centroid) = joint_aabb_of_shapes(child_l_indices, shapes);
            let (child_r_aabb, child_r_centroid) = joint_aabb_of_shapes(child_r_indices, shapes);

            (
                (child_l_aabb, child_l_centroid, child_l_indices),
                (child_r_aabb, child_r_centroid, child_r_indices),
            )
        } else {
            BvhNode::build_buckets(
                shapes,
                indices,
                split_axis,
                split_axis_size,
                &centroid_bounds,
                &aabb_bounds,
            )
        };

        let left_len = child_l_indices.len() * 2 - 1;
        let child_l_index = node_index + 1;
        let child_r_index = child_l_index + left_len;

        // Construct the actual data structure and replace the dummy node.
        nodes[0].write(BvhNode::Node {
            parent_index,
            child_l_aabb,
            child_l_index,
            child_r_aabb,
            child_r_index,
        });

        let next_nodes = &mut nodes[1..];
        let (l_nodes, r_nodes) = next_nodes.split_at_mut(left_len);

        Some((
            BvhNodeBuildArgs::new(
                shapes,
                child_l_indices,
                l_nodes,
                node_index,
                depth + 1,
                child_l_index,
                child_l_aabb,
                child_l_centroid,
            ),
            BvhNodeBuildArgs::new(
                shapes,
                child_r_indices,
                r_nodes,
                node_index,
                depth + 1,
                child_r_index,
                child_r_aabb,
                child_r_centroid,
            ),
        ))
    }

    #[allow(clippy::type_complexity)]
    fn build_buckets<'a, S: BHShape<T, D>>(
        shapes: &Shapes<S>,
        indices: &'a mut [ShapeIndex],
        split_axis: usize,
        split_axis_size: T,
        centroid_bounds: &Aabb<T, D>,
        aabb_bounds: &Aabb<T, D>,
    ) -> (
        (Aabb<T, D>, Aabb<T, D>, &'a mut [ShapeIndex]),
        (Aabb<T, D>, Aabb<T, D>, &'a mut [ShapeIndex]),
    ) {
        // Create six `Bucket`s, and six index assignment vector.
        // let mut buckets = [Bucket::empty(); NUM_BUCKETS];
        // let mut bucket_assignments: [SmallVec<[usize; 1024]>; NUM_BUCKETS] = Default::default();
        BUCKETS.with(move |buckets| {
            let bucket_assignments = &mut *buckets.borrow_mut();
            let mut buckets = [Bucket::empty(); NUM_BUCKETS];
            buckets.fill(Bucket::empty());
            for b in bucket_assignments.iter_mut() {
                b.clear();
            }

            // In this branch the `split_axis_size` is large enough to perform meaningful splits.
            // We start by assigning the shapes to `Bucket`s.
            for idx in indices.iter() {
                let shape = shapes.get(*idx);
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid `[0.0..1.0]`.
                let bucket_num_relative =
                    (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

                // Convert that to the actual `Bucket` number.
                let bucket_num = (bucket_num_relative
                    * (T::from(NUM_BUCKETS).unwrap() - T::from(0.01).unwrap()))
                .to_usize()
                .unwrap();

                // Extend the selected `Bucket` and add the index to the actual bucket.
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }

            // Compute the costs for each configuration and select the best configuration.
            let mut min_bucket = 0;
            let mut min_cost = T::infinity();
            let mut child_l_aabb = Aabb::empty();
            let mut child_l_centroid = Aabb::empty();
            let mut child_r_aabb = Aabb::empty();
            let mut child_r_centroid = Aabb::empty();
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let child_l = l_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);
                let child_r = r_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);

                let cost = (T::from(child_l.size).unwrap() * child_l.aabb.surface_area()
                    + T::from(child_r.size).unwrap() * child_r.aabb.surface_area())
                    / aabb_bounds.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.aabb;
                    child_l_centroid = child_l.centroid;
                    child_r_aabb = child_r.aabb;
                    child_r_centroid = child_r.centroid;
                }
            }
            // Join together all index buckets.
            // split input indices, loop over assignments and assign
            let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);

            let mut l_count = 0;
            for group in l_assignments.iter() {
                l_count += group.len();
            }

            let (child_l_indices, child_r_indices) = indices.split_at_mut(l_count);
            let mut i = 0;
            for group in l_assignments.iter() {
                for x in group {
                    child_l_indices[i] = *x;
                    i += 1;
                }
            }
            i = 0;
            for group in r_assignments.iter() {
                for x in group {
                    child_r_indices[i] = *x;
                    i += 1;
                }
            }

            (
                (child_l_aabb, child_l_centroid, child_l_indices),
                (child_r_aabb, child_r_centroid, child_r_indices),
            )
        })
    }

    /// Traverses the [`Bvh`] recursively and returns all shapes whose [`Aabb`] is
    /// intersected by the given [`Ray`].
    ///
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    /// [`Bvh`]: struct.Bvh.html
    /// [`Ray`]: ../ray/struct.Ray.html
    ///
    pub fn traverse_recursive(
        nodes: &[BvhNode<T, D>],
        node_index: usize,
        ray: &Ray<T, D>,
        indices: &mut Vec<usize>,
    ) {
        match nodes[node_index] {
            BvhNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
                ..
            } => {
                if ray.intersects_aabb(child_l_aabb) {
                    BvhNode::traverse_recursive(nodes, child_l_index, ray, indices);
                }
                if ray.intersects_aabb(child_r_aabb) {
                    BvhNode::traverse_recursive(nodes, child_r_index, ray, indices);
                }
            }
            BvhNode::Leaf { shape_index, .. } => {
                indices.push(shape_index);
            }
        }
    }
}

/// Shapes holds a mutable ptr to the slice of Shapes passed in to build. It is accessed only through a ShapeIndex.
/// These are a set of unique indices into Shapes that are generated at the start of the build process. Because they
/// are all unique they guarantee that access into Shapes is safe to do in parallel.
pub struct Shapes<'a, S> {
    ptr: *mut S,
    len: usize,
    marker: PhantomData<&'a S>,
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct ShapeIndex(pub usize);

impl<S> Shapes<'_, S> {
    pub(crate) fn set_node_index<T: BHValue, const D: usize>(
        &self,
        shape_index: ShapeIndex,
        node_index: usize,
    ) where
        S: BHShape<T, D>,
    {
        assert!(shape_index.0 < self.len);
        unsafe {
            self.ptr
                .add(shape_index.0)
                .as_mut()
                .unwrap()
                .set_bh_node_index(node_index);
        }
    }
    pub(crate) fn get<T: BHValue, const D: usize>(&self, shape_index: ShapeIndex) -> &S
    where
        S: BHShape<T, D>,
    {
        assert!(shape_index.0 < self.len);
        unsafe { self.ptr.add(shape_index.0).as_ref().unwrap() }
    }

    pub(crate) fn from_slice<T: BHValue, const D: usize>(slice: &mut [S]) -> Shapes<S>
    where
        S: BHShape<T, D>,
    {
        Shapes {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            marker: PhantomData,
        }
    }
}

unsafe impl<S: Send> Send for Shapes<'_, S> {}
unsafe impl<S> Sync for Shapes<'_, S> {}

/// Holds the arguments for calling build.
pub struct BvhNodeBuildArgs<'a, S, T: BHValue, const D: usize> {
    shapes: &'a Shapes<'a, S>,
    indices: &'a mut [ShapeIndex],
    nodes: &'a mut [MaybeUninit<BvhNode<T, D>>],
    parent_index: usize,
    depth: u32,
    node_index: usize,
    aabb_bounds: Aabb<T, D>,
    centroid_bounds: Aabb<T, D>,
}

impl<'a, S, T: BHValue, const D: usize> BvhNodeBuildArgs<'a, S, T, D> {
    /// Creates the args
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        shapes: &'a Shapes<'a, S>,
        indices: &'a mut [ShapeIndex],
        nodes: &'a mut [MaybeUninit<BvhNode<T, D>>],
        parent_index: usize,
        depth: u32,
        node_index: usize,
        aabb_bounds: Aabb<T, D>,
        centroid_bounds: Aabb<T, D>,
    ) -> Self {
        Self {
            shapes,
            indices,
            nodes,
            parent_index,
            depth,
            node_index,
            aabb_bounds,
            centroid_bounds,
        }
    }

    /// Finish building this portion of the bvh.
    pub fn build(self)
    where
        S: BHShape<T, D>,
    {
        BvhNode::<T, D>::build(self)
    }

    /// Finish building this portion of the bvh using a custom executor.
    pub fn build_with_executor(
        self,
        executor: impl FnMut(BvhNodeBuildArgs<'_, S, T, D>, BvhNodeBuildArgs<'_, S, T, D>),
    ) where
        S: BHShape<T, D>,
    {
        BvhNode::<T, D>::build_with_executor(self, executor)
    }

    /// Returns the number of nodes that are part of this build.
    pub fn node_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns the current depth in the Bvh.
    pub fn depth(&self) -> usize {
        self.depth as usize
    }
}

impl<T: BHValue, const D: usize> PartialEq for BvhNode<T, D> {
    // TODO Consider also comparing [`Aabbs`]
    fn eq(&self, other: &BvhNode<T, D>) -> bool {
        match (self, other) {
            (
                &BvhNode::Node {
                    parent_index: self_parent_index,
                    child_l_index: self_child_l_index,
                    child_r_index: self_child_r_index,
                    ..
                },
                &BvhNode::Node {
                    parent_index: other_parent_index,
                    child_l_index: other_child_l_index,
                    child_r_index: other_child_r_index,
                    ..
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_child_l_index == other_child_l_index
                    && self_child_r_index == other_child_r_index
            }
            (
                &BvhNode::Leaf {
                    parent_index: self_parent_index,
                    shape_index: self_shape_index,
                },
                &BvhNode::Leaf {
                    parent_index: other_parent_index,
                    shape_index: other_shape_index,
                },
            ) => self_parent_index == other_parent_index && self_shape_index == other_shape_index,
            _ => false,
        }
    }
}

impl<T: BHValue, const D: usize> BvhNode<T, D> {
    /// Returns the index of the parent node.
    pub fn parent(&self) -> usize {
        match *self {
            BvhNode::Node { parent_index, .. } | BvhNode::Leaf { parent_index, .. } => parent_index,
        }
    }

    /// Returns a mutable reference to the parent node index.
    pub fn parent_mut(&mut self) -> &mut usize {
        match *self {
            BvhNode::Node {
                ref mut parent_index,
                ..
            }
            | BvhNode::Leaf {
                ref mut parent_index,
                ..
            } => parent_index,
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l(&self) -> usize {
        match *self {
            BvhNode::Node { child_l_index, .. } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l_mut(&mut self) -> &mut usize {
        match *self {
            BvhNode::Node {
                ref mut child_l_index,
                ..
            } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }

    /// Returns the `Aabb` of the right child node.
    pub fn child_l_aabb(&self) -> Aabb<T, D> {
        match *self {
            BvhNode::Node { child_l_aabb, .. } => child_l_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the [`Aabb`] of the left child node.
    pub fn child_l_aabb_mut(&mut self) -> &mut Aabb<T, D> {
        match *self {
            BvhNode::Node {
                ref mut child_l_aabb,
                ..
            } => child_l_aabb,
            _ => panic!("Tried to get the left child's `Aabb` of a leaf node."),
        }
    }

    /// Returns the index of the right child node.
    pub fn child_r(&self) -> usize {
        match *self {
            BvhNode::Node { child_r_index, .. } => child_r_index,
            _ => panic!("Tried to get the right child of a leaf node."),
        }
    }

    /// Returns the index of the right child node.
    pub fn child_r_mut(&mut self) -> &mut usize {
        match *self {
            BvhNode::Node {
                ref mut child_r_index,
                ..
            } => child_r_index,
            _ => panic!("Tried to get the right child of a leaf node."),
        }
    }

    /// Returns the [`Aabb`] of the right child node.
    pub fn child_r_aabb(&self) -> Aabb<T, D> {
        match *self {
            BvhNode::Node { child_r_aabb, .. } => child_r_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the [`Aabb`] of the right child node.
    pub fn child_r_aabb_mut(&mut self) -> &mut Aabb<T, D> {
        match *self {
            BvhNode::Node {
                ref mut child_r_aabb,
                ..
            } => child_r_aabb,
            _ => panic!("Tried to get the right child's `Aabb` of a leaf node."),
        }
    }

    /// Gets the [`Aabb`] for a [`BvhNode`].
    /// Returns the shape's [`Aabb`] for leaves, and the joined [`Aabb`] of
    /// the two children's [`Aabb`]'s for non-leaves.
    pub fn get_node_aabb<Shape: BHShape<T, D>>(&self, shapes: &[Shape]) -> Aabb<T, D> {
        match *self {
            BvhNode::Node {
                child_l_aabb,
                child_r_aabb,
                ..
            } => child_l_aabb.join(&child_r_aabb),
            BvhNode::Leaf { shape_index, .. } => shapes[shape_index].aabb(),
        }
    }

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or [`None`] if it is an interior node.
    pub fn shape_index(&self) -> Option<usize> {
        match *self {
            BvhNode::Leaf { shape_index, .. } => Some(shape_index),
            _ => None,
        }
    }

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or `None` if it is an interior node.
    pub fn shape_index_mut(&mut self) -> Option<&mut usize> {
        match *self {
            BvhNode::Leaf {
                ref mut shape_index,
                ..
            } => Some(shape_index),
            _ => None,
        }
    }
}
