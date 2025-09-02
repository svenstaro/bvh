use crate::bvh::ShapeIndex;
use alloc::vec::Vec;

pub const NUM_BUCKETS: usize = 6;
pub type BucketArray = [Vec<ShapeIndex>; NUM_BUCKETS];

#[cfg(feature = "std")]
mod inner {
    use super::BucketArray;
    use std::cell::RefCell;

    thread_local! {
        /// Thread local for the buckets used while building to reduce allocations during build
        pub static BUCKETS: RefCell<BucketArray> = RefCell::new(Default::default());
    }
}

#[cfg(not(feature = "std"))]
mod inner {
    use crate::bvh::bucket::BucketArray;
    use alloc::vec::Vec;

    pub fn alloc_buckets() -> BucketArray {
        [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]
    }
}

// re-export specific implementation
pub use inner::*;
