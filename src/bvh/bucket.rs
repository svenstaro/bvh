use alloc::vec::Vec;

use crate::bvh::ShapeIndex;

pub const NUM_BUCKETS: usize = 6;
pub type BucketArray = [Vec<ShapeIndex>; NUM_BUCKETS];

#[cfg(feature = "std")]
mod inner {
    use std::{cell::RefCell, thread_local};

    use super::BucketArray;

    thread_local! {
        /// Thread local for the buckets used while building to reduce allocations during build
        pub static BUCKETS: RefCell<BucketArray> = RefCell::new(Default::default());
    }

    pub fn with_buckets<R>(closure: impl FnOnce(&mut BucketArray) -> R) -> R {
        BUCKETS.with(move |buckets| {
            let mut buckets = buckets.borrow_mut();
            closure(&mut buckets)
        })
    }
}

#[cfg(not(feature = "std"))]
mod inner {
    use alloc::vec::Vec;

    use crate::bvh::{ShapeIndex, bucket::BucketArray};

    pub fn with_buckets<R>(closure: impl FnOnce(&mut BucketArray) -> R) -> R {
        const EMPTY: Vec<ShapeIndex> = Vec::new();
        closure(&mut [EMPTY; 6])
    }
}

// re-export specific implementation
pub use inner::*;
