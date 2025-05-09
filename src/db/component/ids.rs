use std::ops::Deref;

use uuid::Uuid;

macro_rules! ids {
    [$head: tt , $($tail: tt),+ ] => {
        ids!($head);
        ids![$($tail),+];
    };
    [$head: tt] => {
        #[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Ord)]
        pub struct $head(pub Uuid);

        impl Deref for $head {
            type Target = Uuid;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl PartialOrd for $head {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }
    };
}

ids![ClusterId, VectorId, PartitionId];
