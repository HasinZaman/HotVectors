use std::ops::Deref;

use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

macro_rules! ids {
    [$head: tt , $($tail: tt),+ ] => {
        ids!($head);
        ids![$($tail),+];
    };
    [$head: tt] => {
        #[derive(Archive, Serialize, Deserialize, Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Ord)]
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

        impl AsRef<[u8]> for $head {
            fn as_ref(&self) -> &[u8] {
                self.0.as_bytes()
            }
        }

        impl $head {
            /// Returns a `[u8; 16]` representing the UUID.
            pub fn to_bytes(self) -> [u8; 16] {
                *self.0.as_bytes()
            }
        }

    };
}

ids![ClusterId, VectorId, PartitionId];
