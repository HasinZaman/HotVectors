use std::ops::Deref;

use uuid::Uuid;

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Ord)]
pub struct VectorId(pub Uuid);

impl Deref for VectorId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialOrd for VectorId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Ord)]
pub struct PartitionId(pub Uuid);

impl Deref for PartitionId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl PartialOrd for PartitionId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
