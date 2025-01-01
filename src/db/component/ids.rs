use std::ops::Deref;

use uuid::Uuid;

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VectorId(pub Uuid);

impl Deref for VectorId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PartitionId(pub Uuid);

impl Deref for PartitionId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
