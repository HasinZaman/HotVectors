use std::collections::HashMap;

use tokio::sync::RwLock;
use uuid::Uuid;

use crate::db::partition::PartitionErr;

pub struct LeastUsedIterator(Vec<(f32, usize, Uuid)>);

impl LeastUsedIterator {
    pub fn new(data: Vec<(f32, usize, Uuid)>) -> Self {
        todo!()
    }
}

impl Iterator for LeastUsedIterator {
    type Item = (usize, Uuid);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct DataBuffer<A: Clone + Sized, const CAP: usize>
where
    for<'a> &'a A: Into<Uuid>,
{
    source: String,

    buffer: [RwLock<Option<A>>; CAP],
    used_stack: [RwLock<Option<(usize, usize)>>; CAP],
    pub buffer_size: RwLock<usize>,

    empty_index_stack: RwLock<[Option<usize>; CAP]>,
    empty_index_size: RwLock<usize>,

    index_map: RwLock<HashMap<Uuid, usize>>,
}

impl<A: Clone + Sized, const CAP: usize> DataBuffer<A, CAP>
where
    for<'a> &'a A: Into<Uuid>,
{
    pub fn new() -> Self {
        todo!()
    }
    pub fn try_access(&mut self, id: &Uuid) -> Result<RwLock<A>, PartitionErr> {
        todo!()
    }

    pub async fn max_try_access<const MAX: usize>(
        &mut self,
        id: &Uuid,
    ) -> Result<RwLock<A>, PartitionErr> {
        todo!()
    }

    pub async fn access(&mut self, id: &Uuid) -> Result<RwLock<A>, PartitionErr> {
        todo!()
    }

    fn least_used(&self) -> Option<usize> {
        todo!()
    }

    pub fn least_used_iter(&self) -> Option<LeastUsedIterator> {
        todo!()
    }

    pub fn remove(&mut self, id: &Uuid) -> Result<(), PartitionErr> {
        todo!()
    }

    pub fn unload(&mut self, id: &Uuid) -> Result<(), PartitionErr> {
        todo!()
    }

    pub fn de_increment(&mut self) {
        todo!()
    }

    pub fn contains(&self, id: &Uuid) -> bool {
        todo!()
    }
}
