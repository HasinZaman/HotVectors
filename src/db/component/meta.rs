use std::{collections::HashMap, marker::PhantomData};

use tokio::sync::RwLock;

use crate::vector::{Field, VectorSpace};

use super::ids::PartitionId;

#[derive(Clone)]
pub struct Meta<A: Field<A>, B: VectorSpace<A> + Clone> {
    pub id: PartitionId,
    pub size: usize,
    pub centroid: B,

    pub _phantom_data: PhantomData<A>,
}

impl<A: Field<A>, B: VectorSpace<A> + Clone> Meta<A, B> {
    pub fn new(id: PartitionId, size: usize, centroid: B) -> Self {
        todo!()
    }

    pub fn load_from_folder() -> Vec<Self> {
        todo!()
    }
}
