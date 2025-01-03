use std::collections::HashMap;

use tokio::sync::RwLock;
use uuid::Uuid;

use crate::vector::{Field, VectorSerial, VectorSpace};

use super::component::{graph::InterPartitionGraph, meta::Meta};

// atomic/async operations
pub mod add;
pub mod knn;
pub mod merge;
pub mod split;

pub mod read;

const SOURCE_DIR: &str = "partitions";
const META_DIR: &str = "partitions";

pub struct LoadedPartitions<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
> {
    source: String,
    // pub loaded_partitions: DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, MAX_LOADED>,
    // pub loaded_min_spanning_tree: DataBuffer<IntraPartitionGraph<A>, MAX_LOADED>,
    pub inter_graph: InterPartitionGraph<A>,
    pub meta_data: RwLock<HashMap<Uuid, RwLock<Meta<A, B>>>>,
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
        const MAX_LOADED: usize,
    > LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>
{
    pub fn new() -> Self {
        todo!()
        // LoadedPartitions {
        //     loaded: RwLock::new(0),
        //     partitions: array::from_fn(|_| RwLock::new(Arc::new(StdRwLock::new(None)))),
        //     load_state: array::from_fn(|_| RwLock::new(None)),

        //     index_map: RwLock::new(HashMap::new()),
        //     inter_partition_graph: RwLock::new(InterPartitionGraph::new()), //
        //     meta_data: RwLock::new(
        //         Meta::load_from_folder()
        //             .into_iter()
        //             .map(|x| (*x.id, RwLock::new(x)))
        //             .collect::<HashMap<Uuid, RwLock<Meta<A, B>>>>(),
        //     ), // Replace by going through partitions/meta
        // }
    }
}
