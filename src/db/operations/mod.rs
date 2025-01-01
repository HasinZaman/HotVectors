use std::{
    array,
    collections::{HashMap, HashSet},
    hash::Hash,
    marker::PhantomData,
    mem,
    ops::{Deref, Index},
    str::FromStr,
    sync::{mpsc::Sender, Arc, Condvar, Mutex},
};

use std::sync::RwLock as StdRwLock;

use heapify::{make_heap, make_heap_iter, make_heap_with, pop_heap_with, HeapIterator};
use petgraph::{
    csr::DefaultIx,
    graph::{EdgeIndex, Node, NodeIndex},
    prelude::StableGraph,
    stable_graph::Edges,
    visit::{EdgeRef, NodeRef},
    Graph, Undirected,
};
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    from_bytes, rancor,
    rend::u32_le,
    to_bytes,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, DeserializeUnsized,
};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSerial, VectorSpace};

use super::component::{graph::InterPartitionGraph, meta::Meta};

// atomic/async operations
pub mod add;
pub mod merge;
pub mod split;

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
    pub meta_data: TokioRwLock<HashMap<Uuid, TokioRwLock<Meta<A, B>>>>,
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
        //     loaded: TokioRwLock::new(0),
        //     partitions: array::from_fn(|_| TokioRwLock::new(Arc::new(StdRwLock::new(None)))),
        //     load_state: array::from_fn(|_| TokioRwLock::new(None)),

        //     index_map: TokioRwLock::new(HashMap::new()),
        //     inter_partition_graph: TokioRwLock::new(InterPartitionGraph::new()), //
        //     meta_data: TokioRwLock::new(
        //         Meta::load_from_folder()
        //             .into_iter()
        //             .map(|x| (*x.id, TokioRwLock::new(x)))
        //             .collect::<HashMap<Uuid, TokioRwLock<Meta<A, B>>>>(),
        //     ), // Replace by going through partitions/meta
        // }
    }
}
