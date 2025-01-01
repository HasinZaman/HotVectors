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

use super::serialization::{PartitionGraphSerial, PartitionSerial};
use component::{
    data_buffer::DataBuffer,
    graph::{InterPartitionGraph, IntraPartitionGraph},
    serial::FileExtension,
};
use heapify::{make_heap, make_heap_iter, make_heap_with, pop_heap_with, HeapIterator};
use meta::Meta;
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

//partition data types
pub mod component;

// atomic/async operations
pub mod add;
pub mod merge;
pub mod meta;
pub mod split;

pub enum Error {
    FileDoesNotExist,
    NotEnoughSpace,
    AllLocksInUse,
    NotInMemory,
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::FileDoesNotExist
    }
}
impl From<rancor::Error> for Error {
    fn from(value: rancor::Error) -> Self {
        todo!()
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VectorId(Uuid);

impl Deref for VectorId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PartitionId(Uuid);

impl Deref for PartitionId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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

#[derive(Debug)]
pub enum PartitionErr {
    Overflow,
    VectorNotFound,
    PartitionEmpty,
    InsufficientSizeForSplits,
}

#[derive(Debug, Clone, Copy)]
pub struct Partition<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
> {
    pub size: usize,

    pub vectors: [Option<VectorEntry<A, B>>; PARTITION_CAP],
    pub centroid: B,

    pub id: Uuid,
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Into<Uuid> for &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    fn into(self) -> Uuid {
        self.id
    }
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    pub fn new() -> Self
    where
        B: Extremes,
    {
        Partition {
            size: 0usize,
            vectors: [None; PARTITION_CAP],
            centroid: B::additive_identity(),
            id: Uuid::new_v4(),
        }
    }

    pub fn centroid(&self) -> B {
        B::scalar_mult(
            &self.centroid,
            &A::div(&A::multiplicative_identity(), &A::from_usize(self.size)),
        )
    }

    pub fn remove_by_vector(&mut self, value: B) -> Result<VectorEntry<A, B>, PartitionErr>
    where
        B: PartialEq,
    {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let index = {
            let mut index = 0;
            let mut found = false;
            for i in 0..self.size {
                if self.vectors[i].unwrap().vector == value {
                    found = true;
                    index = i;

                    break;
                }
            }

            if !found {
                return Err(PartitionErr::VectorNotFound);
            }

            index
        };

        let mut removed_vec = None;

        mem::swap(&mut removed_vec, &mut self.vectors[index]);

        for i in index..self.size {
            match i + 1 == self.size {
                true => {
                    self.vectors[i] = None;
                }
                false => {
                    self.vectors[i] = self.vectors[i + 1];
                }
            }
        }
        self.size -= 1;

        Ok(removed_vec.unwrap())
    }
    pub fn remove_by_id(&mut self, id: Uuid) -> Result<VectorEntry<A, B>, PartitionErr> {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let index = {
            let mut index = 0;
            let mut found = false;
            for i in 0..self.size {
                if self.vectors[i].unwrap().id == id {
                    found = true;
                    index = i;

                    break;
                }
            }

            if !found {
                return Err(PartitionErr::VectorNotFound);
            }

            index
        };

        let mut removed_vec = None;

        mem::swap(&mut removed_vec, &mut self.vectors[index]);

        for i in index..self.size {
            match i + 1 == self.size {
                true => {
                    self.vectors[i] = None;
                }
                false => {
                    self.vectors[i] = self.vectors[i + 1];
                }
            }
        }
        self.size -= 1;

        Ok(removed_vec.unwrap())
    }

    pub fn pop(&mut self) -> Result<VectorEntry<A, B>, PartitionErr> {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let index = self.size - 1;

        let mut removed_vec = None;

        mem::swap(&mut removed_vec, &mut self.vectors[index]);

        for i in index..self.size {
            match i + 1 == self.size {
                true => {
                    self.vectors[i] = None;
                }
                false => {
                    self.vectors[i] = self.vectors[i + 1];
                }
            }
        }
        self.size -= 1;

        Ok(removed_vec.unwrap())
    }

    pub fn remove_by_func(
        &mut self,
        func: fn(&VectorEntry<A, B>) -> bool,
    ) -> Result<(), PartitionErr> {
        if self.size == 0 {
            return Ok(());
        }

        todo!();
    }

    pub fn split(
        &mut self,
        graph_1: &mut IntraPartitionGraph<A>,
        inter_graph: &mut InterPartitionGraph<A>,
    ) -> Result<
        (
            Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
            IntraPartitionGraph<A>,
        ),
        PartitionErr,
    >
    where
        A: PartialOrd + Ord,
        B: Extremes,
    {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let mut new_partition = Partition::new();
        let mut new_graph = IntraPartitionGraph::new(PartitionId(new_partition.id));

        todo!();
        // let _ = new_partition.add(self.pop().unwrap(), &mut new_graph);

        if cfg!(feature = "gpu_processing") {
            todo!()
        } else {
            let traitors = (0..self.size)
                .map(|i| {
                    (
                        i,
                        B::dist(&self.vectors[i].unwrap().vector, &self.centroid),
                        B::dist(&self.vectors[i].unwrap().vector, &new_partition.centroid),
                    )
                })
                .filter(|(_i, old_dist, new_dist)| new_dist < old_dist)
                .map(|(i, _old_dist, _new_dist)| i)
                .collect::<Vec<usize>>();

            // move values from graph old -> new
            todo!();

            traitors
                .iter()
                .enumerate()
                .map(|(i1, i2)| (i1, i2 + 1))
                .for_each(|(i1, i2)| {
                    mem::swap(&mut self.vectors[i1], &mut new_partition.vectors[i2]);
                });
            new_partition.size = 1 + traitors.len();

            self.fix_holes();

            self.size = self.size - traitors.len();
        }

        Ok((new_partition, new_graph))
    }

    fn fix_holes(&mut self) {
        for i1 in 0..self.size {
            if self.vectors[i1].is_some() {
                continue;
            }

            let mut next_pos = i1;

            while self.vectors[next_pos].is_none() && next_pos < self.size {
                next_pos += 1;
            }

            if next_pos >= self.size {
                break;
            }
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = &VectorEntry<A, B>> + '_> {
        Box::new(self.vectors.iter().take(self.size).map(vec_unwrap))
    }
}

fn vec_unwrap<'a, A: Field<A>, B: VectorSpace<A>>(
    vector: &'a Option<VectorEntry<A, B>>,
) -> &'a VectorEntry<A, B> {
    match vector {
        Some(vector) => vector,
        None => panic!(),
    }
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Index<usize> for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    type Output = VectorEntry<A, B>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.size);

        match self.vectors.get(index) {
            Some(Some(vector)) => vector,
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorEntry<A: Field<A>, B: VectorSpace<A> + Sized> {
    pub vector: B,
    pub id: Uuid,

    _phantom_data: PhantomData<A>,
}

impl<A: Field<A>, B: VectorSpace<A> + Sized> VectorEntry<A, B> {
    pub fn from_str_id(vector: B, id: &str) -> Self {
        Self {
            vector: vector,
            id: Uuid::from_str(id).unwrap(),
            _phantom_data: PhantomData,
        }
    }
    pub fn from_uuid(vector: B, id: Uuid) -> Self {
        Self {
            vector: vector,
            id: id,
            _phantom_data: PhantomData,
        }
    }
}
