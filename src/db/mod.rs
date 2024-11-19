use std::{
    array,
    collections::HashMap,
    marker::PhantomData,
    mem,
    ops::Index,
    sync::{
        mpsc::{Receiver, Sender},
        RwLock,
    },
};

use chrono::format::Pad;
use petgraph::{csr::DefaultIx, Graph, Undirected};
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    deserialize, from_bytes, rancor,
    rend::u32_le,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, DeserializeUnsized,
};
use serde::{Deserialize, Serialize};
use serialization::{PartitionGraphSerial, PartitionSerial};
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSerial, VectorSpace};

mod serialization;

fn db_loop() -> ! {
    // initialize partitions
    // load external graphs
    // initialize internal graphs
    // initialize locks
    loop {}
}

fn split_partition<
    'a,
    A: Clone + Copy + PartialEq + Field<A>,
    B: Clone + Copy + VectorSpace<A> + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    partition: &'a mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    internal_graphs: &[&'a mut PartitionGraph<A, Internal>],
    external_graphs: &[&'a mut PartitionGraph<A, External>],
) -> (
    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    Vec<PartitionGraph<A, Internal>>,
) {
    // select random
    let mut new_partition: Partition<A, B, PARTITION_CAP, VECTOR_CAP> = Partition::new();

    // select to random points

    // divide

    todo!()
}

fn merge_partition<
    'a,
    A: Clone + Copy + PartialEq + Field<A>,
    B: Clone + Copy + VectorSpace<A>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    sink_partitions: &'a mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    sink_internal_graphs: &[&'a mut PartitionGraph<A, Internal>],

    source_partitions: &[&'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>],
    source_internal_graphs: &[&[&'a mut PartitionGraph<A, Internal>]],

    external_graphs: &[&'a mut PartitionGraph<A, External>],
) -> (
    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    Vec<PartitionGraph<A, Internal>>,
) {
    // select random
    todo!()
}

async fn update_partition<
    A: Clone + Copy + PartialEq + Field<A>,
    B: Clone + Copy + VectorSpace<A>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    start_partitions: Uuid,
    external_graphs: Vec<PartitionGraph<A, External>>,

    partition_receiver: Receiver<(
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        Vec<PartitionGraph<A, Internal>>,
    )>,
    partition_request: Sender<Uuid>,
) -> (
    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    Vec<PartitionGraph<A, Internal>>,
) {
    todo!()
}

#[derive(Debug, Clone, Copy)]
enum PartitionState {
    Read(bool, bool),
    Write,
    Free,
}

pub struct LoadedPartitions<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
> {
    loaded: RwLock<usize>,
    partitions: [RwLock<
        Option<(
            Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
            PartitionGraph<A, Internal>,
        )>,
    >; MAX_LOADED],
    load_state: [RwLock<Option<PartitionState>>; MAX_LOADED],

    // internal_graphs: [Option<Box<PartitionGraph<A, Internal>>>; MAX_LOADED],

    // partitions: [Option<Partition<A, B, PARTITION_CAP, VECTOR_CAP>>; MAX_LOADED],
    // partitions: [Option<Partition<A, B, PARTITION_CAP, VECTOR_CAP>>; MAX_LOADED],
    hash_map: HashMap<Uuid, usize>,
}

pub enum Error {
    FileDoesNotExist,
    NotEnoughSpace,
    AllLocksInUse,
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

#[derive(Clone, Copy)]
struct Internal(Uuid);

#[derive(Clone, Copy)]
struct External(Uuid);

#[derive(Debug, Archive)]
pub struct PartitionGraph<A: Field<A>, B>(Graph<Uuid, A, Undirected, DefaultIx>, PhantomData<B>);

const SOURCE_DIR: &str = "partitions";
const META_DIR: &str = "partitions";

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
        const MAX_LOADED: usize,
    > LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>
{
    pub fn new() -> Self {
        LoadedPartitions {
            loaded: RwLock::new(0),
            partitions: array::from_fn(|_| RwLock::new(None)),
            load_state: array::from_fn(|_| RwLock::new(None)),

            hash_map: HashMap::new(),
        }
    }

    pub async fn load(&mut self, id: Uuid) -> Result<(), Error>
    where
        A: Archive,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
        [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,

        [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
            DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    {
        if *self.loaded.read().unwrap() >= PARTITION_CAP {
            return Err(Error::NotEnoughSpace);
        }

        // find earliest empty value
        let i1 = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_index, partition)| match partition.try_read() {
                Ok(partition) => match *partition {
                    Some(_) => false,
                    None => true,
                },
                Err(_err) => false,
            })
            .map(|(index, _partition)| index)
            .next();

        let Some(i1) = i1 else {
            return Err(Error::AllLocksInUse);
        };

        //get access to RwLock
        let mut partition_block = self.partitions[i1].write().unwrap();
        let mut partition_block = partition_block.as_mut();

        let mut load_state = self.load_state[i1].write().unwrap();
        let mut load_state = load_state.as_mut();

        //load files
        let new_partition: Partition<A, B, PARTITION_CAP, VECTOR_CAP> = {
            let bytes = tokio::fs::read(&format!("{SOURCE_DIR}//{id}.partition")).await?;

            from_bytes::<PartitionSerial<A>, rancor::Error>(&bytes)?.into()
        };
        let new_internal_graph: PartitionGraph<A, Internal> = {
            let bytes = tokio::fs::read(&format!("{SOURCE_DIR}//{id}.graph")).await?;

            from_bytes::<PartitionGraphSerial<A>, rancor::Error>(&bytes)?.into()
        };

        //assign values
        partition_block.replace(&mut (new_partition, new_internal_graph));
        load_state.replace(&mut PartitionState::Free);

        let mut loaded = *self.loaded.write().unwrap();

        loaded += 1;

        Ok(())
    }
    pub fn unload_by_id(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
    pub fn unload_by_least_used(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
    pub fn unload_by_youngest(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }

    pub fn read(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
}

#[derive(Debug)]
pub enum PartitionErr {
    Overflow,
    VectorNotFound,
    PartitionEmpty,
}

#[derive(Debug, Clone, Copy)]
pub struct Partition<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
> {
    size: usize,

    vectors: [Option<VectorEntry<A, B, VECTOR_CAP>>; PARTITION_CAP],
    centroid: B,

    id: Uuid,
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

    pub fn add(&mut self, value: VectorEntry<A, B, VECTOR_CAP>) -> Result<(), PartitionErr> {
        if self.size + 1 >= PARTITION_CAP {
            return Err(PartitionErr::Overflow);
        };

        self.vectors[self.size] = Some(value);
        self.size += 1;

        self.centroid = B::add(&self.centroid, &value.vector);

        Ok(())
    }

    pub fn remove_by_vector(
        &mut self,
        value: B,
    ) -> Result<VectorEntry<A, B, VECTOR_CAP>, PartitionErr>
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
    pub fn remove_by_id(
        &mut self,
        id: Uuid,
    ) -> Result<VectorEntry<A, B, VECTOR_CAP>, PartitionErr> {
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

    pub fn pop(&mut self) -> Result<VectorEntry<A, B, VECTOR_CAP>, PartitionErr> {
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
        func: fn(&VectorEntry<A, B, VECTOR_CAP>) -> bool,
    ) -> Result<(), PartitionErr> {
        if self.size == 0 {
            return Ok(());
        }

        todo!();
    }

    pub fn split(&mut self) -> Result<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionErr>
    where
        B: Extremes,
    {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let mut new_partition = Partition::new();

        new_partition.add(self.pop().unwrap()).unwrap();

        Ok(new_partition)
    }

    pub fn merge(&mut self, other: Self) -> Result<(), PartitionErr> {
        if self.size + other.size >= VECTOR_CAP {
            return Err(PartitionErr::Overflow);
        }

        (0..other.size)
            .map(|i| (i, other.vectors[i]))
            .for_each(|(i, x)| {
                self.centroid = B::add(&self.centroid, &x.unwrap().vector);

                let _ = mem::replace(&mut self.vectors[self.size + i], x);
            });

        self.size += other.size;

        Ok(())
    }
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Index<usize> for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    type Output = Option<VectorEntry<A, B, VECTOR_CAP>>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index >= self.size);

        &self.vectors[index]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorEntry<A: Field<A>, B: VectorSpace<A> + Sized, const CAP: usize> {
    vector: B,
    id: Uuid,

    _phantom_data: PhantomData<A>,
}
