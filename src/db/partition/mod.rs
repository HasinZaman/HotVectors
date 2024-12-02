use std::{
    array,
    collections::{HashMap, HashSet},
    marker::PhantomData,
    mem,
    ops::{Deref, Index},
    str::FromStr,
    sync::{mpsc::Sender, Arc, Condvar, Mutex},
};

use std::sync::RwLock as StdRwLock;

use super::serialization::{PartitionGraphSerial, PartitionSerial};
use petgraph::{
    csr::DefaultIx,
    graph::{EdgeIndex, NodeIndex},
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

pub mod add;

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

#[derive(Debug, Clone, Copy)]
pub enum PartitionState {
    Read(usize),
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
    loaded: TokioRwLock<usize>,
    partitions: [TokioRwLock<
        Arc<
            StdRwLock<
                Option<(
                    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
                    IntraPartitionGraph<A>,
                )>,
            >,
        >,
    >; MAX_LOADED],
    load_state: [TokioRwLock<Option<(usize, usize)>>; MAX_LOADED],

    // internal_graphs: [Option<Box<PartitionGraph<A, Internal>>>; MAX_LOADED],

    // partitions: [Option<Partition<A, B, PARTITION_CAP, VECTOR_CAP>>; MAX_LOADED],
    // partitions: [Option<Partition<A, B, PARTITION_CAP, VECTOR_CAP>>; MAX_LOADED],
    hash_map: TokioRwLock<HashMap<Uuid, usize>>,
}

#[derive(Debug, Default, Archive, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VectorId(Uuid);

impl Deref for VectorId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Default, Archive, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PartitionId(Uuid);

impl Deref for PartitionId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Default, Archive)]
pub struct InterPartitionGraph<A: Field<A>>(
    pub StableGraph<PartitionId, (A, VectorId, VectorId), Undirected, DefaultIx>,
    pub HashMap<PartitionId, NodeIndex<DefaultIx>>,
);

#[derive(Debug, Archive)]
pub struct IntraPartitionGraph<A: Field<A>>(
    pub StableGraph<VectorId, A, Undirected, DefaultIx>,
    pub HashMap<VectorId, NodeIndex<DefaultIx>>,
);

impl<A: Field<A>> Default for IntraPartitionGraph<A> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

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
            loaded: TokioRwLock::new(0),
            partitions: array::from_fn(|_| TokioRwLock::new(Arc::new(StdRwLock::new(None)))),
            load_state: array::from_fn(|_| TokioRwLock::new(None)),

            hash_map: TokioRwLock::new(HashMap::new()),
        }
    }

    pub async fn load(&mut self, id: &Uuid) -> Result<(), Error>
    where
        A: Archive,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
        [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,

        [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
            DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    {
        if *self.loaded.read().await >= PARTITION_CAP {
            return Err(Error::NotEnoughSpace);
        }

        // find earliest empty value
        let i1 = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_index, partition)| match partition.try_read() {
                Ok(partition) => {
                    let Ok(partition) = (**partition).read() else {
                        return false;
                    };
                    match *partition {
                        Some(_) => false,
                        None => true,
                    }
                }
                Err(_err) => false,
            })
            .map(|(index, _partition)| index)
            .next();

        let Some(i1) = i1 else {
            return Err(Error::AllLocksInUse);
        };

        //get access to RwLock
        let partition_block = self.partitions[i1].write().await;
        let load_state = self.load_state[i1].write().await;
        let mut loaded = self.loaded.write().await;

        //load files
        let new_partition: Partition<A, B, PARTITION_CAP, VECTOR_CAP> = {
            let bytes = tokio::fs::read(&format!("{SOURCE_DIR}//{id}.partition")).await?;

            from_bytes::<PartitionSerial<A>, rancor::Error>(&bytes)?.into()
        };
        let new_internal_graph: IntraPartitionGraph<A> = {
            let bytes = tokio::fs::read(&format!("{SOURCE_DIR}//{id}.graph")).await?;

            from_bytes::<PartitionGraphSerial<A>, rancor::Error>(&bytes)?.into()
        };
        //assign values
        let partition_block = partition_block.write();
        let mut partition_block = partition_block.unwrap();
        partition_block.replace((new_partition, new_internal_graph));

        let mut load_state = *load_state;
        load_state.replace((0usize, 0usize));

        *loaded += 1;

        Ok(())
    }
    pub async fn unload_at_index(&mut self, index: usize) -> Result<(), Error>
    where
        VectorSerial<A>: From<B>,
        A: for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        >, // for<'a> rkyv::Serialize<Strategy<rkyv::ser::Serializer<AlignedVec, ArenaHandle<'a>, Share>, _>>
    {
        let partition_block = self.partitions[index].write().await;
        let mut hash_map = self.hash_map.write().await;
        let load_state = self.load_state[index].write().await;
        let mut loaded = self.loaded.write().await;

        let id = {
            let (partition, graph) = {
                let mut tmp_value = None;

                let mut partition_block = partition_block.write().unwrap();

                mem::swap(&mut tmp_value, &mut partition_block);

                match tmp_value {
                    Some(value) => value,
                    None => {
                        return Err(Error::NotInMemory);
                    }
                }
            };

            // serialize & save data
            {
                let partition_serial: PartitionSerial<A> = partition.into();
                let bytes = to_bytes::<rancor::Error>(&partition_serial)?;

                tokio::fs::write(
                    &format!("{SOURCE_DIR}//{}.partition", partition.id.to_string()),
                    bytes.as_slice(),
                )
                .await?;
            }
            {
                let graph_serial: PartitionGraphSerial<A> = graph.into();
                let bytes = to_bytes::<rancor::Error>(&graph_serial)?;

                tokio::fs::write(
                    &format!("{SOURCE_DIR}//{}.graph", partition.id.to_string()),
                    bytes.as_slice(),
                )
                .await?;
            }

            let mut load_state = *load_state;
            mem::replace(&mut load_state, None);

            partition.id
        };

        hash_map.remove(&id);

        *loaded -= 1;

        Ok(())
    }

    pub async fn access(
        &mut self,
        id: &Uuid,
        // tx: &mut Sender<(
        //     Arc<
        //         StdRwLock<
        //             Option<(
        //                 Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        //                 IntraPartitionGraph<A>,
        //             )>,
        //         >,
        //     >,
        //     Arc<(Condvar, Mutex<bool>)>,
        // )>,
    ) -> Result<
        Arc<
            StdRwLock<
                Option<(
                    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
                    IntraPartitionGraph<A>,
                )>,
            >,
        >,
        Error,
    > {
        let index = {
            let hash_map = self.hash_map.read().await;

            match hash_map.get(&id) {
                Some(index) => *index,
                None => return Err(Error::NotInMemory),
            }
        };
        let mut use_count = self.load_state[index].write().await;

        if let None = *use_count {
            // invalid state
            todo!()
        };

        if let Some(use_count) = use_count.as_mut() {
            use_count.1 = use_count.1 + 1;
        };

        Ok(self.partitions[index].read().await.clone())
    }

    pub fn least_used(&self) -> Option<usize> {
        let mut iter = self
            .load_state
            .iter()
            .map(|x| x.try_read())
            .enumerate()
            .filter(|(_index, x)| x.is_ok())
            .map(|(index, x)| (index, x.unwrap()))
            .filter(|(_index, x)| x.is_some())
            .map(|(index, x)| (index, x.unwrap()));

        let Some(head) = iter.next() else {
            return None;
        };

        Some(
            iter.fold(
                head,
                |(acc_index, (acc_prev, acc_cur)), (next_index, (next_prev, next_cur))| {
                    if next_prev + next_cur < acc_prev + acc_cur {
                        (next_index, (next_prev, next_cur))
                    } else {
                        (acc_index, (acc_prev, acc_cur))
                    }
                },
            )
            .0,
        )
    }

    pub async fn de_increment(&mut self) {
        for x in self.load_state.iter_mut() {
            let x = x.write().await;

            if x.is_none() {
                continue;
            }

            let mut x = x.unwrap();

            let replace = (x.1, 0usize);
            let _ = mem::replace(&mut x, replace);
        }
    }

    pub async fn insert_partition(
        &mut self,
        new_partition: Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        new_partition_graph: IntraPartitionGraph<A>,
    ) -> Result<(), Error> {
        if *self.loaded.read().await >= PARTITION_CAP {
            return Err(Error::NotEnoughSpace);
        }

        let index_load_state_pair = self
            .load_state
            .iter()
            .enumerate()
            .map(|(index, state)| (index, state.try_read()))
            .filter(|(_index, state)| state.is_ok())
            .map(|(index, state)| (index, state.unwrap()))
            .filter(|(_index, state)| state.is_none())
            .map(|(index, _state)| index)
            .next();

        let Some(index) = index_load_state_pair else {
            return Err(Error::AllLocksInUse);
        };

        let partition_block = self.partitions[index].write().await;
        let mut load_state = self.load_state[index].write().await;

        let mut partition_block = (*partition_block).write().unwrap();
        partition_block.replace((new_partition, new_partition_graph));

        (*load_state).replace((0usize, 0usize));

        Ok(())
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
    pub size: usize,

    pub vectors: [Option<VectorEntry<A, B>>; PARTITION_CAP],
    pub centroid: B,

    pub id: Uuid,
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

    pub fn add(
        &mut self,
        value: VectorEntry<A, B>,
        intra_graph: &mut IntraPartitionGraph<A>,

        inter_graph: &mut InterPartitionGraph<A>,

        neighbors: &[(&Self, &mut IntraPartitionGraph<A>)],
    ) -> Result<(), PartitionErr>
    where
        A: PartialOrd + Ord,
    {
        if self.size + 1 >= PARTITION_CAP {
            return Err(PartitionErr::Overflow);
        };

        self.vectors[self.size] = Some(value);
        self.size += 1;

        self.centroid = B::add(&self.centroid, &value.vector);

        // insert node into a minimum spanning tree
        {
            //need to get distance of all neighbor vectors
            let idx = intra_graph.0.add_node(value.id);
            intra_graph.1.insert(value.id, idx);

            let partition_splits = [self.size]
                .iter()
                .chain(neighbors.iter().map(|x| &x.0.size))
                .collect::<Vec<&usize>>();

            let dist = self
                .vectors
                .iter()
                .chain(neighbors.iter().map(|x| x.0.vectors.iter()).flatten())
                .filter_map(|x| *x)
                .enumerate()
                .map(|(index, x)| (x.id, (index, B::dist(&x.vector, &value.vector))))
                .collect::<HashMap<Uuid, (usize, A)>>();

            // find closet inserted vector
            let (id, index, _dist) = {
                let Some((id, (index, _dist))) = dist
                    .iter()
                    // .enumerate()
                    .min_by(|(_, (_, x1)), (_, (_, x2))| x1.cmp(x2))
                else {
                    todo!()
                };

                (*id, *index, *_dist)
            };

            // determine search area

            // two list
            //  -> internal nodes
            //      -> new intra-graph edge
            //      -> destroy intra-graph edge
            //  -> external nodes
            //      -> new inter-graph edge
            //      -> destroy foreign intra-graph edge
            //      -> destroy inter-graph edge

            todo!();
            if &index < partition_splits[0] {
                let start_index = intra_graph.1[&id];

                let neighbors = intra_graph
                    .0
                    .edges(start_index)
                    .map(|x| [(x.target(), x.weight()), (x.source(), x.weight())])
                    .flatten()
                    .filter(|(x, _)| x != &start_index)
                    .map(|(x, y)| (*intra_graph.0.node_weight(x).unwrap(), *y))
                    .collect::<HashMap<Uuid, A>>();

                let replace_edges = dist
                    .iter()
                    .filter(|(id, _)| neighbors.contains_key(id))
                    .filter(|(id, (_, x))| neighbors[id] > *x)
                    .filter_map(|(id, (_, dist))| {
                        intra_graph
                            .0
                            .find_edge(intra_graph.1[&id], start_index)
                            .map(|edge| (*id, edge, *dist))
                    })
                    .collect::<Vec<(Uuid, EdgeIndex, A)>>();

                replace_edges
                    .into_iter()
                    .for_each(|(target, remove_edge, dist)| {
                        intra_graph.0.add_edge(intra_graph.1[&target], idx, dist);
                        intra_graph.0.remove_edge(remove_edge);
                    });

                intra_graph.0.add_edge(idx, start_index, _dist);

                //check if any of the replaced affected vertex are
                todo!()
            } else {
            }

            // if index in (0, partition_splits[0]) -> home shit
            // if index (partition_splits[0], partition_splits[1]) -> closet node in another partition
        }

        Ok(())
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
        let mut new_graph = IntraPartitionGraph::default();

        todo!();
        let _ = new_partition.add(self.pop().unwrap(), &mut new_graph);

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
    type Output = Option<VectorEntry<A, B>>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index >= self.size);

        &self.vectors[index]
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
