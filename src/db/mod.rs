use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
};

use crate::{db::operations::LoadedPartitions, vector::VectorSerial};
use component::{
    data_buffer::DataBuffer,
    graph::{GraphSerial, InterPartitionGraph, IntraPartitionGraph},
    meta::Meta,
    partition::{Partition, PartitionSerial, VectorEntry},
};
use log::{Log, State};
use operations::{add::add, read::stream_meta_data};
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    ptr_meta::metadata,
    rend::u32_le,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    vec, Archive, DeserializeUnsized,
};
use tokio::{
    runtime,
    sync::{oneshot, Mutex, RwLock},
};
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSpace};

pub mod component;
pub mod log;
pub mod operations;

#[derive(Clone)]
pub enum AtomicCmd<A: Field<A>, B: VectorSpace<A> + Sized> {
    // transaction commands
    StartTransaction(Uuid),
    EndTransaction(Uuid),
    UndoTransaction(Uuid),

    // Write
    InsertVector {
        vector: B,
        transaction_id: Option<Uuid>,
    },
    DeleteVector {
        uuid: Uuid,
        transaction_id: Option<Uuid>,
    },

    // projections
    GetIds {
        transaction_id: Option<Uuid>,
    },
    GetVectors {
        transaction_id: Option<Uuid>,
    },
    GetMetaData {
        transaction_id: Option<Uuid>,
    },

    // Filters
    Knn {
        transaction_id: Option<Uuid>,
    },
    SelectedUUID {
        transaction_id: Option<Uuid>,
    },

    PhantomData(PhantomData<A>),
}

#[derive(Clone)]
pub struct ChainedCmd<A: Field<A>, B: VectorSpace<A> + Sized>(Vec<AtomicCmd<A, B>>);

#[derive(Clone)]
pub enum Cmd<A: Field<A>, B: VectorSpace<A> + Sized> {
    Atomic(AtomicCmd<A, B>),
    Chained(ChainedCmd<A, B>),
}

#[derive(Clone, Debug)]
pub enum Response<A: Clone + Copy> {
    Success(Success<A>),
    Fail,
    Done,
}

#[derive(Clone, Debug)]
pub enum Success<A: Clone + Copy> {
    MetaData(usize, VectorSerial<A>),
}

pub fn db_loop<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Ord
        + Field<A>
        + Send
        + Sync
        + 'static
        + rkyv::Archive
        + for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        >,
    B: VectorSpace<A>
        + Sized
        + Clone
        + Copy
        + Send
        + Sync
        + Sized
        + From<VectorSerial<A>>
        + Extremes
        + PartialEq
        + 'static,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    // load data buffers
    partition_buffer: DataBuffer<
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        PartitionSerial<A>,
        MAX_LOADED,
    >,
    min_spanning_tree_buffer: DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, MAX_LOADED>,

    // load persistent data
    inter_spanning_graph: InterPartitionGraph<A>,
    meta_data: HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,

    // input channel
    cmd_input: Receiver<(Cmd<A, B>, Sender<Response<A>>)>,
    logger: Sender<State<AtomicCmd<A, B>>>,
    // log: Log<A, B, 5000>,
) -> !
where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,

    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    VectorSerial<A>: From<B>,
{
    // initialize internal data
    let partition_buffer = Arc::new(RwLock::new(partition_buffer));
    let min_spanning_tree_buffer = Arc::new(RwLock::new(min_spanning_tree_buffer));
    let inter_spanning_graph = Arc::new(RwLock::new(inter_spanning_graph));
    let meta_data = Arc::new(RwLock::new(meta_data));

    // initialize locks
    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();

    rt.block_on(async {
        loop {
            let (cmd, sender) = cmd_input.recv().unwrap();

            match cmd {
                Cmd::Atomic(atomic_cmd) => {
                    match atomic_cmd {
                        AtomicCmd::StartTransaction(uuid) => {
                            // send to write log
                            todo!()
                        }
                        AtomicCmd::EndTransaction(uuid) => {
                            // write to log
                            todo!()
                        }
                        AtomicCmd::UndoTransaction(uuid) => {
                            // kill all affected read threads
                            // undo all write threads
                            todo!()
                        }
                        AtomicCmd::InsertVector {
                            vector,
                            transaction_id,
                        } => {
                            let inter_spanning_graph = inter_spanning_graph.clone();

                            let partition_buffer = partition_buffer.clone();
                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

                            let meta_data = meta_data.clone();

                            rt.spawn(async move {
                                let value = VectorEntry::from_uuid(vector, Uuid::new_v4());

                                match add(
                                    value,
                                    inter_spanning_graph,
                                    partition_buffer,
                                    min_spanning_tree_buffer,
                                    meta_data,
                                )
                                .await
                                {
                                    Ok(_) => {
                                        // log success
                                        // send success
                                    }
                                    Err(_) => todo!(),
                                };

                                // let _ = sender.send(Response::Success);
                                let _ = sender.send(Response::Done);
                            });
                        }
                        AtomicCmd::DeleteVector {
                            uuid,
                            transaction_id,
                        } => {
                            // create new write thread
                            todo!()
                        }

                        AtomicCmd::GetIds { transaction_id } => {
                            // read all ids from partitions
                            todo!()
                        }
                        AtomicCmd::GetVectors { transaction_id } => {
                            // read all vectors from partitions
                            todo!()
                        }
                        AtomicCmd::GetMetaData { transaction_id } => {
                            let meta_data = meta_data.clone();

                            rt.spawn(async move {
                                let _ = stream_meta_data(meta_data, &sender).await;

                                let _ = sender.send(Response::Done);
                            });
                        }

                        AtomicCmd::Knn { transaction_id } => {
                            // read all vectors and maybe a cache file
                            todo!()
                        }
                        AtomicCmd::SelectedUUID { transaction_id } => {
                            // create a cache of vectors in Uuid
                            todo!()
                        }
                        AtomicCmd::PhantomData(phantom_data) => panic!(""),
                    }
                }
                Cmd::Chained(chained_cmd) => {
                    todo!()
                }
            }
        }
    });
    panic!();
}
