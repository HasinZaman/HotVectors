use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{
        mpsc::{Receiver, Sender},
        Arc, RwLock as StdRwLock,
    },
};

use crate::{db::operations::LoadedPartitions, vector::VectorSerial};
use log::{Log, State};
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
    sync::{oneshot, Mutex, RwLock as TokioRwLock},
};
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSpace};

pub mod log;
pub mod operations;
pub mod component;

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

    // Filters
    Knn {
        transaction_id: Option<Uuid>,
    },
    SelectedUUID {
        transaction_id: Option<Uuid>,
    },

    PhantomData(PhantomData<A>),
}

pub struct ChainedCmd<A: Field<A>, B: VectorSpace<A> + Sized>(Vec<AtomicCmd<A, B>>);

pub enum Cmd<A: Field<A>, B: VectorSpace<A> + Sized> {
    Atomic(AtomicCmd<A, B>),
    Chained(ChainedCmd<A, B>),
}

// pub fn db_loop<
//     A: PartialEq
//         + PartialOrd
//         + Clone
//         + Copy
//         + Field<A>
//         + Send
//         + Sync
//         + 'static
//         + rkyv::Archive
//         + for<'a> rkyv::Serialize<
//             rancor::Strategy<
//                 rkyv::ser::Serializer<
//                     rkyv::util::AlignedVec,
//                     rkyv::ser::allocator::ArenaHandle<'a>,
//                     rkyv::ser::sharing::Share,
//                 >,
//                 rancor::Error,
//             >,
//         >,
//     B: VectorSpace<A>
//         + Sized
//         + Clone
//         + Copy
//         + Send
//         + Sync
//         + From<VectorSerial<A>>
//         + Extremes
//         + PartialEq
//         + 'static,
//     const PARTITION_CAP: usize,
//     const VECTOR_CAP: usize,
//     const MAX_LOADED: usize,
// >(
//     loaded_partitions: LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>,
//     inter_partition_graph: InterPartitionGraph<A>,
//     meta_data: HashMap<Uuid, StdRwLock<B>>,
//     cmd_input: Receiver<(Cmd<A, B>, Sender<u32>)>,
//     logger: Sender<State<AtomicCmd<A, B>>>,
//     // log: Log<A, B, 5000>,
// ) -> !
// where
//     for<'a> <A as Archive>::Archived:
//         CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
//     [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,

//     [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
//         DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
//     VectorSerial<A>: From<B>,
// {
//     // initialize internal graphs
//     // initialize locks
//     let rt = runtime::Builder::new_multi_thread()
//         .worker_threads(1)
//         .build()
//         .unwrap();

//     // let read_rt = runtime::Builder::new_multi_thread()
//     //     .worker_threads(1)
//     //     .build()
//     //     .unwrap();

//     // let write_rt = runtime::Builder::new_multi_thread()
//     //     .worker_threads(1)
//     //     .build()
//     //     .unwrap();
//     let loaded_partitions = Arc::new(Mutex::new(loaded_partitions));
//     let inter_partition_graph = Arc::new(TokioRwLock::new(inter_partition_graph));
//     let meta_data: Arc<TokioRwLock<HashMap<Uuid, StdRwLock<B>>>> =
//         Arc::new(TokioRwLock::new(meta_data));
//     // let log = Arc::new(Mutex::new(Vec::<String>::new()));

//     loop {
//         let (cmd, sender) = cmd_input.recv().unwrap();

//         match &cmd {
//             Cmd::Atomic(atomic_cmd) => {
//                 match atomic_cmd {
//                     AtomicCmd::StartTransaction(uuid) => {
//                         // send to write log
//                         todo!()
//                     }
//                     AtomicCmd::EndTransaction(uuid) => {
//                         // write to log
//                         todo!()
//                     }
//                     AtomicCmd::UndoTransaction(uuid) => {
//                         // kill all affected read threads
//                         // undo all write threads
//                         todo!()
//                     }
//                     AtomicCmd::InsertVector {
//                         vector,
//                         transaction_id,
//                     } => {
//                         let meta_data = meta_data.clone();
//                         let loaded_partitions = loaded_partitions.clone();

//                         let (tx, rx) = oneshot::channel();
//                         let _ = tx.send((vector.clone(), transaction_id.clone()));

//                         rt.spawn(async move {
//                             todo!()
//                             // insert_vector(rx, logger, meta_data, loaded_partitions).await;
//                         });
//                     }
//                     AtomicCmd::DeleteVector {
//                         uuid,
//                         transaction_id,
//                     } => {
//                         // create new write thread
//                         todo!()
//                     }

//                     AtomicCmd::GetIds { transaction_id } => {
//                         // read all ids from partitions
//                         todo!()
//                     }
//                     AtomicCmd::GetVectors { transaction_id } => {
//                         // read all vectors from partitions
//                         todo!()
//                     }
//                     AtomicCmd::Knn { transaction_id } => {
//                         // read all vectors and maybe a cache file
//                         todo!()
//                     }
//                     AtomicCmd::SelectedUUID { transaction_id } => {
//                         // create a cache of vectors in Uuid
//                         todo!()
//                     }
//                     AtomicCmd::PhantomData(phantom_data) => panic!(""),
//                 }
//                 todo!()
//             }
//             Cmd::Chained(chained_cmd) => {
//                 todo!()
//             }
//         }
//     }
//     panic!()
// }
