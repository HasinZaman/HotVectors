use std::{
    cmp::{min, Ordering},
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs,
    marker::PhantomData,
    mem,
    path::Path,
    sync::Arc,
    time::Duration,
};

use crate::{
    db::{
        banker::{AccessMode, AccessResponse, BankerMessage},
        component::{partition::PartitionMembership, umap::incremental_model::UMapStrategy},
    },
    resolve_buffer,
    vector::{Vector, VectorSerial},
};

use burn::{prelude::Backend, tensor::backend::AutodiffBackend};
#[cfg(feature = "benchmark")]
use component::benchmark::{benchmark_logger, Benchmark, BenchmarkId};

use banker::banker;
use component::{
    cluster::ClusterSet,
    data_buffer::{DataBuffer, Global},
    graph::{GraphSerial, InterGraphSerial, InterPartitionGraph, IntraPartitionGraph},
    ids::{ClusterId, PartitionId, VectorId},
    meta::Meta,
    partition::{Partition, PartitionSerial, VectorEntry},
    serial::FileExtension,
};
use log::State;
use operations::{
    add::{batch, single},
    cluster::build_clusters,
    read::{
        knn::stream_exact_knn, stream_inter_graph/*, stream_meta_data */, stream_partition_graph,
        stream_vectors_from_partition,
    },
};
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    rend::u32_le,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, DeserializeUnsized,
};
use spade::HasPosition;
use tokio::{
    join, runtime,
    sync::{
        mpsc::{self, Receiver, Sender},
        oneshot, Notify, RwLock,
    },
    time::sleep,
};
use tracing::{event, Level};
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSpace};

mod banker;
pub mod component;
pub mod log;
pub mod operations;

#[derive(Clone, Copy, Debug)]
pub enum Source<F> {
    VectorId(VectorId),
    PartitionId(PartitionId),
    ClusterId(ClusterId, F),
}

// maybe convert into a struct with bools and can be built?
#[derive(Clone, Copy, Debug)]
pub enum ProjectionMode {
    Default,
    IdOnly,
    VectorOnly,
}

impl From<(bool, bool)> for ProjectionMode {
    fn from(value: (bool, bool)) -> Self {
        match value {
            (true, true) => ProjectionMode::Default,
            (true, false) => ProjectionMode::IdOnly,
            (false, true) => ProjectionMode::VectorOnly,
            (false, false) => todo!(),
        }
    }
}

impl Default for ProjectionMode {
    fn default() -> Self {
        ProjectionMode::Default
    }
}

#[derive(Clone, Debug)]
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
    BatchInsertVectors {
        vectors: Vec<B>,
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

        source: Source<A>,

        dim_projection: Option<usize>,
        projection_mode: ProjectionMode,
    },
    GetMetaData {
        transaction_id: Option<Uuid>,

        source: Source<A>,
        dim_projection: Option<usize>,
    },
    // Graphs
    GetPartitionGraph {
        transaction_id: Option<Uuid>,
        partition_id: PartitionId,
    },
    GetInterPartitionGraph {
        transaction_id: Option<Uuid>,
        partition_id: PartitionId,
    },

    // Filters
    Knn {
        vector: B,
        k: usize,
        transaction_id: Option<Uuid>,
    },

    // Clustering
    CreateCluster {
        threshold: A,
    },
}

#[derive(Clone, Debug)]
pub struct ChainedCmd<A: Field<A>, B: VectorSpace<A> + Sized>(Vec<AtomicCmd<A, B>>);

#[derive(Clone, Debug)]
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
    UInt(usize),
    Knn(VectorId, A),
    Partition(PartitionId),
    Cluster(ClusterId),
    Vector(VectorId, VectorSerial<A>),
    Edge(VectorId, VectorId, A),
    InterEdge((PartitionId, VectorId), (PartitionId, VectorId), A),
}

pub fn dir_initialized(dir: &str) -> bool {
    // Check if the directory exists
    let directory = Path::new(dir);
    if !directory.exists() {
        fs::create_dir_all(directory).unwrap();
        return false;
    }

    return true;
}

pub fn dir_initialized_with_files(dir: &str) -> bool {
    // Check if the directory exists
    let directory = Path::new("data");
    if !directory.exists() {
        fs::create_dir_all(directory).unwrap();
    }

    let directory = Path::new(dir);
    if !directory.exists() {
        // Create the directory
        match fs::create_dir_all(directory) {
            Ok(_) => {
                println!("Directory created at {}", dir);
                return false;
            }
            Err(_e) => panic!(),
        }
    }
    return true;
}

pub fn file_exists(dir: &str, file_name: &str, extension: &str) -> bool {
    // Ensure the directory exists
    let directory = Path::new(dir);
    if !directory.exists() {
        println!("Directory created at {}", dir);
        fs::create_dir_all(directory).expect("Failed to create directory");
        return false;
    }

    // Check if the file exists
    let file_path = directory.join(&format!("{file_name}.{extension}"));

    file_path.exists()
}

pub fn db_loop<
    B: Backend + AutodiffBackend + Send + Sync,
    // F: PartialEq
    //     + PartialOrd
    //     + Clone
    //     + Copy
    //     + Extremes
    //     + Field<f32>
    //     + Send
    //     + Sync
    //     + 'static
    //     + rkyv::Archive
    //     + for<'a> rkyv::Serialize<
    //         rancor::Strategy<
    //             rkyv::ser::Serializer<
    //                 rkyv::util::AlignedVec,
    //                 rkyv::ser::allocator::ArenaHandle<'a>,
    //                 rkyv::ser::sharing::Share,
    //             >,
    //             rancor::Error,
    //         >,
    //     >
    //     + Debug = f32,
    V: VectorSpace<f32>
        + Sized
        + Clone
        + Copy
        + Send
        + Sync
        + Sized
        + From<VectorSerial<f32>>
        + Extremes
        + PartialEq
        + 'static
        + Debug
        + From<Vec<f32>>
        + Into<Vec<f32>>,
    LV: VectorSpace<f32> + Send + Sync + for<'a> From<&'a [f32]> + From<Vec<f32>> + Into<Vec<f32>>,
    K: Send + Sync + Clone + Debug + 'static + From<usize>,
    // U: UMapStrategy<V, LV, Global, K, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>,
    // + HasPosition<Scalar = f32>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
    const MAX_THREADS: usize,
>(
    // input channel
    mut cmd_input: Receiver<(Cmd<f32, V>, Sender<Response<f32>>)>,
    // u_map_projections: U,
    // logger: Sender<State<AtomicCmd<A, B>>>,
    // log: Log<A, B, 5000>,
) -> !
where
    // for<'a> <F as Archive>::Archived:
    //     CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    // [<F as Archive>::Archived]: DeserializeUnsized<[F], Strategy<Pool, rancor::Error>>,

    // [ArchivedTuple3<u32_le, u32_le, <F as Archive>::Archived>]:
    //     DeserializeUnsized<[(usize, usize, F)], Strategy<Pool, rancor::Error>>,
    VectorSerial<f32>: From<V>,
    // <F as Archive>::Archived: Deserialize<F, Strategy<Pool, rancor::Error>>,
    // f32: From<F>,
    Vector<f32, VECTOR_CAP>: From<V>,
    Vector<f32, 2>: From<V>,
    // Vec<f32>: From<LV>,
{
    event!(Level::INFO, "🔥 HOT VECTOR START UP 👠👠");

    // initialize internal data
    const PARTITION_DIR: &str = "data//partitions";
    const MIN_SPAN_DIR: &str = "data//min_span_trees";
    const PERSISTENT_DIR: &str = "data//persistent";
    const META_DATA_DIR: &str = "data//meta";
    const CLUSTER_DATA_DIR: &str = "data//clusters";

    const GLOBAL_MIN_SPAN_FILE: &str = "global_min_span";

    let partition_buffer = Arc::new(RwLock::new(DataBuffer::<
        Partition<f32, V, PARTITION_CAP, VECTOR_CAP>,
        PartitionSerial<f32>,
        Global,
        MAX_LOADED,
    >::new(PARTITION_DIR.into())));
    let min_spanning_tree_buffer = Arc::new(RwLock::new(DataBuffer::<
        IntraPartitionGraph<f32>,
        GraphSerial<f32>,
        Global,
        MAX_LOADED,
    >::new(MIN_SPAN_DIR.into())));
    let inter_spanning_graph: Arc<RwLock<InterPartitionGraph<f32>>> =
        Arc::new(RwLock::new(InterPartitionGraph::<f32>::new()));
    let meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<f32, V>>>>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let cluster_sets: Arc<RwLock<Vec<ClusterSet<f32>>>> = Arc::new(RwLock::new(Vec::new()));

    let partition_membership: Arc<RwLock<PartitionMembership>> = Arc::new(RwLock::new(
        PartitionMembership::new(format!("data//{}//{}", PARTITION_DIR, "membership")),
    ));

    // check if file environment
    event!(Level::INFO, "FILE CHECK🗄️🗄️");
    let all_initialized: &[bool] = &[
        // dir_initialized_with_files(PARTITION_DIR),
        // dir_initialized_with_files(MIN_SPAN_DIR),
        dir_initialized_with_files(META_DATA_DIR),
        dir_initialized_with_files(CLUSTER_DATA_DIR),
        file_exists(
            PERSISTENT_DIR,
            GLOBAL_MIN_SPAN_FILE,
            InterGraphSerial::<f32>::extension(),
        ),
    ];

    event!(
        Level::INFO,
        "File system initialized: {:?}",
        all_initialized
    );

    // initialize locks
    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(MAX_THREADS)
        // .enable_io()
        .enable_time()
        .build()
        .unwrap();

    let (access_tx, rx) = tokio::sync::mpsc::channel(64);
    rt.spawn(async move {
        banker(rx).await;
    });

    rt.block_on(async {
        #[cfg(feature = "benchmark")]
        let benchmark_writer = {
            let (tx, rx) = std::sync::mpsc::channel::<(BenchmarkId, u64, u64, String)>();
            let writer = std::sync::Arc::new(tx);

            rt.spawn(async move { benchmark_logger(rx).await });

            writer
        };

        let all_true = all_initialized.iter().all(|x| *x);
        let all_false = all_initialized.iter().all(|x| !*x);
        match (all_false, all_true) {
            (true, _) => {
                event!(Level::INFO, "🔨📈 CREATING FILING CABINET 🔨📈");
                {
                    let partition = Partition::new();

                    let id = partition.id;
                    partition_buffer
                        .write()
                        .await
                        .push(partition)
                        .await
                        .unwrap();

                    min_spanning_tree_buffer
                        .write()
                        .await
                        .push(IntraPartitionGraph::new(PartitionId(id)))
                        .await
                        .unwrap();

                    meta_data.write().await.insert(
                        id,
                        Arc::new(RwLock::new(Meta::new(
                            PartitionId(id),
                            0,
                            V::additive_identity(),
                            (<f32 as Extremes>::max(), <f32 as Extremes>::min()),
                        ))),
                    );

                    inter_spanning_graph.write().await.add_node(PartitionId(id));
                }
                event!(Level::INFO, "JUST MADE A FILING CABINET🔨📈");
                event!(Level::INFO, "💋📄 FILLING FILING CABINET 💋📄");
                {
                    partition_buffer.read().await.try_save().await;
                    min_spanning_tree_buffer.read().await.try_save().await;

                    inter_spanning_graph
                        .read()
                        .await
                        .save(PERSISTENT_DIR, GLOBAL_MIN_SPAN_FILE)
                        .await;

                    for (id, data) in meta_data.read().await.iter() {
                        let data = &*data.read().await;

                        data.save(META_DATA_DIR, &id.to_string()).await;
                    }
                }
            }
            (_, true) => {
                event!(Level::INFO, "CHECKING FILES💋📓");

                // Load inter_graph
                {
                    let path = format!(
                        "{PERSISTENT_DIR}//{GLOBAL_MIN_SPAN_FILE}.{}",
                        InterGraphSerial::<f32>::extension()
                    );
                    let (mut x, y) = join!(
                        inter_spanning_graph.write(),
                        InterPartitionGraph::load(&path)
                    );

                    let x = &mut *x;

                    let _ = mem::replace(x, y);
                }

                // Load meta_data
                {
                    let meta_data = &mut *meta_data.write().await;

                    let loaded_data = Meta::<f32, V>::load_from_folder(META_DATA_DIR).await;

                    loaded_data.into_iter().for_each(|data| {
                        meta_data.insert(*data.id, Arc::new(RwLock::new(data)));
                    });
                }

                // load clusters
                {
                    let cluster_data = &mut *cluster_sets.write().await;

                    let mut loaded_data = ClusterSet::<f32>::load_from_folder(CLUSTER_DATA_DIR).await;

                    loaded_data.sort();

                    *cluster_data = loaded_data;
                }
            }
            (false, false) => panic!("Messed up file environment"),
        }

        {
            let partition_buffer = partition_buffer.clone();
            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

            rt.spawn(async move {
                loop {
                    sleep(Duration::from_secs(10)).await;

                    {
                        event!(Level::INFO, "DECREMENTING partition_buffer 👠😎");
                        partition_buffer.write().await.de_increment();
                    }
                    {
                        event!(Level::INFO, "DECREMENTING min_spanning_tree_buffer 👠😎");
                        min_spanning_tree_buffer.write().await.de_increment();
                    }
                }
            });
        }

        let notify_update = Arc::new(Notify::new());
        {
            let meta_data = meta_data.clone();
            let inter_spanning_graph = inter_spanning_graph.clone();
            let cluster_data = cluster_sets.clone();
            let notify_update = notify_update.clone();

            rt.spawn(async move {
                loop {
                    notify_update.notified().await;
                    sleep(Duration::from_secs(20)).await; //60 * 1)).await;
                    {
                        event!(Level::INFO, "Saving inter_spanning_graph 👠😎");
                        let inter_spanning_graph = {
                            let rwlock = &*inter_spanning_graph.read().await;

                            rwlock.clone()
                        };
                        inter_spanning_graph
                            .save(PERSISTENT_DIR, GLOBAL_MIN_SPAN_FILE)
                            .await;
                    }
                    {
                        event!(Level::INFO, "Saving meta_data 👠😎");
                        for (id, data) in meta_data.read().await.iter() {
                            let data = &*data.read().await;

                            data.save(META_DATA_DIR, &id.to_string()).await;
                        }
                    }
                    // might not be nessearry due to no cached data
                    {
                        event!(Level::INFO, "Saving cluster_data 👠😎");
                        for data in cluster_data.read().await.iter() {
                            data.save(CLUSTER_DATA_DIR, &data.id.to_string()).await;
                        }
                    }
                }
            });
        }
        {
            let notify_update = notify_update.clone();

            let partition_buffer = partition_buffer.clone();
            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

            rt.spawn(async move {
                loop {
                    notify_update.notified().await;
                    sleep(Duration::from_secs(40)).await; //60 * 1)).await;
                    {
                        event!(Level::INFO, "saving contents of partition_buffer");
                        let r_partition_buffer_lock = partition_buffer.read().await;
                        let partition_buffer = &*r_partition_buffer_lock;
                        partition_buffer.try_save().await;
                    }
                    {
                        event!(Level::INFO, "saving contents of min_spanning_tree_buffer");
                        let r_min_spanning_tree_buffer = min_spanning_tree_buffer.read().await;
                        let min_spanning_tree_buffer = &*r_min_spanning_tree_buffer;
                        min_spanning_tree_buffer.try_save().await;
                    }
                }
            });
        }
        event!(Level::INFO, "I'M ALL HOT TO GO👠👠");
        loop {
            let (cmd, tx) = cmd_input.recv().await.unwrap();

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
                            let id = Uuid::new_v4();
                            event!(
                                Level::INFO,
                                "Insert Vector :- ({})",
                                id.to_string()
                            );

                            let inter_spanning_graph = inter_spanning_graph.clone();

                            let partition_buffer = partition_buffer.clone();
                            let partition_membership = partition_membership.clone();
                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

                            let meta_data = meta_data.clone();

                            let cluster_sets = cluster_sets.clone();

                            let notify_update = notify_update.clone();

                            let access_tx = access_tx.clone();

                            #[cfg(feature = "benchmark")]
                            let benchmark_writer = benchmark_writer.clone();

                            rt.spawn(async move {
                                let value = VectorEntry::from_uuid(vector, id);

                                match single::add::<B, f32, V, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>(
                                    value,
                                    transaction_id,
                                    meta_data,
                                    partition_buffer,
                                    partition_membership,
                                    min_spanning_tree_buffer,
                                    inter_spanning_graph,
                                    cluster_sets,
                                    access_tx,
                                    #[cfg(feature = "benchmark")]
                                    Benchmark::new("Insert Vector".to_string(), benchmark_writer),
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
                                println!("(Done) Insert Vector :- {vector:?}");
                                tx.send(Response::Success(Success::Vector(
                                    VectorId(id),
                                    VectorSerial::<f32>::from(vector),
                                )))
                                .await
                                .unwrap();
                                tx.send(Response::Done).await.unwrap();

                                notify_update.notify_waiters();
                            });
                        }
                        AtomicCmd::BatchInsertVectors {
                            vectors,
                            transaction_id,
                        } => {
                            let inter_spanning_graph = inter_spanning_graph.clone();

                            let partition_buffer = partition_buffer.clone();
                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

                            let meta_data = meta_data.clone();

                            let cluster_sets = cluster_sets.clone();

                            let notify_update = notify_update.clone();

                            let access_tx = access_tx.clone();

                            #[cfg(feature = "benchmark")]
                            let benchmark_writer = benchmark_writer.clone();
                            rt.spawn(async move {
                                let new_vectors: Vec<_> = vectors
                                    .into_iter()
                                    .map(|vector| VectorEntry::from_uuid(vector, Uuid::new_v4()))
                                    .collect();

                                todo!();
                                // match batch::add(
                                //     new_vectors.clone(),
                                //     transaction_id,
                                //     meta_data,
                                //     partition_buffer,
                                //     min_spanning_tree_buffer,
                                //     inter_spanning_graph,
                                //     cluster_sets,
                                //     access_tx,
                                //     #[cfg(feature = "benchmark")]
                                //     Benchmark::new("Insert Vector".to_string(), benchmark_writer),
                                // )
                                // .await
                                // {
                                //     Ok(_) => {
                                //         // log success
                                //         // send success
                                //     }
                                //     Err(_) => todo!(),
                                // };

                                // let _ = sender.send(Response::Success);
                                // for VectorEntry { id, vector, .. } in new_vectors {
                                //     tx.send(Response::Success(Success::Vector(
                                //         VectorId(id),
                                //         vector.into(),
                                //     )))
                                //     .await
                                //     .unwrap();
                                // }
                                tx.send(Response::Done).await.unwrap();

                                notify_update.notify_waiters();

                                notify_update.notify_waiters();
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
                        AtomicCmd::GetVectors {
                            transaction_id,
                            source,
                            dim_projection,
                            projection_mode,
                        } => {
                            let (vector_tx, mut vector_rx) = mpsc::channel(64);
                            let _vector_loader_thread = match source {
                                Source::VectorId(vector_id) => {
                                    let access_tx = access_tx.clone();
                                    let partition_buffer = partition_buffer.clone();
                                    let partition_membership = partition_membership.clone();

                                    rt.spawn(async move {
                                        let transaction_id =
                                            transaction_id.unwrap_or_else(Uuid::new_v4);

                                        let partition_id = loop {
                                            let partition_membership =
                                                &*partition_membership.read().await;

                                            let partition_id: PartitionId = partition_membership
                                                .get_partition_id(vector_id)
                                                .unwrap();

                                            let (tx, rx) = oneshot::channel();

                                            let _ = access_tx
                                                .send(BankerMessage::RequestAccess {
                                                    transaction_id: transaction_id,
                                                    partitions: vec![(
                                                        partition_id,
                                                        AccessMode::Read,
                                                    )],
                                                    respond_to: tx,
                                                })
                                                .await;

                                            match rx.await {
                                                Ok(AccessResponse::Granted) => break partition_id,
                                                _ => {}
                                            }
                                        };

                                        let partition_buffer = &mut *partition_buffer.write().await;
                                        let partition =
                                            resolve_buffer!(ACCESS, partition_buffer, partition_id);

                                        let Some(partition) = &*partition.read().await else {
                                            todo!()
                                        };


                                        for i in 0..partition.size {
                                            let Some(vector_entry) = partition.vectors[i] else {
                                                todo!()
                                            };

                                             if vector_id != VectorId(vector_entry.id) {
                                                continue;
                                            }

                                            let _ = vector_tx.send(Some(vector_entry.clone())).await;

                                            break;

                                        }
                                        let _ = vector_tx.send(None).await;
                                    })
                                }
                                Source::PartitionId(partition_id) => {
                                    let access_tx = access_tx.clone();
                                    let partition_buffer = partition_buffer.clone();

                                    rt.spawn(async move {
                                        let transaction_id =
                                            transaction_id.unwrap_or_else(Uuid::new_v4);

                                        // Request access to the partition
                                        let granted_partition = loop {
                                            let (tx, rx) = oneshot::channel();

                                            access_tx
                                                .send(BankerMessage::RequestAccess {
                                                    transaction_id,
                                                    partitions: vec![(
                                                        partition_id,
                                                        AccessMode::Read,
                                                    )],
                                                    respond_to: tx,
                                                })
                                                .await
                                                .expect("failed to send access request");

                                            match rx.await {
                                                Ok(AccessResponse::Granted) => break partition_id,
                                                _ => {
                                                    // Retry until access is granted
                                                    // Optionally add delay or backoff
                                                }
                                            }
                                        };

                                        // Load the partition
                                        let partition_buffer = &mut *partition_buffer.write().await;
                                        let partition = resolve_buffer!(
                                            ACCESS,
                                            partition_buffer,
                                            granted_partition
                                        );

                                        let Some(partition) = &*partition.read().await else {
                                            todo!("Partition buffer resolve failed");
                                        };

                                        // Send all vectors in the partition
                                        
                                        for i in 0..partition.size {
                                            let Some(vector_entry) = partition.vectors[i] else {
                                                todo!()
                                            };

                                            let _ = vector_tx.send(Some(vector_entry.clone())).await;
                                        }
                                        let _ = vector_tx.send(None).await;
                                    })
                                }
                                Source::ClusterId(cluster_id, threshold) => {
                                    let cluster_sets = cluster_sets.clone();
                                    let access_tx = access_tx.clone();
                                    let partition_buffer = partition_buffer.clone();
                                    let partition_membership = partition_membership.clone();

                                    rt.spawn(async move {
                                        let vectors = {
                                            let cluster_sets = &*cluster_sets.read().await;

                                            match cluster_sets.binary_search_by(|x| {
                                                x.threshold
                                                    .partial_cmp(&threshold)
                                                    .unwrap_or(Ordering::Equal) // == Ordering::Equal
                                            }) {
                                                Ok(pos) => {
                                                    let cluster_set = &cluster_sets[pos];

                                                    cluster_set
                                                        .get_cluster_members(cluster_id)
                                                        .unwrap()
                                                }
                                                Err(_) => {
                                                    // let _ = tx.send(Response::Fail).await;
                                                    return;
                                                }
                                            }
                                        };

                                        let transaction_id = Uuid::new_v4();
                                        let required_partitions: HashMap<
                                            PartitionId,
                                            HashSet<VectorId>,
                                        > = loop {
                                            let partition_membership =
                                                &*partition_membership.read().await;

                                            let mut partition_vectors_pairs: HashMap<
                                                PartitionId,
                                                HashSet<VectorId>,
                                            > = HashMap::new();

                                            for vector_id in &vectors {
                                                let partition_id: PartitionId =
                                                    partition_membership
                                                        .get_partition_id(*vector_id)
                                                        .unwrap();

                                                partition_vectors_pairs
                                                    .entry(partition_id)
                                                    .or_default()
                                                    .insert(*vector_id);
                                            }

                                            let (tx, rx) = oneshot::channel();

                                            access_tx
                                                .send(BankerMessage::RequestAccess {
                                                    transaction_id: transaction_id,
                                                    partitions: partition_vectors_pairs
                                                        .keys()
                                                        .map(|id| (*id, AccessMode::Read))
                                                        .collect(),
                                                    respond_to: tx,
                                                })
                                                .await;

                                            match rx.await {
                                                Ok(AccessResponse::Granted) => {
                                                    break partition_vectors_pairs
                                                }
                                                _ => {}
                                            }
                                        };

                                        for (partition_id, required_vectors) in
                                            required_partitions.into_iter()
                                        {
                                            let partition_buffer =
                                                &mut *partition_buffer.write().await;
                                            let partition = resolve_buffer!(
                                                ACCESS,
                                                partition_buffer,
                                                partition_id
                                            );

                                            let Some(partition) = &*partition.read().await else {
                                                todo!()
                                            };

                                            for i in 0..partition.size {
                                                let Some(vector_entry) = partition.vectors[i] else {
                                                    todo!()
                                                };

                                                if !required_vectors
                                                    .contains(&VectorId(vector_entry.id))
                                                {
                                                    continue;
                                                }

                                                let _ = vector_tx.send(Some(vector_entry.clone())).await;
                                            }
                                        }
                                        
                                        let _ = vector_tx.send(None).await;
                                    })
                                }
                            };

                            // project points if required
                            let _vector_tx_thread = match dim_projection {
                                Some(dim) => {
                                    todo!()
                                },
                                None => {
                                    rt.spawn(async move {
                                        match projection_mode {
                                            ProjectionMode::Default => {
                                                while let Some(vector_entry) = vector_rx.recv().await.unwrap() {
                                                    let _ = tx.send(Response::Success(
                                                        Success::Vector(
                                                            VectorId(vector_entry.id),
                                                            VectorSerial::<f32>::from(vector_entry.vector),
                                                        )
                                                    )).await;
                                                }
                                            }
                                            ProjectionMode::IdOnly => {
                                                while let Some(vector_entry) = vector_rx.recv().await.unwrap() {

                                                    let _ = tx.send(Response::Success(
                                                        Success::Vector(
                                                            VectorId(vector_entry.id),
                                                            VectorSerial::<f32>::from(vector_entry.vector),
                                                        )
                                                    )).await;
                                                }
                                            },
                                            ProjectionMode::VectorOnly => {
                                                while let Some(vector_entry) = vector_rx.recv().await.unwrap() {

                                                    let _ = tx.send(Response::Success(
                                                        Success::Vector(
                                                            VectorId(Uuid::nil()),
                                                            VectorSerial::<f32>::from(vector_entry.vector),
                                                        )
                                                    )).await;
                                                }
                                            },
                                        }
                                        let _ = tx.send(Response::Done).await;
                                    })
                                }
                            };
                        }
                        AtomicCmd::GetMetaData {
                            transaction_id,
                            source,
                            dim_projection
                        } => {
                            match source {
                                Source::VectorId(vector_id) => {
                                    // maybe just return vector dim?
                                    let _ = tx.send(
                                        Response::Success(
                                            Success::UInt(VECTOR_CAP)
                                        )
                                    ).await;

                                    // if projects are available -> return available reductions
                                    //  -> provide train pool (# of approximated points)

                                    // if valid id -> return partition membership & cluster membership
                                    todo!();

                                    let _ = tx.send(Response::Done).await;
                                },
                                Source::PartitionId(partition_id) => {
                                    if partition_id.0 == Uuid::nil() {
                                        let meta_data = meta_data.clone();
                                        rt.spawn(async move {
                                            let meta_data = &*meta_data.read().await;

                                            let mut visited = HashSet::new();

                                            while visited.len() != meta_data.len() {
                                                let iter: Vec<_> = meta_data
                                                    .iter()
                                                    .filter(|(id, _)| !visited.contains(*id))
                                                    .collect();

                                                for (id, data) in iter {
                                                    let data = &*data.read().await;
                                                    let _ = tx
                                                        .send(
                                                            Response::Success(
                                                                Success::Partition(data.id)
                                                            )
                                                        )
                                                        .await;

                                                    let _ = tx
                                                        .send(
                                                            Response::Success(
                                                                Success::UInt(data.size)
                                                            )
                                                        )
                                                        .await;

                                                    let _ = tx
                                                        .send(
                                                            Response::Success(
                                                                Success::Vector(
                                                                    VectorId(Uuid::nil()),
                                                                    VectorSerial::from(data.centroid.clone())
                                                                )
                                                            )
                                                        )
                                                        .await;

                                                    visited.insert(*id);
                                                }
                                            }

                                            let _ = tx.send(Response::Done).await;
                                        });
                                    }
                                    else {
                                        let meta_data = meta_data.clone();
                                        rt.spawn(async move {
                                            let meta_data = &*meta_data.read().await;

                                            let Some(data) = meta_data.get(&partition_id.0) else {
                                                todo!()
                                            };

                                            let data = &*data.read().await;

                                            let _ = tx
                                                .send(
                                                    Response::Success(
                                                        Success::Partition(partition_id)
                                                    )
                                                )
                                                .await;

                                            let _ = tx
                                                .send(
                                                    Response::Success(
                                                        Success::UInt(data.size)
                                                    )
                                                )
                                                .await;

                                            let _ = tx
                                                .send(
                                                    Response::Success(
                                                        Success::Vector(
                                                            VectorId(Uuid::nil()),
                                                            VectorSerial::from(data.centroid.clone())
                                                        )
                                                    )
                                                )
                                                .await;

                                            let _ = tx.send(Response::Done).await;
                                        });
                                    }
                                },
                                Source::ClusterId(cluster_id, threshold) => {
                                    let cluster_sets = cluster_sets.clone();
                                
                                    rt.spawn(async move {
                                        let cluster_sets = &*cluster_sets.read().await;
                                        match cluster_sets.binary_search_by(|x| {
                                            x.threshold
                                                .partial_cmp(&threshold)
                                                .unwrap_or(Ordering::Equal) // == Ordering::Equal
                                        }) {
                                            Ok(pos) => {
                                                let cluster_set = &cluster_sets[pos];
                                                
                                                if cluster_id.0 == Uuid::nil() {
                                                    for cluster_id in cluster_set.get_clusters() {
                                                        // cluster id
                                                        let _ = tx
                                                            .send(Response::Success(Success::Cluster(
                                                                cluster_id,
                                                            )))
                                                            .await;
                                                        
                                                        // get all vectors in cluster
                                                        let vector_ids = cluster_set.get_cluster_members(
                                                                cluster_id
                                                            ).unwrap();
                                                        
                                                        // send cluster side
                                                        let _ = tx
                                                            .send(Response::Success(Success::UInt(
                                                                vector_ids.len(),
                                                            )))
                                                            .await;
                                                        
                                                        // send vector_ids
                                                        for vec_id in vector_ids {
                                                            let _ = tx
                                                                .send(Response::Success(Success::Vector(
                                                                    vec_id,
                                                                    VectorSerial(Vec::new()),
                                                                )))
                                                                .await;
                                                        }

                                                        // send cluster centroid
                                                        let _ = tx
                                                            .send(Response::Success(Success::Vector(
                                                                VectorId(Uuid::nil()),
                                                                VectorSerial(Vec::new()),
                                                            )))
                                                            .await;
                                                    }
                                
                                                    let _ = tx.send(Response::Done).await;
                                                }
                                                else {
                                                    let _ = tx
                                                        .send(Response::Success(Success::Cluster(
                                                            cluster_id,
                                                        )))
                                                        .await;
                                                    
                                                    // get all vectors in cluster
                                                    let vector_ids = cluster_set.get_cluster_members(
                                                            cluster_id
                                                        ).unwrap();
                                                    
                                                    // send cluster side
                                                    let _ = tx
                                                        .send(Response::Success(Success::UInt(
                                                            vector_ids.len(),
                                                        )))
                                                        .await;
                                                    
                                                    // send vector_ids
                                                    for vec_id in vector_ids {
                                                        let _ = tx
                                                            .send(Response::Success(Success::Vector(
                                                                vec_id,
                                                                VectorSerial(Vec::new()),
                                                            )))
                                                            .await;
                                                    }

                                                    // send cluster centroid
                                                    let _ = tx
                                                        .send(Response::Success(Success::Vector(
                                                            VectorId(Uuid::nil()),
                                                            VectorSerial(Vec::new()),
                                                        )))
                                                        .await;
                                
                                                    let _ = tx.send(Response::Done).await;
                                                }
                                            }
                                            Err(_) => {
                                                let _ = tx.send(Response::Fail).await;
                                            }
                                        }
                                    });
                                },
                            };
                        }
                        AtomicCmd::GetPartitionGraph {
                            transaction_id,
                            partition_id,
                        } => {
                            let meta_data = meta_data.clone();
                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

                            rt.spawn(async move {
                                let _ = stream_partition_graph(
                                    partition_id,
                                    meta_data,
                                    min_spanning_tree_buffer,
                                    &tx,
                                )
                                .await;

                                let _ = tx.send(Response::Done).await;
                            });
                        }
                        AtomicCmd::GetInterPartitionGraph {
                            transaction_id,
                            partition_id,
                        } => {
                            let meta_data = meta_data.clone();
                            let inter_graph = inter_spanning_graph.clone();

                            rt.spawn(async move {
                                let _ =
                                    stream_inter_graph(partition_id, meta_data, inter_graph, &tx)
                                        .await;

                                let _ = tx.send(Response::Done).await;
                            });
                        }
                        AtomicCmd::Knn {
                            vector,
                            k,
                            transaction_id,
                        } => {
                            // should replace with a way to choose knn method
                            let meta_data = meta_data.clone();
                            let inter_graph = inter_spanning_graph.clone();
                            let partition_buffer = partition_buffer.clone();

                            rt.spawn(async move {
                                let Ok(_) = stream_exact_knn(
                                    vector,
                                    k,
                                    inter_graph,
                                    partition_buffer,
                                    meta_data,
                                    &tx,
                                )
                                .await
                                else {
                                    todo!()
                                };

                                let _ = tx.send(Response::Done).await;
                            });
                        }
                        AtomicCmd::CreateCluster { threshold } => {
                            // generate cluster data
                            let cluster_sets = cluster_sets.clone();
                            let meta_data = meta_data.clone();

                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();
                            let inter_spanning_graph: Arc<RwLock<InterPartitionGraph<f32>>> =
                                inter_spanning_graph.clone();

                            let notify_update = notify_update.clone();

                            #[cfg(feature = "benchmark")]
                            let benchmark_writer = benchmark_writer.clone();

                            rt.spawn(async move {
                                #[cfg(feature = "benchmark")]
                                let _benchmark =
                                    Benchmark::new("Create Cluster".to_string(), benchmark_writer);
                                // let cluster_data = &mut *cluster_data.write().await;
                                
                                build_clusters(
                                    threshold,
                                    meta_data,
                                    cluster_sets,
                                    inter_spanning_graph,
                                    min_spanning_tree_buffer,
                                )
                                .await;

                                let _ = tx.send(Response::Done).await;
                                notify_update.notify_waiters();
                            });
                        }
                        // AtomicCmd::GetClusters { threshold } => {
                        //     let cluster_sets = cluster_sets.clone();
                        //
                        //     rt.spawn(async move {
                        //         let cluster_sets = &*cluster_sets.read().await;
                        //         match cluster_sets.binary_search_by(|x| {
                        //             x.threshold
                        //                 .partial_cmp(&threshold)
                        //                 .unwrap_or(Ordering::Equal) // == Ordering::Equal
                        //         }) {
                        //             Ok(pos) => {
                        //                 let cluster_set = &cluster_sets[pos];
                        //
                        //                 for cluster_id in cluster_set.get_clusters() {
                        //                     let _ = tx
                        //                         .send(Response::Success(Success::Cluster(
                        //                             cluster_id,
                        //                         )))
                        //                         .await;
                        //                     let vector_ids = cluster_set
                        //                         .get_cluster_members::<5>(cluster_id)
                        //                         .unwrap();
                        //
                        //                     for vec_id in vector_ids {
                        //                         let _ = tx
                        //                             .send(Response::Success(Success::Vector(
                        //                                 vec_id,
                        //                                 VectorSerial(Vec::new()),
                        //                             )))
                        //                             .await;
                        //                     }
                        //                 }
                        //
                        //                 let _ = tx.send(Response::Done).await;
                        //             }
                        //             Err(_) => {
                        //                 let _ = tx.send(Response::Fail).await;
                        //             }
                        //         }
                        //     });
                        // }
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
