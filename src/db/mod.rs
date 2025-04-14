use std::{
    cmp::{min, Ordering},
    collections::HashMap,
    fmt::Debug,
    fs,
    marker::PhantomData,
    mem,
    path::Path,
    sync::Arc,
    time::Duration,
};

use crate::vector::VectorSerial;
use component::{
    cluster::ClusterSet,
    data_buffer::DataBuffer,
    graph::{GraphSerial, InterGraphSerial, InterPartitionGraph, IntraPartitionGraph},
    ids::{ClusterId, PartitionId, VectorId},
    meta::Meta,
    partition::{Partition, PartitionSerial, VectorEntry},
    serial::FileExtension,
};
use log::State;
use operations::{
    add::add,
    cluster::build_clusters_from_scratch,
    read::{
        knn::stream_exact_knn, stream_inter_graph, stream_meta_data, stream_partition_graph,
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
        mpsc::{Receiver, Sender},
        RwLock,
    },
    time::sleep,
};
use tracing::{event, Level};
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSpace};

pub mod component;
pub mod log;
pub mod operations;

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
    Partitions {
        ids: Vec<PartitionId>,
    },
    SelectedUUID {
        transaction_id: Option<Uuid>,
    },

    // Clustering
    CreateCluster {
        threshold: A,
    },
    GetClusters {
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
    MetaData(PartitionId, usize, VectorSerial<A>),
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
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Extremes
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
        >
        + Debug,
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
        + 'static
        + Debug
        + HasPosition<Scalar = f32>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
    const MAX_THREADS: usize,
>(
    // input channel
    mut cmd_input: Receiver<(Cmd<A, B>, Sender<Response<A>>)>,
    // logger: Sender<State<AtomicCmd<A, B>>>,
    // log: Log<A, B, 5000>,
) -> !
where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,

    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    VectorSerial<A>: From<B>,
    <A as Archive>::Archived: Deserialize<A, Strategy<Pool, rancor::Error>>,
    f32: From<A>,
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
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        PartitionSerial<A>,
        MAX_LOADED,
    >::new(PARTITION_DIR.into())));
    let min_spanning_tree_buffer = Arc::new(RwLock::new(DataBuffer::<
        IntraPartitionGraph<A>,
        GraphSerial<A>,
        MAX_LOADED,
    >::new(MIN_SPAN_DIR.into())));
    let inter_spanning_graph: Arc<RwLock<InterPartitionGraph<A>>> =
        Arc::new(RwLock::new(InterPartitionGraph::<A>::new()));
    let meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let cluster_sets: Arc<RwLock<Vec<ClusterSet<A>>>> = Arc::new(RwLock::new(Vec::new()));

    // check if file environment
    event!(Level::INFO, "FILE CHECK🗄️🗄️");
    let all_initialized: &[bool] = &[
        dir_initialized_with_files(PARTITION_DIR),
        dir_initialized_with_files(MIN_SPAN_DIR),
        dir_initialized_with_files(META_DATA_DIR),
        dir_initialized_with_files(CLUSTER_DATA_DIR),
        file_exists(
            PERSISTENT_DIR,
            GLOBAL_MIN_SPAN_FILE,
            InterGraphSerial::<A>::extension(),
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

    rt.block_on(async {
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
                            B::additive_identity(),
                            (A::max(), A::min()),
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
                        InterGraphSerial::<A>::extension()
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

                    let loaded_data = Meta::<A, B>::load_from_folder(META_DATA_DIR).await;

                    loaded_data.into_iter().for_each(|data| {
                        meta_data.insert(*data.id, Arc::new(RwLock::new(data)));
                    });
                }

                // load clusters
                {
                    let cluster_data = &mut *cluster_sets.write().await;

                    let mut loaded_data = ClusterSet::<A>::load_from_folder(CLUSTER_DATA_DIR).await;

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

        {
            let meta_data = meta_data.clone();
            let inter_spanning_graph = inter_spanning_graph.clone();
            let cluster_data = cluster_sets.clone();

            rt.spawn(async move {
                loop {
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
            let partition_buffer = partition_buffer.clone();
            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

            rt.spawn(async move {
                loop {
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
                                "Insert Vector :- ({}) {vector:?}",
                                id.to_string()
                            );

                            let inter_spanning_graph = inter_spanning_graph.clone();

                            let partition_buffer = partition_buffer.clone();
                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();

                            let meta_data = meta_data.clone();

                            let cluster_sets = cluster_sets.clone();

                            rt.spawn(async move {
                                let value = VectorEntry::from_uuid(vector, id);

                                match add(
                                    value,
                                    inter_spanning_graph,
                                    partition_buffer,
                                    min_spanning_tree_buffer,
                                    meta_data,
                                    cluster_sets,
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
                                tx.send(Response::Done).await.unwrap();
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
                                let _ = stream_meta_data(meta_data, &tx).await;

                                let _ = tx.send(Response::Done).await;
                            });
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
                        AtomicCmd::Partitions { ids } => {
                            // should replace with a way to choose knn method
                            let meta_data = meta_data.clone();
                            let partition_buffer = partition_buffer.clone();

                            rt.spawn(async move {
                                let Ok(_) = stream_vectors_from_partition(
                                    ids,
                                    meta_data,
                                    partition_buffer,
                                    &tx,
                                )
                                .await
                                else {
                                    todo!()
                                };

                                let _ = tx.send(Response::Done).await;
                            });
                        }
                        AtomicCmd::SelectedUUID { transaction_id } => {
                            // create a cache of vectors in Uuid
                            todo!()
                        }
                        AtomicCmd::CreateCluster { threshold } => {
                            // generate cluster data
                            let cluster_sets = cluster_sets.clone();
                            let meta_data = meta_data.clone();

                            let min_spanning_tree_buffer = min_spanning_tree_buffer.clone();
                            let inter_spanning_graph: Arc<RwLock<InterPartitionGraph<A>>> =
                                inter_spanning_graph.clone();

                            rt.spawn(async move {
                                // let cluster_data = &mut *cluster_data.write().await;
                                build_clusters_from_scratch(
                                    threshold,
                                    meta_data,
                                    cluster_sets,
                                    inter_spanning_graph,
                                    min_spanning_tree_buffer,
                                )
                                .await;

                                tx.send(Response::Done).await
                            });
                        }
                        AtomicCmd::GetClusters { threshold } => {
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

                                        for cluster_id in cluster_set.get_clusters().await {
                                            let _ = tx
                                                .send(Response::Success(Success::Cluster(
                                                    cluster_id,
                                                )))
                                                .await;
                                            let vector_ids =
                                                cluster_set.get_cluster_members::<5>(cluster_id).await.unwrap();

                                            for vec_id in vector_ids {
                                                let _ = tx
                                                    .send(Response::Success(Success::Vector(
                                                        vec_id,
                                                        VectorSerial(Vec::new()),
                                                    )))
                                                    .await;
                                            }
                                        }

                                        let _ = tx.send(Response::Done).await;
                                    }
                                    Err(_) => {
                                        let _ = tx.send(Response::Fail).await;
                                    }
                                }
                            });
                        }
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
