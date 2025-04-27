use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    sync::Arc,
};

use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use spade::{DelaunayTriangulation, HasPosition, Triangulation};
use tokio::sync::{mpsc::Sender, oneshot, RwLock};
use tracing::{event, Level};
use uuid::Uuid;

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    rancor,
    rend::u32_le,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, DeserializeUnsized,
};

#[cfg(feature = "benchmark")]
use crate::db::component::benchmark::Benchmark;
use crate::{
    db::{
        component::{
            cluster::ClusterSet,
            data_buffer::{flush, BufferError, DataBuffer, Global, Local},
            graph::{GraphSerial, IntraPartitionGraph, ReConstruct, UpdateTree},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{
                self, ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionSerial,
                VectorEntry, VectorEntrySerial,
            },
        },
        operations::{
            cluster::{remove_cluster_edge, update_cluster},
            split::{
                calculate_number_of_trees, split_partition, split_partition_into_trees,
                FirstTreeSplitStrategy, KMean, MaxAttempt, BFS,
            },
        },
        scheduler::{AccessCommand, AccessPermit, AccessType},
    },
    resolve_buffer,
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;
macro_rules! lock {
    // READ
    [max_attempts = $max_attempts:expr; $( ($lock:expr, READ) ),+ $(,)? ] => {{
        let mut attempts = 0;
        loop {
            if attempts >= $max_attempts {
                event!(Level::WARN, "Failed to acquire READ locks after {} attempts", $max_attempts);
                break None;
            }
            attempts += 1;

            let result = async {
                let guards = (
                    $(
                        match $lock.try_read() {
                            Ok(g) => g,
                            Err(_) => {
                                event!(Level::DEBUG, "Failed to acquire READ lock on {}", stringify!($lock));
                                return None;
                            }
                        }
                    ),+
                );
                Some(guards)
            }.await;

            if let Some(guards) = result {
                break Some(guards);
            }

            tokio::task::yield_now().await;
        }
    }};

    // WRITE
    [max_attempts = $max_attempts:expr; $( ($lock:expr, WRITE) ),+ $(,)? ] => {{
        let mut attempts = 0;
        loop {
            if attempts >= $max_attempts {
                event!(Level::WARN, "Failed to acquire WRITE locks after {} attempts", $max_attempts);
                break None;
            }
            attempts += 1;

            let result = async {
                let guards = (
                    $(
                        match $lock.try_write() {
                            Ok(g) => g,
                            Err(_) => {
                                event!(Level::DEBUG, "Failed to acquire WRITE lock on {}", stringify!($lock));
                                return None;
                            }
                        }
                    ),+
                );
                Some(guards)
            }.await;

            if let Some(guards) = result {
                break Some(guards);
            }

            tokio::task::yield_now().await;
        }
    }};

    // LOCK (mutex)
    [max_attempts = $max_attempts:expr; $( ($lock:expr, LOCK) ),+ $(,)? ] => {{
        let mut attempts = 0;
        loop {
            if attempts >= $max_attempts {
                event!(Level::WARN, "Failed to acquire LOCKs after {} attempts", $max_attempts);
                break None;
            }
            attempts += 1;

            let result = async {
                let guards = (
                    $(
                        match $lock.try_lock() {
                            Ok(g) => g,
                            Err(_) => {
                                event!(Level::DEBUG, "Failed to acquire LOCK on {}", stringify!($lock));
                                return None;
                            }
                        }
                    ),+
                );
                Some(guards)
            }.await;

            if let Some(guards) = result {
                break Some(guards);
            }

            tokio::task::yield_now().await;
        }
    }};

    [$( ($lock:expr, READ) ),+ $(,)? ] => {{
        loop {
            let mut success = true;
            #[allow(unused_mut)]
            let result = async {
                let guards = (
                    $(
                        match $lock.try_read() {
                            Ok(g) => g,
                            Err(_) => {
                                event!(Level::DEBUG, "Failed to acquire READ lock on {}", stringify!($lock));
                                success = false;
                                return None;
                            }
                        }
                    ),+
                );
                Some(guards)
            }.await;

            if let Some(guards) = result {
                break guards;
            }
            drop(result);
            tokio::task::yield_now().await;
        }
    }};
    [$( ($lock:expr, WRITE) ),+ $(,)? ] => {{
        loop {
            let mut success = true;
            #[allow(unused_mut)]
            let result = async {
                let guards = (
                    $(
                        match $lock.try_write() {
                            Ok(g) => g,
                            Err(_) => {
                                event!(Level::DEBUG, "Failed to acquire WRITE lock on {}", stringify!($lock));
                                success = false;
                                return None;
                            }
                        }
                    ),+
                );
                Some(guards)
            }.await;

            if let Some(guards) = result {
                break guards;
            }
            drop(result);

            tokio::task::yield_now().await;
        }
    }};
    [$( ($lock:expr, LOCK) ),+ $(,)? ] => {{
        loop {
            let mut success = true;
            #[allow(unused_mut)]
            let result = async {
                let guards = (
                    $(
                        match $lock.try_lock() {
                            Ok(g) => g,
                            Err(_) => {
                                event!(Level::DEBUG, "Failed to acquire LOCK on {}", stringify!($lock));
                                success = false;
                                return None;
                            }
                        }
                    ),+
                );
                Some(guards)
            }.await;

            if let Some(guards) = result {
                break guards;
            }
            drop(result);

            tokio::task::yield_now().await;
        }
    }};
}

pub async fn add<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<A>
        + for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        > + Debug
        + Extremes,
    B: VectorSpace<A>
        + Sized
        + Clone
        + Copy
        + PartialEq
        + Extremes
        + From<VectorSerial<A>>
        + Debug
        + HasPosition<Scalar = f32>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    new_vector: VectorEntry<A, B>,

    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<A>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
    mst_buffer: Arc<RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Global, MAX_LOADED>>>,
    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,
    cluster_sets: Arc<RwLock<Vec<ClusterSet<A>>>>,

    access_tx: Sender<AccessCommand>,

    #[cfg(feature = "benchmark")] benchmark: Benchmark,
) -> Result<(), PartitionErr>
where
    VectorSerial<A>: From<B>,
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<A>]:
        DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    f32: From<A>,
{
    println!(
        "{:?} :- finding closet_partition & required partitions",
        new_vector.id
    );
    let (mut closet_partition_id, required_partitions) = 'lock_partitions: loop {
        {
            for i in 0..10 {
                tokio::task::yield_now().await;
            }
            let r_meta_data = lock![(meta_data, READ)];
            let meta_data = &*r_meta_data;
            // drop(r_meta_data);

            // find the closet partition
            let mut iter = meta_data.iter();
            let (mut closet_partition_id, mut min_distance, mut min_size) = {
                let (id, meta) = iter.next().unwrap();
                let w_meta = meta.read().await;
                let meta: &Meta<A, B> = &*w_meta;

                (
                    PartitionId(*id),
                    B::dist(&meta.centroid, &new_vector.vector),
                    meta.size,
                )
            };
            for (id, meta) in iter {
                let r_meta = meta.read().await;
                let meta: &Meta<A, B> = &*r_meta;

                let dist = B::dist(&meta.centroid, &new_vector.vector);

                if dist < min_distance {
                    closet_partition_id = PartitionId(*id);
                    min_distance = dist;
                    min_size = meta.size;
                };
            }

            // if the size = 0 then first_add
            if min_size == 0 {
                let Some((
                    mut partition_buffer,
                    mut min_spanning_tree_buffer,
                    mut inter_graph,
                    mut cluster_sets,
                )) = lock![
                    max_attempts = 1;
                    (partition_buffer, WRITE),
                    (mst_buffer, WRITE),
                    (inter_graph, WRITE),
                    (cluster_sets, WRITE),
                ]
                else {
                    println!("failed intial insert");
                    tokio::task::yield_now().await;
                    continue 'lock_partitions;
                };

                let meta = meta_data[&*closet_partition_id].clone();
                let mut meta = meta.write().await;
                let meta = &mut *meta;

                if meta.size != 0 {
                    continue 'lock_partitions;
                }

                return Ok(first_insert(
                    new_vector,
                    closet_partition_id,
                    meta,
                    &mut *partition_buffer,
                    &mut *min_spanning_tree_buffer,
                    &mut *inter_graph,
                    &mut *cluster_sets,
                )
                .await
                .unwrap());
            }

            // update closet partition id based on closet vector_id, partition_id
            // {
            //     let neighbors = {
            //         let Some(inter_graph) = lock![max_attempts = 1; (inter_graph, READ)] else {
            //             tokio::task::yield_now().await;
            //             continue 'lock_partitions;
            //         };
            //
            //         let neighbors = get_neighbors::<A, B, VECTOR_CAP>(
            //             &*inter_graph,
            //             &meta_data,
            //             PartitionId(*closet_partition_id),
            //         )
            //         .await;
            //
            //         neighbors
            //     };
            //     let mut required_partitions: HashSet<PartitionId> = neighbors.into_iter().collect();
            //     required_partitions.insert(closet_partition_id);
            //
            //     let (mut min_partition_id, mut min_dist) = (PartitionId(Uuid::nil()), A::max());
            //
            //     while required_partitions.len() > 0 {
            //         {
            //             let mut collected_partition_locks = Vec::new();
            //
            //             let mut partition_buffer =
            //                 lock![(partition_buffer, WRITE)];
            //             let partition_buffer = &mut *partition_buffer;
            //             for id in required_partitions.iter() {
            //                 let partition = match partition_buffer.try_access(id) {
            //                     Ok(val) => val,
            //                     Err(BufferError::DataNotFound) => {
            //                         //try to load and access
            //                         match partition_buffer.load(id).await {
            //                             Ok(_) => match partition_buffer.try_access(id) {
            //                                 Ok(val) => val,
            //                                 Err(BufferError::FileNotFound) => {
            //                                     tokio::task::yield_now().await;
            //                                     continue 'lock_partitions;
            //                                 }
            //                                 Err(_) => {
            //                                     continue;
            //                                 }
            //                             },
            //                             Err(BufferError::OutOfSpace) => {
            //                                 // try to unload and access
            //                                 let mut iter = partition_buffer
            //                                     .least_used_iter()
            //                                     .await
            //                                     .unwrap()
            //                                     .filter(|(_, id)| {
            //                                         !required_partitions.contains(&PartitionId(*id))
            //                                     });
            //
            //                                 let Some((_, unload_id)) = iter.next() else {
            //                                     continue;
            //                                 };
            //
            //                                 if let Err(_) = partition_buffer
            //                                     .try_unload_and_load(&unload_id, &id)
            //                                     .await
            //                                 {
            //                                     continue;
            //                                 }
            //
            //                                 match partition_buffer.try_access(&id) {
            //                                     Ok(val) => val,
            //                                     Err(_) => {
            //                                         continue;
            //                                     }
            //                                 }
            //                             }
            //                             Err(BufferError::FileNotFound) => {
            //                                 continue 'lock_partitions;
            //                             }
            //                             Err(_) => todo!(),
            //                         }
            //                     },
            //                     Err(BufferError::FileNotFound) => {
            //                         continue 'lock_partitions;
            //                     },
            //                     Err(tmp) => {
            //                         println!("{:?} :- {tmp:?}", new_vector.id);
            //                         continue;
            //                     }
            //                 };
            //                 // let partition = partition.clone();
            //                 collected_partition_locks.push(partition.clone());
            //             }
            //
            //             let mut partition_guards =
            //                 Vec::with_capacity(collected_partition_locks.len());
            //             for partition_lock in collected_partition_locks.iter() {
            //                 let partition_guard = partition_lock.read().await;
            //                 partition_guards.push(partition_guard);
            //             }
            //
            //             let mut collected_partitions =
            //                 Vec::with_capacity(collected_partition_locks.len());
            //             for partition_guard in partition_guards.iter() {
            //                 let Some(partition) = &**partition_guard else {
            //                     todo!()
            //                 };
            //                 collected_partitions.push(partition);
            //             }
            //
            //             // get min dist
            //             // possible multi-threading solution
            //             // possible GPU optimization
            //             for partition in collected_partitions.iter() {
            //                 for vector in partition.iter() {
            //                     let dist = B::dist(&vector.vector, &new_vector.vector);
            //
            //                     if dist < min_dist {
            //                         min_dist = dist;
            //
            //                         min_partition_id = PartitionId(partition.id);
            //                     }
            //                 }
            //             }
            //
            //             for partition in collected_partitions.iter() {
            //                 required_partitions.remove(&PartitionId(partition.id));
            //             }
            //         }
            //
            //         tokio::task::yield_now().await;
            //     }
            //
            //     if *min_partition_id != Uuid::nil() && min_partition_id != closet_partition_id {
            //         closet_partition_id = min_partition_id;
            //     }
            // };

            // get required partitions
            let required_partitions = {
                let Some(inter_graph) = lock![max_attempts = 1; (inter_graph, READ)] else {
                    tokio::task::yield_now().await;
                    continue 'lock_partitions;
                };

                let neighbors = get_neighbors::<A, B, VECTOR_CAP>(
                    &*inter_graph,
                    &meta_data,
                    PartitionId(*closet_partition_id),
                )
                .await;

                find_required_partitions(
                    &neighbors,
                    &PartitionId(*closet_partition_id),
                    &*inter_graph,
                )
            };

            drop(meta_data);
            drop(r_meta_data);

            let (tx, rx) = oneshot::channel();
            let cmd = AccessCommand::Acquire {
                resource_ids: required_partitions.iter().map(|x| *x).collect(),
                access_type: AccessType::Write,
                response: tx,
            };

            let _ = access_tx.send(cmd).await;

            if let Ok(AccessPermit::Granted) = rx.await {
                println!(
                    "{:?} :- (Granted access) {closet_partition_id:?} \n{required_partitions:?}",
                    new_vector.id
                );

                break (closet_partition_id, required_partitions);
            }
        }
    };

    // used for updating min-spanning tree
    let mut neighbor_partitions: HashSet<PartitionId> = required_partitions.clone();

    println!("{:?} :- Copy into local buffer", new_vector.id);

    let id = Uuid::new_v4();

    let mut local_meta_data: HashMap<Uuid, Arc<RwLock<Meta<A, B>>>> = {
        let meta_data = &*meta_data.read().await;

        let mut local_meta_data = HashMap::new();

        let iter = meta_data
            .iter()
            .filter(|(id, _)| required_partitions.contains(&PartitionId(**id)));

        for (id, meta) in iter {
            local_meta_data.insert(*id, Arc::new(RwLock::new((*meta.read().await).clone())));
        }

        local_meta_data
    };

    let mut local_partition_buffer =
        DataBuffer::<
            Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
            PartitionSerial<A>,
            Local,
            MAX_LOADED,
        >::new(format!("data/local/{}/partitions", id.to_string()));
    {
        let partition_buffer = &mut *partition_buffer.write().await;

        for partition_id in required_partitions.iter() {
            let partition = resolve_buffer!(ACCESS, partition_buffer, *partition_id);

            let Some(partition) = &*partition.read().await else {
                todo!();
            };

            let partition = partition.clone();

            resolve_buffer!(PUSH, local_partition_buffer, partition);
        }
    }

    let mut local_mst_buffer =
        DataBuffer::<IntraPartitionGraph<A>, GraphSerial<A>, Local, MAX_LOADED>::new(format!(
            "data/local/{}/min_span_trees",
            id.to_string()
        ));
    {
        let mst_buffer = &mut *mst_buffer.write().await;

        for partition_id in required_partitions.iter() {
            let mst = resolve_buffer!(ACCESS, mst_buffer, *partition_id);

            let Some(mst) = &*mst.read().await else {
                todo!();
            };

            let mst = mst.clone();

            resolve_buffer!(PUSH, local_mst_buffer, mst);
        }
    }

    let mut local_inter_graph = {
        // might be faster to or cheaper to just copy a projection of the inter_graph
        let inter_graph = &*inter_graph.read().await;
        // should attempt reconstruct inter_graph using required partitions
        let mut local_inter_graph = InterPartitionGraph::new();

        let mut inserted_edges = HashSet::new();

        for partition_id in required_partitions.iter() {
            local_inter_graph.add_node(*partition_id);
        }

        for partition_id in required_partitions.iter() {
            for edge_ref in inter_graph.0.edges(inter_graph.1[partition_id]) {
                let (weight, (partition_id_1, vector_id_1), (partition_id_2, vector_id_2)) =
                    edge_ref.weight();

                if inserted_edges
                    .contains(&((partition_id_1, vector_id_1), (partition_id_2, vector_id_2)))
                    || inserted_edges
                        .contains(&((partition_id_2, vector_id_2), (partition_id_1, vector_id_1)))
                {
                    continue;
                }

                if !required_partitions.contains(partition_id_1) {
                    continue;
                }

                if !required_partitions.contains(partition_id_2) {
                    continue;
                }

                local_inter_graph.add_edge(
                    *partition_id_1,
                    *partition_id_2,
                    (
                        *weight,
                        (*partition_id_1, *vector_id_1),
                        (*partition_id_2, *vector_id_2),
                    ),
                );
                inserted_edges
                    .insert(((partition_id_1, vector_id_1), (partition_id_2, vector_id_2)));
            }
        }

        local_inter_graph
    };

    {
        // acquire closet partition
        let partition = resolve_buffer!(ACCESS, local_partition_buffer, closet_partition_id);

        let mut partition = lock![(partition, WRITE)];

        let Some(partition) = &mut *partition else {
            todo!()
        };

        // check if needs to be split
        match partition.size + 1 == PARTITION_CAP {
            true => {
                let (mst, pairs) = {
                    let w_mst = resolve_buffer!(ACCESS, local_mst_buffer, closet_partition_id);
                    let mut mst = lock![(w_mst, WRITE)];
                    let Some(mst) = &mut *mst else {
                        todo!();
                    };

                    // possible race condition
                    //  create a list of new PartitionIds & lock them before updating inter_graph

                    // split partitions
                    let pairs =
                        split_partition::<A, B, KMean<MaxAttempt<128>>, PARTITION_CAP, VECTOR_CAP>(
                            partition,
                            mst,
                            &mut local_inter_graph,
                        )
                        .unwrap();

                    //  split by disjoined sets
                    let pairs: Vec<_> = pairs
                        .into_iter()
                        .map(|(mut partition, mut mst)| {
                            if calculate_number_of_trees(&mst) == 1 {
                                return vec![(partition, mst)];
                            }

                            split_partition_into_trees(
                                &mut partition,
                                &mut mst,
                                &mut local_inter_graph,
                            )
                            .unwrap()
                        })
                        .flatten()
                        .collect();

                    (w_mst.clone(), pairs)
                };
                let mut mst = mst.write().await;
                let Some(mst) = &mut *mst else {
                    todo!();
                };
                let mst = &mut *mst;

                let mut new_closet_partition_id = pairs[0].0.id;
                let mut new_closet_dist = B::dist(&new_vector.vector, &pairs[0].0.centroid());

                for (new_partition, new_mst) in pairs {
                    // update new closet_partition_id

                    let new_dist = B::dist(&new_vector.vector, &new_partition.centroid());

                    if new_dist < new_closet_dist {
                        new_closet_partition_id = new_partition.id;
                        new_closet_dist = new_dist;
                    }

                    // update partition if new_partition matches
                    if new_partition.id == partition.id {
                        *partition = new_partition;
                        *mst = new_mst;

                        let meta = local_meta_data[&partition.id].clone();

                        let mut meta = lock![(meta, WRITE)];
                        let meta = &mut *meta;

                        meta.size = partition.size;
                        meta.centroid = partition.centroid();

                        continue;
                    }

                    // push value into buffer
                    println!("{:?} :- Push {:?}", new_vector.id, new_partition.id);
                    println!(
                        "{:?} :- Got buffer access to push {:?}",
                        new_vector.id, new_partition.id
                    );

                    println!(
                        "{:?} :- Trying to push {:?} without unloading {:?}",
                        new_vector.id,
                        new_partition.id,
                        [partition.id, new_closet_partition_id]
                    );
                    resolve_buffer!(PUSH, local_partition_buffer, new_partition, [partition.id]);
                    resolve_buffer!(PUSH, local_mst_buffer, new_mst, [partition.id]);
                    println!(
                        "{:?} :- Finished pushing {:?}",
                        new_vector.id, new_partition.id
                    );

                    // push into meta data
                    local_meta_data.insert(
                        new_partition.id,
                        Arc::new(RwLock::new(Meta {
                            id: PartitionId(new_partition.id),
                            size: new_partition.size,
                            centroid: new_partition.centroid(),
                            edge_length: (
                                A::max(), // should be replaced with smallest edge
                                A::min(), // should be replaced with largest edge
                            ),
                        })),
                    );

                    neighbor_partitions.insert(PartitionId(new_partition.id));
                }

                println!(
                    "{:?} :- finished splitting - Inserting into partition",
                    new_vector.id
                );
                closet_partition_id = PartitionId(new_closet_partition_id);
                {
                    let meta = local_meta_data[&*closet_partition_id].clone();
                    let mut meta = lock![(meta, WRITE)];
                    let meta = &mut *meta;

                    println!(
                        "{:?} :- Trying to get access to closet_partition({closet_partition_id:?}) from local buffer for insertion",
                        new_vector.id
                    );
                    if partition.id == *closet_partition_id {
                        let _ = add_into_partition(new_vector.clone(), partition).unwrap();
                        meta.size = partition.size;
                        meta.centroid = partition.centroid();
                    } else {
                        let insert_partition = resolve_buffer!(
                            ACCESS,
                            local_partition_buffer,
                            closet_partition_id,
                            [partition.id]
                        );

                        //  match local_partition_buffer.try_access(&*closet_partition_id){
                        // Ok(val) => val,
                        // Err(BufferError::DataNotFound) => {
                        //     match local_partition_buffer.load(&*closet_partition_id).await {
                        //         Ok(()) => local_partition_buffer.try_access(&*closet_partition_id).unwrap(),
                        //         Err(_) => todo!()
                        //     }

                        // },
                        // Err(_) => todo!()
                        // };
                        let mut insert_partition = insert_partition.write().await;
                        let Some(insert_partition) = &mut *insert_partition else {
                            todo!()
                        };
                        let _ = add_into_partition(new_vector.clone(), insert_partition).unwrap();

                        meta.size = insert_partition.size;
                        meta.centroid = insert_partition.centroid();
                    }
                }
            }
            false => {
                let meta = local_meta_data[&partition.id].clone();
                let mut meta = lock![(meta, WRITE)];
                let meta = &mut *meta;
                let _ = add_into_partition(new_vector.clone(), partition).unwrap();

                meta.size = partition.size;
                meta.centroid = partition.centroid();
            }
        };
    }
    // update min_spanning tree
    {
        println!("{:?} :- Update min-spanning tree", new_vector.id);
        // calculate distance
        let dist_map = {
            let mut dist_map = HashMap::new();

            let mut required_partitions = required_partitions.clone();

            while required_partitions.len() > 0 {
                {
                    let mut collected_partition_locks = Vec::new();

                    for id in required_partitions.iter() {
                        let partition = match local_partition_buffer.try_access(id) {
                            Ok(val) => val,
                            Err(BufferError::DataNotFound) => {
                                //try to load and access
                                match local_partition_buffer.load(id).await {
                                    Ok(_) => match local_partition_buffer.try_access(id) {
                                        Ok(val) => val,
                                        Err(_) => {
                                            continue;
                                        }
                                    },
                                    Err(BufferError::OutOfSpace) => {
                                        // try to unload and access
                                        let mut iter = local_partition_buffer
                                            .least_used_iter()
                                            .await
                                            .unwrap()
                                            .filter(|(_, id)| {
                                                !required_partitions.contains(&PartitionId(*id))
                                            });

                                        let Some((_, unload_id)) = iter.next() else {
                                            continue;
                                        };

                                        if let Err(_) = local_partition_buffer
                                            .try_unload_and_load(&unload_id, &id)
                                            .await
                                        {
                                            continue;
                                        }

                                        match local_partition_buffer.try_access(&id) {
                                            Ok(val) => val,
                                            Err(_) => {
                                                continue;
                                            }
                                        }
                                    }
                                    Err(BufferError::FileNotFound) => {
                                        todo!()
                                    }
                                    Err(_) => todo!(),
                                }
                            }
                            Err(tmp) => {
                                continue;
                            }
                        };
                        // let partition = partition.clone();
                        collected_partition_locks.push(partition.clone());
                    }

                    let mut partition_guards = Vec::with_capacity(collected_partition_locks.len());
                    for partition_lock in collected_partition_locks.iter() {
                        let partition_guard = partition_lock.read().await;
                        partition_guards.push(partition_guard);
                    }

                    let mut collected_partitions =
                        Vec::with_capacity(collected_partition_locks.len());
                    for partition_guard in partition_guards.iter() {
                        let Some(partition) = &**partition_guard else {
                            todo!()
                        };
                        collected_partitions.push(partition);
                    }

                    // get min dist
                    // possible multi-threading solution
                    // possible GPU optimization
                    for partition in collected_partitions.iter() {
                        for vector in partition.iter() {
                            let dist = B::dist(&vector.vector, &new_vector.vector);

                            dist_map.insert(VectorId(vector.id), dist);
                        }
                    }

                    for partition in collected_partitions.iter() {
                        required_partitions.remove(&PartitionId(partition.id));
                    }
                }

                tokio::task::yield_now().await;
            }

            dist_map
        };

        // Update closest partition edges
        {
            let mst = resolve_buffer!(ACCESS, local_mst_buffer, closet_partition_id);
            let mut mst = lock![(mst, WRITE)];

            let Some(mst) = &mut *mst else { todo!() };

            //ReConstruct
            match <IntraPartitionGraph<A> as UpdateTree<ReConstruct, A>>::update(
                mst,
                VectorId(new_vector.id),
                &dist_map
                    .iter()
                    .filter(|(id, _)| mst.1.contains_key(id))
                    .map(|(id, dist)| (*dist, *id))
                    .collect::<Vec<_>>(),
            ) {
                Ok(new_edges) => {
                    for (weight, id_1, id_2) in new_edges {
                        // update_cluster(cluster_sets, &weight, id_1, id_2).await;
                    }
                }
                Err(_) => {
                    todo!()
                }
            };
        };

        // neighboring partitions
        {
            let _ = neighbor_partitions.remove(&closet_partition_id);
            // let mut cached_trails: HashMap<((PartitionId, VectorId), (VectorId, PartitionId)), _> =
            //     HashMap::new();
            // let mut cached_partition_trail_keys: HashMap<
            //     PartitionId,
            //     ((PartitionId, VectorId), (VectorId, PartitionId)),
            // > = HashMap::new();

            let mut visited_vectors: HashSet<VectorId> = HashSet::new();

            // loop through neighboring partitions
            while let Some(start_partition) = neighbor_partitions.iter().next().cloned() {
                let start_partition = neighbor_partitions.take(&start_partition).unwrap();

                let search_vectors: Vec<VectorId> = {
                    let start_mst = resolve_buffer!(ACCESS, local_mst_buffer, start_partition);
                    let Some(start_mst) = &*start_mst.read().await else {
                        todo!();
                    };

                    start_mst.1.keys().map(|x| *x).collect()
                };

                // get partition_trail
                println!("{start_partition:?}->{closet_partition_id:?}\n{local_inter_graph:#?}");
                let mut skipping_trail: Vec<((PartitionId, VectorId), (PartitionId, VectorId), A)> =
                    local_inter_graph
                        .find_trail(start_partition, closet_partition_id)
                        .unwrap()
                        .unwrap()
                        .1;

                println!("{skipping_trail:?}");

                for start_vector in search_vectors {
                    if visited_vectors.contains(&start_vector) {
                        continue;
                    }
                    {
                        // already loaded - might want to keep in buffer to minimize loading & unloading
                        let mst = resolve_buffer!(ACCESS, local_mst_buffer, start_partition);
                        let Some(mst) = &*mst.read().await else {
                            todo!();
                        };

                        if !mst.1.contains_key(&start_vector) {
                            continue;
                        }
                    }
                    visited_vectors.insert(start_vector);

                    // generate trail
                    let edges = {
                        // start to intermediate
                        let mut edges: Vec<_> = 'initialize_trail: {
                            let ((_, end_vector), _, _weight) =
                                skipping_trail.iter().next().unwrap();

                            if start_vector == *end_vector {
                                break 'initialize_trail Vec::new()
                            }

                            let mst = resolve_buffer!(ACCESS, local_mst_buffer, start_partition);
                            let Some(mst) = &*mst.read().await else {
                                todo!();
                            };

                            println!("{start_vector:?} -> {end_vector:?}\n{mst:#?}");

                            mst.find_trail(start_vector, *end_vector)
                                .unwrap()
                                .unwrap()
                                .1
                                .into_iter()
                                .map(|(id_1, id_2, weight)| {
                                    ((start_partition, id_1), (start_partition, id_2), weight)
                                })
                                .collect()
                        };

                        // intermediate
                        {
                            let mut iter = skipping_trail.iter().peekable();

                            while let Some(cur) = iter.next() {
                                let (_, (partition_id, start_vector), _) = cur;

                                let Some(((_, end_vector), _, _)) = iter.peek() else {
                                    continue;
                                };

                                if start_vector == end_vector {
                                    continue;
                                }

                                // insert caching

                                let mst = resolve_buffer!(ACCESS, local_mst_buffer, *partition_id);
                                let Some(mst) = &*mst.read().await else {
                                    todo!();
                                };
                                println!("{start_vector:?} -> {end_vector:?}\n{mst:#?}");

                                edges.extend(
                                    mst.find_trail(*start_vector, *end_vector)
                                        .unwrap()
                                        .unwrap()
                                        .1
                                        .into_iter()
                                        .map(|(id_1, id_2, weight)| {
                                            ((*partition_id, id_1), (*partition_id, id_2), weight)
                                        }),
                                );
                            }
                        }

                        // last edge
                        {
                            let (_, (_, start_vector), _weight) = skipping_trail.last().unwrap();

                            if start_vector == &VectorId(new_vector.id) {
                                continue;
                            }

                            let mst =
                                resolve_buffer!(ACCESS, local_mst_buffer, closet_partition_id);
                            let Some(mst) = &*mst.read().await else {
                                todo!();
                            };
                            println!(
                                "{start_vector:?} -> {:?}\n{mst:#?}",
                                VectorId(new_vector.id)
                            );

                            edges.extend(
                                mst.find_trail(*start_vector, VectorId(new_vector.id))
                                    .unwrap()
                                    .unwrap()
                                    .1
                                    .into_iter()
                                    .map(|(id_1, id_2, weight)| {
                                        ((start_partition, id_1), (start_partition, id_2), weight)
                                    }),
                            );
                        };

                        // add inter_edges
                        edges.extend(skipping_trail.clone());

                        edges
                    };

                    // get max edge
                    let max_edge = edges.into_iter().fold(
                        (
                            (PartitionId(Uuid::nil()), VectorId(Uuid::nil())),
                            (PartitionId(Uuid::nil()), VectorId(Uuid::nil())),
                            A::min(),
                        ),
                        |acc, next| match acc.2 < next.2 {
                            true => next,
                            false => acc,
                        },
                    );

                    let ((partition_id_1, vector_id_1), (partition_id_2, vector_id_2), max_weight) =
                        max_edge;

                    if *partition_id_1 == Uuid::nil() {
                        continue;
                    }

                    if max_weight <= dist_map[&start_vector] {
                        continue;
                    }

                    println!("Remove edge: {:?}", ((partition_id_1, vector_id_1), (partition_id_2, vector_id_2), max_weight));

                    // delete edge
                    match partition_id_1 == partition_id_2 {
                        true => {
                            let split_partition_id = partition_id_1;

                            let mst = resolve_buffer!(ACCESS, local_mst_buffer, split_partition_id);
                            let Some(mst) = &mut *mst.write().await else {
                                todo!();
                            };

                            let partition =
                                resolve_buffer!(ACCESS, local_partition_buffer, split_partition_id);
                            let Some(partition) = &mut *partition.write().await else {
                                todo!();
                            };

                            // remove edge
                            mst.remove_edge(vector_id_1, vector_id_2);

                            // split graph
                            let Ok([pair_1, pair_2]) = split_partition::<
                                A,
                                B,
                                FirstTreeSplitStrategy,
                                PARTITION_CAP,
                                VECTOR_CAP,
                            >(
                                partition, mst, &mut local_inter_graph
                            ) else {
                                todo!()
                            };

                            let (pair_1, pair_2) = match pair_1.0.id == *split_partition_id {
                                true => (pair_1, pair_2),
                                false => (pair_2, pair_1),
                            };

                            {
                                let (new_partition, new_mst) = pair_1;

                                if new_partition.id != *split_partition_id {
                                    todo!()
                                }

                                *partition = new_partition;
                                *mst = new_mst;

                                let target_meta: &mut Meta<A, B> =
                                    &mut *local_meta_data[&*split_partition_id].write().await;

                                target_meta.size = partition.size;
                                target_meta.centroid = partition.centroid();

                                target_meta.edge_length = (
                                    match mst.smallest_edge() {
                                        Some(x) => x.2,
                                        None => A::max(),
                                    },
                                    match mst.largest_edge() {
                                        Some(x) => x.2,
                                        None => A::min(),
                                    },
                                );
                            }

                            if split_partition_id == start_partition {
                                neighbor_partitions.insert(PartitionId(pair_2.0.id));
                            }

                            let (new_partition, new_mst) = pair_2;

                            if new_mst.1.contains_key(&VectorId(new_vector.id)) {
                                closet_partition_id = PartitionId(new_partition.id);
                            }

                            local_meta_data.insert(
                                new_partition.id,
                                Arc::new(RwLock::new(Meta::new(
                                    PartitionId(new_partition.id),
                                    new_partition.size,
                                    new_partition.centroid(),
                                    (
                                        match new_mst.smallest_edge() {
                                            Some(x) => x.2,
                                            None => A::max(),
                                        },
                                        match new_mst.largest_edge() {
                                            Some(x) => x.2,
                                            None => A::min(),
                                        },
                                    ),
                                ))),
                            );

                            resolve_buffer!(
                                PUSH,
                                local_mst_buffer,
                                new_mst,
                                [*start_partition, *split_partition_id]
                            );
                            resolve_buffer!(
                                PUSH,
                                local_partition_buffer,
                                new_partition,
                                [*start_partition, *split_partition_id]
                            );
                        }
                        false => {
                            local_inter_graph.remove_edge(
                                (partition_id_1, vector_id_1),
                                (partition_id_2, vector_id_2),
                            );
                        }
                    }

                    // add edge
                    local_inter_graph.add_edge(
                        start_partition,
                        closet_partition_id,
                        (
                            dist_map[&start_vector],
                            (start_partition, start_vector),
                            (closet_partition_id, VectorId(new_vector.id)),
                        ),
                    );

                    skipping_trail = vec![(
                        (start_partition, start_vector),
                        (closet_partition_id, VectorId(new_vector.id)),
                        dist_map[&start_vector],
                    )];
                }
            }
        }
    }

    println!("{:?} - flush local into global", new_vector.id);
    {
        let partition_buffer = &mut *partition_buffer.write().await;

        flush(local_partition_buffer, partition_buffer).await;
    }
    {
        let mst_buffer = &mut *mst_buffer.write().await;

        flush(local_mst_buffer, mst_buffer).await;
    }
    {
        let meta_data = &mut *meta_data.write().await;

        for key in local_meta_data.keys() {
            match meta_data.contains_key(&key) {
                true => {
                    println!("{:?} - update meta data {key:}", new_vector.id);
                    let mut meta = lock![(meta_data[key], WRITE)];
                    println!("{:?} - got global meta {key:}", new_vector.id);
                    let data = lock![(local_meta_data[key], READ)];
                    println!("{:?} - got local meta {key:}", new_vector.id);
                    *meta = (*data).clone();
                }
                false => {
                    println!("{:?} - insert meta data {key:}", new_vector.id);
                    let data = lock![(local_meta_data[key], READ)];
                    meta_data.insert(*key, Arc::new(RwLock::new((*data).clone())));
                }
            }
        }
    }
    {
        let inter_graph = &mut *inter_graph.write().await;

        for node in local_inter_graph.1.keys() {
            if inter_graph.1.contains_key(node) {
                continue;
            }
            inter_graph.add_node(*node);
        }

        let mut existing_edges: HashSet<((PartitionId, VectorId), (PartitionId, VectorId))> =
            HashSet::new();
        for edge in inter_graph.0.edge_references() {
            let (w, id1, id2) = edge.weight();
            let key = (*id1, *id2);
            let rev_key = (*id2, *id1); // because it's undirected
            existing_edges.insert(key);
            existing_edges.insert(rev_key);
        }

        for edge in local_inter_graph.0.edge_references() {
            let id1 = local_inter_graph.0[edge.source()];
            let id2 = local_inter_graph.0[edge.target()];
            let weight = edge.weight().clone();
            let key = (weight.1, weight.2);

            if !existing_edges.contains(&key) {
                inter_graph.add_edge(id1, id2, weight);
                existing_edges.insert(key);
                existing_edges.insert((weight.2, weight.1)); // undirected
            }
        }
    }

    std::fs::remove_dir_all(format!("data/local/{}", id.to_string())).unwrap();

    println!("{:?} :- Done", new_vector.id);

    let cmd = AccessCommand::Release {
        resource_ids: required_partitions.iter().map(|x| *x).collect(),
        access_type: AccessType::Write,
    };
    let _ = access_tx.send(cmd).await;

    // todo!()
    Ok(())
}

// pub async fn batch_add<
//     A: PartialEq
//         + PartialOrd
//         + Clone
//         + Copy
//         + Field<A>
//         + for<'a> rkyv::Serialize<
//             rancor::Strategy<
//                 rkyv::ser::Serializer<
//                     rkyv::util::AlignedVec,
//                     rkyv::ser::allocator::ArenaHandle<'a>,
//                     rkyv::ser::sharing::Share,
//                 >,
//                 rancor::Error,
//             >,
//         > + Debug
//         + Extremes,
//     B: VectorSpace<A>
//         + Sized
//         + Clone
//         + Copy
//         + PartialEq
//         + Extremes
//         + From<VectorSerial<A>>
//         + Debug
//         + HasPosition<Scalar = f32>,
//     const PARTITION_CAP: usize,
//     const VECTOR_CAP: usize,
//     const MAX_LOADED: usize,
// >(
//     value: Vec<VectorEntry<A, B>>,
//
//     inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,
//
//     partition_buffer: Arc<
//         RwLock<
//             DataBuffer<
//                 Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
//                 PartitionSerial<A>,
//                 Global,
//                 MAX_LOADED,
//             >,
//         >,
//     >,
//     min_spanning_tree_buffer: Arc<
//         RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Global, MAX_LOADED>>,
//     >,
//
//     meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
//
//     cluster_sets: Arc<RwLock<Vec<ClusterSet<A>>>>,
//
//     access_tx: Sender<AccessCommand>,
//
//     #[cfg(feature = "benchmark")] benchmark: Benchmark,
// ) -> Result<(), PartitionErr>
// where
//     VectorSerial<A>: From<B>,
//     for<'a> <A as Archive>::Archived:
//         CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
//     [ArchivedVectorEntrySerial<A>]:
//         DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
//     [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
//     [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
//         DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
//     f32: From<A>,
// {
//     todo!()
// }

async fn first_insert<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<A>
        + for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        > + Debug
        + Extremes,
    B: VectorSpace<A>
        + Sized
        + Clone
        + Copy
        + PartialEq
        + Extremes
        + From<VectorSerial<A>>
        + Debug
        + HasPosition<Scalar = f32>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    vector_entry: VectorEntry<A, B>,

    partition_id: PartitionId,

    meta: &mut Meta<A, B>,
    partition_buffer: &mut DataBuffer<
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        PartitionSerial<A>,
        Global,
        MAX_LOADED,
    >,
    min_spanning_tree_buffer: &mut DataBuffer<
        IntraPartitionGraph<A>,
        GraphSerial<A>,
        Global,
        MAX_LOADED,
    >,
    inter_graph: &mut InterPartitionGraph<A>,
    cluster_sets: &mut Vec<ClusterSet<A>>,
) -> Result<(), ()>
where
    VectorSerial<A>: From<B>,
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<A>]:
        DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    f32: From<A>,
{
    let partition = resolve_buffer!(ACCESS, partition_buffer, partition_id);

    let Some(partition) = &mut *partition.write().await else {
        todo!()
    };
    add_into_partition(vector_entry, partition)
        .expect("Unable to insert value into empty partition");

    let tree = resolve_buffer!(ACCESS, min_spanning_tree_buffer, partition_id);

    let Some(tree) = &mut *tree.write().await else {
        todo!()
    };

    tree.add_node(VectorId(vector_entry.id));

    meta.size = partition.size;
    meta.centroid = partition.centroid();
    // meta.edge_length = (todo!(), todo!());

    for cluster_set in cluster_sets.iter_mut() {
        event!(Level::DEBUG, "cluster_set({:?})", cluster_set.threshold);
        let cluster_id = cluster_set.new_cluster().await.unwrap();
        let _ = cluster_set
            .new_cluster_from_vector(VectorId(vector_entry.id), cluster_id)
            .await
            .unwrap();
    }

    // access_tx
    //     .send(AccessCommand::Release {
    //         resource_ids: required_ids.iter().map(|x| *x).collect(),
    //         access_type: AccessType::Write,
    //     })
    //     .await;

    return Ok(());
}

fn add_into_partition<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    vector: VectorEntry<A, B>,
    partition: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
) -> Result<(), PartitionErr> {
    if partition.size == PARTITION_CAP {
        return Err(PartitionErr::Overflow);
    };

    partition.vectors[partition.size] = Some(vector);

    partition.centroid = B::add(&partition.centroid, &vector.vector);

    partition.size = partition.size + 1;

    Ok(())
}

struct DelaunayVertex<A: Into<f32>, B: HasPosition<Scalar = f32>> {
    id: PartitionId,
    vertex: B,
    _phantom: PhantomData<A>,
}
impl<A: Debug + Into<f32>, B: HasPosition<Scalar = f32>> HasPosition for DelaunayVertex<A, B> {
    type Scalar = f32;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        self.vertex.position()
    }
}

async fn get_neighbors<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<A>
        + for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        > + Debug
        + Extremes,
    B: VectorSpace<A>
        + Sized
        + Clone
        + Copy
        + PartialEq
        + Extremes
        + From<VectorSerial<A>>
        + Debug
        + HasPosition<Scalar = f32>,
    const VECTOR_CAP: usize,
>(
    inter_graph: &InterPartitionGraph<A>,
    meta_data: &HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,

    closet_partition_id: PartitionId,
) -> Vec<PartitionId>
where
    f32: From<A>,
{
    let graph_neighbors = inter_graph
        .0
        .edges(inter_graph.1[&closet_partition_id])
        .map(|edge| edge.weight())
        .map(|edge| (edge.1 .0, edge.2 .0))
        .map(
            |(partition_id_1, partition_id_2)| match partition_id_1 == closet_partition_id {
                true => partition_id_2,
                false => partition_id_1,
            },
        )
        // .flatten()
        .collect::<HashSet<PartitionId>>()
        .drain()
        .collect();

    if inter_graph.1.len() < 3 {
        return graph_neighbors;
    }

    let mut triangulation: DelaunayTriangulation<DelaunayVertex<A, B>> =
        DelaunayTriangulation::new();

    if VECTOR_CAP > 2 {
        // let data: Vec<Vec<f32>> = Vec::new();

        // for (id, meta_data) in meta_data {
        //     let meta_data = &*meta_data.read().await;

        //     let centroid = meta_data.centroid;

        //     let vertex = DelaunayVertex {
        //         id: PartitionId(*id),
        //         vertex: centroid,
        //         _phantom: PhantomData::<A>,
        //     };
        //     data.append(vertex.into());
        // }

        // let model = umap(data);

        // let transformed = model.transform(data);

        // for data in transformed {
        //     let _ = triangulation.insert(data);
        // }
        todo!()
    } else {
        for (id, meta) in meta_data {
            let meta = &*meta.read().await;

            let centroid = meta.centroid;

            let vertex = DelaunayVertex {
                id: PartitionId(*id),
                vertex: centroid,
                _phantom: PhantomData::<A>,
            };
            let _ = triangulation.insert(vertex);
        }
    }

    triangulation
        .inner_faces()
        .filter(|face_handle| {
            let vertices = face_handle.vertices();

            vertices
                .iter()
                .any(|vertex| vertex.data().id == closet_partition_id)
        })
        .flat_map(|face_handle| {
            face_handle
                .adjacent_edges()
                .iter()
                .map(|edge_handle| [edge_handle.from().data().id, edge_handle.to().data().id])
                .filter(|[from, to]| from == &closet_partition_id || to == &closet_partition_id)
                .flatten()
                .collect::<Vec<_>>()
        })
        .chain(graph_neighbors.into_iter())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn find_required_partitions<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<A>
        + for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        > + Debug
        + Extremes,
>(
    sources: &[PartitionId],
    sink: &PartitionId,
    inter_graph: &InterPartitionGraph<A>,
) -> HashSet<PartitionId> {
    let mut required_partitions = HashSet::new();

    for source in sources {
        if source == sink {
            continue;
        }

        let Ok(Some((_, path))) = inter_graph.find_trail(*source, *sink) else {
            todo!();
        };

        path.iter().for_each(|((partition_id, _), ..)| {
            required_partitions.insert(*partition_id);
        });
    }
    required_partitions.insert(*sink);

    required_partitions
}
