use std::{
    cmp::{min, Ordering},
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs,
    sync::Arc,
};

use spade::HasPosition;
use tokio::sync::{mpsc::Sender, oneshot, RwLock};
use tracing::{event, Level};
use uuid::Uuid;

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    rancor,
    rend::u32_le,
    ser::{allocator::ArenaHandle, sharing::Share, Serializer},
    tuple::ArchivedTuple3,
    util::AlignedVec,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, DeserializeUnsized, Serialize,
};

#[cfg(feature = "benchmark")]
use crate::db::component::benchmark::Benchmark;
use crate::{
    db::{
        banker::{AccessMode, AccessResponse, BankerMessage},
        component::{
            cluster::ClusterSet,
            data_buffer::{flush, BufferError, DataBuffer, Global, Local},
            graph::{GraphSerial, IntraPartitionGraph, UpdateTree},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{
                ArchivedVectorEntrySerial, Partition, PartitionSerial, VectorEntry,
                VectorEntrySerial,
            },
        },
        operations::{
            add::{create_local_inter_graph, create_local_meta, expand, get_required_partitions},
            cluster::{remove_cluster_edge, update_cluster},
            split::{
                calculate_number_of_trees, split_partition, split_partition_into_trees,
                FirstTreeSplitStrategy, KMean, MaxAttempt,
            },
        },
    },
    resolve_buffer,
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;
use crate::db::operations::add::get_neighbors;

const LOCAL_BUFFER_SIZE: usize = 16;

pub async fn add<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<A>
        + Archive
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>
        + Debug
        + Extremes
        + std::marker::Send
        + std::marker::Sync
        + 'static,
    B: VectorSpace<A>
        + Sized
        + Clone
        + Copy
        + PartialEq
        + Extremes
        + From<VectorSerial<A>>
        + Debug
        + HasPosition<Scalar = f32>
        + std::marker::Send
        + std::marker::Sync
        + 'static,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    new_vectors: Vec<VectorEntry<A, B>>,

    transaction_id: Option<Uuid>,

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

    access_tx: Sender<BankerMessage>,

    #[cfg(feature = "benchmark")] benchmark: Benchmark,
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
    <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<Pool, rancor::Error>>,
{
    let transaction_id = match transaction_id {
        Some(transaction_id) => transaction_id,
        None => Uuid::new_v4(),
    };

    // acquire required partitions
    println!("{:?} :- Acquire partitions", transaction_id);
    let mut batches = Vec::new();
    let mut un_assigned_vectors: HashSet<_> = new_vectors.clone().into_iter().collect();
    while !un_assigned_vectors.is_empty() {
        tokio::task::yield_now().await;

        let mut meta_data = match meta_data.try_write() {
            Ok(val) => val,
            Err(_) => continue,
        };
        let meta_data = &mut *meta_data;
        event!(Level::DEBUG, "🔒 Locked `meta_data`");

        let inter_graph = &*inter_graph.read().await;
        event!(Level::DEBUG, "🔒 Locked `inter_graph`");

        // find closet id
        let remaining_vectors: HashSet<_> = un_assigned_vectors.clone();

        // maybe make two version of groups
        let groups: Vec<(PartitionId, Vec<VectorEntry<_, _>>)> = group_vectors(
            &mut *partition_buffer.write().await,
            meta_data,
            inter_graph,
            &remaining_vectors,
        )
        .await;
        // for each group try to send jobs
        for (partition_id, new_vectors) in groups {
            // for each group try to send jobs
            let neighbor_ids: Vec<PartitionId> = get_neighbors::<A, B, VECTOR_CAP>(
                inter_graph,
                partition_id,
                meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
            )
            .await;

            let write_partitions: HashSet<PartitionId> =
                get_required_partitions(&neighbor_ids, &partition_id, &inter_graph);

            let read_partitions: HashSet<PartitionId> = expand(
                &write_partitions.iter().map(|x| *x).collect::<Vec<_>>(),
                &inter_graph,
            )
            .difference(&write_partitions)
            .into_iter()
            .map(|x| *x)
            .collect();

            let (tx, rx) = oneshot::channel();

            let _ = access_tx
                .send(BankerMessage::RequestAccess {
                    transaction_id,
                    partitions: read_partitions
                        .clone()
                        .into_iter()
                        .map(|id| (id, AccessMode::Read))
                        .chain(
                            write_partitions
                                .clone()
                                .into_iter()
                                .map(|id| (id, AccessMode::Write)),
                        )
                        .collect(),
                    respond_to: tx,
                })
                .await;

            let acquired_partitions: HashSet<PartitionId> = read_partitions
                .clone()
                .into_iter()
                .map(|id| id)
                .chain(write_partitions.clone().into_iter().map(|id| id))
                .collect();

            match rx.await {
                Ok(AccessResponse::Granted) => {
                    for vector in new_vectors.iter() {
                        un_assigned_vectors.remove(vector);
                    }

                    // todo!() ->  create a local env for batch insert
                    //  -> let insert_batch handle creating a local env for the batch isnert

                    // new_vectors: Vec<VectorEntry<A, B>>,
                    // transaction_id: Option<Uuid>,
                    let batch_transaction_id = Uuid::new_v4();

                    let (input_tx, input_rx) = oneshot::channel();
                    let (output_tx, output_rx) = oneshot::channel();

                    let batch = tokio::spawn(async move {
                        let (
                            new_vectors,
                            partition_id,
                            write_partitions,
                            read_partitions,
                            batch_transaction_id,
                            local_meta_data,
                            local_partition_buffer,
                            local_mst_buffer,
                            local_inter_graph,
                        ) = input_rx.await.unwrap();

                        let data = batch(
                            new_vectors,
                            partition_id,
                            write_partitions,
                            read_partitions,
                            batch_transaction_id,
                            local_meta_data,
                            local_partition_buffer,
                            local_mst_buffer,
                            local_inter_graph,
                        )
                        .await;

                        let _ = output_tx
                            .send(data)
                            .expect("Failed to send data back to parent");
                    });

                    {
                        let local_meta_data: HashMap<Uuid, Arc<RwLock<Meta<A, B>>>> =
                            create_local_meta(&meta_data, &acquired_partitions).await;
                        let local_inter_graph: InterPartitionGraph<A> =
                            create_local_inter_graph(&acquired_partitions, &inter_graph);

                        let local_partition_buffer: DataBuffer<
                            Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
                            PartitionSerial<A>,
                            Local,
                            LOCAL_BUFFER_SIZE,
                        > = partition_buffer
                            .write()
                            .await
                            .copy_local(batch_transaction_id, "partition", &write_partitions)
                            .await;

                        let local_mst_buffer: DataBuffer<
                            IntraPartitionGraph<A>,
                            GraphSerial<A>,
                            Local,
                            LOCAL_BUFFER_SIZE,
                        > = mst_buffer
                            .write()
                            .await
                            .copy_local(batch_transaction_id, "graph", &write_partitions)
                            .await;

                        let _ = input_tx.send((
                            new_vectors,
                            partition_id,
                            write_partitions,
                            read_partitions,
                            batch_transaction_id,
                            local_meta_data,
                            local_partition_buffer,
                            local_mst_buffer,
                            local_inter_graph,
                        ));
                    }
                    batches.push((batch, output_rx));
                }
                _ => {
                    // println!(
                    //     "{:?} :- Access denied for {:?}",
                    //     new_vector.id,
                    //     read_partitions
                    //         .clone()
                    //         .into_iter()
                    //         .chain(write_partitions.clone().into_iter())
                    //         .collect::<HashSet<_>>()
                    // );
                }
            };
        }
    }

    // let mut data = Vec::new();
    for (_batch, rx) in batches {
        // batch.await;

        // data.push(rx.await);

        let (
            local_meta_data,
            local_partition_buffer,
            local_mst_buffer,
            local_inter_graph,
            (new_edges, deleted_edges),
            acquired_partitions,
            assigned_vectors,
            batch_transaction_id,
        ) = rx.await.unwrap();

        {
            // flush partition buffer
            let partition_buffer = &mut *partition_buffer.write().await;

            let _ = flush(local_partition_buffer, partition_buffer)
                .await
                .unwrap();

            // let _ = fs::remove_dir_all(&format!("data/local/{}/partitions", transaction_id.to_string())).unwrap();
        }
        {
            // flush mst buffer
            let mst_buffer = &mut *mst_buffer.write().await;

            let _ = flush(local_mst_buffer, mst_buffer).await.unwrap();
            // let _ = fs::remove_dir_all(&format!("data/local/{}/graph", transaction_id.to_string())).unwrap();
        }
        {
            // update meta_data & inter_graph together
            let meta_data = &mut *meta_data.write().await;
            let inter_graph = &mut *inter_graph.write().await;
            let cluster_sets = &mut *cluster_sets.write().await;
            {
                for (id, local_data) in local_meta_data {
                    match meta_data.contains_key(&id) {
                        true => {
                            let data = &mut *meta_data[&id].write().await;

                            *data = (&*local_data.read().await).clone()
                        }
                        false => {
                            meta_data.insert(
                                id,
                                Arc::new(RwLock::new((&*local_data.read().await).clone())),
                            );
                        }
                    }
                }
            }

            {
                // println!("global:\n{inter_graph:#?}local:\n{local_inter_graph:#?}");
                let mut local_edges = HashSet::new();
                let mut edges = HashMap::new();
                for (id, idx) in &local_inter_graph.1 {
                    if !inter_graph.1.contains_key(&id) {
                        inter_graph.add_node(*id);
                    }

                    for edge_ref in local_inter_graph.0.edges(*idx) {
                        let (weight, id_1, id_2) = edge_ref.weight();

                        local_edges.insert(((*id_1), (*id_2)));
                        edges.insert(((*id_1), (*id_2)), *weight);

                        local_edges.insert(((*id_2), (*id_1)));
                        edges.insert(((*id_2), (*id_1)), *weight);
                    }
                }

                let mut global_edges = HashSet::new();
                for (id, _) in &local_inter_graph.1 {
                    for edge_ref in inter_graph.0.edges(inter_graph.1[&id]) {
                        let (_, id_1, id_2) = edge_ref.weight();

                        if !acquired_partitions.contains(&id_1.0) {
                            continue;
                        }
                        if !acquired_partitions.contains(&id_2.0) {
                            continue;
                        }

                        global_edges.insert(((*id_1), (*id_2)));
                        global_edges.insert(((*id_2), (*id_1)));
                    }
                }

                // remove edges
                for (id_1, id_2) in global_edges.difference(&local_edges) {
                    let _ = inter_graph.remove_edge(*id_1, *id_2);
                    let _ = inter_graph.remove_edge(*id_2, *id_1);
                }

                // add new edges
                let mut visited_edges = HashSet::new();
                for (id_1, id_2) in local_edges.difference(&global_edges) {
                    if visited_edges.contains(&(id_1, id_2)) {
                        continue;
                    }
                    visited_edges.insert((id_1, id_2));
                    visited_edges.insert((id_2, id_1));

                    inter_graph.add_edge(id_1.0, id_2.0, (edges[&(*id_1, *id_2)], *id_1, *id_2))
                }
                // println!("final global: {inter_graph:#?}");
            }

            {
                for cluster_set in cluster_sets.iter_mut() {
                    for vector_id in &assigned_vectors {
                        let cluster_id = cluster_set.new_cluster().unwrap();
                        let _ = cluster_set
                            .new_cluster_from_vector(*vector_id, cluster_id)
                            .unwrap();
                    }
                }

                for ((id_1, id_2), dist) in new_edges {
                    update_cluster(cluster_sets, &dist, id_1, id_2).await;
                }

                for (id_1, id_2) in deleted_edges {
                    remove_cluster_edge(cluster_sets, id_1, id_2).await;
                }
            }
        }
        let _ = fs::remove_dir_all(&format!("data/local/{}", batch_transaction_id.to_string()))
            .unwrap();

        let _ = access_tx
            .send(BankerMessage::ReleaseAccess {
                transaction_id,
                partitions: acquired_partitions,
            })
            .await
            .unwrap();
    }

    Ok(())
}

#[inline(always)]
async fn group_vectors<
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
    S,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    partition_buffer: &mut DataBuffer<
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        PartitionSerial<A>,
        S,
        MAX_LOADED,
    >,
    meta_data: &mut HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,
    inter_graph: &InterPartitionGraph<A>,
    remaining_vectors: &HashSet<VectorEntry<A, B>>,
) -> Vec<(PartitionId, Vec<VectorEntry<A, B>>)>
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
    // find closet partitions
    let mut partition_assignments: HashMap<Uuid, Vec<VectorEntry<A, B>>> = HashMap::new();
    for vector_entry in remaining_vectors.iter() {
        let mut closest_id: Option<Uuid> = None;
        let mut min_dist: Option<A> = None;

        for (id, data) in meta_data.iter() {
            let data = data.read().await;
            let dist = B::dist(&vector_entry.vector, &data.centroid);

            match min_dist {
                None => {
                    min_dist = Some(dist);
                    closest_id = Some(*id);
                }
                Some(current_min) if dist < current_min => {
                    min_dist = Some(dist);
                    closest_id = Some(*id);
                }
                _ => {}
            }
        }

        if let Some(id) = closest_id {
            partition_assignments
                .entry(id)
                .or_default()
                .push(vector_entry.clone());
        }
    }
    // find closet partition by vector
    let partition_assignments = {
        let mut new_partition_assignments: HashMap<Uuid, HashSet<VectorEntry<_, _>>> =
            HashMap::new();
        let mut vector_assignments: HashMap<Uuid, (Uuid, A)> = HashMap::new();

        for (id, vectors) in partition_assignments {
            // getting neighbor ids
            let neighbor_ids: Vec<PartitionId> = get_neighbors::<A, B, VECTOR_CAP>(
                inter_graph,
                PartitionId(id),
                meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
            )
            .await;

            let mut required_partitions: HashSet<PartitionId> = [PartitionId(id)]
                .into_iter()
                .chain(neighbor_ids.iter().cloned())
                .collect();

            loop {
                let mut acquired_partitions = Vec::new();
                for id in required_partitions.iter() {
                    // Replace with try access and/or batch access
                    let Ok(partition) = partition_buffer.access(id).await else {
                        continue;
                    };

                    acquired_partitions.push(partition);
                }

                let mut acquired_partitions_locks = Vec::new();
                {
                    for partition in acquired_partitions.iter() {
                        if let Ok(lock) = partition.try_read() {
                            acquired_partitions_locks.push(lock);
                        }
                    }
                }

                let mut partitions = Vec::new();
                {
                    for partition in acquired_partitions_locks.iter() {
                        if let Some(inner) = &**partition {
                            partitions.push(inner);
                        }
                    }
                }

                if partitions.len() > 0 {
                    for new_vector in &vectors {
                        let mut closet_id: Option<Uuid> = None;
                        let mut closet_dist: Option<A> = None;

                        for partition in &partitions {
                            if partition.size == 0 && closet_id.is_none() && closet_dist.is_none() {
                                closet_dist = Some(A::max());
                                closet_id = Some(partition.id);
                                continue;
                            }
                            for vector_entry in partition.iter() {
                                let dist = B::dist(&new_vector.vector, &vector_entry.vector);

                                if let (Some(min_dist), Some(min_id)) =
                                    (&mut closet_dist, &mut closet_id)
                                {
                                    if &dist < min_dist {
                                        *min_dist = dist;
                                        *min_id = partition.id;
                                    }
                                } else {
                                    closet_dist = Some(dist);
                                    closet_id = Some(partition.id);
                                };
                            }
                        }

                        let closet_id = closet_id.unwrap();
                        let closet_dist = closet_dist.unwrap();

                        if vector_assignments.contains_key(&new_vector.id) {
                            if closet_dist < vector_assignments[&new_vector.id].1 {
                                new_partition_assignments
                                    .get_mut(&vector_assignments[&new_vector.id].0)
                                    .unwrap()
                                    .remove(&new_vector);
                                new_partition_assignments
                                    .entry(closet_id)
                                    .and_modify(|set| {
                                        set.insert(new_vector.clone());
                                    })
                                    .or_insert_with(|| {
                                        let mut set = HashSet::new();
                                        set.insert(new_vector.clone());
                                        set
                                    });

                                vector_assignments.insert(new_vector.id, (closet_id, closet_dist));
                            }
                        } else {
                            new_partition_assignments
                                .entry(closet_id)
                                .and_modify(|set| {
                                    set.insert(new_vector.clone());
                                })
                                .or_insert_with(|| {
                                    let mut set: HashSet<VectorEntry<A, B>> = HashSet::new();
                                    set.insert(new_vector.clone());
                                    set
                                });
                            vector_assignments.insert(new_vector.id, (closet_id, closet_dist));
                        }
                    }

                    // let mut remove_index: Vec<_> =
                    partitions
                        .iter()
                        .map(|partition| PartitionId(partition.id))
                        .for_each(|id| {
                            required_partitions.remove(&id);
                        });

                    if required_partitions.len() == 0 {
                        break;
                    }
                    partitions.clear();
                    acquired_partitions_locks = Vec::new();
                    acquired_partitions = Vec::new();
                }

                //unload and swap
                let mut least_used = partition_buffer.least_used_iter().await;

                for id in required_partitions.clone() {
                    match partition_buffer.load(&*id).await {
                        Ok(_) => {
                            event!(Level::DEBUG, "📦 Load buffer space");
                            // partitions.push(partition_buffer.access(&*id).await.unwrap());
                        }
                        Err(BufferError::OutOfSpace) => {
                            event!(Level::DEBUG, "📦 Unload and Load buffer space");

                            let Some(least_used) = &mut least_used else {
                                continue;
                            };

                            let Some(unload_id) = least_used.next() else {
                                break;
                            };

                            partition_buffer
                                .unload_and_load(&unload_id.1, &*id)
                                .await
                                .unwrap();
                            // partitions.push(partition_buffer.access(&*id).await.unwrap());
                        }
                        Err(BufferError::FileNotFound) => {
                            todo!()
                        }
                        Err(_) => todo!(),
                    };
                }
            }
        }

        new_partition_assignments
    };
    partition_assignments
        .into_iter()
        .map(|(id, val)| (PartitionId(id), val.into_iter().collect::<Vec<_>>()))
        .collect()
}

#[inline(always)]
async fn batch<
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
    new_vectors: Vec<VectorEntry<A, B>>,
    closet_partition: PartitionId,

    write_partitions: HashSet<PartitionId>,
    read_partitions: HashSet<PartitionId>,

    transaction_id: Uuid,

    mut meta_data: HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,
    mut partition_buffer: DataBuffer<
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        PartitionSerial<A>,
        Local,
        MAX_LOADED,
    >,
    mut mst_buffer: DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Local, MAX_LOADED>,

    mut inter_graph: InterPartitionGraph<A>,

    #[cfg(feature = "benchmark")] benchmark: Benchmark,
) -> (
    HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,
    DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionSerial<A>, Local, MAX_LOADED>,
    DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Local, MAX_LOADED>,
    InterPartitionGraph<A>,
    (
        HashMap<(VectorId, VectorId), A>,
        HashSet<(VectorId, VectorId)>,
    ),
    Vec<PartitionId>,
    Vec<VectorId>,
    Uuid,
)
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
    let assigned_vectors = new_vectors.clone();

    let mut new_edges: HashMap<(VectorId, VectorId), A> = HashMap::new();
    let mut deleted_edges: HashSet<(VectorId, VectorId)> = HashSet::new();

    let mut groups = vec![(closet_partition, new_vectors)];

    while groups.len() > 0 {
        let (insert_partition_id, mut vectors) = groups.pop().unwrap();

        println!("transaction_id -: {transaction_id:?}\ninsert_partition_id:{insert_partition_id:?}\nvectors:{vectors:?}");

        println!("transaction_id -: {transaction_id:?} Finding max insert size");
        let (insert_vectors, remaining_vectors) = {
            let size: usize = meta_data[&closet_partition].read().await.size;

            let remaining: Vec<VectorEntry<A, B>> =
                vectors.split_off(min(PARTITION_CAP - size, vectors.len()));
            let insert: Vec<VectorEntry<A, B>> = vectors;

            println!(
                "{}\t{:?}\t{:?}",
                PARTITION_CAP - size,
                insert.len(),
                remaining.len()
            );

            (insert, remaining)
        };

        // insert vectors
        println!("transaction_id -: {transaction_id:?} Inserting vectors into partition");
        {
            let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition);

            let Some(partition) = &mut *partition.write().await else {
                todo!();
            };

            debug_assert!(partition.size + insert_vectors.len() <= PARTITION_CAP);

            partition.vectors[partition.size..partition.size + insert_vectors.len()]
                .clone_from_slice(
                    &insert_vectors
                        .iter()
                        .map(|x| Some(x.clone()))
                        .collect::<Vec<_>>(),
                );
            println!(
                "transaction_id -: {transaction_id:?}\t{:?}",
                partition.vectors
            );
            for vector in &insert_vectors {
                partition.centroid = B::add(&partition.centroid, &vector.vector);
            }

            partition.size = partition.size + insert_vectors.len();

            let mut data = &mut *meta_data[&closet_partition].write().await;

            data.size = partition.size;
            data.centroid = partition.centroid();
        };

        // update mst
        println!("transaction_id -: {transaction_id:?} Update MST");
        {
            println!("transaction_id -: {transaction_id:?} Getting neighbors");
            let neighbor_ids: HashSet<PartitionId> = {
                get_neighbors::<A, B, VECTOR_CAP>(
                    &inter_graph,
                    insert_partition_id,
                    &meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
                )
                .await
                .into_iter()
                .filter(|id| !read_partitions.contains(id))
                .collect()
            };

            println!("transaction_id -: {transaction_id:?} Update dist_map");
            let dist_map: HashMap<(VectorId, VectorId), A> = {
                let mut missing_partitions = [insert_partition_id]
                    .into_iter()
                    .chain(neighbor_ids.clone())
                    .collect::<HashSet<_>>();
                let mut dist_map: HashMap<(VectorId, VectorId), A> = HashMap::new();

                while missing_partitions.len() > 0 {
                    let mut acquired_partitions = Vec::new();
                    event!(
                        Level::DEBUG,
                        "Attempt to acquired required ids that aren't loaded"
                    );
                    for id in missing_partitions.iter() {
                        // Replace with try access and/or batch access
                        let Ok(partition) = partition_buffer.access(&**id).await else {
                            event!(Level::WARN, "⚠️ Failed to access partition {id:?}");
                            continue;
                        };

                        acquired_partitions.push(partition);
                    }

                    let mut acquired_partitions_locks = Vec::new();
                    {
                        for partition in acquired_partitions.iter() {
                            if let Ok(lock) = partition.try_read() {
                                acquired_partitions_locks.push(lock);
                            }
                        }
                    }

                    let mut partitions = Vec::new();
                    {
                        for partition in acquired_partitions_locks.iter() {
                            if let Some(inner) = &**partition {
                                partitions.push(inner);
                            }
                        }
                    }

                    if !partitions.is_empty() {
                        event!(Level::DEBUG, "📥 Processing newly acquired partitions");
                        // get closet_id & dist
                        // get get closet dist for each partition
                        for partition in &partitions {
                            for VectorEntry { id, vector, .. } in partition.iter() {
                                for VectorEntry {
                                    id: new_id,
                                    vector: new_vector,
                                    ..
                                } in &insert_vectors
                                {
                                    if id == new_id {
                                        continue;
                                    }
                                    dist_map.insert(
                                        (VectorId(*id), VectorId(*new_id)),
                                        B::dist(&new_vector, vector),
                                    );
                                    dist_map.insert(
                                        (VectorId(*new_id), VectorId(*id)),
                                        B::dist(&new_vector, vector),
                                    );
                                }
                            }

                            let id = PartitionId(partition.id);
                            missing_partitions.remove(&id);
                        }

                        if missing_partitions.is_empty() {
                            event!(Level::DEBUG, "✅ All required partitions loaded");
                            break;
                        }
                        partitions.clear();
                        acquired_partitions_locks = Vec::new();
                        acquired_partitions = Vec::new();
                    }

                    //unload and swap
                    let mut least_used = partition_buffer.least_used_iter().await;

                    for id in missing_partitions.clone() {
                        match partition_buffer.load(&*id).await {
                            Ok(_) => {
                                event!(Level::DEBUG, "📦 Load buffer space");
                                // partitions.push(partition_buffer.access(&*id).await.unwrap());
                            }
                            Err(BufferError::OutOfSpace) => {
                                event!(
                                    Level::DEBUG,
                                    "📦 Unloading and loading buffer space for partition {id:?}"
                                );

                                let Some(least_used) = &mut least_used else {
                                    continue;
                                };

                                let Some((_unload_idx, unload_id)) = least_used.next() else {
                                    break;
                                };

                                if missing_partitions.contains(&PartitionId(unload_id)) {
                                    continue;
                                }

                                partition_buffer
                                    .unload_and_load(&unload_id, &*id)
                                    .await
                                    .unwrap();
                                // partitions.push(partition_buffer.access(&*id).await.unwrap());
                            }
                            Err(BufferError::FileNotFound) => {
                                event!(Level::ERROR, "🛑 Partition {id:?} file not found!");
                                todo!()
                            }
                            Err(_) => {
                                event!(
                                    Level::ERROR,
                                    "🛑 Unexpected error loading partition {id:?}"
                                );
                                todo!()
                            }
                        };
                    }
                }

                dist_map
            };

            if dist_map.len() == 0 {
                todo!()
            }

            println!("transaction_id -: {transaction_id:?} Update closet_partition");
            // Update closest partition edges
            {
                println!("transaction_id -: {transaction_id:?} Getting buffer");
                let min_span_tree = resolve_buffer!(ACCESS, mst_buffer, insert_partition_id);

                let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                    todo!()
                };

                for VectorEntry { id, .. } in &insert_vectors {
                    min_span_tree.add_node(VectorId(*id));
                }

                let Ok(tmp_new_edges) = min_span_tree.batch_update(
                    insert_vectors
                        .iter()
                        .map(|x| VectorId(x.id))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    &dist_map
                        .iter()
                        .filter(|(id, _)| {
                            min_span_tree.1.contains_key(&id.0)
                                && min_span_tree.1.contains_key(&id.1)
                        })
                        .map(|(id, val)| ((id.0, id.1), *val))
                        .fold(HashMap::new(), |mut acc, ((id_1, id_2), val)| {
                            if acc.contains_key(&(id_1, id_2)) || acc.contains_key(&(id_2, id_1)) {
                                return acc;
                            }

                            acc.insert((id_1, id_2), val);

                            acc
                        })
                        .into_iter()
                        .map(|(id, val)| (val, id.0, id.1))
                        .collect::<Vec<_>>(),
                ) else {
                    todo!();
                };

                for (weight, id_1, id_2) in tmp_new_edges {
                    new_edges.insert((id_1, id_2), weight);
                }
            }

            // update neighbor partitions
            // {
            //     let mut start_partitions = HashSet::new();
            //     start_partitions.insert(insert_partition_id);

            //     let mut visited_new_vectors = HashSet::new();

            //     let mut partition_path_cache: HashMap<(PartitionId, VectorId, VectorId), Vec<_>> =
            //         HashMap::new();
            //     let mut cached_partitions: HashMap<
            //         PartitionId,
            //         HashSet<(PartitionId, VectorId, VectorId)>,
            //     > = HashMap::new();

            //     'vector_loop: for new_vector in &insert_vectors {
            //         if visited_new_vectors.contains(&new_vector.id) {
            //             continue;
            //         }
            //         let mut hints_cache: HashMap<(PartitionId, VectorId), HashMap<VectorId, A>> =
            //             HashMap::new();
            //         let mut cached_partition_hints: HashMap<
            //             PartitionId,
            //             HashSet<(PartitionId, VectorId)>,
            //         > = HashMap::new();

            //         while start_partitions.len() > 0 {
            //             let start_partition = *start_partitions.iter().next().unwrap();
            //             let mut start_partition = start_partitions.take(&start_partition).unwrap();

            //             {
            //                 let mst = resolve_buffer!(ACCESS, mst_buffer, start_partition);

            //                 let Some(mst) = &*mst.read().await else {
            //                     todo!()
            //                 };

            //                 if !mst.1.contains_key(&VectorId(new_vector.id)) {
            //                     continue 'vector_loop;
            //                 }
            //                 visited_new_vectors.insert(new_vector.id);
            //             }

            //             let mut visit_requirements: HashSet<PartitionId> = neighbor_ids.clone();
            //             visit_requirements.take(&insert_partition_id);

            //             let mut visited_vertices: HashSet<VectorId> = HashSet::new();

            //             while let Some(target_partition_id) =
            //                 visit_requirements.iter().next().cloned()
            //             {
            //                 let target_vector_ids = {
            //                     let min_span_tree =
            //                         resolve_buffer!(ACCESS, mst_buffer, target_partition_id);

            //                     let Some(min_span_tree) = &mut *min_span_tree.write().await else {
            //                         todo!()
            //                     };

            //                     min_span_tree.1.clone()
            //                 };

            //                 let mut partition_trail = {
            //                     let Ok(partition_trail) =
            //                         inter_graph.find_trail(start_partition, target_partition_id)
            //                     else {
            //                         todo!()
            //                     };

            //                     let Some((_, partition_trail)) = partition_trail else {
            //                         panic!("Failed to find partition trail from {start_partition:?} -> {target_partition_id:?}\n{inter_graph:#?}");
            //                     };

            //                     event!(Level::DEBUG, "partition_trail:\n{partition_trail:?}");

            //                     partition_trail
            //                 };

            //                 for (target_vector_id, _) in target_vector_ids {
            //                     if visited_vertices.contains(&target_vector_id) {
            //                         continue;
            //                     }
            //                     visited_vertices.insert(target_vector_id);
            //                     {
            //                         let min_span_tree = resolve_buffer!(
            //                             ACCESS,
            //                             mst_buffer,
            //                             target_partition_id,
            //                             [*target_partition_id]
            //                         );

            //                         let Some(min_span_tree) = &mut *min_span_tree.write().await
            //                         else {
            //                             todo!()
            //                         };

            //                         if !min_span_tree.1.contains_key(&target_vector_id) {
            //                             continue;
            //                         }
            //                     }

            //                     let Some(weight) =
            //                         dist_map.get(&(target_vector_id, VectorId(new_vector.id)))
            //                     else {
            //                         continue;
            //                     };

            //                     let trails = {
            //                         let mut trails: Vec<(
            //                             (PartitionId, VectorId),
            //                             (PartitionId, VectorId),
            //                             A,
            //                         )> = 'initialize_trail: {
            //                             let inter_edge = partition_trail[0];

            //                             if VectorId(new_vector.id) == inter_edge.0 .1 {
            //                                 break 'initialize_trail vec![inter_edge];
            //                             }
            //                             let min_span_tree = resolve_buffer!(
            //                                 ACCESS,
            //                                 mst_buffer,
            //                                 start_partition,
            //                                 [*target_partition_id]
            //                             );

            //                             let Some(min_span_tree) = &mut *min_span_tree.write().await
            //                             else {
            //                                 todo!()
            //                             };

            //                             // djikstra's hint
            //                             // Sink (value.id)
            //                             let hints = match hints_cache.contains_key(&(
            //                                 start_partition,
            //                                 VectorId(new_vector.id),
            //                             )) {
            //                                 true => {
            //                                     &hints_cache
            //                                         [&(start_partition, VectorId(new_vector.id))]
            //                                 }
            //                                 false => {
            //                                     hints_cache.insert(
            //                                         (start_partition, VectorId(new_vector.id)),
            //                                         min_span_tree
            //                                             .dijkstra_weights(VectorId(new_vector.id))
            //                                             .unwrap(),
            //                                     );

            //                                     cached_partition_hints
            //                                         .entry(start_partition)
            //                                         .or_insert_with(|| {
            //                                             let mut set = HashSet::new();
            //                                             set.insert((
            //                                                 start_partition,
            //                                                 VectorId(new_vector.id),
            //                                             ));

            //                                             set
            //                                         })
            //                                         .insert((
            //                                             start_partition,
            //                                             VectorId(new_vector.id),
            //                                         ));

            //                                     &hints_cache
            //                                         [&(start_partition, VectorId(new_vector.id))]
            //                                 }
            //                             };

            //                             let Ok(trail) = min_span_tree.find_trail_with_hints(
            //                                 inter_edge.0 .1,
            //                                 VectorId(new_vector.id),
            //                                 &hints,
            //                             ) else {
            //                                 event!(
            //                                     Level::DEBUG,
            //                                     "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
            //                                     VectorId(new_vector.id),
            //                                     inter_edge.0 .1,
            //                                     inter_edge.1,
            //                                 );

            //                                 todo!()
            //                             };

            //                             let Some(trail) = trail else {
            //                                 event!(
            //                                     Level::DEBUG,
            //                                     "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
            //                                     VectorId(new_vector.id),
            //                                     inter_edge.0 .1,
            //                                     inter_edge.1,
            //                                 );
            //                                 todo!()
            //                             };

            //                             trail
            //                                 .1
            //                                 .into_iter()
            //                                 .map(|(vector_id_1, vector_id_2, weight)| {
            //                                     (
            //                                         (start_partition, vector_id_1),
            //                                         (start_partition, vector_id_2),
            //                                         weight,
            //                                     )
            //                                 })
            //                                 .chain(vec![inter_edge].into_iter())
            //                                 .collect()
            //                         };

            //                         for i in 1..partition_trail.len() {
            //                             let (_, (partition_id_1, vector_id_1), _) =
            //                                 partition_trail[i - 1];
            //                             let (
            //                                 (partition_id_2, vector_id_2),
            //                                 (partition_id_3, vector_id_3),
            //                                 edge_2_weight,
            //                             ) = partition_trail[i];

            //                             if vector_id_1 == vector_id_2 {
            //                                 trails.push((
            //                                     (partition_id_2, vector_id_2),
            //                                     (partition_id_3, vector_id_3),
            //                                     edge_2_weight,
            //                                 ));
            //                                 continue;
            //                             }

            //                             if partition_path_cache.contains_key(&(
            //                                 partition_id_1,
            //                                 vector_id_1,
            //                                 vector_id_2,
            //                             )) {
            //                                 trails.extend(
            //                                     partition_path_cache
            //                                         [&(partition_id_1, vector_id_1, vector_id_2)]
            //                                         .clone(),
            //                                 );
            //                                 trails.push((
            //                                     (partition_id_2, vector_id_2),
            //                                     (partition_id_3, vector_id_3),
            //                                     edge_2_weight,
            //                                 ));
            //                                 continue;
            //                             }
            //                             if partition_path_cache.contains_key(&(
            //                                 partition_id_1,
            //                                 vector_id_2,
            //                                 vector_id_1,
            //                             )) {
            //                                 trails.extend(
            //                                     partition_path_cache
            //                                         [&(partition_id_1, vector_id_2, vector_id_1)]
            //                                         .clone(),
            //                                 );
            //                                 trails.push((
            //                                     (partition_id_2, vector_id_2),
            //                                     (partition_id_3, vector_id_3),
            //                                     edge_2_weight,
            //                                 ));
            //                                 continue;
            //                             }

            //                             let min_span_tree =
            //                                 resolve_buffer!(ACCESS, mst_buffer, partition_id_1);

            //                             let Some(min_span_tree) = &mut *min_span_tree.write().await
            //                             else {
            //                                 todo!()
            //                             };

            //                             let Ok(trail) =
            //                                 min_span_tree.find_trail(vector_id_1, vector_id_2)
            //                             else {
            //                                 event!(
            //                                     Level::DEBUG,
            //                                     "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
            //                                     VectorId(new_vector.id),
            //                                     (partition_id_2, vector_id_2),
            //                                     (partition_id_3, vector_id_3)
            //                                 );
            //                                 todo!()
            //                             };

            //                             let Some((_, trail)) = trail else { todo!() };

            //                             let trail: Vec<_> = trail
            //                                 .iter()
            //                                 .map(|(vector_id_1, vector_id_2, weight)| {
            //                                     (
            //                                         (partition_id_1, *vector_id_1),
            //                                         (partition_id_1, *vector_id_2),
            //                                         *weight,
            //                                     )
            //                                 })
            //                                 .collect();
            //                             partition_path_cache.insert(
            //                                 (partition_id_1, vector_id_1, vector_id_2),
            //                                 trail.clone(),
            //                             );
            //                             cached_partitions
            //                                 .entry(partition_id_1)
            //                                 .or_insert_with(|| {
            //                                     let mut set = HashSet::new();
            //                                     set.insert((
            //                                         partition_id_1,
            //                                         vector_id_1,
            //                                         vector_id_2,
            //                                     ));

            //                                     set
            //                                 })
            //                                 .insert((partition_id_1, vector_id_1, vector_id_2));
            //                             trails.push((
            //                                 (partition_id_2, vector_id_2),
            //                                 (partition_id_3, vector_id_3),
            //                                 edge_2_weight,
            //                             ));
            //                             trails.extend(trail)
            //                         }

            //                         'final_trail: {
            //                             // if partition_trail.len() == 1 {
            //                             //     break 'final_trail;
            //                             // }
            //                             let (_, (partition_id_1, vector_id_1), _) =
            //                                 partition_trail.last().unwrap();

            //                             if vector_id_1 == &target_vector_id {
            //                                 break 'final_trail;
            //                             }

            //                             let min_span_tree = resolve_buffer!(
            //                                 ACCESS,
            //                                 mst_buffer,
            //                                 *partition_id_1,
            //                                 [*target_partition_id]
            //                             );

            //                             let Some(min_span_tree) = &mut *min_span_tree.write().await
            //                             else {
            //                                 todo!()
            //                             };

            //                             let Ok(path) = min_span_tree.find_trail(
            //                                 target_vector_id,
            //                                 *vector_id_1,
            //                                 // &hints,
            //                             ) else {
            //                                 event!(
            //                                     Level::DEBUG,
            //                                     "Failed to find trail:\n{min_span_tree:?}\n{:?}\n{:?} => {:?}",
            //                                     VectorId(new_vector.id),
            //                                     vector_id_1,
            //                                     target_vector_id
            //                                 );
            //                                 todo!()
            //                             };
            //                             let Some(path) = path else { todo!() };

            //                             trails.extend(
            //                                 path.1.into_iter().map(
            //                                     |(vector_id_1, vector_id_2, weight)| {
            //                                         (
            //                                             (*partition_id_1, vector_id_1),
            //                                             (*partition_id_1, vector_id_2),
            //                                             weight,
            //                                         )
            //                                     },
            //                                 ), // .collect::<Vec<_>>()
            //                             )
            //                         };

            //                         trails
            //                     };

            //                     let ((partition_id_1, vector_id_1), (partition_id_2, vector_id_2), max_weight) =
            //                         trails.into_iter().fold(
            //                             (
            //                                 (PartitionId(Uuid::nil()), VectorId(Uuid::nil())),
            //                                 (PartitionId(Uuid::nil()), VectorId(Uuid::nil())),
            //                                 A::min(),
            //                             ),
            //                             |(acc_id_1, acc_id_2, acc_weight),
            //                             (next_id_1, next_id_2, next_weight)| {
            //                                 match next_weight.partial_cmp(&acc_weight) {
            //                                     Some(Ordering::Greater) => (next_id_1, next_id_2, next_weight),
            //                                     _ => (acc_id_1, acc_id_2, acc_weight),
            //                                 }
            //                             },
            //                         );
            //                     if partition_id_1 == PartitionId(Uuid::nil()) {
            //                         continue;
            //                     }
            //                     if partition_id_2 == PartitionId(Uuid::nil()) {
            //                         continue;
            //                     }

            //                     event!(
            //                         Level::DEBUG,
            //                         "Max edge {:?}",
            //                         (
            //                             (partition_id_1, vector_id_1),
            //                             (partition_id_2, vector_id_2),
            //                             max_weight,
            //                             weight
            //                         )
            //                     );
            //                     // if max weight < weight
            //                     if weight >= &max_weight {
            //                         continue;
            //                     };

            //                     if cached_partitions.contains_key(&partition_id_1) {
            //                         for (partition_id, vector_1, vector_2) in
            //                             cached_partitions.remove(&partition_id_1).unwrap()
            //                         {
            //                             partition_path_cache.remove(&(
            //                                 partition_id,
            //                                 vector_1,
            //                                 vector_2,
            //                             ));
            //                             partition_path_cache.remove(&(
            //                                 partition_id,
            //                                 vector_2,
            //                                 vector_1,
            //                             ));
            //                         }
            //                     }
            //                     if cached_partitions.contains_key(&partition_id_2) {
            //                         for (partition_id, vector_1, vector_2) in
            //                             cached_partitions.remove(&partition_id_2).unwrap()
            //                         {
            //                             partition_path_cache.remove(&(
            //                                 partition_id,
            //                                 vector_1,
            //                                 vector_2,
            //                             ));
            //                             partition_path_cache.remove(&(
            //                                 partition_id,
            //                                 vector_2,
            //                                 vector_1,
            //                             ));
            //                         }
            //                     }

            //                     if cached_partition_hints.contains_key(&partition_id_1) {
            //                         for id in
            //                             cached_partition_hints.remove(&partition_id_1).unwrap()
            //                         {
            //                             cached_partition_hints.remove(&partition_id_1);
            //                             hints_cache.remove(&id);
            //                         }
            //                     };

            //                     if cached_partition_hints.contains_key(&partition_id_2) {
            //                         for id in
            //                             cached_partition_hints.remove(&partition_id_2).unwrap()
            //                         {
            //                             cached_partition_hints.remove(&partition_id_2);
            //                             hints_cache.remove(&id);
            //                         }
            //                     };

            //                     inter_graph.add_edge(
            //                         start_partition,
            //                         target_partition_id,
            //                         (
            //                             *weight,
            //                             (start_partition, VectorId(new_vector.id)),
            //                             (target_partition_id, target_vector_id),
            //                         ),
            //                     );

            //                     if new_edges.contains_key(&(vector_id_1, vector_id_2))
            //                         || new_edges.contains_key(&(vector_id_2, vector_id_1))
            //                     {
            //                         new_edges.remove(&(vector_id_1, vector_id_2));
            //                         new_edges.remove(&(vector_id_2, vector_id_1));
            //                     }
            //                     if !deleted_edges.contains(&(vector_id_2, vector_id_1)) {
            //                         deleted_edges.insert((vector_id_1, vector_id_2));
            //                     }

            //                     if deleted_edges
            //                         .contains(&(VectorId(new_vector.id), target_vector_id))
            //                         || deleted_edges
            //                             .contains(&(target_vector_id, VectorId(new_vector.id)))
            //                     {
            //                         deleted_edges
            //                             .remove(&(VectorId(new_vector.id), target_vector_id));
            //                         deleted_edges
            //                             .remove(&(target_vector_id, VectorId(new_vector.id)));
            //                     }

            //                     if !new_edges
            //                         .contains_key(&(target_vector_id, VectorId(new_vector.id)))
            //                     {
            //                         new_edges.insert(
            //                             (VectorId(new_vector.id), target_vector_id),
            //                             *weight,
            //                         );
            //                     }

            //                     match partition_id_1 == partition_id_2 {
            //                         true => {
            //                             let split_partition_id = partition_id_1;

            //                             let min_span_tree = resolve_buffer!(
            //                                 ACCESS,
            //                                 mst_buffer,
            //                                 split_partition_id,
            //                                 [*target_partition_id]
            //                             );
            //                             let Some(min_span_tree) = &mut *min_span_tree.write().await
            //                             else {
            //                                 todo!()
            //                             };

            //                             let partition = resolve_buffer!(
            //                                 ACCESS,
            //                                 partition_buffer,
            //                                 split_partition_id,
            //                                 [*target_partition_id]
            //                             );
            //                             let Some(partition) = &mut *partition.write().await else {
            //                                 todo!()
            //                             };

            //                             let _ = min_span_tree
            //                                 .remove_edge(vector_id_1, vector_id_2)
            //                                 .unwrap();

            //                             let [pair_1, pair_2] =
            //                                 split_partition::<
            //                                     A,
            //                                     B,
            //                                     FirstTreeSplitStrategy,
            //                                     PARTITION_CAP,
            //                                     VECTOR_CAP,
            //                                 >(
            //                                     partition, min_span_tree, &mut inter_graph
            //                                 )
            //                                 .unwrap();
            //                             // split_partition_into_trees(partition, min_span_tree, inter_graph)
            //                             //     .unwrap();

            //                             let (pair_1, pair_2) =
            //                                 match pair_1.0.id == *split_partition_id {
            //                                     true => (pair_1, pair_2),
            //                                     false => (pair_2, pair_1),
            //                                 };

            //                             event!(
            //                                 Level::DEBUG,
            //                                 "split data: {:#?}\n{inter_graph:#?}",
            //                                 [&pair_1, &pair_2]
            //                             );

            //                             {
            //                                 let (new_partition, new_min_span_tree) = pair_1;

            //                                 if new_partition.id != *split_partition_id {
            //                                     todo!()
            //                                 }

            //                                 *partition = new_partition;
            //                                 *min_span_tree = new_min_span_tree;

            //                                 let target_meta: &mut Meta<A, B> =
            //                                     &mut *meta_data[&*split_partition_id].write().await;

            //                                 target_meta.size = partition.size;
            //                                 target_meta.centroid = partition.centroid();

            //                                 target_meta.edge_length = (
            //                                     match min_span_tree.smallest_edge() {
            //                                         Some(x) => x.2,
            //                                         None => A::max(),
            //                                     },
            //                                     match min_span_tree.largest_edge() {
            //                                         Some(x) => x.2,
            //                                         None => A::min(),
            //                                     },
            //                                 );
            //                             }

            //                             // drop(partition);
            //                             // drop(min_span_tree);

            //                             if split_partition_id == target_partition_id {
            //                                 visit_requirements.insert(PartitionId(pair_2.0.id));
            //                             }

            //                             if split_partition_id == start_partition {
            //                                 start_partitions.insert(split_partition_id);
            //                             }

            //                             let (new_partition, new_min_span_tree) = pair_2;

            //                             if new_min_span_tree
            //                                 .1
            //                                 .contains_key(&VectorId(new_vector.id))
            //                             {
            //                                 event!(
            //                                     Level::DEBUG,
            //                                     "Updating closet_partition_id: {start_partition:?}->{split_partition_id:?}"
            //                                 );
            //                                 start_partition = PartitionId(new_partition.id);
            //                             }

            //                             meta_data.insert(
            //                                 new_partition.id,
            //                                 Arc::new(RwLock::new(Meta::new(
            //                                     PartitionId(new_partition.id),
            //                                     new_partition.size,
            //                                     new_partition.centroid(),
            //                                     (
            //                                         match new_min_span_tree.smallest_edge() {
            //                                             Some(x) => x.2,
            //                                             None => A::max(),
            //                                         },
            //                                         match new_min_span_tree.largest_edge() {
            //                                             Some(x) => x.2,
            //                                             None => A::min(),
            //                                         },
            //                                     ),
            //                                 ))),
            //                             );

            //                             resolve_buffer!(
            //                                 PUSH,
            //                                 mst_buffer,
            //                                 new_min_span_tree,
            //                                 [*target_partition_id, *split_partition_id]
            //                             );
            //                             resolve_buffer!(
            //                                 PUSH,
            //                                 partition_buffer,
            //                                 new_partition,
            //                                 [*target_partition_id, *split_partition_id]
            //                             );
            //                         }
            //                         false => {
            //                             event!(
            //                                 Level::DEBUG,
            //                                 "Removing edge inter edge: {:?}",
            //                                 (
            //                                     (partition_id_1, vector_id_1),
            //                                     (partition_id_2, vector_id_2)
            //                                 )
            //                             );
            //                             let _ = inter_graph
            //                                 .remove_edge(
            //                                     (partition_id_1, vector_id_1),
            //                                     (partition_id_2, vector_id_2),
            //                                 )
            //                                 .unwrap();
            //                         }
            //                     };

            //                     partition_trail = {
            //                         let Ok(partition_trail) = inter_graph
            //                             .find_trail(start_partition, target_partition_id)
            //                         else {
            //                             todo!()
            //                             // continue 'ordered_edges_loop; // Should be replaced with panic as this edge case should never happen
            //                         };

            //                         let Some((_, partition_trail)) = partition_trail else {
            //                             panic!("Failed to find partition trail from {start_partition:?} -> {target_partition_id:?}\n{inter_graph:#?}");
            //                         };

            //                         event!(Level::DEBUG, "partition_trail:\n{partition_trail:?}");

            //                         partition_trail
            //                     };
            //                 }

            //                 visit_requirements.take(&target_partition_id);
            //             }
            //         }
            //     }
            // }
        }

        // split if required
        println!("transaction_id -: {transaction_id:?} Splitting partitions");
        {
            let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition);

            let Some(partition) = &mut *partition.write().await else {
                todo!();
            };

            // splits partition if required
            if partition.size == PARTITION_CAP {
                let mut push_pairs = Vec::new();

                println!("transaction_id -: {transaction_id:?} Split Partition getting mst");
                let mst = resolve_buffer!(ACCESS, mst_buffer, closet_partition);
                let Some(mst) = &mut *mst.write().await else {
                    todo!();
                };

                let Ok(splits) =
                    split_partition::<A, B, KMean<MaxAttempt<128>>, PARTITION_CAP, VECTOR_CAP>(
                        partition,
                        mst,
                        &mut inter_graph,
                    )
                else {
                    panic!()
                };

                for (mut new_partition, mut new_graph) in splits {
                    // fix any forests
                    let pairs = match calculate_number_of_trees(&new_graph) > 1 {
                        true => {
                            event!(Level::DEBUG, "Fixing forest");
                            split_partition_into_trees(
                                &mut new_partition,
                                &mut new_graph,
                                &mut inter_graph,
                            )
                            .unwrap()
                        }
                        false => vec![(new_partition, new_graph)],
                    };

                    // if last value is closet_partition (then maybe skip that value and handle the edge case after the for loop)
                    for (new_partition, new_mst) in pairs {
                        // update partition if new_partition matches
                        if new_partition.id == partition.id {
                            *partition = new_partition;
                            *mst = new_mst;

                            let target_meta: &mut Meta<A, B> =
                                &mut *meta_data[&partition.id].write().await;

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
                        } else {
                            meta_data.insert(
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
                                        // new_graph.smallest_edge().unwrap().2,
                                        // new_graph.largest_edge().unwrap().2,
                                    ),
                                ))),
                            );

                            push_pairs.push((new_partition, new_mst));
                        }
                    }
                }

                for (new_partition, new_graph) in push_pairs {
                    resolve_buffer!(PUSH, partition_buffer, new_partition, [partition.id]);
                    resolve_buffer!(PUSH, mst_buffer, new_graph, [*mst.2]);
                }
            }
        }
        {
            if remaining_vectors.len() == 0 {
                continue;
            }

            println!(
                "transaction_id -: {transaction_id:?} Finding new groups with remaining vectors"
            );
            let new_groups = group_vectors(
                &mut partition_buffer,
                &mut meta_data
                    .clone()
                    .into_iter()
                    .filter(|(id, _)| !read_partitions.contains(&PartitionId(*id)))
                    .collect(),
                &inter_graph,
                &remaining_vectors.into_iter().collect(),
            )
            .await;

            groups.extend(new_groups);
        }
    }

    (
        // new_vectors: Vec<VectorEntry<A, B>>,
        // closet_partition: PartitionId,

        // write_partitions: HashSet<PartitionId>,
        // read_partitions: HashSet<PartitionId>,

        // transaction_id: Uuid,
        meta_data,
        partition_buffer,
        mst_buffer,
        inter_graph,
        (new_edges, deleted_edges),
        read_partitions
            .into_iter()
            .chain(write_partitions.into_iter())
            .collect(),
        assigned_vectors
            .iter()
            .map(|VectorEntry { id, .. }| VectorId(*id))
            .collect(),
        transaction_id,
    )
}
