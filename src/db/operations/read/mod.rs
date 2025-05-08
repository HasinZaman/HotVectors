use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    rend::u32_le,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, DeserializeUnsized,
};
use tokio::sync::{mpsc::Sender, RwLock};
use tracing::{event, Level};
use uuid::Uuid;

use crate::{
    db::{
        component::{
            data_buffer::{BufferError, DataBuffer, Global},
            graph::{GraphSerial, InterPartitionGraph, IntraPartitionGraph},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{ArchivedVectorEntrySerial, Partition, PartitionSerial, VectorEntrySerial},
        },
        Response, Success,
    },
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

pub mod knn;

pub async fn stream_meta_data<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + Into<VectorSerial<A>>,
>(
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
    sender: &Sender<Response<A>>,
) -> Result<(), ()> {
    let meta_data = &*meta_data.read().await;

    let mut visited = HashSet::new();

    while visited.len() != meta_data.len() {
        let iter: Vec<_> = meta_data
            .iter()
            .filter(|(id, _)| !visited.contains(*id))
            // .map(|(id, data)| (id, data.try_read()))
            // .filter(|(_, data)| data.is_ok())
            // .map(|(id, data)| (id, data.unwrap()))
            .collect();

        for (id, data) in iter {
            let data = &*data.read().await;

            let _ = sender
                .send(Response::Success(Success::MetaData(
                    PartitionId(*id),
                    data.size,
                    data.centroid.clone().into(),
                )))
                .await;

            visited.insert(*id);
        }
    }

    Ok(())
}

pub async fn stream_vectors_from_partition<
    A: PartialEq
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
        > + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + From<VectorSerial<A>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    mut requested_partitions: Vec<PartitionId>,
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
    sender: &Sender<Response<A>>,
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
{
    event!(Level::INFO, "Starting to stream vectors from partitions");
    let r_meta_data_lock = meta_data.read().await;

    while requested_partitions.len() > 0 {
        event!(Level::DEBUG, requested_partitions = ?requested_partitions, "Processing partitions");

        loop {
            let mut remove_index = vec![];

            for (i1, partition_id) in requested_partitions.clone().iter().enumerate() {
                event!(
                    Level::DEBUG,
                    ?partition_id,
                    "Attempting to access partition"
                );

                let partition = {
                    let w_partition_buffer = &mut *partition_buffer.write().await;

                    let rwl_partition = match w_partition_buffer.try_access(&*partition_id) {
                        Ok(val) => val,
                        Err(_) => {
                            event!(Level::WARN, ?partition_id, "Partition access failed");
                            continue;
                        }
                    };

                    let r_partition = rwl_partition.read().await;

                    let Some(partition) = &*r_partition else {
                        event!(Level::ERROR, ?partition_id, "Partition not found");
                        continue;
                    };

                    partition.clone()
                };

                event!(
                    Level::INFO,
                    ?partition_id,
                    "Successfully accessed partition"
                );
                remove_index.push(i1);

                let _ = sender
                    .send(Response::Success(Success::Partition(PartitionId(
                        partition.id,
                    ))))
                    .await;

                let iter = partition
                    .vectors
                    .iter()
                    .take(partition.size)
                    .map(|x| x.unwrap());

                for vector in iter {
                    event!(Level::DEBUG, ?vector, "Sending vector");
                    let _ = sender
                        .send(Response::Success(Success::Vector(
                            VectorId(vector.id),
                            vector.vector.into(),
                        )))
                        .await;
                }
            }

            remove_index.sort();
            remove_index.reverse();
            remove_index.into_iter().for_each(|idx| {
                event!(Level::DEBUG, index = idx, "Removing processed partition");
                requested_partitions.remove(idx);
            });

            if requested_partitions.len() <= 0 {
                break;
            }

            {
                let mut replace_count = 0;
                let partition_buffer = &mut *partition_buffer.write().await;

                let mut least_used = partition_buffer.least_used_iter().await;

                for i1 in 0..min(MAX_LOADED, requested_partitions.len()) {
                    let id = requested_partitions[i1];

                    match (partition_buffer.load(&id).await, replace_count < 1) {
                        (Ok(_), _) => {
                            event!(Level::DEBUG, ?id, "Loaded partition buffer");
                        }
                        (Err(BufferError::OutOfSpace), true) => {
                            event!(Level::WARN, "Out of space, unloading least used partition");

                            let Some(least_used) = &mut least_used else {
                                continue;
                            };

                            let Some(unload_id) = least_used.next() else {
                                break;
                            };

                            partition_buffer
                                .unload_and_load(&unload_id.1, &id)
                                .await
                                .unwrap();

                            replace_count += 1;
                            event!(Level::INFO, ?unload_id, ?id, "Replaced partition");
                        }
                        (Err(BufferError::OutOfSpace), false) => {
                            event!(Level::WARN, "Out of space, skipping replacement");
                            break;
                        }
                        (Err(BufferError::FileNotFound), _) => {
                            event!(Level::ERROR, ?id, "Partition file not found");
                            todo!()
                        }
                        (Err(_), _) => {
                            event!(Level::ERROR, ?id, "Unexpected buffer error");
                            todo!()
                        }
                    };
                }
            }
        }
    }

    event!(Level::INFO, "Ending read of DB and cleaning up");
    drop(r_meta_data_lock);
    Ok(())
}

pub async fn stream_partition_graph<
    A: PartialEq
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
        > + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + From<VectorSerial<A>> + Debug,
    const MAX_LOADED: usize,
>(
    requested_partition: PartitionId,
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
    min_spanning_tree_buffer: Arc<
        RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Global, MAX_LOADED>>,
    >,
    sender: &Sender<Response<A>>,
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
{
    event!(Level::INFO, "Starting to stream vectors from partitions");
    let r_meta_data_lock = meta_data.read().await;
    let mut w_min_spanning_tree_buffer = min_spanning_tree_buffer.write().await;
    let graph = {
        let min_spanning_tree_buffer = &mut *w_min_spanning_tree_buffer;

        match min_spanning_tree_buffer.access(&*requested_partition).await {
            Ok(graph) => graph,
            Err(_) => {
                let Some(mut least_used) = min_spanning_tree_buffer.least_used_iter().await else {
                    todo!()
                };

                loop {
                    let unload_id = match least_used.next() {
                        Some((_, unload_id)) => unload_id,
                        None => {
                            least_used = min_spanning_tree_buffer.least_used_iter().await.unwrap();
                            continue;
                        }
                    };
                    match min_spanning_tree_buffer
                        .unload_and_load(&unload_id, &*requested_partition)
                        .await
                    {
                        Ok(_) => {
                            break min_spanning_tree_buffer
                                .access(&*requested_partition)
                                .await
                                .unwrap();
                        }
                        Err(_) => {
                            continue;
                        }
                    }
                }
            }
        }
    };

    drop(w_min_spanning_tree_buffer);
    drop(r_meta_data_lock);
    let Some(graph) = &*graph.read().await else {
        todo!()
    };

    for edge_idx in graph.0.edge_indices() {
        let (source, target) = graph.0.edge_endpoints(edge_idx).unwrap();

        let source = graph.0.node_weight(source).unwrap();
        let target = graph.0.node_weight(target).unwrap();

        let dist = graph.0.edge_weight(edge_idx).unwrap();

        let _ = sender
            .send(Response::Success(Success::Edge(*source, *target, *dist)))
            .await;
    }

    event!(Level::INFO, "Ending read of DB and cleaning up");
    Ok(())
}

pub async fn stream_inter_graph<
    A: PartialEq
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
        > + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + From<VectorSerial<A>> + Debug,
>(
    requested_partition: PartitionId,
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,
    sender: &Sender<Response<A>>,
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
{
    event!(Level::INFO, "Starting to stream vectors from partitions");
    let r_meta_data_lock = meta_data.read().await;
    let inter_graph = &*inter_graph.read().await;

    let edges = inter_graph.0.edges(
        *inter_graph
            .1
            .get(&requested_partition)
            .expect(&format!("{requested_partition:?} not in graph")),
    );

    for edge_ref in edges {
        let (dist, source, target) = edge_ref.weight();

        let _ = sender
            .send(Response::Success(Success::InterEdge(
                *source, *target, *dist,
            )))
            .await;
    }

    drop(r_meta_data_lock);

    event!(Level::INFO, "Ending read of DB and cleaning up");
    Ok(())
}
