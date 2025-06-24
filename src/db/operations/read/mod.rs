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
// pub mod meta;
pub mod vector;

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
