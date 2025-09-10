use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    sync::Arc,
    vec,
};

use petgraph::visit::EdgeRef;
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    ser::{allocator::ArenaHandle, sharing::Share, Serializer},
    util::AlignedVec,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, Serialize,
};
use sled::Db;
use tokio::{
    spawn,
    sync::{mpsc::Sender, oneshot, RwLock, RwLockWriteGuard},
};
use tracing::{debug, event, info, trace, Level};
use uuid::Uuid;

use crate::{
    db::{banker::{AccessMode, AccessResponse, BankerMessage}, component::{
        cluster::{ClusterSet, MergeClusterError},
        data_buffer::{DataBuffer, Global},
        graph::{GraphSerial, InterPartitionGraph, IntraPartitionGraph},
        ids::{ClusterId, PartitionId, VectorId},
        meta::Meta,
    }}, resolve_buffer, vector::{Field, VectorSpace}
};

pub async fn build_clusters<
    A: Field<A>
        + Debug
        + Clone
        + Copy
        + PartialOrd
        + Archive
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>
        + 'static,
    B: VectorSpace<A> + Clone,
    const CAP: usize,
>(
    transaction_id: Option<Uuid>,
    
    threshold: A,

    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,

    cluster_sets: Arc<RwLock<Vec<ClusterSet<A>>>>, // maybe replace with a databuffer in the future??

    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,
    intra_graph_buffer: Arc<
        RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Global, CAP>>,
    >,

    access_tx: Sender<BankerMessage>,

) where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<rkyv::de::Pool, rancor::Error>>,
{
    
    let transaction_id = match transaction_id {
        Some(transaction_id) => transaction_id,
        None => Uuid::new_v4(),
    };

    let cluster_sets = &mut *cluster_sets.write().await;
    match cluster_sets.binary_search_by(|x| {
        x.threshold
            .partial_cmp(&threshold)
            .unwrap_or(Ordering::Equal) // == Ordering::Equal
    }) {
        Ok(_) => {
            trace!("Cluster set with threshold {:?} already exists.", threshold);
        }
        Err(0) => {
            info!("Creating a new cluster set at position {}", 0);
            // Get all partitions as read
            let partition_ids = loop {
                let meta_data = &*meta_data.read().await;

                let ids: Vec<PartitionId> = meta_data.iter().map(|(uuid, _)| PartitionId(*uuid)).collect();

                let (tx, rx) = oneshot::channel();
                let _ = access_tx
                    .send(BankerMessage::RequestAccess {
                        transaction_id,
                        partitions: ids.clone().into_iter().map(|id| (id, AccessMode::Read)).collect(),
                        respond_to: tx,
                    })
                    .await;

                match rx.await {
                    Ok(AccessResponse::Granted) => {
                        break ids
                    }
                    _ => {
                        event!(
                            Level::DEBUG,
                            "Access denied for {:?}",
                            ids
                        );
                    }
                };
            };

            // assert!(partition_ids.len() == 0);

            // start at random partition
            let mut cluster_seed: Vec<(VectorId, PartitionId, Option<ClusterId>)> = Vec::new();
            let mut visited_partitions: HashSet<PartitionId> = HashSet::new();
            let mut cluster_edges: Vec<(A, VectorId, VectorId)> = Vec::new();
            
            let mut cluster_set = ClusterSet::new(threshold, "data/clusters/".to_string());

            let inter_graph = &*inter_graph.read().await;
            // expand outwards from partition
            // df traversal from partition
            // when done with partition release from Bank
            //initialize state
            {
                let partition_id = partition_ids[0];

                let graph = {
                    let mut rw_graph_buffer = intra_graph_buffer.write().await;
                    let graph_buffer = &mut rw_graph_buffer;

                    let graph = resolve_buffer!(ACCESS, graph_buffer, partition_id);

                    let graph = graph.clone();
                            
                    graph
                };

                let Some(graph) = &*graph.read().await else {
                    todo!()
                };

                // temp solution (should check at very beginning)
                if graph.1.len() == 0 {
                    cluster_sets.insert(0, cluster_set);

                    let _ = access_tx.send(
                        BankerMessage::ReleaseAccess { transaction_id, partitions: vec![partition_id] }
                    ).await
                    .unwrap();
                
                    return;
                }

                trace!(
                    "Graph found for partition {:?}, starting clustering.",
                    partition_id
                );

                let start_vector: VectorId = graph.1.iter()
                    .map(|(vector_id, _)| vector_id)
                    .next()
                    .expect("No vectors in graph")
                    .clone();

                cluster_seed.push((start_vector, partition_id, None));
            }

            while let Some((start_vector_id, partition_id, cluster_id)) = cluster_seed.pop() {
                if visited_partitions.contains(&partition_id) {
                    continue;
                }
                
                let mut cluster_id = match cluster_id {
                    Some(cluster_id) => cluster_id,
                    None => cluster_set.new_cluster().unwrap()
                };

                let graph = {
                    let mut rw_graph_buffer = intra_graph_buffer.write().await;
                    let graph_buffer = &mut rw_graph_buffer;

                    let graph = resolve_buffer!(ACCESS, graph_buffer, partition_id);

                    let graph = graph.clone();
                            
                    graph
                };

                let Some(graph) = &*graph.read().await else {
                    todo!()
                };

                let mut visited_vectors: HashSet<VectorId> = HashSet::new();
                let mut partition_seeds: Vec<VectorId> = Vec::new();
                let mut vector_stack: Vec<VectorId> = vec![start_vector_id];

                loop {
                    if let Some(seed_id) = partition_seeds.pop() {
                        cluster_id = cluster_set.new_cluster().unwrap();
                        vector_stack.push(seed_id);
                    }

                    while let Some(vector_id) = vector_stack.pop() {

                        // add vector_id to cluster_id
                        let _ = cluster_set.insert_vector_into_cluster(vector_id, Some(cluster_id)).unwrap();
                        visited_vectors.insert(vector_id);

                        // check inter_edges
                        inter_graph.0
                            .edges(inter_graph.1[&partition_id])
                            .map(|edge_ref| edge_ref.weight())
                            .filter(|(_, (source_partition_id, _), (target_partition_id, _))| {
                                !visited_partitions.contains(&source_partition_id) && !visited_partitions.contains(&target_partition_id)
                            })
                            .filter(|(_, (_, source_vector_id), (_, target_vector_id))| {
                                &vector_id == source_vector_id || &vector_id == target_vector_id
                            })
                            .for_each(|(weight, (source_partition_id, source_vector_id), (target_partition_id, target_vector_id))| {
                                let (out_bound_partition_id, out_bound_vector_id) = match &partition_id == source_partition_id {
                                    true => (target_partition_id, target_vector_id),
                                    false => (source_partition_id, source_vector_id),
                                };

                                if weight < &threshold {
                                    cluster_seed.push((
                                        *out_bound_vector_id,
                                        *out_bound_partition_id,
                                        Some(cluster_id)
                                    ));
                                } else {
                                    cluster_seed.push((
                                        *out_bound_vector_id,
                                        *out_bound_partition_id,
                                        None
                                    ));
                                    cluster_edges.push((*weight, *source_vector_id, *target_vector_id));
                                }
                            });

                        // check local edges
                        for edge_ref in graph.0.edges(graph.1[&vector_id]) {
                            let weight = edge_ref.weight();

                            let source_id = graph.0.node_weight(edge_ref.source()).unwrap();
                            let target_id = graph.0.node_weight(edge_ref.target()).unwrap();

                            let out_bound_vector_id = match source_id == &vector_id {
                                true => target_id,
                                false => source_id,
                            };

                            if visited_vectors.contains(out_bound_vector_id) {
                                continue;
                            }

                            if weight < &threshold {
                                vector_stack.push(*out_bound_vector_id);
                            } else {
                                cluster_edges.push((*weight, *source_id, *target_id));
                                partition_seeds.push(*out_bound_vector_id)
                            }
                        }
                    }

                    if partition_seeds.len() == 0 {
                        break;
                    }
                }

                visited_partitions.insert(partition_id);

                let _ = access_tx.send(
                    BankerMessage::ReleaseAccess { transaction_id, partitions: vec![partition_id] }
                ).await
                .unwrap();
            }

            for (weight, vector_id_1, vector_id_2) in cluster_edges {
                let _ = cluster_set.add_edge(vector_id_1, vector_id_2, weight).unwrap();
            }

            cluster_sets.insert(0, cluster_set);

            
            // let inter_graph = &*inter_graph.read().await;
            // 
            // // should find first partition with vectors
            // let partition_id = *meta_data.iter().next().unwrap().0;
            // debug!("Processing partition ID: {:?}", partition_id);
            // 
            // let mut partition_stack = Vec::new();
            // let mut visited_partitions = HashSet::new();
            // let mut cluster_edges = Vec::new();
            // 
            // visited_partitions.insert(PartitionId(partition_id));
            // {
            //     let (start_vector, _) = graph.1.iter().next().unwrap();
            // 
            //     debug!(
            //         "Initial cluster ID: {:?} assigned to vector {:?}",
            //         cluster_id, start_vector
            //     );
            //     let _ = cluster_set.insert_vector_into_cluster(*start_vector, Some(cluster_id));
            // 
            //     let mut vector_to_cluster = HashMap::new();
            //     vector_to_cluster.insert(start_vector, cluster_id);
            // 
            //     let mut visit_stack = vec![start_vector];
            //     let mut cluster_seeds = Vec::new();
            // 
            //     while vector_to_cluster.len() < graph.1.len() {
            //         let vector = match (visit_stack.len() > 0, cluster_seeds.len() > 0) {
            //             (true, _) => visit_stack.pop().unwrap(),
            //             (false, true) => {
            //                 let (vector, new_cluster_id) = cluster_seeds.pop().unwrap();
            // 
            //                 cluster_id = new_cluster_id;
            // 
            //                 vector
            //             }
            //             (false, false) => {
            //                 trace!("All vectors processed, breaking loop.");
            //                 todo!()
            //             }
            //         };
            //         for edge_ref in graph.0.edges(graph.1[vector]) {
            //             let weight = edge_ref.weight();
            // 
            //             let source = graph.0.node_weight(edge_ref.source()).unwrap();
            //             let target = graph.0.node_weight(edge_ref.target()).unwrap();
            // 
            //             let out_bound_vec = match source == vector {
            //                 true => target,
            //                 false => source,
            //             };
            // 
            //             if vector_to_cluster.contains_key(out_bound_vec) {
            //                 continue;
            //             }
            // 
            //             if weight < &threshold {
            //                 let _ = cluster_set.insert_vector_into_cluster(*out_bound_vec, Some(cluster_id));
            //                 visit_stack.push(out_bound_vec);
            //             } else {
            //                 cluster_edges.push((*vector, *out_bound_vec));
            // 
            //                 let new_cluster = cluster_set.new_cluster().unwrap();
            //                 let _ =
            //                     cluster_set.insert_vector_into_cluster(*out_bound_vec, Some(new_cluster));
            //                 cluster_seeds.push((out_bound_vec, new_cluster));
            //             }
            //         }
            // 
            //         vector_to_cluster.insert(vector, cluster_id);
            //         info!("Cluster set added at position {}", 0);
            //     }
            // 
            //     for edge_ref in inter_graph
            //         .0
            //         .edges(inter_graph.1[&PartitionId(partition_id)])
            //     {
            //         let (weight, source, target) = edge_ref.weight();
            // 
            //         let ((_, source_vector), (target_partition, target_vector)) =
            //             match source.0 == PartitionId(partition_id) {
            //                 true => (source, target),
            //                 false => (target, source),
            //             };
            // 
            //         let source_cluster = vector_to_cluster[&source_vector];
            // 
            //         if weight < &threshold {
            //             partition_stack.push((source_cluster, target_partition, target_vector));
            //         } else {
            //             cluster_edges.push((*source_vector, *target_vector));
            //             let cluster_id = cluster_set.new_cluster().unwrap();
            //             partition_stack.push((cluster_id, target_partition, target_vector));
            //         }
            //     }
            // }
            // 
            // while partition_stack.len() > 0 {
            //     let (mut cluster_id, partition_id, start_vector) = partition_stack.pop().unwrap();
            // 
            //     visited_partitions.insert(*partition_id);
            // 
            //     let mut rw_graph_buffer: RwLockWriteGuard<
            //         '_,
            //         DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Global, CAP>,
            //     > = intra_graph_buffer.write().await;
            //     let graph_buffer: &mut RwLockWriteGuard<
            //         '_,
            //         DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, Global, CAP>,
            //     > = &mut rw_graph_buffer;
            // 
            //     let r_graph: Arc<RwLock<Option<IntraPartitionGraph<A>>>> =
            //         graph_buffer.access(&partition_id).await.unwrap();
            //     let Some(graph) = &*r_graph.read().await else {
            //         todo!()
            //     };
            // 
            //     let _ = rw_graph_buffer.downgrade();
            // 
            //     let mut vector_to_cluster = HashMap::new();
            // 
            //     let mut visit_stack = vec![start_vector];
            //     let mut cluster_seeds = Vec::new();
            // 
            //     while vector_to_cluster.len() < graph.1.len() {
            //         let vector = match (visit_stack.len() > 0, cluster_seeds.len() > 0) {
            //             (true, _) => visit_stack.pop().unwrap(),
            //             (false, true) => {
            //                 let (vector, new_cluster_id) = cluster_seeds.pop().unwrap();
            // 
            //                 cluster_id = new_cluster_id;
            // 
            //                 vector
            //             }
            //             (false, false) => todo!(),
            //         };
            // 
            //         for edge_ref in graph.0.edges(graph.1[vector]) {
            //             let weight = edge_ref.weight();
            // 
            //             let source = graph.0.node_weight(edge_ref.source()).unwrap();
            //             let target = graph.0.node_weight(edge_ref.target()).unwrap();
            // 
            //             let out_bound_vec = match source == vector {
            //                 true => target,
            //                 false => source,
            //             };
            // 
            //             if vector_to_cluster.contains_key(out_bound_vec) {
            //                 continue;
            //             }
            // 
            //             if weight < &threshold {
            //                 let _ = cluster_set.insert_vector_into_cluster(*out_bound_vec, Some(cluster_id));
            //                 visit_stack.push(out_bound_vec);
            //             } else {
            //                 cluster_edges.push((*vector, *out_bound_vec));
            // 
            //                 let new_cluster = cluster_set.new_cluster().unwrap();
            //                 let _ =
            //                     cluster_set.insert_vector_into_cluster(*out_bound_vec, Some(new_cluster));
            //                 cluster_seeds.push((out_bound_vec, new_cluster));
            //             }
            //         }
            // 
            //         vector_to_cluster.insert(vector, cluster_id);
            //     }
            // 
            //     for edge_ref in inter_graph.0.edges(inter_graph.1[partition_id]) {
            //         let (weight, source, target) = edge_ref.weight();
            // 
            //         let ((_, source_vector), (target_partition, target_vector)) =
            //             match source.0 == *partition_id {
            //                 true => (source, target),
            //                 false => (target, source),
            //             };
            // 
            //         if visited_partitions.contains(target_partition) {
            //             continue;
            //         }
            // 
            //         let source_cluster = vector_to_cluster[&source_vector];
            // 
            //         if weight < &threshold {
            //             partition_stack.push((source_cluster, target_partition, target_vector));
            //         } else {
            //             cluster_edges.push((*source_vector, *target_vector));
            // 
            //             let cluster_id = cluster_set.new_cluster().unwrap();
            //             partition_stack.push((cluster_id, target_partition, target_vector));
            //         }
            //     }
            // }
            // 
            // cluster_sets.insert(0, cluster_set);
        }
        Err(n) => {
            trace!("Nearest cluster({threshold:?}) for exists {:?}.", n - 1);
            cluster_sets.insert(
                n,
                ClusterSet::from_smaller_cluster_set(threshold, &cluster_sets[n - 1]).unwrap(),
            );
        }
    };
    info!("Finished building clusters.");
}

pub async fn create_local<
    A: Field<A>
        + Debug
        + Clone
        + Copy
        + 'static
        + Archive
        + PartialOrd
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
>(
    cluster_sets: &[ClusterSet<A>],
    vectors: &[VectorId],
    transaction_id: Uuid,
) -> Vec<ClusterSet<A>>
where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<rkyv::de::Pool, rancor::Error>>,
{
    let mut local_sets = Vec::new();

    for cluster_set in cluster_sets {
        let local = cluster_set.create_local(vectors, transaction_id).await;

        local_sets.push(local);
    }

    local_sets
}
pub async fn update_cluster<
    A: Field<A>
        + Debug
        + Clone
        + Copy
        + 'static
        + Archive
        + PartialOrd
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
>(
    cluster_sets: &mut Vec<ClusterSet<A>>,
    dist: &A,
    id_1: VectorId,
    id_2: VectorId,
) where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<rkyv::de::Pool, rancor::Error>>,
{
    for cluster_set in cluster_sets.iter_mut() {
        if &cluster_set.threshold < dist {
            let _ = cluster_set.add_edge(id_1, id_2, dist.clone());

            continue;
        }

        let cluster_id_1 = cluster_set.get_cluster(id_1).unwrap();
        let cluster_id_2 = cluster_set.get_cluster(id_2).unwrap();

        if cluster_id_1 == cluster_id_2 {
            continue;
        }

        match cluster_set.merge_clusters::<5>(cluster_id_1, cluster_id_2) {
            Ok(_) | Err(MergeClusterError::SameCluster) => {},
            Err(_) => todo!(),
        }
    }
}
pub async fn remove_cluster_edge<
    A: Field<A>
        + Debug
        + Clone
        + Copy
        + 'static
        + Archive
        + PartialOrd
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
>(
    cluster_sets: &mut Vec<ClusterSet<A>>,
    id_1: VectorId,
    id_2: VectorId,
) where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<rkyv::de::Pool, rancor::Error>>,
{
    for cluster_set in cluster_sets.iter_mut() {
        let _ = cluster_set.delete_edge(id_1, id_2);
    }
}
