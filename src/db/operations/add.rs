use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem,
    sync::Arc,
};

use heapify::{make_heap_with, pop_heap_with};
use petgraph::{csr::DefaultIx, graph::EdgeIndex, visit::EdgeRef};

use tokio::{
    runtime::Runtime,
    sync::{Mutex, RwLock},
};
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

use crate::{
    db::{
        component::{
            self,
            cluster::ClusterSet,
            data_buffer::{BufferError, DataBuffer},
            graph::{GraphSerial, IntraPartitionGraph},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{
                ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionSerial, VectorEntry,
                VectorEntrySerial,
            },
        },
        operations::split::{
            calculate_number_of_trees, split_partition, split_partition_into_trees, KMean, BFS,
        },
    },
    resolve_buffer,
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;

fn find_closet_vectors<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    vector: VectorEntry<A, B>,
    partitions: &[&Partition<A, B, PARTITION_CAP, VECTOR_CAP>],
    // dist_map: &mut HashMap<(PartitionId, VectorId), A>,
) -> (
    Vec<((PartitionId, VectorId), A)>,
    HashMap<(PartitionId, VectorId), A>,
) {
    if cfg!(feature = "gpu_processing") {
        todo!()
    } else {
        let mut dist_map: HashMap<(PartitionId, VectorId), A> = HashMap::new();
        (
            partitions
                .iter()
                .map(|partition| {
                    partition
                        .iter()
                        .map(|y| ((PartitionId(partition.id), VectorId(y.id)), y.vector))
                        .map(|(id, vec)| (id, B::dist(&vec, &vector.vector)))
                        .map(|(id, dist)| {
                            dist_map.insert(id, dist);

                            (id, dist)
                        })
                        .min_by(|(_, dist_1), (_, dist_2)| {
                            (*dist_1).partial_cmp(dist_2).unwrap_or(Ordering::Equal)
                        })
                        .unwrap()
                })
                .collect(),
            dist_map,
        )
    }
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
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + From<VectorSerial<A>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    value: VectorEntry<A, B>,

    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,

    partition_buffer: Arc<
        RwLock<
            DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionSerial<A>, MAX_LOADED>,
        >,
    >,
    min_spanning_tree_buffer: Arc<
        RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, MAX_LOADED>>,
    >,

    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,

    cluster_sets: Arc<RwLock<Vec<ClusterSet<A>>>>,
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
{
    event!(Level::INFO, "🔥 Adding vector: {value:?}");

    let meta_data = &mut *meta_data.write().await;
    event!(Level::DEBUG, "🔒 Locked `meta_data`");

    let mut w_partition_buffer = partition_buffer.write().await;

    event!(Level::DEBUG, "🔒 Locked `partition_buffer`");

    let mut w_min_spanning_tree_buffer = min_spanning_tree_buffer.write().await;
    event!(Level::DEBUG, "🔒 Locked `min_spanning_tree_buffer`");

    let inter_graph = &mut *inter_graph.write().await;
    event!(Level::DEBUG, "🔒 Locked `inter_graph`");

    let cluster_sets = &mut *cluster_sets.write().await;
    event!(Level::DEBUG, "🔒 Locked `inter_graph`");

    // find closet id
    let (mut closet_partition_id, mut closet_size) = {
        event!(Level::INFO, "✨ Finding closest partition");
        let mut meta_data = meta_data.iter();
        let mut closet_size = 0;
        let mut closest = {
            let (_, data) = meta_data.next().unwrap();

            let Meta {
                id, centroid, size, ..
            } = &*data.read().await;

            closet_size = *size;

            (*id, B::dist(centroid, &value.vector))
        };

        for (_, data) in meta_data {
            let Meta {
                id, centroid, size, ..
            } = &*data.read().await;

            let dist = B::dist(centroid, &value.vector);

            if dist < closest.1 {
                closet_size = *size;

                closest = (*id, dist)
            }
        }

        (closest.0, closet_size)
    };
    event!(
        Level::INFO,
        "💎 Closest partition: {closet_partition_id:?} - {closet_size:?}"
    );

    if closet_size == 0 {
        let partition_buffer = &mut *w_partition_buffer;

        let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition_id);

        let Some(partition) = &mut *partition.write().await else {
            todo!()
        };
        add_into_partition(value.clone(), partition)
            .expect("Unable to insert value into empty partition");

        let tree_buffer = &mut *w_min_spanning_tree_buffer;

        let tree = resolve_buffer!(ACCESS, tree_buffer, closet_partition_id);

        let Some(tree) = &mut *tree.write().await else {
            todo!()
        };

        tree.add_node(VectorId(value.id));

        let meta = &mut *meta_data[&*closet_partition_id].write().await;

        meta.size = partition.size;
        meta.centroid = partition.centroid();
        // meta.edge_length = (todo!(), todo!());

        for cluster_set in cluster_sets.iter_mut() {
            event!(Level::DEBUG, "cluster_set({:?})", cluster_set.threshold);
            let cluster_id = cluster_set.new_cluster().await.unwrap();
            let _ = cluster_set
                .insert(VectorId(value.id), cluster_id)
                .await
                .unwrap();
        }

        return Ok(());
    }

    // getting neighbor ids
    let neighbor_ids: Vec<PartitionId> = inter_graph
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
        // .filter()
        .collect();
    event!(Level::INFO, "🤝 Neighbors: {neighbor_ids:?}");

    // get closet vectors in partition & generate dist_map
    let mut dist_map = HashMap::new();

    let required_ids: Vec<&PartitionId> = [&closet_partition_id]
        .into_iter()
        .chain(neighbor_ids.iter())
        .collect();

    let mut smallest_partition_edge_length: HashMap<PartitionId, A> = HashMap::new();

    let closet_vector_id = {
        let partition_buffer = &mut *w_partition_buffer;

        event!(Level::DEBUG, "📦 Finding closet vector");
        let mut required_partitions: HashSet<&PartitionId> =
            required_ids.clone().into_iter().collect();

        let mut closet_id = None;
        let mut closet_dist = None;

        loop {
            let mut acquired_partitions = Vec::new();
            event!(
                Level::DEBUG,
                "Attempt to acquired required ids that aren't loaded"
            );
            for id in required_partitions.iter() {
                // Replace with try access and/or batch access
                let Ok(partition) = partition_buffer.access(*id).await else {
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
                // possible future optimization to remove size of dist map
                // get closet_id & dist
                // get get closet dist for each partition
                let (partition_distances, new_dist_map) =
                    find_closet_vectors(value, partitions.as_slice());

                dist_map.extend(new_dist_map);

                partition_distances
                    .iter()
                    .for_each(|((partition_id, _), dist)| {
                        smallest_partition_edge_length.insert(*partition_id, *dist);
                    });

                let (id, dist) = partition_distances
                    .into_iter()
                    .min_by(|(_, dist_1), (_, dist_2)| {
                        (*dist_1).partial_cmp(dist_2).unwrap_or(Ordering::Equal)
                    })
                    .unwrap();

                (closet_id, closet_dist) = match closet_id {
                    None => (Some(id), Some(dist)),
                    Some(_current_id) => {
                        if dist < closet_dist.unwrap() {
                            (Some(id), Some(dist))
                        } else {
                            (closet_id, closet_dist)
                        }
                    }
                };

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

        event!(Level::DEBUG, "📦 Found closet vector");
        closet_id.unwrap()
    };

    // needs to be updated (Requires loading in more partitions in order to get distance of all nearby vectors)
    closet_partition_id = closet_vector_id.0;
    {
        let data = meta_data
            .get(&closet_partition_id)
            .expect(&format!("Expect {closet_partition_id:?} in meta_data"));

        let Meta { size, .. } = &*data.try_read().unwrap();

        closet_size = *size;
    };
    event!(Level::INFO, "💎 closet_vector_id: {closet_vector_id:?}");

    // insert into partition
    {
        let partition_buffer = &mut *w_partition_buffer;
        let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

        event!(Level::DEBUG, "Insert partition size: {closet_size}");
        if closet_size >= PARTITION_CAP {
            let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition_id);

            let Some(partition) = &mut *partition.try_write().unwrap() else {
                todo!()
            };

            // split partition

            let graph = resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);
            let Some(graph) = &mut *graph.try_write().unwrap() else {
                todo!()
            };

            // split based on trees
            let Ok(splits) = split_partition::<A, B, KMean, PARTITION_CAP, VECTOR_CAP>(
                partition,
                graph,
                inter_graph,
            ) else {
                panic!()
            };

            let mut new_closet_id = (
                splits[0].0.id,
                B::dist(&splits[0].0.centroid(), &value.vector),
            );
            let mut push_pairs = Vec::new();

            // can easily be unwrapped & gets rid of if statements
            for (mut new_partition, mut new_graph) in splits {
                // fix any forests
                let pairs = match calculate_number_of_trees(&new_graph) > 1 {
                    true => {
                        event!(Level::DEBUG, "Fixing forest");
                        split_partition_into_trees(&mut new_partition, &mut new_graph, inter_graph)
                            .unwrap()
                    }
                    false => vec![(new_partition, new_graph)],
                };

                // if last value is closet_partition (then maybe skip that value and handle the edge case after the for loop)
                for (new_partition, new_graph) in pairs {
                    let dist = B::dist(&new_partition.centroid(), &value.vector);

                    if dist < new_closet_id.1 {
                        new_closet_id = (new_partition.id, dist);
                    }

                    if partition.id == new_partition.id {
                        event!(Level::DEBUG, "Found original partition");
                        *partition = new_partition;
                        *graph = new_graph;

                        let target_meta: &mut Meta<A, B> =
                            &mut *meta_data[&partition.id].write().await;

                        target_meta.size = partition.size;
                        target_meta.centroid = partition.centroid();

                        target_meta.edge_length = (
                            match graph.smallest_edge() {
                                Some(x) => x.2,
                                None => A::max(),
                            },
                            match graph.largest_edge() {
                                Some(x) => x.2,
                                None => A::min(),
                            },
                        );
                    } else {
                        event!(Level::DEBUG, "Updating new partition");
                        // update dist map
                        new_partition
                            .iter()
                            .map(|vector| VectorId(vector.id))
                            .for_each(|vector_id| {
                                dist_map.insert(
                                    (PartitionId(new_partition.id), vector_id),
                                    *dist_map
                                        .get(&(PartitionId(partition.id), vector_id))
                                        .expect(&format!(
                                            "Expect {:?} in {:?}",
                                            (PartitionId(partition.id), vector_id),
                                            dist_map
                                        )),
                                );
                                dist_map.remove(&(PartitionId(partition.id), vector_id));
                            });

                        meta_data.insert(
                            new_partition.id,
                            Arc::new(RwLock::new(Meta::new(
                                PartitionId(new_partition.id),
                                new_partition.size,
                                new_partition.centroid(),
                                (
                                    match new_graph.smallest_edge() {
                                        Some(x) => x.2,
                                        None => A::max(),
                                    },
                                    match new_graph.largest_edge() {
                                        Some(x) => x.2,
                                        None => A::min(),
                                    },
                                    // new_graph.smallest_edge().unwrap().2,
                                    // new_graph.largest_edge().unwrap().2,
                                ),
                            ))),
                        );

                        push_pairs.push((new_partition, new_graph));
                    }
                }
            }

            // drop(partition);
            let id = partition.id;
            for (new_partition, new_graph) in push_pairs {
                resolve_buffer!(PUSH, partition_buffer, new_partition, [id]);
                resolve_buffer!(PUSH, min_span_tree_buffer, new_graph, [id]);
            }

            closet_partition_id = PartitionId(new_closet_id.0);
        }

        let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition_id);

        let Some(partition) = &mut *partition.try_write().unwrap() else {
            todo!()
        };

        let _ = add_into_partition(value.clone(), partition).unwrap();

        let target_meta: &mut Meta<A, B> = &mut *meta_data[&partition.id].write().await;

        target_meta.size = partition.size;
        target_meta.centroid = partition.centroid();
    }

    {
        for cluster_set in cluster_sets.iter_mut() {
            event!(Level::DEBUG, "cluster_set({:?})", cluster_set.threshold);
            let cluster_id = cluster_set.new_cluster().await.unwrap();
            let _ = cluster_set
                .insert(VectorId(value.id), cluster_id)
                .await
                .unwrap();
        }
    }

    //update min_spanning
    'update_min_span: {
        // filter out partitions where there is no overlap partition_max < vector_min
        // let mut neighbor_ids = HashSet::new();
        let neighbor_ids: HashSet<PartitionId> = inter_graph
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
            // .filter()
            .collect();

        println!("{neighbor_ids:#?}");

        // vector_id -> partition_id
        println!("{dist_map:#?}");

        // update dist_map to include missing partitions
        let (dist_map, vec_to_partition, partition_to_vectors) = {
            let mut vec_to_partition: HashMap<VectorId, PartitionId> = dist_map
                .iter()
                .map(|((partition_id, vector_id), _)| (*vector_id, *partition_id))
                .chain([(VectorId(value.id), closet_partition_id)])
                .collect();
            let mut partition_to_vectors: HashMap<PartitionId, HashSet<VectorId>> = HashMap::new();
            for (vector_id, partition_id) in &vec_to_partition {
                partition_to_vectors
                    .entry(*partition_id)
                    .or_insert_with(HashSet::new)
                    .insert(*vector_id);
            }

            let mut missing_partitions = neighbor_ids.clone();
            let mut dist_map: HashMap<VectorId, A> = dist_map
                .iter()
                .map(|((partition_id, vector_id), dist)| {
                    missing_partitions.remove(partition_id);
                    (*vector_id, *dist)
                })
                .collect();

            let partition_buffer = &mut *w_partition_buffer;
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

                if !missing_partitions.is_empty() {
                    event!(Level::DEBUG, "📥 Processing newly acquired partitions");
                    // get closet_id & dist
                    // get get closet dist for each partition
                    let (_, new_dist_map) = find_closet_vectors(value, partitions.as_slice());

                    dist_map.extend(new_dist_map.iter().map(
                        |((partition_id, vector_id), dist)| {
                            vec_to_partition.insert(*vector_id, *partition_id);

                            partition_to_vectors
                                .entry(*partition_id)
                                .or_insert_with(HashSet::new)
                                .insert(*vector_id);

                            (*vector_id, *dist)
                        },
                    ));

                    partitions
                        .iter()
                        .map(|partition| PartitionId(partition.id))
                        .for_each(|id| {
                            missing_partitions.remove(&id);
                        });

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

                            let Some((unload_idx, unload_id)) = least_used.next() else {
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
                            event!(Level::ERROR, "🛑 Unexpected error loading partition {id:?}");
                            todo!()
                        }
                    };
                }
            }

            (dist_map, vec_to_partition, partition_to_vectors)
        };

        // list of ids in graph
        // get list of all edges and sort by distance (can be chunks)
        let mut edges = {
            let mut edges = Vec::new();
            let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

            // add inter_edge
            inter_graph
                .0
                .edges(inter_graph.1[&closet_partition_id])
                .map(|edge| (edge.weight()))
                .map(|(dist, source, target)| (source.1, target.1, dist))
                .for_each(|(source, target, dist)| {
                    edges.push((source, target, *dist));
                });

            // add local
            {
                let min_span_tree: Arc<RwLock<Option<IntraPartitionGraph<A>>>> =
                    resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);

                let Some(min_span_tree) = &*min_span_tree.read().await else {
                    todo!()
                };

                min_span_tree
                    .0
                    .edge_indices()
                    .map(|idx| {
                        (
                            min_span_tree.0.edge_endpoints(idx).unwrap(),
                            min_span_tree.0.edge_weight(idx).unwrap(),
                        )
                    })
                    .map(|((source, target), dist)| {
                        (
                            min_span_tree.0.node_weight(source).unwrap(),
                            min_span_tree.0.node_weight(target).unwrap(),
                            dist,
                        )
                    })
                    .for_each(|(source, target, dist)| {
                        edges.push((*source, *target, *dist));
                    });
            }

            // candidate edges
            dist_map
                .iter()
                .map(|(target, dist)| (VectorId(value.id), *target, *dist))
                .filter(|(_, id, _)| vec_to_partition.get(id).is_some())
                .filter(|(_, id, _)| {
                    neighbor_ids.contains(vec_to_partition.get(id).unwrap())
                        || vec_to_partition.get(id).unwrap() == &closet_partition_id
                })
                .for_each(|(source, target, dist)| {
                    edges.push((source, target, dist));
                });

            if edges.len() == 0 {
                break 'update_min_span;
            }

            make_heap_with(&mut edges, |x, y| y.2.partial_cmp(&x.2));

            edges
        };

        // if edges.len() == 0 {}

        println!("{dist_map:#?}");
        println!("{edges:#?}");
        println!("{vec_to_partition:#?}");

        // vec_to_partition

        let mut sets: Vec<HashSet<VectorId>> = dist_map
            .iter()
            .map(|(key, _)| key)
            .chain(&[VectorId(value.id)])
            .map(|key| {
                let mut set = HashSet::new();
                set.insert(*key);

                set
            })
            .collect();

        let mut vec_to_set: HashMap<VectorId, usize> = dist_map
            .iter()
            .map(|(key, _)| key)
            .chain(&[VectorId(value.id)])
            .map(|id| *id)
            .enumerate()
            .map(|(idx, id)| (id, idx))
            .collect();

        // println!("PRE\n{sets:?}\t{vec_to_set:?}");

        for (vector_id, partition_id) in &vec_to_partition {
            if *partition_id != closet_partition_id {
                let source_set_idx = *vec_to_set
                    .get(vector_id)
                    .expect("Vector not found in vec_to_set");

                for (other_vector_id, other_partition_id) in &vec_to_partition {
                    if other_partition_id == partition_id && other_vector_id != vector_id {
                        let target_set_idx = *vec_to_set
                            .get(other_vector_id)
                            .expect("Vector not found in vec_to_set");

                        if source_set_idx != target_set_idx {
                            // Merge smaller set into the larger one
                            let (smaller_idx, larger_idx) =
                                if sets[source_set_idx].len() < sets[target_set_idx].len() {
                                    (source_set_idx, target_set_idx)
                                } else {
                                    (target_set_idx, source_set_idx)
                                };

                            let tmp: Vec<VectorId> = sets[smaller_idx].drain().collect();
                            for id in tmp {
                                sets[larger_idx].insert(id);
                                vec_to_set.insert(id, larger_idx);
                            }
                        }
                    }
                }
            }
        }
        // println!("POST\n{sets:?}\t{vec_to_set:?}");

        let mut remove_edges = Vec::new();
        let mut new_edges: Vec<(VectorId, VectorId, A)> = Vec::new();
        let new_id = VectorId(value.id);
        while edges.len() > 0 {
            pop_heap_with(&mut edges, |x, y| y.2.partial_cmp(&x.2));

            let (source, target, dist) = edges.pop().unwrap();

            let same_tree = sets[*vec_to_set
                .get(&source)
                .expect(&format!("Failed to extract {source:?} from {vec_to_set:?}"))]
            .contains(&target);

            if same_tree {
                remove_edges.push((source, target))
            } else if target == new_id || source == new_id {
                new_edges.push((source, target, dist));

                // merge sets
                let source_set_idx = *vec_to_set
                    .get(&source)
                    .expect("Source not found in vec_to_set");
                let target_set_idx = *vec_to_set
                    .get(&target)
                    .expect("Target not found in vec_to_set");

                if source_set_idx != target_set_idx {
                    // Merge smaller set into the larger one for efficiency
                    let (smaller_idx, larger_idx) =
                        if sets[source_set_idx].len() < sets[target_set_idx].len() {
                            (source_set_idx, target_set_idx)
                        } else {
                            (target_set_idx, source_set_idx)
                        };

                    let tmp: Vec<VectorId> = sets[smaller_idx].drain().collect();
                    for id in tmp {
                        sets[larger_idx].insert(id);
                        vec_to_set.insert(id, larger_idx);
                    }
                }
            } else {
                // merge sets
                let source_set_idx = *vec_to_set
                    .get(&source)
                    .expect(&format!("Failed to extract {source:?} from {vec_to_set:?}"));
                let target_set_idx = *vec_to_set
                    .get(&target)
                    .expect(&format!("Failed to extract {source:?} from {vec_to_set:?}"));

                if source_set_idx != target_set_idx {
                    // Merge smaller set into the larger one for efficiency
                    let (smaller_idx, larger_idx) =
                        if sets[source_set_idx].len() < sets[target_set_idx].len() {
                            (source_set_idx, target_set_idx)
                        } else {
                            (target_set_idx, source_set_idx)
                        };

                    let tmp: Vec<VectorId> = sets[smaller_idx].drain().collect();
                    for id in tmp {
                        sets[larger_idx].insert(id);
                        vec_to_set.insert(id, larger_idx);
                    }
                }
            }
        }
        {
            let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

            for (vec_id_1, vec_id_2) in remove_edges {
                let partition_id_1 = vec_to_partition.get(&vec_id_1).expect(&format!(
                    "Failed to extract {vec_id_1:?} from {vec_to_partition:?}"
                ));
                let partition_id_2 = vec_to_partition.get(&vec_id_2).expect(&format!(
                    "Failed to extract {vec_id_2:?} from {vec_to_partition:?}"
                ));

                match partition_id_1 == partition_id_2 {
                    true => {
                        let rw_graph =
                            resolve_buffer!(ACCESS, min_span_tree_buffer, *partition_id_1);

                        let Some(graph) = &mut *rw_graph.try_write().unwrap() else {
                            todo!()
                        };

                        let _ = graph.remove_edge(vec_id_1, vec_id_2);
                    }
                    false => {
                        let _ = inter_graph
                            .remove_edge((*partition_id_1, vec_id_1), (*partition_id_2, vec_id_2));
                    }
                }
            }
        }
        {
            let new_edges_iter = new_edges
                .iter()
                .filter(|(id_1, id_2, _)| vec_to_partition.get(id_1) != vec_to_partition.get(id_2))
                .filter(|(id_1, id_2, _)| {
                    vec_to_partition.get(id_1).unwrap() == &closet_partition_id
                        || vec_to_partition.get(id_2).unwrap() == &closet_partition_id
                })
                .map(|(id_1, id_2, dist)| {
                    (
                        *vec_to_partition.get(id_1).unwrap(),
                        *vec_to_partition.get(id_2).unwrap(),
                        (
                            *dist,
                            (*vec_to_partition.get(id_1).unwrap(), *id_1),
                            (*vec_to_partition.get(id_2).unwrap(), *id_2),
                        ),
                    )
                });

            for (id_1, id_2, weight) in new_edges_iter {
                inter_graph.add_edge(id_1, id_2, weight);

                // room for optimization
                for cluster_set in cluster_sets.iter_mut() {
                    if cluster_set.threshold < weight.0 {
                        continue;
                    }

                    let (_, (_, vector_id_1), (_, vector_id_2)) = weight;

                    let cluster_id_1 = cluster_set.get_cluster(vector_id_1).await.unwrap();
                    let cluster_id_2 = cluster_set.get_cluster(vector_id_2).await.unwrap();

                    if cluster_id_1 == cluster_id_2 {
                        continue;
                    }

                    let _ = cluster_set
                        .merge_clusters(cluster_id_1, cluster_id_2)
                        .await
                        .unwrap();
                }
            }
        }
        'new_local: {
            let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;
            let partition_buffer = &mut *w_partition_buffer;

            // add new new vector to local graph
            {
                let rw_graph = resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);

                if let Some(graph) = &mut *rw_graph.try_write().unwrap() {
                    graph.add_node(VectorId(value.id));
                };
            }

            let mut new_local: Vec<(VectorId, VectorId, A)> = new_edges
                .iter()
                .filter(|(id_1, id_2, _)| vec_to_partition.get(id_1) == vec_to_partition.get(id_2))
                .map(|(id_1, id_2, dist)| (*id_1, *id_2, *dist))
                .collect();

            new_local.sort_by(|(x, ..), (y, ..)| {
                vec_to_partition
                    .get(x)
                    .unwrap()
                    .cmp(&vec_to_partition.get(y).unwrap())
            });

            let mut iter = new_local.into_iter();

            let (mut current_partition_id, mut rw_graph) = {
                let Some((id_1, id_2, dist)) = iter.next() else {
                    break 'new_local;
                };

                let partition_id = vec_to_partition.get(&id_1).unwrap();
                let rw_graph = resolve_buffer!(ACCESS, min_span_tree_buffer, *partition_id);

                if let Some(graph) = &mut *rw_graph.write().await {
                    graph.add_edge(id_1, id_2, dist);

                    // room for optimization
                    for cluster_set in cluster_sets.iter_mut() {
                        if cluster_set.threshold < dist {
                            continue;
                        }

                        let cluster_id_1 = cluster_set.get_cluster(id_1).await.unwrap();
                        let cluster_id_2 = cluster_set.get_cluster(id_2).await.unwrap();

                        if cluster_id_1 == cluster_id_2 {
                            continue;
                        }

                        let _ = cluster_set
                            .merge_clusters(cluster_id_1, cluster_id_2)
                            .await
                            .unwrap();
                    }
                    // TODO - Update cluster (possible merge cluster(id_1) & cluster(id_2))
                };

                (partition_id, rw_graph)
            };

            for (id_1, id_2, dist) in iter {
                let partition_id = vec_to_partition.get(&id_1).unwrap();
                if current_partition_id != partition_id {
                    current_partition_id = partition_id;

                    if let Some(graph) = &mut *rw_graph.try_write().unwrap() {
                        if calculate_number_of_trees(&graph) > 1 {
                            let w_partition =
                                resolve_buffer!(ACCESS, partition_buffer, *current_partition_id);
                            let Some(partition) = &mut *w_partition.try_write().unwrap() else {
                                todo!()
                            };

                            let pairs =
                                split_partition_into_trees(partition, graph, inter_graph).unwrap();

                            for (new_partition, new_graph) in pairs {
                                if new_partition.id == partition.id {
                                    *partition = new_partition;
                                    *graph = new_graph;

                                    let target_meta: &mut Meta<A, B> =
                                        &mut *meta_data[&partition.id].write().await;

                                    target_meta.size = partition.size;
                                    target_meta.centroid = partition.centroid();

                                    target_meta.edge_length = (
                                        match graph.smallest_edge() {
                                            Some(x) => x.2,
                                            None => A::max(),
                                        },
                                        match graph.largest_edge() {
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
                                                match new_graph.smallest_edge() {
                                                    Some(x) => x.2,
                                                    None => A::max(),
                                                },
                                                match new_graph.largest_edge() {
                                                    Some(x) => x.2,
                                                    None => A::min(),
                                                },
                                            ),
                                        ))),
                                    );

                                    resolve_buffer!(
                                        PUSH,
                                        partition_buffer,
                                        new_partition,
                                        [partition.id]
                                    );
                                    resolve_buffer!(
                                        PUSH,
                                        min_span_tree_buffer,
                                        new_graph,
                                        [partition.id]
                                    );
                                }
                            }
                        };
                    };

                    rw_graph = resolve_buffer!(ACCESS, min_span_tree_buffer, *current_partition_id);
                }

                // already in the buffer and shouldn't need to wait
                if let Some(graph) = &mut *rw_graph.try_write().unwrap() {
                    graph.add_edge(id_1, id_2, dist);

                    // room for optimization
                    for cluster_set in cluster_sets.iter_mut() {
                        if cluster_set.threshold < dist {
                            continue;
                        }

                        let cluster_id_1 = cluster_set.get_cluster(id_1).await.unwrap();
                        let cluster_id_2 = cluster_set.get_cluster(id_2).await.unwrap();

                        if cluster_id_1 == cluster_id_2 {
                            continue;
                        }

                        let _ = cluster_set
                            .merge_clusters(cluster_id_1, cluster_id_2)
                            .await
                            .unwrap();
                    }
                };
            }

            if let Some(graph) = &mut *rw_graph.try_write().unwrap() {
                if calculate_number_of_trees(&graph) > 1 {
                    let w_partition =
                        resolve_buffer!(ACCESS, partition_buffer, *current_partition_id);
                    let Some(partition) = &mut *w_partition.try_write().unwrap() else {
                        todo!()
                    };

                    let pairs = split_partition_into_trees(partition, graph, inter_graph).unwrap();

                    for (new_partition, new_graph) in pairs {
                        if new_partition.id == partition.id {
                            *partition = new_partition;
                            *graph = new_graph;

                            let target_meta: &mut Meta<A, B> =
                                &mut *meta_data[&partition.id].write().await;

                            target_meta.size = partition.size;
                            target_meta.centroid = partition.centroid();

                            target_meta.edge_length = (
                                match graph.smallest_edge() {
                                    Some(x) => x.2,
                                    None => A::max(),
                                },
                                match graph.largest_edge() {
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
                                        match new_graph.smallest_edge() {
                                            Some(x) => x.2,
                                            None => A::max(),
                                        },
                                        match new_graph.largest_edge() {
                                            Some(x) => x.2,
                                            None => A::min(),
                                        },
                                    ),
                                ))),
                            );

                            resolve_buffer!(PUSH, partition_buffer, new_partition, [partition.id]);
                            resolve_buffer!(PUSH, min_span_tree_buffer, new_graph, [partition.id]);
                        }
                    }
                };
            };
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {}
