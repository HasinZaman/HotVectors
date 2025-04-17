use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    sync::Arc,
};

use spade::{DelaunayTriangulation, HasPosition, Triangulation};
use tokio::sync::RwLock;
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
            data_buffer::{BufferError, DataBuffer},
            graph::{GraphSerial, IntraPartitionGraph},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{
                ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionSerial, VectorEntry,
                VectorEntrySerial,
            },
        },
        operations::{
            cluster::{remove_cluster_edge, update_cluster},
            split::{
                calculate_number_of_trees, split_partition, split_partition_into_trees,
                FirstTreeSplitStrategy, KMean, MaxAttempt, BFS,
            },
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
        #[cfg(feature = "benchmark")]
        let _child_benchmark =
            Benchmark::spawn_child("Finding closet partition".to_string(), &benchmark);

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
                .new_cluster_from_vector(VectorId(value.id), cluster_id)
                .await
                .unwrap();
        }

        return Ok(());
    }

    // getting neighbor ids
    let neighbor_ids: Vec<PartitionId> = get_neighbors::<A, B, VECTOR_CAP>(
        inter_graph,
        closet_partition_id,
        meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
    )
    .await;
    event!(Level::INFO, "🤝 Neighbors: {neighbor_ids:?}");

    // get closet vectors in partition & generate dist_map
    let mut dist_map = HashMap::new();

    let required_ids: Vec<&PartitionId> = [&closet_partition_id]
        .into_iter()
        .chain(neighbor_ids.iter())
        .collect();

    let mut smallest_partition_edge_length: HashMap<PartitionId, A> = HashMap::new();

    let closet_vector_id = {
        #[cfg(feature = "benchmark")]
        let _child_benchmark =
            Benchmark::spawn_child("Finding closet vector".to_string(), &benchmark);
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
        #[cfg(feature = "benchmark")]
        let _child_benchmark = Benchmark::spawn_child("Insert partition".to_string(), &benchmark);
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

            event!(Level::DEBUG, "Beginning KMean split");
            let Ok(splits) =
                split_partition::<A, B, KMean<MaxAttempt<128>>, PARTITION_CAP, VECTOR_CAP>(
                    partition,
                    graph,
                    inter_graph,
                )
            else {
                panic!()
            };
            event!(Level::DEBUG, "Finished KMean split");

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
                            .filter(|vector_id| {
                                !dist_map.contains_key(&(PartitionId(partition.id), *vector_id))
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
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

            for (new_partition, new_graph) in push_pairs {
                resolve_buffer!(PUSH, partition_buffer, new_partition, [partition.id]);
                resolve_buffer!(PUSH, min_span_tree_buffer, new_graph, [*graph.2]);
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
                .new_cluster_from_vector(VectorId(value.id), cluster_id)
                .await
                .unwrap();
        }
    }

    //update min_spanning
    'update_min_span: {
        #[cfg(feature = "benchmark")]
        let _child_benchmark =
            Benchmark::spawn_child("Update min-spanning tree".to_string(), &benchmark);
        // filter out partitions where there is no overlap partition_max < vector_min
        // let mut neighbor_ids = HashSet::new();
        let neighbor_ids: HashSet<PartitionId> = {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Find neighbors".to_string(), &_child_benchmark);

            get_neighbors::<A, B, VECTOR_CAP>(
                inter_graph,
                closet_partition_id,
                meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
            )
            .await
            .into_iter()
            .collect()
        };
        event!(
            Level::DEBUG,
            "Neighbors of {:?}\n{:?}",
            closet_partition_id,
            neighbor_ids
        );

        // update dist_map to include missing partitions
        // might not be required
        let (dist_map, mut vec_to_partition, mut partition_to_vectors) = {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Generate distance map".to_string(), &_child_benchmark);

            let mut vec_to_partition: HashMap<VectorId, PartitionId> = HashMap::new();
            let mut partition_to_vectors: HashMap<PartitionId, HashSet<VectorId>> = HashMap::new();
            for (vector_id, partition_id) in &vec_to_partition {
                partition_to_vectors
                    .entry(*partition_id)
                    .or_insert_with(HashSet::new)
                    .insert(*vector_id);
            }

            let mut missing_partitions = [closet_partition_id]
                .into_iter()
                .into_iter()
                .chain(neighbor_ids.clone())
                .collect::<HashSet<_>>();
            let mut dist_map: HashMap<VectorId, A> = HashMap::new();

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

                if !partitions.is_empty() {
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

            dist_map.remove(&VectorId(value.id));

            (dist_map, vec_to_partition, partition_to_vectors)
        };

        // println!("{:#?}", dist_map);
        if dist_map.len() == 0 {
            let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

            let min_span_tree: Arc<RwLock<Option<IntraPartitionGraph<A>>>> =
                resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);

            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                todo!()
            };

            min_span_tree.add_node(VectorId(value.id));

            break 'update_min_span;
        }

        let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

        let partition_buffer = &mut *w_partition_buffer;
        // Update closest partition edges
        {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Updating local graph".to_string(), &_child_benchmark);

            let min_span_tree = resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);

            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                todo!()
            };

            let vector_iter = min_span_tree.1.iter().map(|(x, _)| *x).collect::<Vec<_>>();
            let mut vector_iter = vector_iter.iter();

            // first edge
            min_span_tree.add_node(VectorId(value.id));

            min_span_tree.add_node(VectorId(value.id));
            match vector_iter.next() {
                Some(vector_id) => {
                    min_span_tree.add_edge(
                        VectorId(value.id),
                        *vector_id,
                        *dist_map.get(vector_id).expect(""),
                    );
                }
                None => todo!(),
            }

            //  -> replace loop with while hashset.is_empty() & for loop for a trail
            // while iterating through trail (add and remove edges based on deleting or adding edges)
            //  This would reduce # of times trails are calculated
            //      Current O(vectors*(BFS)) = O(|V|*|V|*|E|))
            //                               = O(|V|*|V|^2)
            //                               = O(|V|^3)
            //      Proposed solution: O(|T|*(BFS)) = O(|T|*|V|*|E|))
            //                                      = O(|T|*|V|*|V|))
            //                                      = O(|T|*|V|^2))
            //      note: |T| := # of searched trails
            //      V(T1) U...U V(Tn) = V (The set of all searched vectors := vectors)
            //      |T| < |V|, therefore O(|T|*(BFS)) < O(vectors*(BFS))
            for vector_id in vector_iter {
                let weight = *dist_map.get(vector_id).expect("");
                event!(Level::DEBUG, "{:?} -> {:?}", vector_id, VectorId(value.id));
                let Ok(path) = min_span_tree.find_trail(*vector_id, VectorId(value.id)) else {
                    event!(
                        Level::DEBUG,
                        "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?}",
                        vector_id,
                        VectorId(value.id)
                    );
                    todo!()
                };
                let Some((_, path)) = path else { todo!() };

                let (max_vector_id_1, max_vector_id_2, max_weight) = path.into_iter().fold(
                    (VectorId(Uuid::nil()), VectorId(Uuid::nil()), A::min()),
                    |(acc_id_1, acc_id_2, acc_weight), (next_id_1, next_id_2, next_weight)| {
                        match next_weight.partial_cmp(&acc_weight) {
                            Some(Ordering::Greater) => (next_id_1, next_id_2, next_weight),
                            _ => (acc_id_1, acc_id_2, acc_weight),
                        }
                    },
                );

                if weight >= max_weight {
                    continue;
                };

                let _ = min_span_tree
                    .remove_edge(max_vector_id_1, max_vector_id_2)
                    .unwrap();

                let _ = remove_cluster_edge(cluster_sets, max_vector_id_1, max_vector_id_2);

                min_span_tree.add_edge(VectorId(value.id), *vector_id, weight);

                update_cluster(cluster_sets, &weight, VectorId(value.id), *vector_id).await;
            }
        }

        // update foreign edges (update delaunay triangulation to shrink search area)
        {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Updating foreign graphs".to_string(), &_child_benchmark);

            let mut visit_requirements: HashSet<PartitionId> = neighbor_ids.clone();
            visit_requirements.take(&closet_partition_id);

            let mut visited_vertices: HashSet<VectorId> = HashSet::new();

            let mut partition_path_cache: HashMap<(PartitionId, VectorId, VectorId), Vec<_>> =
                HashMap::new();
            let mut cached_partitions: HashMap<
                PartitionId,
                HashSet<(PartitionId, VectorId, VectorId)>,
            > = HashMap::new();

            let mut hints_cache: HashMap<(PartitionId, VectorId), HashMap<VectorId, A>> =
                HashMap::new();
            let mut cached_partition_hints: HashMap<PartitionId, HashSet<(PartitionId, VectorId)>> =
                HashMap::new();

            while let Some(target_partition_id) = visit_requirements.iter().next().cloned() {
                let target_vector_ids = {
                    let min_span_tree =
                        resolve_buffer!(ACCESS, min_span_tree_buffer, target_partition_id);

                    let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                        todo!()
                    };

                    min_span_tree.1.clone()
                };

                let mut partition_trail = {
                    let Ok(partition_trail) =
                        inter_graph.find_trail(closet_partition_id, target_partition_id)
                    else {
                        todo!()
                    };

                    let Some((_, partition_trail)) = partition_trail else {
                        panic!("Failed to find partition trail from {closet_partition_id:?} -> {target_partition_id:?}\n{inter_graph:#?}");
                        todo!()
                    };

                    event!(Level::DEBUG, "partition_trail:\n{partition_trail:?}");

                    partition_trail
                };

                for (target_vector_id, _) in target_vector_ids {
                    if visited_vertices.contains(&target_vector_id) {
                        continue;
                    }
                    visited_vertices.insert(target_vector_id);
                    {
                        let min_span_tree = resolve_buffer!(
                            ACCESS,
                            min_span_tree_buffer,
                            target_partition_id,
                            [*target_partition_id]
                        );

                        let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                            todo!()
                        };

                        if !min_span_tree.1.contains_key(&target_vector_id) {
                            continue;
                        }
                    }

                    let Some(weight) = dist_map.get(&target_vector_id) else {
                        continue;
                    };

                    let trails = {
                        let mut trails: Vec<((PartitionId, VectorId), (PartitionId, VectorId), A)> = 'initialize_trail: {
                            let inter_edge = partition_trail[0];

                            if VectorId(value.id) == inter_edge.0 .1 {
                                break 'initialize_trail vec![inter_edge];
                            }
                            let min_span_tree = resolve_buffer!(
                                ACCESS,
                                min_span_tree_buffer,
                                closet_partition_id,
                                [*target_partition_id]
                            );

                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            // djikstra's hint
                            // Sink (value.id)
                            let hints = match hints_cache
                                .contains_key(&(closet_partition_id, VectorId(value.id)))
                            {
                                true => &hints_cache[&(closet_partition_id, VectorId(value.id))],
                                false => {
                                    hints_cache.insert(
                                        (closet_partition_id, VectorId(value.id)),
                                        min_span_tree.dijkstra_weights(VectorId(value.id)).unwrap(),
                                    );

                                    cached_partition_hints
                                        .entry(closet_partition_id)
                                        .or_insert_with(|| {
                                            let mut set = HashSet::new();
                                            set.insert((closet_partition_id, VectorId(value.id)));

                                            set
                                        })
                                        .insert((closet_partition_id, VectorId(value.id)));

                                    &hints_cache[&(closet_partition_id, VectorId(value.id))]
                                }
                            };

                            let Ok(trail) = min_span_tree.find_trail_with_hints(
                                inter_edge.0 .1,
                                VectorId(value.id),
                                &hints,
                            ) else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
                                    VectorId(value.id),
                                    inter_edge.0 .1,
                                    inter_edge.1,
                                );

                                todo!()
                            };

                            let Some(trail) = trail else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
                                    VectorId(value.id),
                                    inter_edge.0 .1,
                                    inter_edge.1,
                                );
                                todo!()
                            };

                            trail
                                .1
                                .into_iter()
                                .map(|(vector_id_1, vector_id_2, weight)| {
                                    (
                                        (closet_partition_id, vector_id_1),
                                        (closet_partition_id, vector_id_2),
                                        weight,
                                    )
                                })
                                .chain(vec![inter_edge].into_iter())
                                .collect()
                        };

                        for i in 1..partition_trail.len() {
                            let (_, (partition_id_1, vector_id_1), _) = partition_trail[i - 1];
                            let (
                                (partition_id_2, vector_id_2),
                                (partition_id_3, vector_id_3),
                                edge_2_weight,
                            ) = partition_trail[i];

                            if vector_id_1 == vector_id_2 {
                                trails.push((
                                    (partition_id_2, vector_id_2),
                                    (partition_id_3, vector_id_3),
                                    edge_2_weight,
                                ));
                                continue;
                            }

                            if partition_path_cache.contains_key(&(
                                partition_id_1,
                                vector_id_1,
                                vector_id_2,
                            )) {
                                trails.extend(
                                    partition_path_cache
                                        [&(partition_id_1, vector_id_1, vector_id_2)]
                                        .clone(),
                                );
                                trails.push((
                                    (partition_id_2, vector_id_2),
                                    (partition_id_3, vector_id_3),
                                    edge_2_weight,
                                ));
                                continue;
                            }
                            if partition_path_cache.contains_key(&(
                                partition_id_1,
                                vector_id_2,
                                vector_id_1,
                            )) {
                                trails.extend(
                                    partition_path_cache
                                        [&(partition_id_1, vector_id_2, vector_id_1)]
                                        .clone(),
                                );
                                trails.push((
                                    (partition_id_2, vector_id_2),
                                    (partition_id_3, vector_id_3),
                                    edge_2_weight,
                                ));
                                continue;
                            }

                            let min_span_tree =
                                resolve_buffer!(ACCESS, min_span_tree_buffer, partition_id_1);

                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            let Ok(trail) = min_span_tree.find_trail(vector_id_1, vector_id_2)
                            else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
                                    VectorId(value.id),
                                    (partition_id_2, vector_id_2),
                                    (partition_id_3, vector_id_3)
                                );
                                todo!()
                            };

                            let Some((_, trail)) = trail else { todo!() };

                            let trail: Vec<_> = trail
                                .iter()
                                .map(|(vector_id_1, vector_id_2, weight)| {
                                    (
                                        (partition_id_1, *vector_id_1),
                                        (partition_id_1, *vector_id_2),
                                        *weight,
                                    )
                                })
                                .collect();
                            partition_path_cache
                                .insert((partition_id_1, vector_id_1, vector_id_2), trail.clone());
                            cached_partitions
                                .entry(partition_id_1)
                                .or_insert_with(|| {
                                    let mut set = HashSet::new();
                                    set.insert((partition_id_1, vector_id_1, vector_id_2));

                                    set
                                })
                                .insert((partition_id_1, vector_id_1, vector_id_2));
                            trails.push((
                                (partition_id_2, vector_id_2),
                                (partition_id_3, vector_id_3),
                                edge_2_weight,
                            ));
                            trails.extend(trail)
                        }

                        'final_trail: {
                            // if partition_trail.len() == 1 {
                            //     break 'final_trail;
                            // }
                            let (_, (partition_id_1, vector_id_1), _) =
                                partition_trail.last().unwrap();

                            if vector_id_1 == &target_vector_id {
                                break 'final_trail;
                            }

                            let min_span_tree = resolve_buffer!(
                                ACCESS,
                                min_span_tree_buffer,
                                *partition_id_1,
                                [*target_partition_id]
                            );

                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            // let hints =
                            //     match hints_cache.contains_key(&(*partition_id_1, *vector_id_1)) {
                            //         true => &hints_cache[&(*partition_id_1, *vector_id_1)],
                            //         false => {
                            //             hints_cache.insert(
                            //                 (*partition_id_1, *vector_id_1),
                            //                 min_span_tree.dijkstra_weights(*vector_id_1).unwrap(),
                            //             );

                            //             cached_partition_hints
                            //                 .entry(*partition_id_1)
                            //                 .or_insert_with(|| {
                            //                     let mut set = HashSet::new();
                            //                     set.insert((*partition_id_1, *vector_id_1));

                            //                     set
                            //                 })
                            //                 .insert((*partition_id_1, *vector_id_1));

                            //             &hints_cache[&(*partition_id_1, *vector_id_1)]
                            //         }
                            //     };

                            let Ok(path) = min_span_tree.find_trail(
                                target_vector_id,
                                *vector_id_1,
                                // &hints,
                            ) else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:?}\n{:?}\n{:?} => {:?}",
                                    VectorId(value.id),
                                    vector_id_1,
                                    target_vector_id
                                );
                                todo!()
                            };
                            let Some(path) = path else { todo!() };

                            trails.extend(
                                path.1
                                    .into_iter()
                                    .map(|(vector_id_1, vector_id_2, weight)| {
                                        (
                                            (*partition_id_1, vector_id_1),
                                            (*partition_id_1, vector_id_2),
                                            weight,
                                        )
                                    }), // .collect::<Vec<_>>()
                            )
                        };

                        trails
                    };

                    let ((partition_id_1, vector_id_1), (partition_id_2, vector_id_2), max_weight) =
                        trails.into_iter().fold(
                            (
                                (PartitionId(Uuid::nil()), VectorId(Uuid::nil())),
                                (PartitionId(Uuid::nil()), VectorId(Uuid::nil())),
                                A::min(),
                            ),
                            |(acc_id_1, acc_id_2, acc_weight),
                             (next_id_1, next_id_2, next_weight)| {
                                match next_weight.partial_cmp(&acc_weight) {
                                    Some(Ordering::Greater) => (next_id_1, next_id_2, next_weight),
                                    _ => (acc_id_1, acc_id_2, acc_weight),
                                }
                            },
                        );
                    if partition_id_1 == PartitionId(Uuid::nil()) {
                        continue;
                    }
                    if partition_id_2 == PartitionId(Uuid::nil()) {
                        continue;
                    }

                    event!(
                        Level::DEBUG,
                        "Max edge {:?}",
                        (
                            (partition_id_1, vector_id_1),
                            (partition_id_2, vector_id_2),
                            max_weight,
                            weight
                        )
                    );
                    // if max weight < weight
                    if weight >= &max_weight {
                        continue;
                    };

                    if cached_partitions.contains_key(&partition_id_1) {
                        for (partition_id, vector_1, vector_2) in
                            cached_partitions.remove(&partition_id_1).unwrap()
                        {
                            partition_path_cache.remove(&(partition_id, vector_1, vector_2));
                            partition_path_cache.remove(&(partition_id, vector_2, vector_1));
                        }
                    }
                    if cached_partitions.contains_key(&partition_id_2) {
                        for (partition_id, vector_1, vector_2) in
                            cached_partitions.remove(&partition_id_2).unwrap()
                        {
                            partition_path_cache.remove(&(partition_id, vector_1, vector_2));
                            partition_path_cache.remove(&(partition_id, vector_2, vector_1));
                        }
                    }

                    if cached_partition_hints.contains_key(&partition_id_1) {
                        for id in cached_partition_hints.remove(&partition_id_1).unwrap() {
                            cached_partition_hints.remove(&partition_id_1);
                            hints_cache.remove(&id);
                        }
                    };

                    if cached_partition_hints.contains_key(&partition_id_2) {
                        for id in cached_partition_hints.remove(&partition_id_2).unwrap() {
                            cached_partition_hints.remove(&partition_id_2);
                            hints_cache.remove(&id);
                        }
                    };

                    inter_graph.add_edge(
                        closet_partition_id,
                        target_partition_id,
                        (
                            *weight,
                            (closet_partition_id, VectorId(value.id)),
                            (target_partition_id, target_vector_id),
                        ),
                    );

                    update_cluster(cluster_sets, &weight, VectorId(value.id), target_vector_id)
                        .await;

                    match partition_id_1 == partition_id_2 {
                        true => {
                            let split_partition_id = partition_id_1;

                            let min_span_tree = resolve_buffer!(
                                ACCESS,
                                min_span_tree_buffer,
                                split_partition_id,
                                [*target_partition_id]
                            );
                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            let partition = resolve_buffer!(
                                ACCESS,
                                partition_buffer,
                                split_partition_id,
                                [*target_partition_id]
                            );
                            let Some(partition) = &mut *partition.write().await else {
                                todo!()
                            };

                            let _ = min_span_tree.remove_edge(vector_id_1, vector_id_2).unwrap();
                            let _ = remove_cluster_edge(cluster_sets, vector_id_1, vector_id_2);

                            let [pair_1, pair_2] = split_partition::<
                                A,
                                B,
                                FirstTreeSplitStrategy,
                                PARTITION_CAP,
                                VECTOR_CAP,
                            >(
                                partition, min_span_tree, inter_graph
                            )
                            .unwrap();
                            // split_partition_into_trees(partition, min_span_tree, inter_graph)
                            //     .unwrap();

                            let (pair_1, pair_2) = match pair_1.0.id == *split_partition_id {
                                true => (pair_1, pair_2),
                                false => (pair_2, pair_1),
                            };

                            event!(
                                Level::DEBUG,
                                "split data: {:#?}\n{inter_graph:#?}",
                                [&pair_1, &pair_2]
                            );

                            {
                                let (new_partition, new_min_span_tree) = pair_1;

                                if new_partition.id != *split_partition_id {
                                    todo!()
                                }

                                *partition = new_partition;
                                *min_span_tree = new_min_span_tree;

                                let target_meta: &mut Meta<A, B> =
                                    &mut *meta_data[&*split_partition_id].write().await;

                                target_meta.size = partition.size;
                                target_meta.centroid = partition.centroid();

                                target_meta.edge_length = (
                                    match min_span_tree.smallest_edge() {
                                        Some(x) => x.2,
                                        None => A::max(),
                                    },
                                    match min_span_tree.largest_edge() {
                                        Some(x) => x.2,
                                        None => A::min(),
                                    },
                                );
                            }

                            // drop(partition);
                            // drop(min_span_tree);

                            if split_partition_id == target_partition_id {
                                visit_requirements.insert(PartitionId(pair_2.0.id));
                            }

                            let (new_partition, new_min_span_tree) = pair_2;

                            if new_min_span_tree.1.contains_key(&VectorId(value.id)) {
                                event!(
                                    Level::DEBUG,
                                    "Updating closet_partition_id: {closet_partition_id:?}->{split_partition_id:?}"
                                );
                                closet_partition_id = PartitionId(new_partition.id);
                            }

                            meta_data.insert(
                                new_partition.id,
                                Arc::new(RwLock::new(Meta::new(
                                    PartitionId(new_partition.id),
                                    new_partition.size,
                                    new_partition.centroid(),
                                    (
                                        match new_min_span_tree.smallest_edge() {
                                            Some(x) => x.2,
                                            None => A::max(),
                                        },
                                        match new_min_span_tree.largest_edge() {
                                            Some(x) => x.2,
                                            None => A::min(),
                                        },
                                    ),
                                ))),
                            );

                            resolve_buffer!(
                                PUSH,
                                min_span_tree_buffer,
                                new_min_span_tree,
                                [*target_partition_id, *split_partition_id]
                            );
                            resolve_buffer!(
                                PUSH,
                                partition_buffer,
                                new_partition,
                                [*target_partition_id, *split_partition_id]
                            );
                        }
                        false => {
                            event!(
                                Level::DEBUG,
                                "Removing edge inter edge: {:?}",
                                ((partition_id_1, vector_id_1), (partition_id_2, vector_id_2))
                            );
                            let _ = inter_graph
                                .remove_edge(
                                    (partition_id_1, vector_id_1),
                                    (partition_id_2, vector_id_2),
                                )
                                .unwrap();

                            let _ = remove_cluster_edge(cluster_sets, vector_id_1, vector_id_2);
                        }
                    };

                    partition_trail = {
                        let Ok(partition_trail) =
                            inter_graph.find_trail(closet_partition_id, target_partition_id)
                        else {
                            todo!()
                            // continue 'ordered_edges_loop; // Should be replaced with panic as this edge case should never happen
                        };

                        let Some((_, partition_trail)) = partition_trail else {
                            panic!("Failed to find partition trail from {closet_partition_id:?} -> {target_partition_id:?}\n{inter_graph:#?}");
                            todo!()
                        };

                        event!(Level::DEBUG, "partition_trail:\n{partition_trail:?}");

                        partition_trail
                    };
                }

                visit_requirements.take(&target_partition_id);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {}

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
    inter_graph: &mut InterPartitionGraph<A>,
    closet_partition_id: PartitionId,
    meta_data: &HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,
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
        for (id, meta_data) in meta_data {
            let meta_data = &*meta_data.read().await;

            let centroid = meta_data.centroid;

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
