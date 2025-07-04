use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs,
    sync::Arc,
};

use burn::prelude::Backend;
use petgraph::visit::EdgeRef;
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
                ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionMembership,
                PartitionSerial, VectorEntry, VectorEntrySerial,
            },
        },
        operations::{
            add::{create_local_inter_graph, create_local_meta, expand, get_required_partitions},
            cluster::{remove_cluster_edge, update_cluster},
            merge::{merge_partition_into, MergeError},
            split::{
                calculate_number_of_trees, split_partition, split_partition_into_trees,
                FirstTreeSplitStrategy, KMean, MaxAttempt,
            },
        },
    },
    resolve_buffer,
    vector::{Extremes, Field, Vector, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;
use crate::db::operations::add::get_neighbors;

pub async fn add<
    B: Backend,
    F: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<F>
        + Archive
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>
        + Debug
        + Extremes
        + std::marker::Send
        + std::marker::Sync
        + 'static,
    V: VectorSpace<F> + Sized + Clone + Copy + PartialEq + Extremes + From<VectorSerial<F>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    new_vector: VectorEntry<F, V>,

    transaction_id: Option<Uuid>,

    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<F, V>>>>>>,
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<F, V, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<F>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
    mut partition_membership: Arc<RwLock<PartitionMembership>>,
    mst_buffer: Arc<RwLock<DataBuffer<IntraPartitionGraph<F>, GraphSerial<F>, Global, MAX_LOADED>>>,

    inter_graph: Arc<RwLock<InterPartitionGraph<F>>>,
    cluster_sets: Arc<RwLock<Vec<ClusterSet<F>>>>,

    access_tx: Sender<BankerMessage>,

    #[cfg(feature = "benchmark")] benchmark: Benchmark,
) -> Result<(), ()>
where
    VectorSerial<F>: From<V>,
    for<'a> <F as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<F>]:
        DeserializeUnsized<[VectorEntrySerial<F>], Strategy<Pool, rancor::Error>>,
    [<F as Archive>::Archived]: DeserializeUnsized<[F], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <F as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, F)], Strategy<Pool, rancor::Error>>,
    f32: From<F>,
    <F as Archive>::Archived: rkyv::Deserialize<F, Strategy<Pool, rancor::Error>>,
    Vector<f32, VECTOR_CAP>: From<V>,
    Vector<f32, 2>: From<V>,
{
    let transaction_id = match transaction_id {
        Some(transaction_id) => transaction_id,
        None => Uuid::new_v4(),
    };

    // acquire required partitions
    println!("{:?} :- Acquire partitions", new_vector.id);
    let (cached_dist_map, acquired_partitions, write_partitions, mut closet_partition_id) = loop {
        tokio::task::yield_now().await;

        let mut meta_data = match meta_data.try_write() {
            Ok(val) => val,
            Err(_) => continue,
        };
        let meta_data = &mut *meta_data;
        event!(Level::DEBUG, "🔒 Locked `meta_data`");

        let inter_graph = &*inter_graph.read().await;
        event!(Level::DEBUG, "🔒 Locked `inter_graph`");

        let cluster_sets = &mut *cluster_sets.write().await;

        // find closet id
        let (mut closet_partition_id, closet_size) = {
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

                (*id, V::dist(centroid, &new_vector.vector))
            };

            for (_, data) in meta_data {
                let Meta {
                    id, centroid, size, ..
                } = &*data.read().await;

                let dist = V::dist(centroid, &new_vector.vector);

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
            // replace with lock
            let partition_buffer = &mut *partition_buffer.write().await;

            let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition_id);

            let Some(partition) = &mut *partition.write().await else {
                todo!()
            };

            let tree_buffer = &mut *mst_buffer.write().await;

            let tree = resolve_buffer!(ACCESS, tree_buffer, closet_partition_id);

            let Some(tree) = &mut *tree.write().await else {
                todo!()
            };

            if partition.size != 0 {
                continue;
            }

            add_into_partition(new_vector.clone(), partition)
                .expect("Unable to insert value into empty partition");

            tree.add_node(VectorId(new_vector.id));

            let meta = &mut *meta_data[&*closet_partition_id].write().await;

            meta.size = partition.size;
            meta.centroid = partition.centroid();

            // for cluster_set in cluster_sets.iter_mut() {
            //     let _ = cluster_set
            //         .new_cluster_from_vector(VectorId(new_vector.id), ClusterId(Uuid::new_v4()))
            //         .await
            //         .unwrap();
            // }

            // meta.edge_length = (todo!(), todo!());

            // update cluster set data
            for cluster_set in cluster_sets.iter_mut() {
                let cluster_id = cluster_set.new_cluster().unwrap();
                let _ = cluster_set
                    .new_cluster_from_vector(VectorId(new_vector.id), cluster_id)
                    .unwrap();
            }

            return Ok(());
        }

        let mut dist_map = HashMap::new();
        let closet_vector_id = {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Finding closet vector".to_string(), &benchmark);

            let mut w_partition_buffer = partition_buffer.write().await;
            let partition_buffer = &mut *w_partition_buffer;

            // getting neighbor ids
            let neighbor_ids: Vec<PartitionId> = get_neighbors::<B, F, V, VECTOR_CAP>(
                inter_graph,
                closet_partition_id,
                meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
            )
            .await;
            event!(Level::INFO, "🤝 Neighbors: {neighbor_ids:?}");

            let mut required_partitions: HashSet<&PartitionId> = [&closet_partition_id]
                .into_iter()
                .chain(neighbor_ids.iter())
                .collect();

            event!(Level::DEBUG, "📦 Finding closet vector");
            // let mut required_partitions: HashSet<&PartitionId> =
            //     required_ids.clone().into_iter().collect();

            let mut closet_id: Option<Uuid> = None;
            let mut closet_dist: Option<F> = None;

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
                    for partition in &partitions {
                        for vector_entry in partition.iter() {
                            let dist = V::dist(&new_vector.vector, &vector_entry.vector);

                            dist_map.insert(vector_entry.id, dist);

                            if let (Some(min_dist), Some(min_id)) =
                                (&mut closet_dist, &mut closet_id)
                            {
                                *min_dist = dist;
                                *min_id = partition.id;
                            } else {
                                closet_dist = Some(dist);
                                closet_id = Some(partition.id);
                            };
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

            event!(Level::DEBUG, "📦 Found closet vector");
            closet_id.unwrap()
        };

        // needs to be updated (Requires loading in more partitions in order to get distance of all nearby vectors)
        closet_partition_id = PartitionId(closet_vector_id);

        let neighbor_ids: Vec<PartitionId> = get_neighbors::<B, F, V, VECTOR_CAP>(
            inter_graph,
            closet_partition_id,
            meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
        )
        .await;

        let write_partitions: HashSet<PartitionId> =
            get_required_partitions(&neighbor_ids, &closet_partition_id, &inter_graph);

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

        match rx.await {
            Ok(AccessResponse::Granted) => {
                break (
                    dist_map,
                    read_partitions
                        .clone()
                        .into_iter()
                        .chain(write_partitions.clone().into_iter())
                        .collect::<HashSet<_>>(),
                    write_partitions.clone(),
                    closet_partition_id,
                )
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
    };

    #[cfg(feature = "benchmark")]
    let _child_benchmark = Benchmark::spawn_child("Total Insertion time".to_string(), &benchmark);

    let read_partitions: HashSet<_> = acquired_partitions
        .difference(&write_partitions)
        .map(|x| *x)
        .collect();

    // create local env
    println!("{:?} :- Create local buffer", new_vector.id);
    let (
        mut local_meta_data,
        mut local_inter_graph,
        mut local_partition_buffer,
        mut local_partition_membership,
        mut local_mst_buffer,
        // mut local_cluster_sets,
    ) = {
        let local_meta_data = {
            let meta_data: &HashMap<Uuid, Arc<RwLock<Meta<F, V>>>> = &*meta_data.read().await;
            create_local_meta(&meta_data, &acquired_partitions).await
        };

        let local_inter_graph = {
            let inter_graph: &InterPartitionGraph<F> = &*inter_graph.read().await;
            create_local_inter_graph(&acquired_partitions, &inter_graph)
        };

        let mut local_partition_buffer: DataBuffer<
            Partition<F, V, PARTITION_CAP, VECTOR_CAP>,
            PartitionSerial<F>,
            Local,
            MAX_LOADED,
        > = partition_buffer
            .write()
            .await
            .copy_local(transaction_id, "partition", &write_partitions)
            .await;

        let mut local_partition_membership: PartitionMembership = PartitionMembership::new(
            format!("data//local//{}//partition//membership", transaction_id),
        );
        for partition_id in &write_partitions {
            let partition = resolve_buffer!(ACCESS, local_partition_buffer, *partition_id);

            let Some(partition) = &*partition.read().await else {
                todo!();
            };

            local_partition_membership.update_membership(&partition);
        }

        let local_mst_buffer: DataBuffer<
            IntraPartitionGraph<F>,
            GraphSerial<F>,
            Local,
            MAX_LOADED,
        > = mst_buffer
            .write()
            .await
            .copy_local(transaction_id, "graph", &write_partitions)
            .await;

        // let local_cluster_sets = 'cluster_sets: {
        //     let cluster_sets = &*cluster_sets.read().await;

        //     if cluster_sets.len() == 0 {
        //         break 'cluster_sets Vec::new();
        //     }

        //     // could be optimized through incremental loading & updating
        //     // so vectors can be loaded while the database is being initialized(pipelined)
        //     // therefore something is being done while files are being loaded or db is updated & less memory in vector_ids
        //     let mut vector_ids = Vec::new();
        //     for id in &write_partitions {
        //         let partition = resolve_buffer!(ACCESS, local_partition_buffer, *id);
        //         let Some(partition) = &*partition.read().await else {
        //             todo!();
        //         };

        //         for VectorEntry{id, ..} in partition.iter() {
        //             vector_ids.push(VectorId(id.clone()));
        //         }
        //     }

        //     cluster::create_local(
        //         cluster_sets,
        //         vector_ids.as_slice(),
        //         transaction_id.clone()
        //     ).await
        // };

        (
            local_meta_data,
            local_inter_graph,
            local_partition_buffer,
            local_partition_membership,
            local_mst_buffer,
            // local_cluster_sets
        )
    };

    // insert into partition
    println!("{:?} :- Insert into local buffer", new_vector.id);
    {
        #[cfg(feature = "benchmark")]
        let _child_benchmark = Benchmark::spawn_child("Insert partition".to_string(), &benchmark);

        // load partition instead
        let size: usize = {
            let partition = resolve_buffer!(ACCESS, local_partition_buffer, closet_partition_id);

            let Some(partition) = &*partition.read().await else {
                todo!()
            };

            event!(Level::DEBUG, "Insert partition size: {}", partition.size);

            partition.size
        };

        if size >= PARTITION_CAP {
            let partition = resolve_buffer!(ACCESS, local_partition_buffer, closet_partition_id);

            let Some(partition) = &mut *partition.try_write().unwrap() else {
                todo!()
            };

            // split partition

            let mst = resolve_buffer!(ACCESS, local_mst_buffer, closet_partition_id);
            let Some(mst) = &mut *mst.try_write().unwrap() else {
                todo!()
            };

            // split based on trees

            event!(Level::DEBUG, "Beginning KMean split");
            let Ok(splits) =
                split_partition::<F, V, KMean<MaxAttempt<128>>, PARTITION_CAP, VECTOR_CAP>(
                    partition,
                    mst,
                    &mut local_inter_graph,
                )
            else {
                panic!()
            };
            event!(Level::DEBUG, "Finished KMean split");

            let mut new_closet_id = (
                splits[0].0.id,
                V::dist(&splits[0].0.centroid(), &new_vector.vector),
            );
            let mut push_pairs = Vec::new();

            // can easily be unwrapped & gets rid of if statements
            for (mut new_partition, mut new_graph) in splits {
                // fix any forests
                let pairs = match calculate_number_of_trees(&new_graph) > 1 {
                    true => {
                        event!(Level::DEBUG, "Fixing forest");
                        split_partition_into_trees(
                            &mut new_partition,
                            &mut new_graph,
                            &mut local_inter_graph,
                        )
                        .unwrap()
                    }
                    false => vec![(new_partition, new_graph)],
                };

                // if last value is closet_partition (then maybe skip that value and handle the edge case after the for loop)
                for (new_partition, new_mst) in pairs {
                    let dist = V::dist(&new_partition.centroid(), &new_vector.vector);

                    if dist < new_closet_id.1 {
                        new_closet_id.0 = new_partition.id;
                        new_closet_id.1 = dist;
                    }

                    // update partition if new_partition matches
                    if new_partition.id == partition.id {
                        *partition = new_partition;
                        *mst = new_mst;

                        let target_meta: &mut Meta<F, V> =
                            &mut *local_meta_data[&partition.id].write().await;

                        target_meta.size = partition.size;
                        target_meta.centroid = partition.centroid();

                        target_meta.edge_length = (
                            match mst.smallest_edge() {
                                Some(x) => x.2,
                                None => F::max(),
                            },
                            match mst.largest_edge() {
                                Some(x) => x.2,
                                None => F::min(),
                            },
                        );
                    } else {
                        event!(Level::DEBUG, "Updating new partition");
                        local_meta_data.insert(
                            new_partition.id,
                            Arc::new(RwLock::new(Meta::new(
                                PartitionId(new_partition.id),
                                new_partition.size,
                                new_partition.centroid(),
                                (
                                    match new_mst.smallest_edge() {
                                        Some(x) => x.2,
                                        None => F::max(),
                                    },
                                    match new_mst.largest_edge() {
                                        Some(x) => x.2,
                                        None => F::min(),
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
                local_partition_membership.update_membership(&new_partition);

                resolve_buffer!(PUSH, local_partition_buffer, new_partition, [partition.id]);
                resolve_buffer!(PUSH, local_mst_buffer, new_graph, [*mst.2]);
            }

            closet_partition_id = PartitionId(new_closet_id.0);
        }

        let partition = resolve_buffer!(ACCESS, local_partition_buffer, closet_partition_id);

        let Some(partition) = &mut *partition.try_write().unwrap() else {
            todo!()
        };

        let _ = add_into_partition(new_vector.clone(), partition).unwrap();

        local_partition_membership.assign(VectorId(new_vector.id), PartitionId(partition.id));

        let local_data = &mut *local_meta_data[&*closet_partition_id].write().await;

        local_data.size = partition.size;
        local_data.centroid = partition.centroid();
    }

    // update min_spanning
    println!("{:?} :- Update MST", new_vector.id);
    let (new_edges, deleted_edges): (
        HashMap<(VectorId, VectorId), F>,
        HashSet<(VectorId, VectorId)>,
    ) = 'update_min_span: {
        #[cfg(feature = "benchmark")]
        let _child_benchmark =
            Benchmark::spawn_child("Update min-spanning tree".to_string(), &benchmark);

        let mut new_edges: HashMap<(VectorId, VectorId), F> = HashMap::new();
        let mut deleted_edges: HashSet<(VectorId, VectorId)> = HashSet::new();
        // filter out partitions where there is no overlap partition_max < vector_min
        // let mut neighbor_ids = HashSet::new();
        let neighbor_ids: HashSet<PartitionId> = {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Find neighbors".to_string(), &_child_benchmark);

            get_neighbors::<B, F, V, VECTOR_CAP>(
                &local_inter_graph,
                closet_partition_id,
                &local_meta_data, // HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
            )
            .await
            .into_iter()
            .filter(|id| !read_partitions.contains(id))
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
        let dist_map: HashMap<VectorId, F> = {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Generate distance map".to_string(), &_child_benchmark);

            let mut missing_partitions = [closet_partition_id]
                .into_iter()
                .chain(neighbor_ids.clone())
                .collect::<HashSet<_>>();
            let mut dist_map: HashMap<VectorId, F> = cached_dist_map
                .into_iter()
                .map(|(id, val)| (VectorId(id), val))
                .collect();

            while missing_partitions.len() > 0 {
                let mut acquired_partitions = Vec::new();
                event!(
                    Level::DEBUG,
                    "Attempt to acquired required ids that aren't loaded"
                );
                for id in missing_partitions.iter() {
                    // Replace with try access and/or batch access
                    let Ok(partition) = local_partition_buffer.access(&**id).await else {
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
                            dist_map.insert(VectorId(*id), V::dist(&new_vector.vector, vector));
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
                let mut least_used = local_partition_buffer.least_used_iter().await;

                for id in missing_partitions.clone() {
                    match local_partition_buffer.load(&*id).await {
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

                            local_partition_buffer
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

            dist_map.remove(&VectorId(new_vector.id));

            dist_map
        };

        // println!("{:#?}", dist_map);
        if dist_map.len() == 0 {
            let min_span_tree: Arc<RwLock<Option<IntraPartitionGraph<F>>>> =
                resolve_buffer!(ACCESS, local_mst_buffer, closet_partition_id);

            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                todo!()
            };

            min_span_tree.add_node(VectorId(new_vector.id));

            break 'update_min_span (new_edges, deleted_edges);
        }

        // Update closest partition edges
        {
            #[cfg(feature = "benchmark")]
            let _child_benchmark =
                Benchmark::spawn_child("Updating local graph".to_string(), &_child_benchmark);

            let min_span_tree = resolve_buffer!(ACCESS, local_mst_buffer, closet_partition_id);

            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                todo!()
            };

            // first edge
            min_span_tree.add_node(VectorId(new_vector.id));

            let Ok(tmp_new_edges) = min_span_tree.update(
                VectorId(new_vector.id),
                &dist_map
                    .iter()
                    .filter(|(id, _)| min_span_tree.1.contains_key(id))
                    .map(|(id, val)| (*val, *id))
                    .collect::<Vec<_>>(),
            ) else {
                todo!();
            };

            for (weight, id_1, id_2) in tmp_new_edges {
                new_edges.insert((id_1, id_2), weight);
            }

            // might become optimal after a certan point
            // match vector_iter.next() {
            //     Some(vector_id) => {
            //         min_span_tree.add_edge(
            //             VectorId(new_vector.id),
            //             *vector_id,
            //             *dist_map.get(vector_id).expect(""),
            //         );
            //     }
            //     None => todo!(),
            // }
            //
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
            // for vector_id in vector_iter {
            //     let weight = *dist_map.get(vector_id).expect("");
            //     event!(Level::DEBUG, "{:?} -> {:?}", vector_id, VectorId(new_vector.id));
            //     let Ok(path) = min_span_tree.find_trail(*vector_id, VectorId(new_vector.id)) else {
            //         event!(
            //             Level::DEBUG,
            //             "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?}",
            //             vector_id,
            //             VectorId(new_vector.id)
            //         );
            //         todo!()
            //     };
            //     let Some((_, path)) = path else { todo!() };
            //
            //     let (max_vector_id_1, max_vector_id_2, max_weight) = path.into_iter().fold(
            //         (VectorId(Uuid::nil()), VectorId(Uuid::nil()), A::min()),
            //         |(acc_id_1, acc_id_2, acc_weight), (next_id_1, next_id_2, next_weight)| {
            //             match next_weight.partial_cmp(&acc_weight) {
            //                 Some(Ordering::Greater) => (next_id_1, next_id_2, next_weight),
            //                 _ => (acc_id_1, acc_id_2, acc_weight),
            //             }
            //         },
            //     );
            //
            //     if weight >= max_weight {
            //         continue;
            //     };
            //
            //     let _ = min_span_tree
            //         .remove_edge(max_vector_id_1, max_vector_id_2)
            //         .unwrap();
            //
            //     let _ = remove_cluster_edge(cluster_sets, max_vector_id_1, max_vector_id_2);
            //
            //     min_span_tree.add_edge(VectorId(new_vector.id), *vector_id, weight);
            //
            //     update_cluster(cluster_sets, &weight, VectorId(new_vector.id), *vector_id).await;
            // }
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

            let mut hints_cache: HashMap<(PartitionId, VectorId), HashMap<VectorId, F>> =
                HashMap::new();
            let mut cached_partition_hints: HashMap<PartitionId, HashSet<(PartitionId, VectorId)>> =
                HashMap::new();

            while let Some(target_partition_id) = visit_requirements.iter().next().cloned() {
                let target_vector_ids = {
                    let min_span_tree =
                        resolve_buffer!(ACCESS, local_mst_buffer, target_partition_id);

                    let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                        todo!()
                    };

                    min_span_tree.1.clone()
                };

                let mut partition_trail = {
                    let Ok(partition_trail) =
                        local_inter_graph.find_trail(closet_partition_id, target_partition_id)
                    else {
                        todo!()
                    };

                    let Some((_, partition_trail)) = partition_trail else {
                        panic!("Failed to find partition trail from {closet_partition_id:?} -> {target_partition_id:?}\n{inter_graph:#?}");
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
                            local_mst_buffer,
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
                        let mut trails: Vec<((PartitionId, VectorId), (PartitionId, VectorId), F)> = 'initialize_trail: {
                            let inter_edge = partition_trail[0];

                            if VectorId(new_vector.id) == inter_edge.0 .1 {
                                break 'initialize_trail vec![inter_edge];
                            }
                            let min_span_tree = resolve_buffer!(
                                ACCESS,
                                local_mst_buffer,
                                closet_partition_id,
                                [*target_partition_id]
                            );

                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            // djikstra's hint
                            // Sink (value.id)
                            let hints = match hints_cache
                                .contains_key(&(closet_partition_id, VectorId(new_vector.id)))
                            {
                                true => {
                                    &hints_cache[&(closet_partition_id, VectorId(new_vector.id))]
                                }
                                false => {
                                    hints_cache.insert(
                                        (closet_partition_id, VectorId(new_vector.id)),
                                        min_span_tree
                                            .dijkstra_weights(VectorId(new_vector.id))
                                            .unwrap(),
                                    );

                                    cached_partition_hints
                                        .entry(closet_partition_id)
                                        .or_insert_with(|| {
                                            let mut set = HashSet::new();
                                            set.insert((
                                                closet_partition_id,
                                                VectorId(new_vector.id),
                                            ));

                                            set
                                        })
                                        .insert((closet_partition_id, VectorId(new_vector.id)));

                                    &hints_cache[&(closet_partition_id, VectorId(new_vector.id))]
                                }
                            };

                            let Ok(trail) = min_span_tree.find_trail_with_hints(
                                inter_edge.0 .1,
                                VectorId(new_vector.id),
                                &hints,
                            ) else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
                                    VectorId(new_vector.id),
                                    inter_edge.0 .1,
                                    inter_edge.1,
                                );

                                todo!()
                            };

                            let Some(trail) = trail else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
                                    VectorId(new_vector.id),
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
                                resolve_buffer!(ACCESS, local_mst_buffer, partition_id_1);

                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            let Ok(trail) = min_span_tree.find_trail(vector_id_1, vector_id_2)
                            else {
                                event!(
                                    Level::DEBUG,
                                    "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?} => {:?}",
                                    VectorId(new_vector.id),
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
                                local_mst_buffer,
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
                                    VectorId(new_vector.id),
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
                                F::min(),
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

                    local_inter_graph.add_edge(
                        closet_partition_id,
                        target_partition_id,
                        (
                            *weight,
                            (closet_partition_id, VectorId(new_vector.id)),
                            (target_partition_id, target_vector_id),
                        ),
                    );

                    if new_edges.contains_key(&(vector_id_1, vector_id_2))
                        || new_edges.contains_key(&(vector_id_2, vector_id_1))
                    {
                        new_edges.remove(&(vector_id_1, vector_id_2));
                        new_edges.remove(&(vector_id_2, vector_id_1));
                    }
                    if !deleted_edges.contains(&(vector_id_2, vector_id_1)) {
                        deleted_edges.insert((vector_id_1, vector_id_2));
                    }

                    if deleted_edges.contains(&(VectorId(new_vector.id), target_vector_id))
                        || deleted_edges.contains(&(target_vector_id, VectorId(new_vector.id)))
                    {
                        deleted_edges.remove(&(VectorId(new_vector.id), target_vector_id));
                        deleted_edges.remove(&(target_vector_id, VectorId(new_vector.id)));
                    }

                    if !new_edges.contains_key(&(target_vector_id, VectorId(new_vector.id))) {
                        new_edges.insert((VectorId(new_vector.id), target_vector_id), *weight);
                    }

                    match partition_id_1 == partition_id_2 {
                        true => {
                            let split_partition_id = partition_id_1;

                            let min_span_tree = resolve_buffer!(
                                ACCESS,
                                local_mst_buffer,
                                split_partition_id,
                                [*target_partition_id]
                            );
                            let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                                todo!()
                            };

                            let partition = resolve_buffer!(
                                ACCESS,
                                local_partition_buffer,
                                split_partition_id,
                                [*target_partition_id]
                            );
                            let Some(partition) = &mut *partition.write().await else {
                                todo!()
                            };

                            let _ = min_span_tree.remove_edge(vector_id_1, vector_id_2).unwrap();

                            let [pair_1, pair_2] =
                                split_partition::<
                                    F,
                                    V,
                                    FirstTreeSplitStrategy,
                                    PARTITION_CAP,
                                    VECTOR_CAP,
                                >(
                                    partition, min_span_tree, &mut local_inter_graph
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

                                let target_meta: &mut Meta<F, V> =
                                    &mut *local_meta_data[&*split_partition_id].write().await;

                                target_meta.size = partition.size;
                                target_meta.centroid = partition.centroid();

                                target_meta.edge_length = (
                                    match min_span_tree.smallest_edge() {
                                        Some(x) => x.2,
                                        None => F::max(),
                                    },
                                    match min_span_tree.largest_edge() {
                                        Some(x) => x.2,
                                        None => F::min(),
                                    },
                                );
                            }

                            // drop(partition);
                            // drop(min_span_tree);

                            if split_partition_id == target_partition_id {
                                visit_requirements.insert(PartitionId(pair_2.0.id));
                            }

                            let (new_partition, new_min_span_tree) = pair_2;

                            if new_min_span_tree.1.contains_key(&VectorId(new_vector.id)) {
                                event!(
                                    Level::DEBUG,
                                    "Updating closet_partition_id: {closet_partition_id:?}->{split_partition_id:?}"
                                );
                                closet_partition_id = PartitionId(new_partition.id);
                            }

                            local_meta_data.insert(
                                new_partition.id,
                                Arc::new(RwLock::new(Meta::new(
                                    PartitionId(new_partition.id),
                                    new_partition.size,
                                    new_partition.centroid(),
                                    (
                                        match new_min_span_tree.smallest_edge() {
                                            Some(x) => x.2,
                                            None => F::max(),
                                        },
                                        match new_min_span_tree.largest_edge() {
                                            Some(x) => x.2,
                                            None => F::min(),
                                        },
                                    ),
                                ))),
                            );

                            local_partition_membership.update_membership(&new_partition);

                            resolve_buffer!(
                                PUSH,
                                local_mst_buffer,
                                new_min_span_tree,
                                [*target_partition_id, *split_partition_id]
                            );
                            resolve_buffer!(
                                PUSH,
                                local_partition_buffer,
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
                            let _ = local_inter_graph
                                .remove_edge(
                                    (partition_id_1, vector_id_1),
                                    (partition_id_2, vector_id_2),
                                )
                                .unwrap();
                        }
                    };

                    partition_trail = {
                        let Ok(partition_trail) =
                            local_inter_graph.find_trail(closet_partition_id, target_partition_id)
                        else {
                            todo!()
                            // continue 'ordered_edges_loop; // Should be replaced with panic as this edge case should never happen
                        };

                        let Some((_, partition_trail)) = partition_trail else {
                            panic!("Failed to find partition trail from {closet_partition_id:?} -> {target_partition_id:?}\n{inter_graph:#?}");
                        };

                        event!(Level::DEBUG, "partition_trail:\n{partition_trail:?}");

                        partition_trail
                    };
                }

                visit_requirements.take(&target_partition_id);
            }
        }

        (new_edges, deleted_edges)
    };

    // merge splits
    {
        for sink_id in &write_partitions {
            let mut delete_ids = Vec::new();

            let sink_partition = resolve_buffer!(ACCESS, local_partition_buffer, *sink_id);
            let Some(sink_partition) = &mut *sink_partition.write().await else {
                todo!()
            };
            let sink_mst = resolve_buffer!(ACCESS, local_mst_buffer, *sink_id);
            let Some(sink_mst) = &mut *sink_mst.write().await else {
                todo!()
            };
            {
                let data = &mut *local_meta_data[&*sink_id].write().await;

                'bfs_loop: loop {
                    // bfs from sink
                    let neighbors: Vec<_> = local_inter_graph
                        .0
                        .edges(local_inter_graph.1[sink_id])
                        .map(|edge_ref| {
                            let source = local_inter_graph.0[edge_ref.source()];
                            let target = local_inter_graph.0[edge_ref.target()];

                            match source.0 == **sink_id {
                                true => target,
                                false => source,
                            }
                        })
                        .filter(|target| !write_partitions.contains(&PartitionId(target.0)))
                        .filter(|target| !read_partitions.contains(&PartitionId(target.0)))
                        .collect();

                    if neighbors.len() == 0 {
                        break 'bfs_loop;
                    }

                    // update sink
                    for source_id in neighbors {
                        let result = {
                            // println!("Getting source partition :- {source_id:?}");
                            let source_partition = resolve_buffer!(
                                ACCESS,
                                local_partition_buffer,
                                source_id,
                                [**sink_id]
                            );
                            let Some(source_partition) = &*source_partition.read().await else {
                                todo!()
                            };

                            // println!("Getting source mst :- {source_id:?}");
                            let source_mst =
                                resolve_buffer!(ACCESS, local_mst_buffer, source_id, [**sink_id]);
                            let Some(source_mst) = &*source_mst.read().await else {
                                todo!()
                            };
                            // println!("{acquired_partitions:?}\n{local_inter_graph:#?}\n{sink_partition:#?}\n{source_partition:#?}");

                            merge_partition_into(
                                (&sink_partition, &sink_mst),
                                (&source_partition, &source_mst),
                                &mut local_inter_graph,
                            )
                        };

                        match result {
                            Ok((new_partition, new_mst)) => {
                                local_partition_membership.update_membership(&new_partition);

                                *sink_partition = new_partition;
                                *sink_mst = new_mst;

                                data.size = sink_partition.size;
                                data.centroid = sink_partition.centroid();
                                // data.edge_length = (
                                //     A::min(data.edge_length.0, ),
                                //     A::max(data.edge_length.1, )
                                // )

                                delete_ids.push(*source_id);
                                let _ = local_partition_buffer
                                    .delete_value(&*source_id)
                                    .await
                                    .unwrap();
                                let _ = local_mst_buffer.delete_value(&*source_id).await.unwrap();
                            }
                            Err(MergeError::Overflow) => {
                                break 'bfs_loop;
                            }
                            _ => {
                                todo!()
                            }
                        }
                    }
                }
            }
            for id in delete_ids {
                local_meta_data.remove(&id);
            }
        }
    }

    // update global using local data
    println!("{:?} :- Update global", new_vector.id);
    {
        {
            // flush partition buffer
            let partition_buffer = &mut *partition_buffer.write().await;

            let partition_membership = &mut *partition_membership.write().await;
            partition_membership.flush(&local_partition_membership);

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
                    let cluster_id = cluster_set.new_cluster().unwrap();
                    let _ = cluster_set
                        .new_cluster_from_vector(VectorId(new_vector.id), cluster_id)
                        .unwrap();
                }

                for ((id_1, id_2), dist) in new_edges {
                    update_cluster(cluster_sets, &dist, id_1, id_2).await;
                }

                for (id_1, id_2) in deleted_edges {
                    remove_cluster_edge(cluster_sets, id_1, id_2).await;
                }
            }
        }
        let _ = fs::remove_dir_all(&format!("data/local/{}", transaction_id.to_string())).unwrap();
    }

    println!("{:?} :- Release acquired_partitions", new_vector.id);
    let _ = access_tx
        .send(BankerMessage::ReleaseAccess {
            transaction_id,
            partitions: acquired_partitions.iter().map(|x| *x).collect(),
        })
        .await
        .unwrap();

    Ok(())
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
