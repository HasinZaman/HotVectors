use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem,
    sync::Arc,
};

use petgraph::{csr::DefaultIx, graph::EdgeIndex, visit::EdgeRef};

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

use crate::{
    db::{
        component::{
            data_buffer::{BufferError, DataBuffer},
            graph::{GraphSerial, IntraPartitionGraph},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{
                ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionSerial, VectorEntry,
                VectorEntrySerial,
            },
        },
        operations::split::split_partition,
    },
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;

fn find_closet_vector<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    vector: VectorEntry<A, B>,
    partitions: &[&Partition<A, B, PARTITION_CAP, VECTOR_CAP>],

    dist_map: &mut HashMap<(PartitionId, VectorId), A>,
) -> ((PartitionId, VectorId), A) {
    if cfg!(feature = "gpu_processing") {
        todo!()
    } else {
        partitions
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| ((PartitionId(x.id), VectorId(y.id)), y.vector))
            })
            .flatten()
            .map(|(id, vec)| {
                let dist = B::dist(&vector.vector, &vec);

                dist_map.insert(id, dist);

                (id, dist)
            })
            .min_by(|(id_1, dist_1), (id_2, dist_2)| {
                dist_1.partial_cmp(dist_2).unwrap_or(Ordering::Equal)
            })
            .unwrap()
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

fn group_edges<A: PartialEq + Clone + Copy + Field<A>>(
    edges: Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
) -> HashMap<
    PartitionId,
    Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
> {
    let mut partition_map = HashMap::new();

    for edge in edges {
        let (idx, dist, (partition_id_1, vector_id_1), (partition_id_2, vector_id_2)) = edge;

        partition_map
            .entry(partition_id_2)
            .or_insert_with(Vec::new)
            .push((
                idx,
                dist,
                (partition_id_1, vector_id_1),
                (partition_id_2, vector_id_2),
            ));
    }

    partition_map
}

fn update_local_min_span_tree<A: PartialEq + Clone + Copy + Field<A> + PartialOrd>(
    vector_id: VectorId,

    partition_id: PartitionId,

    target_vector_id: VectorId,

    inter_graph: &InterPartitionGraph<A>, // Used to determine all foreign edges that need to be updated
    min_spanning_graph: &mut IntraPartitionGraph<A>,

    dist_map: HashMap<VectorId, A>,
) -> Result<
    Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
    PartitionErr,
>
where
    IntraPartitionGraph<A>: Debug,
{
    let node_index = min_spanning_graph.add_node(vector_id);

    // check if new inter_graph to replace intra_edges
    let replace_edges: Vec<(EdgeIndex, VectorId)> = min_spanning_graph
        .0
        .edges(match min_spanning_graph.1.get(&target_vector_id) {
            Some(val) => *val,
            None => {
                panic!("Failed to extract {target_vector_id:?} from {min_spanning_graph:#?} ")
            }
        })
        .map(|edge| {
            (
                edge.id(),
                min_spanning_graph.0.node_weight(edge.target()).unwrap(),
                min_spanning_graph.0.node_weight(edge.source()).unwrap(),
                edge.weight(),
            )
        })
        .map(
            |(edge_index, target, source, dist)| match target == &target_vector_id {
                true => (edge_index, source, dist),
                false => (edge_index, target, dist),
            },
        )
        .filter(|(_, id, dist)| dist < &&dist_map[*id])
        .map(|(edge_index, id, _)| (edge_index, *id))
        .collect();

    // add new edges
    replace_edges
        .iter()
        .map(|(_, target_id)| (dist_map[target_id], *target_id, vector_id))
        .for_each(|(dist, id_1, id_2)| {
            min_spanning_graph.add_edge(id_1, id_2, dist);
        });

    // remove edges
    replace_edges
        .into_iter()
        .map(|(edge_index, _)| edge_index)
        .for_each(|edge_index| {
            min_spanning_graph.0.remove_edge(edge_index);
        });

    min_spanning_graph.0.add_edge(
        node_index,
        min_spanning_graph.1[&target_vector_id],
        dist_map[&target_vector_id],
    );

    let replace_inter_edges: Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )> = inter_graph
        .0
        .edges(inter_graph.1[&partition_id])
        .map(|edge| (edge.id(), edge.weight()))
        .filter_map(|(idx, (dist, id_1, id_2))| {
            match (
                id_1 == &(partition_id, target_vector_id),
                id_2 == &(partition_id, target_vector_id),
            ) {
                (true, false) => Some((idx, *dist, *id_1, *id_2)),
                (false, true) => Some((idx, *dist, *id_2, *id_1)),
                _ => None,
            }
        })
        .collect();

    Ok(replace_inter_edges)
}

fn update_foreign_min_span_tree<A: PartialEq + Clone + Copy + Field<A> + PartialOrd>(
    source_id: PartitionId,
    vector_id: VectorId,

    target_partition_id: PartitionId,
    target_vector_id: VectorId,

    inter_graph: &mut InterPartitionGraph<A>, // Used to determine all foreign edges that need to be updated
    min_spanning_graph: &mut IntraPartitionGraph<A>,

    dist_map: HashMap<VectorId, A>,
) -> Result<
    Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
    PartitionErr,
>
where
    IntraPartitionGraph<A>: Debug,
{
    let replace_inter_edges: Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )> = inter_graph
        .0
        .edges(inter_graph.1[&target_partition_id])
        .map(|edge| (edge.id(), edge.weight()))
        .filter_map(|(idx, (dist, id_1, id_2))| {
            match (
                id_1 == &(target_partition_id, target_vector_id),
                id_2 == &(target_partition_id, target_vector_id),
            ) {
                // (target_id, other)
                (true, false) => Some((idx, *dist, *id_1, *id_2)),
                (false, true) => Some((idx, *dist, *id_2, *id_1)),
                _ => None,
            }
        })
        .collect();

    // check if new inter_graph to replace intra_edges
    let replace_edges: Vec<(EdgeIndex, VectorId)> = min_spanning_graph
        .0
        .edges(match min_spanning_graph.1.get(&target_vector_id) {
            Some(val) => *val,
            None => {
                panic!("Failed to extract {target_vector_id:?} from {min_spanning_graph:#?} ")
            }
        })
        .map(|edge| {
            (
                edge.id(),
                min_spanning_graph.0.node_weight(edge.target()).unwrap(),
                min_spanning_graph.0.node_weight(edge.source()).unwrap(),
                edge.weight(),
            )
        })
        .map(
            |(edge_index, target, source, dist)| match target == &target_vector_id {
                true => (edge_index, source, dist),
                false => (edge_index, target, dist),
            },
        )
        .filter(|(_, id, dist)| dist < &&dist_map[*id])
        .map(|(edge_index, id, _)| (edge_index, *id))
        .collect();

    // add new edges
    replace_edges
        .iter()
        .map(|(_, target_id)| (dist_map[target_id], *target_id))
        .for_each(|(dist, id_1)| {
            inter_graph.add_edge(
                source_id,
                target_partition_id,
                (dist, (target_partition_id, id_1), (source_id, vector_id)),
            );
        });

    // remove edges
    replace_edges
        .into_iter()
        .map(|(edge_index, _)| edge_index)
        .for_each(|edge_index| {
            min_spanning_graph.0.remove_edge(edge_index);
        });

    Ok(replace_inter_edges)
}

macro_rules! resolve_buffer {
    (ACCESS, $buffer:expr, $id:expr) => {
        'buffer_access: {
            match $buffer.access(&$id).await {
                Ok(value) => value,
                Err(_) => {
                    if let Ok(_) = $buffer.load(&$id).await {
                        break 'buffer_access $buffer.access(&$id).await.unwrap();
                    };

                    let mut least_used = $buffer.least_used_iter().await.unwrap();

                    loop {
                        event!(Level::DEBUG, "Attempt get least used");
                        let Some(next_unload) = least_used.next() else {
                            event!(Level::DEBUG, "Restarting least used iter");
                            least_used = match $buffer.least_used_iter().await {
                                Some(val) => val,
                                None => continue,
                            };
                            continue;
                        };

                        event!(
                            Level::DEBUG,
                            "Filtering values any value that is equal to load goal"
                        );
                        if $id == PartitionId(next_unload.1) {
                            continue;
                        }

                        event!(
                            Level::DEBUG,
                            "Attempt to unload({:?}) & load({:?})",
                            next_unload.1,
                            $id
                        );
                        if let Err(err) = $buffer.unload_and_load(&next_unload.1, &$id).await {
                            event!(Level::DEBUG, "Err({err:?})");
                            continue;
                        };

                        event!(Level::DEBUG, "Break loop and return");
                        break $buffer.access(&$id).await.unwrap();
                    }
                }
            }
        }
    };
    (ACCESS, $buffer:expr, $id:expr, $loaded_ids:expr) => {
        'buffer_access: {
            match $buffer.access(&$id).await {
                Ok(partition) => partition,
                Err(_) => {
                    if let Ok(_) = $buffer.load(&$id).await {
                        break 'buffer_access $buffer.access(&$id).await.unwrap();
                    };

                    let mut least_used = $buffer.least_used_iter().await.unwrap();

                    loop {
                        event!(Level::DEBUG, "Attempt get least used");
                        let Some(next_unload) = least_used.next() else {
                            event!(Level::DEBUG, "Restarting least used iter");
                            least_used = match $buffer.least_used_iter().await {
                                Some(val) => val,
                                None => continue,
                            };
                            continue;
                        };

                        event!(
                            Level::DEBUG,
                            "Filtering values any value that is equal to load goal"
                        );
                        if $id == PartitionId(next_unload.1) {
                            continue;
                        }
                        if $loaded_ids.iter().any(|id| id == &next_unload.1) {
                            let unload_id = next_unload.1;
                            let loaded_ids = $loaded_ids;
                            event!(
                                Level::DEBUG,
                                "unload_id:({unload_id}) in {loaded_ids:?} - Must skip."
                            );
                            continue;
                        }

                        event!(
                            Level::DEBUG,
                            "Attempt to unload({:?}) & load({:?})",
                            next_unload.1,
                            $id
                        );
                        if let Err(err) = $buffer.unload_and_load(&next_unload.1, &$id).await {
                            event!(Level::DEBUG, "Err({err:?})");
                            continue;
                        };

                        event!(Level::DEBUG, "Break loop and return");
                        break $buffer.access(&$id).await.unwrap();
                    }
                }
            }
        }
    };

    (PUSH, $buffer:expr, $value:expr, $loaded_ids:expr) => {{
        'primary_loop: while let Err(_) = $buffer.push($value.clone()).await {
            let mut least_used = $buffer.least_used_iter().await.unwrap();

            loop {
                event!(Level::DEBUG, "Attempt get least used");
                let Some(next_unload) = least_used.next() else {
                    event!(Level::DEBUG, "Restarting least used iter");
                    least_used = $buffer.least_used_iter().await.unwrap();
                    continue;
                };

                event!(
                    Level::DEBUG,
                    "Filtering values any value that is equal to loaded values"
                );
                if $loaded_ids.iter().any(|id| id == &next_unload.1) {
                    let unload_id = next_unload.1;
                    let loaded_ids = $loaded_ids;
                    event!(
                        Level::DEBUG,
                        "unload_id:({unload_id}) in {loaded_ids:?} - Must skip."
                    );
                    continue;
                }

                event!(Level::DEBUG, "Attempt to unload({:?})", next_unload.1);
                if let Err(err) = $buffer
                    .unload_and_push(&next_unload.1, $value.clone())
                    .await
                {
                    event!(Level::DEBUG, "Err({err:?})");
                    continue;
                };

                event!(Level::DEBUG, "Break loop");
                break 'primary_loop;
            }
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
        > + Debug,
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

    // find closet id
    let (mut closet_partition_id, closet_size) = {
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

            if closest.1 < dist {
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

    let mut closet_vector_id = {
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
                let (id, dist) = find_closet_vector(value, partitions.as_slice(), &mut dist_map);

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

    // insert into partition
    {
        let partition_buffer = &mut *w_partition_buffer;

        let partition = resolve_buffer!(ACCESS, partition_buffer, closet_partition_id);

        let Some(partition) = &mut *partition.write().await else {
            todo!()
        };

        if closet_size + 1 >= PARTITION_CAP {
            // pre_mature split operation so don't have to do it after add_into_partition fails
        }

        let result = add_into_partition(value.clone(), partition);

        'add_into_partition: {
            event!(Level::DEBUG, "Add result: {result:?}");
            if let Err(_) = result {
                event!(Level::DEBUG, "Splitting partition");

                let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

                let min_span_tree =
                    resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);
                let min_span_tree = &mut *min_span_tree.write().await;
                let Some(min_span_tree) = min_span_tree else {
                    todo!()
                };

                let Ok([new_split_2, new_split_1]) =
                    split_partition(partition, min_span_tree, inter_graph)
                else {
                    panic!()
                };
                // let Ok(mut new_splits) = split_partition(partition, min_span_tree, 2, inter_graph)
                // else {
                //     panic!()
                // };

                // event!(Level::DEBUG, "New split size: {}", new_splits.len());

                event!(Level::DEBUG, "OG graph: {min_span_tree:#?}");

                {
                    let (new_partition, new_graph) = new_split_1;

                    *partition = new_partition;

                    *min_span_tree = new_graph;
                    event!(Level::DEBUG, "partition - 1: {partition:?}");
                    event!(Level::DEBUG, "New graph - 1: {min_span_tree:#?}");
                }

                let (mut new_partition, new_graph) = new_split_2;
                event!(Level::DEBUG, "partition - 2: {new_partition:?}");
                event!(Level::DEBUG, "New graph - 2: {new_graph:#?}");
                event!(Level::DEBUG, "inter: {inter_graph:#?}");

                //add into partition or new_partition
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

                let dist_old = B::dist(&value.vector, &partition.centroid());
                let dist_new = B::dist(&value.vector, &new_partition.centroid());

                event!(Level::DEBUG, "closet_vector_id = {closet_vector_id:?}");
                event!(
                    Level::DEBUG,
                    target_partition_id = %partition.id,
                    target_partition_size = partition.size,
                    new_partition_id = %new_partition.id,
                    new_partition_size = new_partition.size,
                    "Calculated distances for new and target partitions"
                );
                event!(Level::DEBUG, "{dist_old:?} < {dist_new:?}");

                if closet_vector_id.0 == closet_partition_id
                    && new_graph.1.contains_key(&closet_vector_id.1)
                {
                    let new_id = (PartitionId(new_partition.id), closet_vector_id.1);
                    event!(
                        Level::DEBUG,
                        "closet_vector_id:{closet_vector_id:?}\tcloset_partition_id:{closet_partition_id:?}\tChanged closet id from = {closet_vector_id:?} -> {new_id:?}"
                    );
                    {
                        let a = closet_partition_id;
                        let b = PartitionId(new_partition.id);
                        let c = closet_vector_id;
                        let d = (PartitionId(new_partition.id), closet_vector_id.1);
                        event!(
                            Level::DEBUG,
                            "Updating closet_partition_id = {a:?} -> {b:?}\tcloset_vector_id={c:?} -> {d:?}"
                        );
                    }
                    closet_vector_id = new_id;
                }

                if dist_old < dist_new {
                    add_into_partition(value, partition)
                        .expect("Partition should now have space but an error occurred");
                } else {
                    add_into_partition(value, &mut new_partition)
                        .expect("Partition should now have space but an error occurred");

                    closet_partition_id = PartitionId(new_partition.id);
                }

                meta_data.insert(
                    new_partition.id,
                    Arc::new(RwLock::new(Meta::new(
                        PartitionId(new_partition.id),
                        new_partition.size,
                        new_partition.centroid(),
                    ))),
                );

                resolve_buffer!(PUSH, partition_buffer, new_partition, [partition.id]);
                resolve_buffer!(PUSH, min_span_tree_buffer, new_graph, [*min_span_tree.2]);
            };
        }

        // update original partition
        {
            let target_meta = &mut *meta_data[&partition.id].write().await;

            target_meta.size = partition.size;
            target_meta.centroid = partition.centroid();
        }
    }

    //update min_spanning
    {
        let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

        let min_span_tree = resolve_buffer!(ACCESS, min_span_tree_buffer, closet_vector_id.0);
        let Some(min_span_tree) = &mut *min_span_tree.write().await else {
            todo!();
        };

        let update_inter_graph_edges = match closet_vector_id.0 == closet_partition_id {
            true => {
                event!(Level::DEBUG, "update_local_min_span_tree");
                update_local_min_span_tree(
                    VectorId(value.id),
                    closet_vector_id.0,
                    closet_vector_id.1,
                    &inter_graph,
                    min_span_tree,
                    // filter and collection operation is unnecessary
                    dist_map
                        .iter()
                        .filter(|((partition_id, vector_id), _dist)| {
                            partition_id == &closet_vector_id.0
                        })
                        .map(|((partition_id, vector_id), dist)| (*vector_id, *dist))
                        .collect(),
                )
            }
            false => {
                event!(Level::DEBUG, "update_foreign_min_span_tree");
                {
                    let min_span_tree = resolve_buffer!(
                        ACCESS,
                        min_span_tree_buffer,
                        closet_partition_id,
                        [*closet_vector_id.0]
                    );
                    let Some(min_span_tree) = &mut *min_span_tree.write().await else {
                        todo!();
                    };

                    min_span_tree.add_node(VectorId(value.id));
                }

                update_foreign_min_span_tree(
                    closet_partition_id,
                    VectorId(value.id),
                    closet_vector_id.0,
                    closet_vector_id.1,
                    inter_graph,
                    min_span_tree,
                    // filter and collection operation is unnecessary
                    dist_map
                        .iter()
                        .filter(|((partition_id, vector_id), _dist)| {
                            partition_id == &closet_vector_id.0
                        })
                        .map(|((partition_id, vector_id), dist)| (*vector_id, *dist))
                        .collect(),
                )
            }
        }
        .unwrap();

        // update_inter_graph.iter()
        //     .filter(|(dist, _, _)| )

        // Vec<(A, (PartitionId, VectorId), (PartitionId, VectorId))>

        {
            let partition_buffer = &mut *w_partition_buffer;

            let mut edges_partitions = group_edges(update_inter_graph_edges);
            loop {
                let mut remove_key = Vec::new();
                edges_partitions.iter().for_each(|(partition, edges)| {
                    if !dist_map.contains_key(&edges[0].3) {
                        return;
                    }

                    edges
                        .iter()
                        .filter(|(_, dist, _, id)| dist < &dist_map[id])
                        .for_each(|(idx, _, _, id_2)| {
                            inter_graph.add_edge(
                                id_2.0,
                                closet_vector_id.0,
                                (dist_map[id_2], *id_2, closet_vector_id),
                            );

                            inter_graph.0.remove_edge(*idx);
                        });

                    remove_key.push(*partition);
                });

                remove_key.into_iter().for_each(|key| {
                    edges_partitions.remove_entry(&key);
                });

                if edges_partitions.len() <= 0 {
                    break;
                }

                event!(Level::DEBUG, "remaining keys: {edges_partitions:?}");

                let partition_id = edges_partitions.iter().map(|(key, _)| *key).next().unwrap();

                event!(Level::DEBUG, "Load: {partition_id:?}");

                let partition = resolve_buffer!(ACCESS, partition_buffer, partition_id);
                event!(Level::DEBUG, "Just resolved buffer");

                let Some(partition) = &*partition.read().await else {
                    todo!()
                };

                partition
                    .iter()
                    .map(|y| ((PartitionId(partition.id), VectorId(y.id)), y.vector))
                    .for_each(|(id, vec)| {
                        let dist = B::dist(&value.vector, &vec);

                        dist_map.insert(id, dist);
                    });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {}
