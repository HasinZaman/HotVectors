use std::{
    cmp::{min, Reverse},
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    sync::Arc,
};

use heapify::{make_heap, make_heap_with, pop_heap, pop_heap_with};
use tokio::sync::{mpsc::Sender, RwLock};
use tracing::{event, Level};
use uuid::Uuid;

use crate::{
    db::{component::{
        data_buffer::{BufferError, DataBuffer},
        graph::{GraphSerial, InterPartitionGraph, IntraPartitionGraph},
        ids::{PartitionId, VectorId},
        meta::Meta,
        partition::{
            ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionSerial, VectorEntry,
            VectorEntrySerial,
        },
    }, Response, Success},
    vector::{Field, Vector, VectorSerial, VectorSpace},
};

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

struct KeyValuePair<A, B: PartialOrd + PartialEq>(pub(crate) A, pub(crate) Reverse<B>);

impl<A, B: PartialOrd + PartialEq> PartialOrd for KeyValuePair<A, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
}
impl<A, B: PartialOrd + PartialEq> PartialEq for KeyValuePair<A, B> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

pub fn knn<'a, A, B: PartialOrd + PartialEq>(
    data: &[&'a [A]],
    metric_func: fn(&A) -> B,
    neighbors: usize,
) -> Vec<&'a A> {
    let mut data = data
        .iter()
        .map(|x| x.iter())
        .flatten()
        .map(|x| x)
        .map(|x| KeyValuePair(x, Reverse(metric_func(x))))
        .collect::<Vec<KeyValuePair<&A, B>>>();

    if data.len() == 0 {
        return vec![];
    }

    make_heap(&mut data);

    (0..min(neighbors, data.len()))
        .map(|_| {
            pop_heap(&mut data);
            data.pop().unwrap()
        })
        .map(|KeyValuePair(key, _val)| key)
        .collect::<Vec<&'a A>>()
}

pub async fn approximate_knn<
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
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    vector: Vector<A, VECTOR_CAP>,
    k: usize,

    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,

    partition_buffer: Arc<
        RwLock<
            DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionSerial<A>, MAX_LOADED>,
        >,
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
    let r_meta_data_lock = meta_data.read().await;
    // find nearest partition

    // do knn (or take all)

    // if not enough propagate outwards to neighbors

    // find all neighbors
    // find do knn
    // return all vectors
    todo!()
}

pub async fn stream_exact_knn<
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
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    vector: B,
    k: usize,

    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,

    partition_buffer: Arc<
        RwLock<
            DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionSerial<A>, MAX_LOADED>,
        >,
    >,

    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,

    sender: &Sender<Response<A>>,
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
    event!(Level::INFO, "🔍 Starting exact_knn execution");

    let r_meta_data_lock = meta_data.read().await;
    let r_inter_graph = inter_graph.read().await;

    event!(Level::INFO, "🔓 Acquired read locks for metadata and inter_graph");
    // find nearest partition

    event!(Level::INFO, "✨ Finding closest partition");
    let ((closet_partition, _), total_size) = {
        let meta_data = &*r_meta_data_lock;

        event!(Level::INFO, "✨ Finding closest partition");
        let mut meta_data = meta_data.iter();

        let mut total_size = 0;
        let mut closet_partition = {
            let Some((_, data)) = meta_data.next() else {
                event!(Level::ERROR, "🚫 No metadata available");
                todo!()
            };

            let Meta {
                id, centroid, size, ..
            } = &*data.read().await;

            total_size += size;

            (*id, B::dist(centroid, &vector))
        };

        for (_, data) in meta_data {
            let Meta {
                id, centroid, size, ..
            } = &*data.read().await;

            let new_dist = B::dist(centroid, &vector);

            total_size += size;

            if new_dist < closet_partition.1 {
                closet_partition = (*id, new_dist);
            };
        }

        (closet_partition, total_size)
    };

    event!(Level::INFO, "📍 Closest partition found: {:?}, Total size: {}", closet_partition, total_size);

    if total_size <= k {
        // just get all vectors (don't need to check inter_graphs)
    }

    let mut core_partitions: HashSet<PartitionId> = HashSet::new();

    let mut core_vector_size = 0;

    let mut cloud_vectors: Vec<(PartitionId, VectorId, Reverse<A>)> = Vec::with_capacity(k);

    let mut search_queue = vec![*closet_partition];

    let mut checked_partitions: HashSet<PartitionId> = HashSet::new();

    while (search_queue.len() > 0 || cloud_vectors.len() > 0) && core_vector_size <= k{
        event!(Level::INFO, "🔄 Search queue length: {}, Core vector count: {}", search_queue.len(), core_vector_size);
        
        loop {
            let mut remove_index = vec![];

            for (i1, partition_id) in search_queue.clone().iter().enumerate() {
                let partition = {
                    let w_partition_buffer = &mut *partition_buffer.write().await;

                    let rwl_partition = match w_partition_buffer.try_access(&*partition_id) {
                        Ok(val) => val,
                        Err(_) => continue,
                    };

                    let r_partition = rwl_partition.read().await;

                    let Some(partition) = &*r_partition else {

                        event!(Level::ERROR, "🚫 Partition not found");
                        todo!()
                    };

                    partition.clone()
                };
                remove_index.push(i1);

                checked_partitions.insert(PartitionId(*partition_id));

                partition
                    .iter()
                    .map(|p_vec| {
                        (
                            PartitionId(partition.id),
                            VectorId(p_vec.id),
                            Reverse(B::dist(&p_vec.vector, &vector)),
                        )
                    })
                    .for_each(|x| {
                        cloud_vectors.push(x);
                    });
            }

            remove_index.sort();
            remove_index.reverse();
            remove_index.into_iter().for_each(|idx| {
                search_queue.remove(idx);
            });

            {
                let mut replace_count = 0;
                let partition_buffer = &mut *partition_buffer.write().await;

                let mut least_used = partition_buffer.least_used_iter().await;

                for i1 in 0..min(MAX_LOADED, search_queue.len()) {
                    let id = search_queue[i1];

                    match (partition_buffer.load(&id).await, replace_count < 1) {
                        (Ok(_), _) => {
                            event!(Level::DEBUG, "📦 Load buffer space");
                            // partitions.push(partition_buffer.access(&*id).await.unwrap());
                        }
                        (Err(BufferError::OutOfSpace), true) => {
                            event!(Level::DEBUG, "📦 Unload and Load buffer space");

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
                            // partitions.push(partition_buffer.access(&*id).await.unwrap());
                        }
                        (Err(BufferError::OutOfSpace), false) => {
                            break;
                        }
                        (Err(BufferError::FileNotFound), _) => {
                            todo!()
                        }
                        (Err(_), _) => todo!(),
                    };
                }
            }

            if search_queue.len() == 0 {
                break;
            }
        }

        
        make_heap_with(&mut cloud_vectors, |(_, _, dist_1), (_, _, dist_2)| {
            dist_1.partial_cmp(dist_2)
        });

        pop_heap_with(&mut cloud_vectors, |(_, _, dist_1), (_, _, dist_2)| {
            dist_1.partial_cmp(dist_2)
        });
        // push values into core_vectors
        while let Some((partition_id, vector_id, Reverse(dist))) = cloud_vectors.pop() {
            match (core_partitions.contains(&partition_id), k <= core_vector_size) {
                (_, true) => {
                    return Ok(())
                }
                (true, false) => {
                    event!(Level::INFO, "Add Vector: {:?}", (vector_id, dist));
                    let _ = sender.send(Response::Success(Success::Knn(vector_id, dist))).await;
                    core_vector_size+=1;
                    // core_vectors.push((vector_id, dist));
                }
                (false, false) => {
                    event!(Level::INFO, "Add Vector: {:?}", (vector_id, dist));
        
                    let _ = sender.send(Response::Success(Success::Knn(vector_id, dist))).await;
                    core_vector_size+=1;

                    core_partitions.insert(partition_id);

                    let mut neighbor_ids: Vec<Uuid> = {
                        let inter_graph = &*r_inter_graph;

                        inter_graph
                            .0
                            .edges(inter_graph.1[&partition_id])
                            .map(|edge| edge.weight())
                            .map(|edge| [edge.1 .0, edge.2 .0])
                            .flatten()
                            .filter(|id| !checked_partitions.contains(id))
                            .collect::<HashSet<PartitionId>>()
                            .drain()
                            .map(|x| *x)
                            .collect()
                    };

                    search_queue.append(&mut neighbor_ids);

                    break;
                }
            }

            pop_heap_with(&mut cloud_vectors, |(_, _, dist_1), (_, _, dist_2)| {
                dist_1.partial_cmp(dist_2)
            });
        }
    }
    event!(Level::INFO, "Core vector count: {}", core_vector_size);
        

    return Ok(());
}

fn update_vectors<
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
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    k: &usize,
    vector: &B,
    vectors: &mut Vec<(VectorId, A)>,
    checked_partitions: &mut HashSet<PartitionId>,
    vector_dist: &mut HashMap<(PartitionId, VectorId), A>,
    partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
) {
    let dist_map: HashMap<(PartitionId, VectorId), A> =
        match checked_partitions.contains(&PartitionId(partition.id)) {
            false => {
                let tmp: HashMap<_, _> = partition
                    .vectors
                    .iter()
                    .take(partition.size)
                    .map(|p_vec| p_vec.unwrap())
                    .map(|p_vec| {
                        (
                            (PartitionId(partition.id), VectorId(p_vec.id)),
                            B::dist(&p_vec.vector, vector),
                        )
                    })
                    .collect();

                vector_dist.extend(tmp.iter());

                tmp
            }
            true => partition
                .vectors
                .iter()
                .take(partition.size)
                .map(|p_vec| p_vec.unwrap())
                .map(|p_vec| {
                    (
                        (PartitionId(partition.id), VectorId(p_vec.id)),
                        vector_dist[&(PartitionId(partition.id), VectorId(p_vec.id))],
                    )
                })
                .collect(),
        };

    checked_partitions.insert(PartitionId(partition.id));

    dist_map.into_iter().for_each(|((_, vec_id), dist)| {
        let pos = vectors
            .binary_search_by(|(_id, vec_dist)| {
                vec_dist
                    .partial_cmp(&dist)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|x| x);

        if k <= &pos {
            return;
        }

        if &vectors.len() >= k {
            vectors.pop();
        }
        vectors.insert(pos, (vec_id, dist));
    });
}

pub fn euclidean_dist<A: PartialOrd + PartialEq + Field<A>, B: VectorSpace<A>>(
    vector_1: &B,
    vector_2: &B,
) -> A {
    let delta = B::sub(vector_1, vector_2);

    B::dot(&delta, &delta)
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{euclidean_dist, knn},
        vector::Vector,
    };

    #[test]
    fn test_knn_single_neighbor() {
        let data = vec![
            vec![Vector([1.0, 2.0]), Vector([2.0, 3.0])],
            vec![Vector([3.0, 4.0]), Vector([5.0, 6.0])],
        ];

        let result = knn(
            vec![data[0].as_slice(), data[1].as_slice()].as_slice(),
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            1,
        );

        // Closest to origin is Vector([1.0, 2.0])
        assert_eq!(*result[0], Vector([1.0, 2.0]));
    }

    #[test]
    fn test_knn_multiple_neighbors() {
        let data = vec![
            Vector([1.0, 2.0]),
            Vector([2.0, 3.0]),
            Vector([3.0, 4.0]),
            Vector([5.0, 6.0]),
        ];

        let result = knn(
            &[data.as_slice()],
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            3,
        );

        // Closest vectors to origin
        let expected = vec![Vector([1.0, 2.0]), Vector([2.0, 3.0]), Vector([3.0, 4.0])];

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_eq!(**res, *exp);
        }
    }

    #[test]
    fn test_knn_with_ties() {
        let data = vec![Vector([1.0, 1.0]), Vector([1.0, 1.0])];

        let result = knn(
            &[data.as_slice()],
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            2,
        );

        // Both vectors are identical, so any order is acceptable
        assert_eq!(*result[0], Vector([1.0, 1.0]));
        assert_eq!(*result[1], Vector([1.0, 1.0]));
    }

    #[test]
    fn test_knn_no_data() {
        let data = vec![];

        let result = knn(
            &[data.as_slice()],
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            3,
        );

        // No neighbors should be returned
        assert!(result.is_empty());
    }
}
