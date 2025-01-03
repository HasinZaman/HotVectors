use std::{
    cmp::{min, Reverse},
    collections::HashMap,
    sync::{Arc, RwLock},
};

use heapify::{make_heap, pop_heap};
use uuid::Uuid;

use crate::{
    db::component::{
        data_buffer::DataBuffer,
        graph::{GraphSerial, InterPartitionGraph, IntraPartitionGraph},
        meta::Meta,
        partition::{
            ArchivedVectorEntrySerial, Partition, PartitionSerial, VectorEntry, VectorEntrySerial,
        },
    },
    vector::{Field, VectorSerial, VectorSpace},
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

pub async fn exact_knn<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    vector: B,

    neighbors: usize,

    partition_buffer: Arc<
        RwLock<
            DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionSerial<A>, MAX_LOADED>,
        >,
    >,
    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
) -> Vec<VectorEntry<A, B>>
where
    A: PartialOrd + Ord,
    VectorSerial<A>: From<B>,
    A: for<'a> rkyv::Serialize<
        rancor::Strategy<
            rkyv::ser::Serializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::ser::sharing::Share,
            >,
            rancor::Error,
        >,
    >,
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<A>]:
        DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
{
    // find nearest partition

    // do knn (or take all)

    // if not enough propagate outwards to neighbors

    // find all neighbors
    // find do knn
    // return all vectors
    todo!()
}

pub async fn approximate_knn<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    vector: B,

    neighbors: usize,

    min_spanning_tree_buffer: Arc<
        RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, MAX_LOADED>>,
    >,
    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
) -> Vec<VectorEntry<A, B>>
where
    A: PartialOrd + Ord,
    VectorSerial<A>: From<B>,
    A: for<'a> rkyv::Serialize<
        rancor::Strategy<
            rkyv::ser::Serializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::ser::sharing::Share,
            >,
            rancor::Error,
        >,
    >,
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<A>]:
        DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
{
    // find nearest partition

    // do knn (or take all)

    // if not enough propagate outwards to neighbors

    // find all neighbors
    // find do knn
    // return all vectors
    todo!()
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
