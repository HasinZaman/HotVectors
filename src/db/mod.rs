use std::array;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::vector::{Vector, VectorSerial};

#[derive(Debug, Clone, Copy)]
pub struct Partition<
    A: PartialEq + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
> {
    size: usize,

    vectors: [VectorEntry<A, VECTOR_CAP>; PARTITION_CAP],
    centroid: Vector<A, VECTOR_CAP>,

    id: Uuid,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PartitionSerial<A: Clone + Copy> {
    vectors: Vec<VectorEntrySerial<A>>,
    centroid: VectorSerial<A>,

    #[serde(with = "uuid::serde::compact")]
    id: Uuid,
}

impl<A: PartialEq + Clone + Copy, const PARTITION_CAP: usize, const VECTOR_CAP: usize>
    From<Partition<A, PARTITION_CAP, VECTOR_CAP>> for PartitionSerial<A>
{
    fn from(value: Partition<A, PARTITION_CAP, VECTOR_CAP>) -> Self {
        PartitionSerial {
            vectors: value
                .vectors
                .iter()
                .map(|x| VectorEntrySerial::<A>::from(*x))
                .collect::<Vec<VectorEntrySerial<A>>>(),
            centroid: value.centroid.into(),
            id: value.id,
        }
    }
}
impl<A: PartialEq + Clone + Copy, const PARTITION_CAP: usize, const VECTOR_CAP: usize>
    From<PartitionSerial<A>> for Partition<A, PARTITION_CAP, VECTOR_CAP>
{
    fn from(value: PartitionSerial<A>) -> Self {
        let mut iter = value
            .vectors
            .iter()
            .map(|x| VectorEntry::<A, VECTOR_CAP>::from(x.clone())); //TODO!() derive from a reference

        Partition {
            size: value.vectors.len(),
            vectors: array::from_fn(|_| iter.next().unwrap()),
            centroid: value.centroid.into(),
            id: value.id,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorEntry<A: PartialEq + Clone + Copy, const CAP: usize> {
    vector: Vector<A, CAP>,
    id: Uuid,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorEntrySerial<A: Clone + Copy> {
    vector: VectorSerial<A>,
    #[serde(with = "uuid::serde::compact")]
    id: Uuid,
}

impl<A: PartialEq + Clone + Copy, const CAP: usize> From<VectorEntry<A, CAP>>
    for VectorEntrySerial<A>
{
    fn from(value: VectorEntry<A, CAP>) -> Self {
        VectorEntrySerial {
            vector: value.vector.into(),
            id: value.id,
        }
    }
}
impl<A: PartialEq + Clone + Copy, const CAP: usize> From<VectorEntrySerial<A>>
    for VectorEntry<A, CAP>
{
    fn from(value: VectorEntrySerial<A>) -> Self {
        VectorEntry {
            vector: value.vector.into(),
            id: value.id,
        }
    }
}
