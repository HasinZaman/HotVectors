use std::{array, marker::PhantomData};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::vector::{Field, VectorSerial, VectorSpace};

use super::{Partition, VectorEntry};

#[derive(Debug, Serialize, Deserialize)]
pub struct PartitionSerial<A: Clone + Copy> {
    vectors: Vec<VectorEntrySerial<A>>,
    centroid: VectorSerial<A>,

    #[serde(with = "uuid::serde::compact")]
    id: Uuid,
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Into<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<Partition<A, B, PARTITION_CAP, VECTOR_CAP>> for PartitionSerial<A>
where
    VectorEntry<A, B, VECTOR_CAP>: Into<VectorEntrySerial<A>>,
{
    fn from(value: Partition<A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
        PartitionSerial {
            vectors: value
                .vectors
                .iter()
                .filter(|x| x.is_some())
                .map(|x| x.unwrap())
                .map(|x| Into::<VectorEntrySerial<A>>::into(x))
                .collect::<Vec<VectorEntrySerial<A>>>(),
            centroid: value.centroid.into(),
            id: value.id,
        }
    }
}
impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + From<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<PartitionSerial<A>> for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
where
    VectorEntrySerial<A>: Into<VectorEntry<A, B, VECTOR_CAP>>,
{
    fn from(value: PartitionSerial<A>) -> Self {
        let mut iter = value
            .vectors
            .iter()
            .map(|x| Into::<VectorEntry<A, B, VECTOR_CAP>>::into(x.clone())); //TODO!() derive from a reference

        Partition {
            size: value.vectors.len(),
            vectors: array::from_fn(|_| iter.next()),
            centroid: value.centroid.into(),
            id: value.id,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorEntrySerial<A: Clone + Copy> {
    vector: VectorSerial<A>,
    #[serde(with = "uuid::serde::compact")]
    id: Uuid,
}

impl<
        A: Clone + Copy + Field<A>,
        B: Clone + Copy + Into<VectorSerial<A>> + VectorSpace<A>,
        const CAP: usize,
    > From<VectorEntry<A, B, CAP>> for VectorEntrySerial<A>
{
    fn from(value: VectorEntry<A, B, CAP>) -> Self {
        VectorEntrySerial {
            vector: value.vector.into(),
            id: value.id,
        }
    }
}
impl<
        A: Clone + Copy + Field<A>,
        B: PartialEq + Clone + Copy + From<VectorSerial<A>> + VectorSpace<A>,
        const CAP: usize,
    > From<VectorEntrySerial<A>> for VectorEntry<A, B, CAP>
{
    fn from(value: VectorEntrySerial<A>) -> Self {
        VectorEntry {
            vector: value.vector.into(),
            id: value.id,
            _phantom_data: PhantomData,
        }
    }
}
