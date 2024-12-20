use crate::vector::{Field, VectorSerial, VectorSpace};

use super::{
    InterPartitionGraph, IntraPartitionGraph, LoadedPartitions, Partition, PartitionErr,
    PartitionId,
};

// testable
fn split_initial<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    target: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph: &mut IntraPartitionGraph<A>,

    splits: usize,

    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<(), PartitionErr> {
    todo!()
}

pub async fn split<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    partition_id: PartitionId,
    splits: usize,

    inter_graph: &mut InterPartitionGraph<A>,
    partition_access: &mut LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>,
) -> Result<(), PartitionErr> {
    // split partition
    // propagate
    todo!()
}
