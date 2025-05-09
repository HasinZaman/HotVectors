use petgraph::visit::EdgeRef;
use std::fmt::Debug;

use crate::{
    db::component::{
        graph::{InterPartitionGraph, IntraPartitionGraph},
        ids::PartitionId,
        partition::Partition,
    },
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

#[derive(Debug)]
pub enum MergeError {
    Overflow,
    NotConnected,
}

pub fn merge_partition_into<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    (sink_partition, sink_mst): (
        &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        &IntraPartitionGraph<A>,
    ),
    (source_partition, source_mst): (
        &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        &IntraPartitionGraph<A>,
    ),
    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<
    (
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        IntraPartitionGraph<A>,
    ),
    MergeError,
> {
    if sink_partition.size + source_partition.size > PARTITION_CAP {
        return Err(MergeError::Overflow);
    }

    // check if both partitions are connected
    let (weight, (partition_id_1, vector_id_1), (partition_id_2, vector_id_2)) = {
        let Some(source_idx) = inter_graph.1.get(&PartitionId(sink_partition.id)) else {
            todo!()
        };

        let connecting_edge = inter_graph
            .0
            .edges(*source_idx)
            .filter(|edge_ref| {
                let edge_source = inter_graph.0[edge_ref.source()];
                let edge_target = inter_graph.0[edge_ref.target()];

                (sink_partition.id == edge_source.0 && source_partition.id == edge_target.0)
                    || (sink_partition.id == edge_target.0 && source_partition.id == edge_source.0)
            })
            .map(|edge_ref| edge_ref.weight())
            .next();

        match connecting_edge {
            Some(edge) => edge,
            None => return Err(MergeError::NotConnected),
        }
    };

    // create a new partition & mst
    let mut new_partition = sink_partition.clone();
    let mut new_mst = sink_mst.clone();

    // add vectors from source into sink
    new_partition.size = new_partition.size + source_partition.size;
    new_partition.vectors[sink_partition.size..new_partition.size]
        .copy_from_slice(&source_partition.vectors[..source_partition.size]);

    // add edges from source to sink
    for (vector_id, _) in &source_mst.1 {
        new_mst.add_node(*vector_id);
    }
    for edge_idx in source_mst.0.edge_indices() {
        let weight = source_mst.0[edge_idx];

        let (source, target) = source_mst.0.edge_endpoints(edge_idx).unwrap();

        let source = source_mst.0[source];
        let target = source_mst.0[target];

        new_mst.add_edge(source, target, weight);
    }

    new_mst.add_edge(*vector_id_1, *vector_id_2, *weight);

    // collapse shared edges into new partition
    let _ = inter_graph
        .remove_edge(
            (*partition_id_1, *vector_id_1),
            (*partition_id_2, *vector_id_2),
        )
        .unwrap();

    // update inter_graph edges from source into sink
    let mut delete_edges = Vec::new();
    for edge_ref in inter_graph
        .0
        .edges(inter_graph.1[&PartitionId(source_partition.id)])
    {
        let weight = edge_ref.weight();

        delete_edges.push(*weight);
    }

    for (
        weight,
        (source_partition_id, source_vector_id),
        (target_partition_id, target_vector_id),
    ) in delete_edges
    {
        let _ = inter_graph
            .remove_edge(
                (source_partition_id, source_vector_id),
                (target_partition_id, target_vector_id),
            )
            .unwrap();

        match *source_partition_id == source_partition.id {
            true => {
                inter_graph.add_edge(
                    PartitionId(sink_partition.id),
                    target_partition_id,
                    (
                        weight,
                        (PartitionId(sink_partition.id), source_vector_id),
                        (target_partition_id, target_vector_id),
                    ),
                );
            }
            false => {
                inter_graph.add_edge(
                    source_partition_id,
                    PartitionId(sink_partition.id),
                    (
                        weight,
                        (source_partition_id, source_vector_id),
                        (PartitionId(sink_partition.id), target_vector_id),
                    ),
                );
            }
        }
    }

    Ok((new_partition, new_mst))
}
