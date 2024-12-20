use std::collections::HashSet;

use petgraph::{graph::EdgeIndex, visit::EdgeRef};
use tokio::try_join;

use crate::vector::{Field, VectorSerial, VectorSpace};

use super::{
    InterPartitionGraph, IntraPartitionGraph, LoadedPartitions, Partition, PartitionErr,
    PartitionId, VectorId,
};

// TESTABLE
fn merge_into<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    partition_1: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph_1: &mut IntraPartitionGraph<A>,

    mut partition_2: Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph_2: IntraPartitionGraph<A>,

    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<(), PartitionErr> {
    if partition_1.size + partition_2.size >= PARTITION_CAP {
        return Err(PartitionErr::Overflow);
    }

    //move vectors from partition_2 into partition_1
    partition_1.vectors[partition_1.size..]
        .swap_with_slice(&mut partition_2.vectors[..partition_2.size]);
    partition_1.size = partition_1.size + partition_2.size;

    intra_graph_2.1.keys().for_each(|id| {
        let idx = intra_graph_1.0.add_node(*id);
        intra_graph_1.1.insert(*id, idx);
    });

    // move edges from intra_graph_2 -> intra_graph_1
    {
        intra_graph_2
            .0
            .edge_indices()
            .map(|edge_index| {
                (
                    intra_graph_2
                        .0
                        .edge_endpoints(edge_index)
                        .map(|(start, end)| {
                            (
                                intra_graph_2.0.node_weight(start).unwrap(),
                                intra_graph_2.0.node_weight(end).unwrap(),
                            )
                        })
                        .unwrap(),
                    intra_graph_2.0.edge_weight(edge_index).unwrap(),
                )
            })
            .for_each(|((start, end), dist)| {
                let _ =
                    intra_graph_1
                        .0
                        .add_edge(intra_graph_1.1[start], intra_graph_1.1[end], *dist);
            });
    }
    // move intra_edge from (partition_1, partition_2) -> intra_graph_1
    {
        let del_index: Vec<EdgeIndex> = inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(partition_2.id)])
            .chain(
                inter_graph
                    .0
                    .edges(inter_graph.1[&PartitionId(partition_1.id)]),
            )
            .filter(|edge| {
                (**inter_graph.0.node_weight(edge.source()).unwrap() == partition_1.id
                    || **inter_graph.0.node_weight(edge.source()).unwrap() == partition_2.id)
                    && (**inter_graph.0.node_weight(edge.target()).unwrap() == partition_1.id
                        || **inter_graph.0.node_weight(edge.target()).unwrap() == partition_2.id)
            })
            .map(|edge_ref| {
                let (dist, start, end) = edge_ref.weight();

                intra_graph_1
                    .0
                    .add_edge(intra_graph_1.1[&start.1], intra_graph_1.1[&end.1], *dist);

                edge_ref.id()
            })
            .collect();

        del_index.into_iter().for_each(|index| {
            inter_graph.0.remove_edge(index);
        });
    }

    // update inter_edge from (partition_2, n)
    {
        let move_index: Vec<EdgeIndex> = inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(partition_2.id)])
            .map(|edge_ref| edge_ref.id())
            .collect();

        let new_edges: Vec<(A, (PartitionId, VectorId), (PartitionId, VectorId))> = move_index
            .iter()
            .map(|edge_id| inter_graph.0.edge_weight(*edge_id).unwrap())
            .map(|(dist, (p_id_1, v_id_1), (p_id_2, v_id_2))| {
                (
                    *dist,
                    {
                        match **p_id_1 == partition_2.id {
                            true => (PartitionId(partition_1.id), *v_id_1),
                            false => (*p_id_1, *v_id_1),
                        }
                    },
                    {
                        match **p_id_2 == partition_2.id {
                            true => (PartitionId(partition_1.id), *v_id_2),
                            false => (*p_id_2, *v_id_2),
                        }
                    },
                )
            })
            .collect::<Vec<(A, (PartitionId, VectorId), (PartitionId, VectorId))>>();

        new_edges.into_iter().for_each(|(dist, id_1, id_2)| {
            inter_graph.0.add_edge(
                inter_graph.1[&id_1.0],
                inter_graph.1[&id_2.0],
                (dist, id_1, id_2),
            );
        });

        move_index.into_iter().for_each(|index| {
            inter_graph.0.remove_edge(index);
        });
    };

    Ok(())
}

// ACTUALLY use
pub async fn merge<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    partition_1: PartitionId,
    partition_2: PartitionId,
    inter_graph: &mut InterPartitionGraph<A>,

    partition_access: &mut LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>,
) -> Result<(), PartitionErr> {
    let partition_1 = match partition_access.access(&partition_1).await {
        Ok(partition) => partition,
        Err(_) => todo!(),
    };

    let mut partition_1 = partition_1.write().unwrap();

    let Some(partition_1) = partition_1.as_mut() else {
        todo!()
    };

    let (partition_2, inter_graph_2) = match partition_access.remove(&partition_2).await {
        Ok(partition) => partition,
        Err(_) => todo!(),
    };

    merge_into(
        &mut partition_1.0,
        &mut partition_1.1,
        partition_2,
        inter_graph_2,
        inter_graph,
    )
}
