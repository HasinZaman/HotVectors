use std::collections::HashMap;

use petgraph::{adj::NodeIndex, csr::DefaultIx, data::DataMap, graph::EdgeIndex, visit::EdgeRef};
use uuid::Uuid;

use crate::{
    db::partition,
    vector::{Field, VectorSerial, VectorSpace},
};

use super::{
    InterPartitionGraph, IntraPartitionGraph, Partition, PartitionErr, PartitionId, VectorEntry,
    VectorId,
};

enum EdgeType<A: Field<A>> {
    Internal(A, A, EdgeIndex<DefaultIx>, VectorId, VectorId),
    NeighborInternal(A, A, EdgeIndex<DefaultIx>, VectorId, VectorId, PartitionId),
    InterEdge(A, A, EdgeIndex<DefaultIx>, VectorId, VectorId, PartitionId),
}

fn get_other<'a, A: Eq>(select: &A, val_1: &'a A, val_2: &'a A) -> &'a A {
    if select == val_1 {
        return val_2;
    }
    val_1
}

pub fn add<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    target: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    value: VectorEntry<A, B>,
    intra_graph: &mut IntraPartitionGraph<A>,

    inter_graph: &mut InterPartitionGraph<A>,

    neighbors: &mut [(
        &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        &mut IntraPartitionGraph<A>,
    )],
) -> Result<(), PartitionErr>
where
    A: PartialOrd + Ord,
{
    if target.size + 1 >= PARTITION_CAP {
        return Err(PartitionErr::Overflow);
    };

    let id_to_graph = neighbors
        .iter()
        .enumerate()
        .map(|(index, (partition, _))| (partition.id, index))
        .collect::<HashMap<Uuid, usize>>();

    target.vectors[target.size] = Some(value);
    target.size += 1;

    target.centroid = B::add(&target.centroid, &value.vector);

    // insert node into a minimum spanning tree
    {
        //need to get distance of all neighbor vectors
        let idx = intra_graph.0.add_node(VectorId(value.id));
        // maybe do this last
        intra_graph.1.insert(VectorId(value.id), idx);

        let partition_splits = {
            let mut tmp = vec![target.size];

            neighbors
                .iter()
                .map(|x| &x.0.size)
                .for_each(|x| tmp.push(*x));

            tmp
        };

        let dist = {
            if cfg!(feature = "gpu_processing") {
                todo!()
            } else {
                target
                    .vectors
                    .iter()
                    .filter_map(|x| *x)
                    .map(|x| (target.id, x))
                    .chain(
                        neighbors
                            .iter()
                            .map(|x| {
                                x.0.vectors
                                    .iter()
                                    .filter_map(|x| *x)
                                    .map(|x| (target.id, x))
                            })
                            .flatten(),
                    )
                    .enumerate()
                    .map(|(index, (partition_id, x))| {
                        (
                            (PartitionId(partition_id), VectorId(x.id)),
                            (index, B::dist(&x.vector, &value.vector)),
                        )
                    })
                    .collect::<HashMap<(PartitionId, VectorId), (usize, A)>>()
            }
        };

        // find closet inserted vector
        let (closet_id, index, closet_dist) = {
            let Some((id, (index, _dist))) = dist
                .iter()
                // .enumerate()
                .min_by(|(_, (_, x1)), (_, (_, x2))| x1.cmp(x2))
            else {
                todo!()
            };

            (*id, *index, *_dist)
        };

        // find edges that need to be replaced
        let replace_edges = match index < partition_splits[0] {
            true => {
                // add to internal nodes search
                let search_edges = intra_graph
                    .0
                    .edges(intra_graph.1[&closet_id.1])
                    .map(|edge| {
                        let other = get_other(
                            &closet_id.1,
                            intra_graph.0.node_weight(edge.source()).unwrap(),
                            intra_graph.0.node_weight(edge.target()).unwrap(),
                        );

                        EdgeType::Internal(
                            *edge.weight(),
                            dist[&(PartitionId(target.id), *other)].1,
                            edge.id(),
                            closet_id.1,
                            *other,
                        )
                    });

                // check if inter-graph connection
                let inter_edge = inter_graph
                    .0
                    .edges(inter_graph.1[&PartitionId(target.id)])
                    .filter(|edge| edge.weight().1 == closet_id.1 || edge.weight().2 == closet_id.1)
                    .map(|edge| {
                        let edge_id = edge.id();
                        let (edge_dist, vertex_1, vertex_2) = edge.weight();

                        let other_partition = get_other(
                            &target.id,
                            inter_graph.0.node_weight(edge.source()).unwrap(),
                            inter_graph.0.node_weight(edge.target()).unwrap(),
                        );

                        let other_vector = get_other(&closet_id.1, vertex_1, vertex_2);

                        EdgeType::InterEdge(
                            *edge_dist,
                            dist[&(PartitionId(*other_partition), *other_vector)].1,
                            edge_id,
                            closet_id.1,
                            *other_vector,
                            PartitionId(*other_partition),
                        )
                    });

                search_edges
                    .chain(inter_edge)
                    .filter(|edge| match edge {
                        EdgeType::Internal(old, new, ..)
                        | EdgeType::NeighborInternal(old, new, ..)
                        | EdgeType::InterEdge(old, new, ..) => new < old,
                    })
                    .collect::<Vec<EdgeType<A>>>()
            }
            false => {
                let (target, intra_graph) = partition_splits
                    .iter()
                    .enumerate()
                    .filter(|(_partition_index, start_index)| index < **start_index)
                    .map(|(partition_index, _start_index)| &neighbors[partition_index - 1])
                    .next()
                    .or_else(|| Some(&neighbors[partition_splits.len() - 1]))
                    .unwrap();

                let search_edges = intra_graph
                    .0
                    .edges(intra_graph.1[&closet_id.1])
                    .map(|edge| {
                        let other = get_other(
                            &closet_id.1,
                            intra_graph.0.node_weight(edge.source()).unwrap(),
                            intra_graph.0.node_weight(edge.target()).unwrap(),
                        );

                        EdgeType::NeighborInternal(
                            *edge.weight(),
                            dist[&(PartitionId(target.id), *other)].1,
                            edge.id(),
                            closet_id.1,
                            *other,
                            PartitionId(target.id),
                        )
                    });

                // check if inter-graph connection
                let inter_edge = inter_graph
                    .0
                    .edges(inter_graph.1[&PartitionId(target.id)])
                    .filter(|edge| edge.weight().1 == closet_id.1 || edge.weight().2 == closet_id.1)
                    .map(|edge| {
                        let edge_id = edge.id();
                        let (edge_dist, vertex_1, vertex_2) = edge.weight();

                        let other_partition = get_other(
                            &target.id,
                            inter_graph.0.node_weight(edge.source()).unwrap(),
                            inter_graph.0.node_weight(edge.target()).unwrap(),
                        );

                        let other_vector = get_other(&closet_id.1, vertex_1, vertex_2);

                        EdgeType::InterEdge(
                            *edge_dist,
                            dist[&(PartitionId(*other_partition), *other_vector)].1,
                            edge_id,
                            closet_id.1,
                            *other_vector,
                            PartitionId(*other_partition),
                        )
                    });

                search_edges
                    .chain(inter_edge)
                    .filter(|edge| match edge {
                        EdgeType::Internal(old, new, ..)
                        | EdgeType::NeighborInternal(old, new, ..)
                        | EdgeType::InterEdge(old, new, ..) => new < old,
                    })
                    .collect::<Vec<EdgeType<A>>>()
            }
        };

        replace_edges.iter().for_each(|edge| match edge {
            EdgeType::Internal(_, dist, edge_index, _, vector_id) => {
                intra_graph.0.add_edge(intra_graph.1[vector_id], idx, *dist);
                intra_graph.0.remove_edge(*edge_index);
                // todo!()
            }
            EdgeType::NeighborInternal(_, dist, edge_index, _, vector_id, partition_id) => {
                let (_, intra_graph) = &mut neighbors[id_to_graph[&**partition_id]];
                intra_graph.0.remove_edge(*edge_index);

                inter_graph.0.add_edge(
                    inter_graph.1[&PartitionId(target.id)],
                    inter_graph.1[partition_id],
                    (*dist, *vector_id, VectorId(value.id)),
                );
            }
            EdgeType::InterEdge(_, dist, edge_index, _, vector_id, partition_id) => {
                inter_graph.0.add_edge(
                    inter_graph.1[&PartitionId(target.id)],
                    inter_graph.1[partition_id],
                    (*dist, *vector_id, VectorId(value.id)),
                );
                inter_graph.0.remove_edge(*edge_index);
            }
        });

        match index < partition_splits[0] {
            true => {
                // new intra_edge
                intra_graph.1.insert(closet_id.1, idx);
                intra_graph
                    .0
                    .add_edge(intra_graph.1[&closet_id.1], idx, closet_dist);
            }
            false => {
                inter_graph.0.add_edge(
                    inter_graph.1[&closet_id.0],
                    inter_graph.1[&PartitionId(target.id)],
                    (closet_dist, closet_id.1, VectorId(value.id)),
                );
            }
        };
    }

    Ok(())
}

pub async fn add_async<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    target: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    value: VectorEntry<A, B>,
    intra_graph: &mut IntraPartitionGraph<A>,

    inter_graph: &mut InterPartitionGraph<A>,

    neighbors: &[(
        &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        &mut IntraPartitionGraph<A>,
    )],
) -> Result<(), PartitionErr>
where
    A: PartialOrd + Ord,
{
    todo!()
}

#[cfg(test)]
mod test {
    // test all cases
    // best Case:
    //  - insert into partition
    //  - Check vector in partition
    //  - Check if MST is correct structure (add one edge)

    // second base case (new vector insert between two other vectors & replaces edge):
    //  - Insert into partition
    //  - Check vector in partition
    //  - Check if MST is correct structure (add 2 edges and removes old MST edge)

    // foreign case 1:
    //  - nearest vector is in target partition
    //  - but must check neighbor

    // foreign case 2:
    //  - nearest vector is in target partition
    //  - but must check neighbor
    //  - Must replace inter_partition graph
    //  - Must replace intra_partition graph

    // foreign case 3:
    //  - nearest vector is in neighbor partition
    //  - Must replace inter_partition graph (new edge)
}
