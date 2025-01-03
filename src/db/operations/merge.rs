use std::{collections::HashSet, fmt::Debug};

use petgraph::{
    graph::EdgeIndex,
    visit::{EdgeRef, IntoEdgeReferences},
};

use crate::{
    db::component::{
        graph::IntraPartitionGraph,
        ids::{PartitionId, VectorId},
        partition::{Partition, PartitionErr, VectorEntry},
    },
    vector::{Field, VectorSerial, VectorSpace},
};

use super::{InterPartitionGraph, LoadedPartitions};

// TESTABLE
pub(self) fn merge_into<
    A: PartialEq + Clone + Copy + Field<A> + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    partition_1: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph_1: &mut IntraPartitionGraph<A>,

    partition_2: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph_2: &IntraPartitionGraph<A>,

    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<(), PartitionErr> {
    if partition_1.size + partition_2.size >= PARTITION_CAP {
        return Err(PartitionErr::Overflow);
    }

    partition_2.iter().for_each(
        |VectorEntry {
             id,
             vector: _,
             _phantom_data: _,
         }| {
            let idx = intra_graph_1.0.add_node(VectorId(*id));
            intra_graph_1.1.insert(VectorId(*id), idx);
        },
    );

    // move edges from intra_graph_2 -> intra_graph_1
    {
        intra_graph_2
            .0
            .edge_references()
            .map(|edge_reference| {
                (
                    intra_graph_2
                        .0
                        .node_weight(edge_reference.source())
                        .unwrap(),
                    intra_graph_2
                        .0
                        .node_weight(edge_reference.target())
                        .unwrap(),
                    edge_reference.weight(),
                )
            })
            .for_each(|(start, end, dist)| {
                let _ = intra_graph_1.add_edge(*start, *end, *dist);
            });
    }
    // move inter_edge from (partition_1, partition_2) -> intra_graph_1
    'move_common_inter_edge: {
        let del_index: Vec<EdgeIndex> = inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(partition_2.id)])
            .filter(|edge| {
                (**inter_graph.0.node_weight(edge.source()).unwrap() == partition_1.id
                    || **inter_graph.0.node_weight(edge.source()).unwrap() == partition_2.id)
                    && (**inter_graph.0.node_weight(edge.target()).unwrap() == partition_1.id
                        || **inter_graph.0.node_weight(edge.target()).unwrap() == partition_2.id)
            })
            .map(|edge_ref| {
                let (dist, start, end) = edge_ref.weight();

                intra_graph_1.add_edge(start.1, end.1, *dist);

                edge_ref.id()
            })
            .collect();

        if del_index.len() == 0 {
            break 'move_common_inter_edge;
        }

        del_index.into_iter().for_each(|index| {
            inter_graph.0.remove_edge(index);
        });
    }

    // update inter_edge from (partition_2, n)
    {
        let move_index: Vec<EdgeIndex> = inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(partition_2.id)])
            .filter(|edge| {
                !((**inter_graph.0.node_weight(edge.source()).unwrap() == partition_1.id
                    || **inter_graph.0.node_weight(edge.source()).unwrap() == partition_2.id)
                    && (**inter_graph.0.node_weight(edge.target()).unwrap() == partition_1.id
                        || **inter_graph.0.node_weight(edge.target()).unwrap() == partition_2.id))
            })
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
            inter_graph.add_edge(id_1.0, id_2.0, (dist, id_1, id_2));
        });

        move_index.into_iter().for_each(|index| {
            inter_graph.0.remove_edge(index);
        });

        inter_graph
            .0
            .remove_node(inter_graph.1[&PartitionId(partition_2.id)]);
        inter_graph.1.remove(&PartitionId(partition_2.id));
    };

    //move vectors from partition_2 into partition_1
    partition_1.vectors[partition_1.size..partition_1.size + partition_2.size]
        .swap_with_slice(&mut partition_2.vectors[..partition_2.size]);
    partition_1.size = partition_1.size + partition_2.size;

    // update centeroid
    partition_1.centroid = B::add(&partition_1.centroid, &partition_2.centroid);

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
    // let partition_1 = match partition_access.access(&partition_1).await {
    //     Ok(partition) => partition,
    //     Err(_) => todo!(),
    // };

    // let mut partition_1 = partition_1.write().unwrap();

    // let Some(partition_1) = partition_1.as_mut() else {
    //     todo!()
    // };

    // let (mut partition_2, inter_graph_2) = match partition_access.remove(&partition_2).await {
    //     Ok(partition) => partition,
    //     Err(_) => todo!(),
    // };
    todo!()
    // merge_into(
    //     &mut partition_1.0,
    //     &mut partition_1.1,
    //     &mut partition_2,
    //     &inter_graph_2,
    //     inter_graph,
    // )
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use petgraph::visit::{EdgeRef, IntoEdgeReferences};
    use rkyv::vec;
    use uuid::Uuid;

    use crate::{
        db::{
            component::{
                graph::IntraPartitionGraph,
                ids::{PartitionId, VectorId},
                partition::{Partition, VectorEntry},
            },
            operations::{add::add_into, merge::merge_into, InterPartitionGraph},
        },
        vector::{Field, Vector, VectorSerial, VectorSpace},
    };

    fn check_partition_neighbors<
        A: PartialEq + Clone + Copy + Field<A> + Debug,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Debug,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    >(
        index: usize,
        intra_graph: &IntraPartitionGraph<A>,
        partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        neighbors: &[Uuid],
    ) {
        let result = intra_graph
            .0
            .edges(intra_graph.1[&VectorId(partition.vectors[index].unwrap().id)])
            .count();
        assert_eq!(result, neighbors.len());

        let result = intra_graph
            .0
            .edges(intra_graph.1[&VectorId(partition.vectors[index].unwrap().id)])
            .map(|edge| {
                match **intra_graph.0.node_weight(edge.source()).unwrap()
                    == partition.vectors[index].unwrap().id
                {
                    true => intra_graph.0.node_weight(edge.target()).unwrap(),
                    false => intra_graph.0.node_weight(edge.source()).unwrap(),
                }
            })
            .map(|id| **id)
            .all(|neighbor_vertex| neighbors.contains(&neighbor_vertex));
        assert!(result);
    }

    // merge two partitions
    #[test]
    fn basic_merge() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new(PartitionId(partition_1.id));
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new(PartitionId(partition_2.id));
        inter_graph.add_node(PartitionId(partition_2.id));

        assert_eq!(inter_graph.1.len(), 2);

        // initialize partitions
        let search_vectors = {
            let mut search_vectors = Vec::new();
            // Insert into partition
            let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());
            let result = add_into(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4());
            let result = add_into(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());

            let vector = VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4());
            search_vectors.push(vector.clone());
            let result = add_into(
                &mut partition_2,
                vector.clone(),
                &mut intra_graph_2,
                &mut inter_graph,
                &mut [(&partition_1, &mut intra_graph_1)],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4());
            search_vectors.push(vector.clone());
            let result = add_into(
                &mut partition_2,
                vector.clone(),
                &mut intra_graph_2,
                &mut inter_graph,
                &mut [(&partition_1, &mut intra_graph_1)],
            );
            assert!(result.is_ok());

            search_vectors
        };

        let result = merge_into(
            &mut partition_1,
            &mut intra_graph_1,
            &mut partition_2,
            &intra_graph_2,
            &mut inter_graph,
        );
        assert!(result.is_ok());

        // check centeroid
        assert_eq!(partition_1.centroid(), Vector::splat(0.));

        // check size of partition
        assert_eq!(partition_1.size, 4);

        // check both vectors are in partition_1
        assert!(search_vectors.iter().all(|search_vector| {
            partition_1.iter().any(|vector| {
                vector.id == search_vector.id && vector.vector == search_vector.vector
            })
        }));

        // check inter_graph
        {
            assert_eq!(inter_graph.1.len(), 1);

            assert_eq!(
                inter_graph
                    .0
                    .edges(inter_graph.1[&PartitionId(partition_1.id)])
                    .count(),
                0
            );
            assert_eq!(inter_graph.1.get(&PartitionId(partition_2.id)), None);
        }

        // check intra_graph
        {
            // get edges for
            check_partition_neighbors(
                0,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[1].unwrap().id,
                    partition_1.vectors[2].unwrap().id,
                ],
            );
            check_partition_neighbors(
                1,
                &intra_graph_1,
                &partition_1,
                &[partition_1.vectors[0].unwrap().id],
            );
            check_partition_neighbors(
                2,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[0].unwrap().id,
                    partition_1.vectors[3].unwrap().id,
                ],
            );
            check_partition_neighbors(
                3,
                &intra_graph_1,
                &partition_1,
                &[partition_1.vectors[2].unwrap().id],
            );
        }
    }

    #[test]
    fn merge_with_neighbors() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new(PartitionId(partition_1.id));
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new(PartitionId(partition_2.id));
        inter_graph.add_node(PartitionId(partition_2.id));

        let mut partition_3 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_3 = IntraPartitionGraph::new(PartitionId(partition_3.id));
        inter_graph.add_node(PartitionId(partition_3.id));

        let mut partition_4 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_4 = IntraPartitionGraph::new(PartitionId(partition_4.id));
        inter_graph.add_node(PartitionId(partition_4.id));

        assert_eq!(inter_graph.1.len(), 4);

        // initialize partitions
        let search_vectors = {
            let mut search_vectors = Vec::new();
            // Insert into partition
            {
                let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());
                let result = add_into(
                    &mut partition_1,
                    vector.clone(),
                    &mut intra_graph_1,
                    &mut inter_graph,
                    &mut [],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4());
                let result = add_into(
                    &mut partition_1,
                    vector.clone(),
                    &mut intra_graph_1,
                    &mut inter_graph,
                    &mut [],
                );
                assert!(result.is_ok());
            }

            {
                let vector = VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4());
                let result = add_into(
                    &mut partition_2,
                    vector.clone(),
                    &mut intra_graph_2,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
                search_vectors.push(vector);
                let vector = VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4());
                let result = add_into(
                    &mut partition_2,
                    vector.clone(),
                    &mut intra_graph_2,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
                search_vectors.push(vector);
            }

            {
                let vector = VectorEntry::from_uuid(Vector([-1., 0.]), Uuid::new_v4());
                let result = add_into(
                    &mut partition_3,
                    vector.clone(),
                    &mut intra_graph_3,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector([-2., 0.]), Uuid::new_v4());
                let result = add_into(
                    &mut partition_3,
                    vector.clone(),
                    &mut intra_graph_3,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
            }

            {
                let vector = VectorEntry::from_uuid(Vector([1., 0.]), Uuid::new_v4());
                let result = add_into(
                    &mut partition_4,
                    vector.clone(),
                    &mut intra_graph_4,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector([2., 0.]), Uuid::new_v4());
                let result = add_into(
                    &mut partition_4,
                    vector.clone(),
                    &mut intra_graph_4,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
            }

            search_vectors
        };

        let result = merge_into(
            &mut partition_1,
            &mut intra_graph_1,
            &mut partition_2,
            &intra_graph_2,
            &mut inter_graph,
        );
        assert!(result.is_ok());

        // check centeroid
        assert_eq!(partition_1.centroid(), Vector::splat(0.));

        // check size of partition
        assert_eq!(partition_1.size, 4);

        // check both vectors are in partition_1
        assert!(search_vectors.iter().all(|search_vector| {
            partition_1.iter().any(|vector| {
                vector.id == search_vector.id && vector.vector == search_vector.vector
            })
        }));

        // check inter_graph
        {
            assert_eq!(inter_graph.1.len(), 3);

            assert_eq!(
                inter_graph
                    .0
                    .edges(inter_graph.1[&PartitionId(partition_1.id)])
                    .count(),
                2
            );

            assert!(
                inter_graph
                    .0
                    .edges(inter_graph.1[&PartitionId(partition_1.id)])
                    .map(|edge| {
                        match **inter_graph.0.node_weight(edge.source()).unwrap() == partition_1.id
                        {
                            true => inter_graph.0.node_weight(edge.target()).unwrap(),
                            false => inter_graph.0.node_weight(edge.source()).unwrap(),
                        }
                    })
                    .map(|id| **id)
                    .all(|neighbor_vertex| [partition_3.id, partition_4.id]
                        .contains(&neighbor_vertex))
            );

            assert_eq!(inter_graph.1.get(&PartitionId(partition_2.id)), None);
        }

        // check intra_graph
        {
            // get edges for
            check_partition_neighbors(
                0,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[1].unwrap().id,
                    partition_1.vectors[2].unwrap().id,
                ],
            );
            check_partition_neighbors(
                1,
                &intra_graph_1,
                &partition_1,
                &[partition_1.vectors[0].unwrap().id],
            );
            check_partition_neighbors(
                2,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[0].unwrap().id,
                    partition_1.vectors[3].unwrap().id,
                ],
            );
            check_partition_neighbors(
                3,
                &intra_graph_1,
                &partition_1,
                &[partition_1.vectors[2].unwrap().id],
            );
        }
    }

    #[test]
    fn merge_overflow() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 3, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new(PartitionId(partition_1.id));
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 3, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new(PartitionId(partition_2.id));
        inter_graph.add_node(PartitionId(partition_2.id));
        // initialize partitions
        {
            // Insert into partition
            let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());
            let result = add_into(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4());
            let result = add_into(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());

            let vector = VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4());
            let result = add_into(
                &mut partition_2,
                vector.clone(),
                &mut intra_graph_2,
                &mut inter_graph,
                &mut [(&partition_1, &mut intra_graph_1)],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4());
            let result = add_into(
                &mut partition_2,
                vector.clone(),
                &mut intra_graph_2,
                &mut inter_graph,
                &mut [(&partition_1, &mut intra_graph_1)],
            );
            assert!(result.is_ok());
        }

        let result = merge_into(
            &mut partition_1,
            &mut intra_graph_1,
            &mut partition_2,
            &intra_graph_2,
            &mut inter_graph,
        );
        assert!(result.is_err());
    }
}
