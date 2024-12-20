use std::collections::HashSet;

use petgraph::{graph::EdgeIndex, visit::EdgeRef};
use tokio::try_join;

use crate::vector::{Field, VectorSerial, VectorSpace};

use super::{
    InterPartitionGraph, IntraPartitionGraph, LoadedPartitions, Partition, PartitionErr,
    PartitionId, VectorId,
};

// TESTABLE
pub(self) fn merge_into<
    A: PartialEq + Clone + Copy + Field<A>,
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

    //move vectors from partition_2 into partition_1
    partition_1.vectors[partition_1.size..]
        .swap_with_slice(&mut partition_2.vectors[..partition_2.size]);
    partition_1.size = partition_1.size + partition_2.size;

    intra_graph_2.1.keys().for_each(|id| {
        let idx = intra_graph_1.0.add_node(*id);
        intra_graph_1.1.insert(*id, idx);
    });

    // update centeroid
    partition_1.centroid = B::add(&partition_1.centroid, &partition_2.centroid);

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

        inter_graph
            .0
            .remove_node(inter_graph.1[&PartitionId(partition_2.id)]);
        inter_graph.1.remove(&PartitionId(partition_2.id));
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

    let (mut partition_2, inter_graph_2) = match partition_access.remove(&partition_2).await {
        Ok(partition) => partition,
        Err(_) => todo!(),
    };

    merge_into(
        &mut partition_1.0,
        &mut partition_1.1,
        &mut partition_2,
        &inter_graph_2,
        inter_graph,
    )
}

#[cfg(test)]
mod test {
    use petgraph::visit::EdgeRef;
    use rkyv::vec;
    use uuid::Uuid;

    use crate::{
        db::partition::{
            add::add,
            merge::{self, merge_into},
            InterPartitionGraph, IntraPartitionGraph, Partition, PartitionId, VectorEntry,
            VectorId,
        },
        vector::{Field, Vector, VectorSerial, VectorSpace},
    };

    
    fn check_partition_neighbors<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    >
    (
        index: usize,
        intra_graph: &IntraPartitionGraph<A>,
        partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        neighbors: &[Uuid],
    ) {
        assert_eq!(
            intra_graph
                .0
                .edges(intra_graph.1[&VectorId(partition.vectors[index].unwrap().id)])
                .count(),
            neighbors.len()
        );
    
        assert!(
            intra_graph
                .0
                .edges(intra_graph.1[&VectorId(partition.vectors[index].unwrap().id)])
                .map(|edge| {
                    match **intra_graph.0.node_weight(edge.source()).unwrap()
                        == partition.vectors[0].unwrap().id
                    {
                        true => intra_graph.0.node_weight(edge.target()).unwrap(),
                        false => intra_graph.0.node_weight(edge.source()).unwrap(),
                    }
                })
                .map(|id| **id)
                .all(|neighbor_vertex| neighbors.contains(&neighbor_vertex))
        );
    }

    // merge two partitions
    #[test]
    fn basic_merge() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_1.id));
        inter_graph.1.insert(PartitionId(partition_1.id), index);

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_2.id));
        inter_graph.1.insert(PartitionId(partition_2.id), index);

        assert_eq!(inter_graph.1.len(), 2);

        // initialize partitions
        {
            // Insert into partition
            let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());
            let result = add(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4());
            let result = add(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());

            let vector = VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4());
            let result = add(
                &mut partition_2,
                vector.clone(),
                &mut intra_graph_2,
                &mut inter_graph,
                &mut [(&partition_1, &mut intra_graph_1)],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4());
            let result = add(
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
        assert!(result.is_ok());

        // check centeroid
        assert_eq!(partition_1.centroid(), Vector::splat(0.));

        // check size of partition
        assert_eq!(partition_1.size, 4);

        // check both vectors are in partition_1
        assert!(partition_2
            .vectors
            .iter()
            .take(partition_2.size)
            .map(|x| x.unwrap())
            .all(|search_vector| {
                partition_1
                    .vectors
                    .iter()
                    .take(partition_1.size)
                    .map(|x| x.unwrap())
                    .any(|vector| {
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
                ]
            );
            check_partition_neighbors(
                1,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[0].unwrap().id,
                ]
            );
            check_partition_neighbors(
                2,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[1].unwrap().id,
                    partition_1.vectors[2].unwrap().id,
                ]
            );
            check_partition_neighbors(
                3,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[2].unwrap().id,
                ]
            );
        }
    }


    #[test]
    fn merge_with_neighbors() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_1.id));
        inter_graph.1.insert(PartitionId(partition_1.id), index);

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_2.id));
        inter_graph.1.insert(PartitionId(partition_2.id), index);

        let mut partition_3 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_3 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_3.id));
        inter_graph.1.insert(PartitionId(partition_3.id), index);

        let mut partition_4 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_4 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_4.id));
        inter_graph.1.insert(PartitionId(partition_3.id), index);

        assert_eq!(inter_graph.1.len(), 4);

        // initialize partitions
        {
            // Insert into partition
            {
                let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());
                let result = add(
                    &mut partition_1,
                    vector.clone(),
                    &mut intra_graph_1,
                    &mut inter_graph,
                    &mut [],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4());
                let result = add(
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
                let result = add(
                    &mut partition_2,
                    vector.clone(),
                    &mut intra_graph_2,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4());
                let result = add(
                    &mut partition_2,
                    vector.clone(),
                    &mut intra_graph_2,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
            }

            {
                let vector = VectorEntry::from_uuid(Vector([-1., 0.]), Uuid::new_v4());
                let result = add(
                    &mut partition_3,
                    vector.clone(),
                    &mut intra_graph_3,
                    &mut inter_graph,
                    &mut [(&partition_2, &mut intra_graph_2)],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector([-2., 0.]), Uuid::new_v4());
                let result = add(
                    &mut partition_3,
                    vector.clone(),
                    &mut intra_graph_3,
                    &mut inter_graph,
                    &mut [(&partition_2, &mut intra_graph_2)],
                );
                assert!(result.is_ok());
            }
            
            {
                let vector = VectorEntry::from_uuid(Vector([1., 0.]), Uuid::new_v4());
                let result = add(
                    &mut partition_4,
                    vector.clone(),
                    &mut intra_graph_4,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
                let vector = VectorEntry::from_uuid(Vector([2., 0.]), Uuid::new_v4());
                let result = add(
                    &mut partition_3,
                    vector.clone(),
                    &mut intra_graph_3,
                    &mut inter_graph,
                    &mut [(&partition_1, &mut intra_graph_1)],
                );
                assert!(result.is_ok());
            }
        }

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
        assert!(partition_2
            .vectors
            .iter()
            .take(partition_2.size)
            .map(|x| x.unwrap())
            .all(|search_vector| {
                partition_1
                    .vectors
                    .iter()
                    .take(partition_1.size)
                    .map(|x| x.unwrap())
                    .any(|vector| {
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
                inter_graph.0
                    .edges(inter_graph.1[&PartitionId(partition_1.id)])
                    .map(|edge| {
                        match **inter_graph.0.node_weight(edge.source()).unwrap()
                            == partition_1.id
                        {
                            true => inter_graph.0.node_weight(edge.target()).unwrap(),
                            false => inter_graph.0.node_weight(edge.source()).unwrap(),
                        }
                    })
                    .map(|id| **id)
                    .all(|neighbor_vertex| [partition_3.id, partition_4.id].contains(&neighbor_vertex))
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
                ]
            );
            check_partition_neighbors(
                1,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[0].unwrap().id,
                ]
            );
            check_partition_neighbors(
                2,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[1].unwrap().id,
                    partition_1.vectors[2].unwrap().id,
                ]
            );
            check_partition_neighbors(
                3,
                &intra_graph_1,
                &partition_1,
                &[
                    partition_1.vectors[2].unwrap().id,
                ]
            );
        }
    }

    #[test]
    fn merge_overflow() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 3, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_1.id));
        inter_graph.1.insert(PartitionId(partition_1.id), index);

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 3, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new();
        let index = inter_graph.0.add_node(PartitionId(partition_2.id));
        inter_graph.1.insert(PartitionId(partition_2.id), index);
        // initialize partitions
        {
            // Insert into partition
            let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());
            let result = add(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4());
            let result = add(
                &mut partition_1,
                vector.clone(),
                &mut intra_graph_1,
                &mut inter_graph,
                &mut [],
            );
            assert!(result.is_ok());

            let vector = VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4());
            let result = add(
                &mut partition_2,
                vector.clone(),
                &mut intra_graph_2,
                &mut inter_graph,
                &mut [(&partition_1, &mut intra_graph_1)],
            );
            assert!(result.is_ok());
            let vector = VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4());
            let result = add(
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
