use std::{
    array,
    cmp::{Ordering, Reverse},
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem,
    ops::{Index, IndexMut},
};

use heapify::{make_heap, pop_heap};
use petgraph::visit::EdgeRef;
use uuid::Uuid;

use crate::{
    db::component::{
        graph::{InterPartitionGraph, IntraPartitionGraph},
        ids::{PartitionId, VectorId},
        partition::{Partition, PartitionErr},
    },
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

struct PartitionSubSet<
    'a,
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
> {
    pub id: Uuid,
    source: &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>,

    pub size: usize,

    pub vectors: [Option<usize>; PARTITION_CAP],
    pub centroid: B,
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>>
    for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    fn from(value: PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
        todo!()
    }
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>
{
    fn new(value: &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
        Self {
            id: Uuid::new_v4(),
            source: value,
            size: 0,
            vectors: [None; PARTITION_CAP],
            centroid: B::additive_identity(),
        }
    }
    pub fn add(&mut self, ref_index: usize) -> Result<(), PartitionErr> {
        if self.source.size <= ref_index {
            return Err(PartitionErr::VectorNotFound);
        };

        if PARTITION_CAP <= self.size {
            return Err(PartitionErr::Overflow);
        };

        self.vectors[self.size] = Some(ref_index);
        self.size += 1;

        self.centroid = B::add(&self.centroid, &self.source[ref_index].vector);

        Ok(())
    }

    pub fn centroid(&self) -> B {
        B::scalar_mult(
            &self.centroid,
            &A::div(&A::multiplicative_identity(), &A::from_usize(self.size)),
        )
    }
    pub fn clear(&mut self) {
        self.size = 0;
        self.centroid = B::additive_identity();
    }
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Index<usize> for PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>
{
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.vectors[index] {
            Some(val) => val,
            None => todo!(),
        }
    }
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > IndexMut<usize> for PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.vectors[index] {
            Some(val) => val,
            None => todo!(),
        }
    }
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<&PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>> for HashSet<VectorId>
{
    fn from(value: &PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
        value
            .vectors
            .iter()
            .take(value.size)
            .map(|x| x.unwrap())
            .map(|index| VectorId(value.source[index].id))
            .collect::<HashSet<VectorId>>()
    }
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<&PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>>
    for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    fn from(value: &PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
        Partition {
            size: value.size,
            vectors: {
                let mut tmp = value.vectors.iter().map(|x| x.map(|x| value.source[x]));

                array::from_fn(move |_| tmp.next().unwrap())
            },
            centroid: value.centroid,
            id: value.id,
        }
    }
}

#[derive(Debug, Clone)]
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

const SPLITS: usize = 2;
// testable
pub fn split_partition<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    target: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph: &mut IntraPartitionGraph<A>,

    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<
    [(
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        IntraPartitionGraph<A>,
    ); SPLITS],
    PartitionErr,
> {
    if target.id != *intra_graph.2 {
        // Should be Err
        panic!("Ids don't match");
    }
    if target.size < SPLITS {
        return Err(PartitionErr::InsufficientSizeForSplits);
    }

    if target.size == SPLITS {
        // simple hard coded solution
        // each vector in target becomes it's own partition
        todo!()
    }

    let new_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>; SPLITS] = 'new_partition_block: {
        let centroid = target.centroid();
        let mut distances = target
            .iter()
            .map(|vector| B::dist(&centroid, &vector.vector))
            .enumerate()
            .map(|(index, dist)| KeyValuePair(index, Reverse(dist)))
            .collect::<Vec<KeyValuePair<usize, A>>>();

        if distances.is_empty() {
            todo!()
        }

        distances.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(Ordering::Equal));

        for i1 in 0..distances.len() {
            for i2 in i1..distances.len() {
                let mut prev_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = array::from_fn(|_| PartitionSubSet::new(&target));
                let mut new_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = [
                    {
                        let mut tmp = PartitionSubSet::new(&target);

                        tmp.add(distances[i1].0);

                        tmp
                    },
                    {
                        let mut tmp = PartitionSubSet::new(&target);

                        tmp.add(distances[i2].0);

                        tmp
                    },
                ];

                // find closet_partition
                while prev_partitions
                    .iter()
                    .zip(new_partitions.iter())
                    .all(|(x, y)| x.vectors != y.vectors)
                {
                    mem::swap(&mut prev_partitions, &mut new_partitions);

                    new_partitions.iter_mut().for_each(|x| x.clear());

                    // CPU solution
                    target
                        .iter()
                        .map(|vector| vector.vector)
                        .map(|vector| {
                            let (index, _) = prev_partitions
                                .iter()
                                .map(|new_partition| B::dist(&new_partition.centroid(), &vector))
                                .enumerate()
                                .min_by(|(_, x1), (_, x2)| {
                                    x1.partial_cmp(x2).unwrap_or(Ordering::Less)
                                })
                                .unwrap();

                            index
                        })
                        .enumerate()
                        .for_each(|(vector_index, partition_index)| {
                            new_partitions[partition_index].add(vector_index).unwrap();
                        });
                }

                if !new_partitions.iter().any(|s| s.size == target.size) {
                    new_partitions[0].id = target.id.clone();

                    break 'new_partition_block new_partitions;
                };
            }
        }

        todo!();
    };
    // take a node from graph & dfs to make all graphs

    let partition_membership: HashMap<VectorId, usize> = new_partitions
        .iter()
        .map(|x| HashSet::<VectorId>::from(x))
        .enumerate()
        .map(|(index, ids)| {
            ids.iter()
                .map(|id| (*id, index))
                .collect::<Vec<(VectorId, usize)>>()
        })
        .flatten()
        .collect();
    let mut intra_graphs = {
        let mut intra_graphs: [IntraPartitionGraph<A>; 2] = array::from_fn(|i1| {
            let x = new_partitions.get(i1).unwrap();
            IntraPartitionGraph::new(PartitionId(x.id))
        });
        // (0..(splits)).map(|_| IntraPartitionGraph::new()).collect();

        let mut new_inter_edges: Vec<((usize, VectorId), (usize, VectorId), A)> = Vec::new();

        let mut inserted_edges = HashSet::new();

        for partition in new_partitions.iter() {
            partition
                .vectors
                .iter()
                .take(partition.size)
                .map(|vector| vector.unwrap())
                .map(|vector_index| target[vector_index].id)
                .for_each(|vector_id| {
                    let edges: Vec<_> = intra_graph
                        .0
                        .edges(*intra_graph.1.get(&VectorId(vector_id)).expect(&format!(
                            "Failed to extract {vector_id:?} from {intra_graph:#?}"
                        )))
                        .map(|edge| {
                            (
                                intra_graph.0.node_weight(edge.source()).unwrap(),
                                intra_graph.0.node_weight(edge.target()).unwrap(),
                                edge.weight(),
                            )
                        })
                        .collect();
                    if edges.len() == 0 {
                        let id = VectorId(vector_id);
                        let idx = partition_membership[&id];

                        if !intra_graphs[idx].1.contains_key(&id) {
                            intra_graphs[idx].add_node(id);
                        }
                    } else {
                        for (id_1, id_2, dist) in edges {
                            if inserted_edges.contains(&(id_1, id_2)) {
                                continue;
                            }

                            let idx_1 = partition_membership[id_1];
                            let idx_2 = partition_membership[id_2];

                            if !intra_graphs[idx_1].1.contains_key(id_1) {
                                intra_graphs[idx_1].add_node(*id_1);
                            }

                            if !intra_graphs[idx_2].1.contains_key(id_2) {
                                intra_graphs[idx_2].add_node(*id_2);
                            }

                            if idx_1 == idx_2 {
                                intra_graphs[idx_1].add_edge(*id_1, *id_2, *dist);
                            } else {
                                new_inter_edges.push(((idx_1, *id_1), (idx_2, *id_2), *dist));
                            }

                            inserted_edges.insert((id_1, id_2));
                            inserted_edges.insert((id_2, id_1));
                        }
                    }
                });
        }

        intra_graphs.iter().skip(1).for_each(|graph| {
            inter_graph.add_node(graph.2);
        });

        let update_inter_edges: Vec<_> = inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(target.id)])
            .map(|edge| (edge.id(), edge.weight()))
            .filter_map(|(idx, (dist, id_1, id_2))| {
                match (
                    (
                        id_1.0 == intra_graphs[0].2,
                        partition_membership.get(&id_1.1),
                    ),
                    (
                        id_2.0 == intra_graphs[0].2,
                        partition_membership.get(&id_2.1),
                    ),
                ) {
                    ((true, Some(0) | None), (false, _)) => None,
                    ((true, Some(n)), (false, _)) => {
                        Some((idx, (*dist, (intra_graphs[*n].2, id_1.1), *id_2)))
                    }

                    ((false, _), (true, Some(0) | None)) => None,
                    ((false, _), (true, Some(n))) => {
                        Some((idx, (*dist, *id_1, (intra_graphs[*n].2, id_2.1))))
                    }

                    _ => todo!(),
                }
            })
            .collect();

        for (idx, (dist, id_1, id_2)) in update_inter_edges {
            inter_graph.0.remove_edge(idx);
            inter_graph.add_edge(id_1.0, id_2.0, (dist, id_1, id_2));
        }

        new_inter_edges
            .into_iter()
            .map(
                |((partition_index_1, vector_id_1), (partition_index_2, vector_id_2), dist)| {
                    (
                        (
                            PartitionId(new_partitions[partition_index_1].id),
                            vector_id_1,
                        ),
                        (
                            PartitionId(new_partitions[partition_index_2].id),
                            vector_id_2,
                        ),
                        dist,
                    )
                },
            )
            .chain({
                // update inter_edges that uses to connect to target partition

                // need to split so we can first borrow read and then mut borrow for deletion
                let mut delete_edges = Vec::new();
                let update_edges: Vec<((PartitionId, VectorId), (PartitionId, VectorId), A)> =
                    inter_graph
                        .0
                        .edges(inter_graph.1[&PartitionId(new_partitions[0].id)])
                        .map(|edge| (edge.id(), edge.weight()))
                        .filter_map(|(id, (dist, (partition_1, vec_1), (partition_2, vec_2)))| {
                            // foreign vec
                            match (
                                **partition_1 == target.id,
                                (partition_1, vec_1),
                                (partition_2, vec_2),
                            ) {
                                (true, (_, target_vec), other_vec)
                                | (false, other_vec, (_, target_vec)) => {
                                    if intra_graphs[0].1.contains_key(target_vec) {
                                        Some((
                                            id,
                                            dist,
                                            (
                                                PartitionId(
                                                    new_partitions
                                                        [partition_membership[target_vec]]
                                                        .id,
                                                ),
                                                *target_vec,
                                            ),
                                            (*other_vec.0, *other_vec.1),
                                        ))
                                    } else {
                                        None
                                    }
                                }
                            }
                        })
                        .map(|(delete_index, dist, source, target)| {
                            // add to delete_edge list
                            delete_edges.push(delete_index);
                            // add to update edge
                            (source, target, *dist)
                        })
                        .collect();

                delete_edges.into_iter().for_each(|edge_index| {
                    inter_graph.0.remove_edge(edge_index);
                });

                update_edges.into_iter()
            })
            .for_each(|(id_1, id_2, dist)| {
                inter_graph.add_edge(id_1.0, id_2.0, (dist, id_1, id_2));
            });

        intra_graphs
    };

    let new_partitions: Vec<Partition<A, B, PARTITION_CAP, VECTOR_CAP>> = {
        let mut tmp: Vec<Partition<A, B, PARTITION_CAP, VECTOR_CAP>> =
            new_partitions.iter().map(|x| Partition::from(x)).collect();

        tmp[0].id = target.id.clone();
        intra_graphs
            .iter_mut()
            .enumerate()
            .for_each(|(i1, graph)| graph.2 = PartitionId(tmp[i1].id.clone()));

        tmp
    };
    // Note: new_partitions[0].id = target.id -> therefore should replace target after split_target call
    let mut result: [_; SPLITS] =
        array::from_fn(|i1| (new_partitions[i1], intra_graphs[i1].clone())); // should be moved
                                                                             //  new_partitions
                                                                             //     .into_iter()
                                                                             //     .zip(intra_graphs.into_iter())
                                                                             //     .collect();

    let i1 = result.len() - 1;
    result.swap(0, i1);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::{db::component::partition::VectorEntry, vector::Vector};

    // #[test]
    // fn basic_split() {
    //     // Setup mock data
    //     type TestField = f32; // Example field type
    //     type TestVector = Vector<f32, 2>; // Example vector type

    //     let mut inter_graph = InterPartitionGraph::new();

    //     let mut partition = Partition::<TestField, TestVector, 500, 500>::new();
    //     let mut intra_graph = IntraPartitionGraph::new(PartitionId(partition.id));
    //     inter_graph.add_node(PartitionId(partition.id));

    //     assert_eq!(inter_graph.1.len(), 1);

    //     let splits = 2; // Example split count

    //     // initialize partitions
    //     let expected_partitions = vec![
    //         vec![
    //             VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4()),
    //             VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4()),
    //         ],
    //         vec![
    //             VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4()),
    //             VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4()),
    //         ],
    //     ];
    //     let expected_centroids = vec![Vector::splat(1.5), Vector::splat(-1.5)];

    //     // expected_partitions
    //     //     .iter()
    //     //     .map(|x| x.iter())
    //     //     .flatten()
    //     //     .for_each(|vector| {
    //     //         let result = add_into(
    //     //             &mut partition,
    //     //             vector.clone(),
    //     //             &mut intra_graph,
    //     //             &mut inter_graph,
    //     //             &mut [],
    //     //         );
    //     //         assert!(result.is_ok());
    //     //     });

    //     // Call split_partition
    //     let result = split_partition(&mut partition, &mut intra_graph, splits, &mut inter_graph);
    //     // Validate results
    //     assert!(result.is_ok());
    //     let new_partitions = result.unwrap();
    //     assert_eq!(new_partitions.len(), splits);

    //     new_partitions.iter().for_each(|(x, _)| {
    //         println!("{:?}", x);
    //     });

    //     let result = new_partitions
    //         .iter()
    //         .all(|(actual_partition, _actual_graph)| {
    //             expected_partitions.iter().any(|expected_partition| {
    //                 partition_check(&expected_partition, actual_partition)
    //             }) && expected_centroids
    //                 .iter()
    //                 .any(|expected_centroid| centroid_check(expected_centroid, actual_partition))
    //         });
    //     assert!(result);

    //     let result = new_partitions
    //         .iter()
    //         .all(|(actual_partition, actual_graph)| {
    //             actual_partition
    //                 .iter()
    //                 .all(|vec| actual_graph.1.contains_key(&VectorId(vec.id)))
    //         });
    //     assert!(result);

    //     let result = new_partitions
    //         .iter()
    //         .all(|(actual_partition, actual_graph)| actual_partition.id == *actual_graph.2);
    //     assert!(result);

    //     let result = new_partitions
    //         .iter()
    //         .all(|(actual_partition, _actual_graph)| {
    //             expected_partitions.iter().any(|expected_partition| {
    //                 partition_check(&expected_partition, actual_partition)
    //             }) && expected_centroids
    //                 .iter()
    //                 .any(|expected_centroid| centroid_check(expected_centroid, actual_partition))
    //         });
    //     assert!(result);

    //     assert_eq!(inter_graph.1.len(), 2);
    //     assert_eq!(
    //         inter_graph
    //             .0
    //             .edges(inter_graph.1[&PartitionId(partition.id)])
    //             .count(),
    //         1
    //     );
    //     // inter_graph.0.edges(inter_graph.1[&PartitionId(partition.id)])
    //     //     .map(|x|x.source())
    //     assert_eq!(
    //         inter_graph
    //             .0
    //             .edges(inter_graph.1[&PartitionId(new_partitions[0].0.id)])
    //             .count(),
    //         1
    //     );
    //     // assert_eq!(inter_graph.1.len(), 2);
    // }

    // fn partition_check<
    //     A: Field<A> + PartialEq + Copy,
    //     B: VectorSpace<A> + Copy + PartialEq,
    //     const PARTITION_CAP: usize,
    //     const VECTOR_CAP: usize,
    // >(
    //     search_vectors: &[VectorEntry<A, B>],
    //     partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    // ) -> bool {
    //     search_vectors.iter().all(|search_vector| {
    //         partition.iter().any(|vector| {
    //             vector.id == search_vector.id && vector.vector == search_vector.vector
    //         })
    //     })
    // }
    // fn centroid_check<
    //     A: Field<A> + PartialEq + Copy,
    //     B: VectorSpace<A> + Copy + PartialEq,
    //     const PARTITION_CAP: usize,
    //     const VECTOR_CAP: usize,
    // >(
    //     centroid: &B,
    //     partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    // ) -> bool {
    //     &partition.centroid() == centroid
    // }

    // #[test]
    // fn invalid_split() {
    //     // Setup mock data
    //     type TestField = f32; // Example field type
    //     type TestVector = Vector<f32, 2>; // Example vector type

    //     let mut inter_graph = InterPartitionGraph::new();

    //     let mut partition = Partition::<TestField, TestVector, 500, 500>::new();
    //     let mut intra_graph = IntraPartitionGraph::new(PartitionId(partition.id));
    //     inter_graph.add_node(PartitionId(partition.id));

    //     assert_eq!(inter_graph.1.len(), 1);

    //     let splits = 5; // Example split count

    //     // initialize partitions
    //     let expected_partitions = vec![
    //         vec![
    //             VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4()),
    //             VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4()),
    //         ],
    //         vec![
    //             VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4()),
    //             VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4()),
    //         ],
    //     ];

    //     // expected_partitions
    //     //     .iter()
    //     //     .map(|x| x.iter())
    //     //     .flatten()
    //     //     .for_each(|vector| {
    //     //         let result = add_into(
    //     //             &mut partition,
    //     //             vector.clone(),
    //     //             &mut intra_graph,
    //     //             &mut inter_graph,
    //     //             &mut [],
    //     //         );
    //     //         assert!(result.is_ok());
    //     //     });

    //     // Call split_partition
    //     let result = split_partition(&mut partition, &mut intra_graph, splits, &mut inter_graph);
    //     // Validate results
    //     assert!(result.is_err());
    // }
}
