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

        // maybe try distance matrix to maximize distance between starting vectors
        for i1 in 0..distances.len() {
            for i2 in (i1 + 1)..distances.len() {
                if i1 == i2 {
                    panic!("End of list - attempted all possible solutions")
                }
                let mut prev_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = array::from_fn(|_| PartitionSubSet::new(&target));
                let mut new_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = [
                    {
                        let mut tmp = PartitionSubSet::new(&target);

                        tmp.add(distances[i1].0).unwrap();

                        tmp
                    },
                    {
                        let mut tmp = PartitionSubSet::new(&target);

                        tmp.add(distances[i2].0).unwrap();

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
        .enumerate()
        .map(|(idx, x)| {
            x.vectors
                .iter()
                .take(x.size)
                .map(|y| y.unwrap())
                .map(|y| VectorId(target[y].id))
                .map(|y| (y, idx))
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
        intra_graph
            .1
            .iter()
            .map(|(vec_id, _)| vec_id)
            .for_each(|vec_id| {
                let index = partition_membership[vec_id];

                intra_graphs[index].add_node(*vec_id);
            });

        let mut new_inter_edges: Vec<((usize, VectorId), (usize, VectorId), A)> = Vec::new();

        intra_graph
            .0
            .edge_indices()
            .map(|edge_idx| {
                (
                    intra_graph.0.edge_endpoints(edge_idx).unwrap(),
                    intra_graph.0.edge_weight(edge_idx).unwrap(),
                )
            })
            .map(|((source, target), weight)| {
                (
                    intra_graph.0.node_weight(source).unwrap(),
                    intra_graph.0.node_weight(target).unwrap(),
                    weight,
                )
            })
            .for_each(|(source, target, weight)| {
                if partition_membership[source] == partition_membership[target] {
                    let idx = partition_membership[source];

                    intra_graphs[idx].add_edge(*source, *target, *weight);
                } else {
                    new_inter_edges.push((
                        (partition_membership[source], *source),
                        (partition_membership[target], *target),
                        *weight,
                    ));
                };
            });

        // for partition in new_partitions.iter() {
        //     partition
        //         .vectors
        //         .iter()
        //         .take(partition.size)
        //         .map(|vector| vector.unwrap())
        //         .map(|vector_index| target[vector_index].id)
        //         .for_each(|vector_id| {
        //             let edges: Vec<_> = intra_graph
        //                 .0
        //                 .edges(*intra_graph.1.get(&VectorId(vector_id)).expect(&format!(
        //                     "Failed to extract {vector_id:?} from {intra_graph:#?}"
        //                 )))
        //                 .map(|edge| {
        //                     (
        //                         intra_graph.0.node_weight(edge.source()).unwrap(),
        //                         intra_graph.0.node_weight(edge.target()).unwrap(),
        //                         edge.weight(),
        //                     )
        //                 })
        //                 .collect();
        //             if edges.len() == 0 {
        //                 let id = VectorId(vector_id);
        //                 let idx = partition_membership[&id];
        //
        //                 if !intra_graphs[idx].1.contains_key(&id) {
        //                     intra_graphs[idx].add_node(id);
        //                 }
        //             } else {
        //                 for (id_1, id_2, dist) in edges {
        //                     if inserted_edges.contains(&(id_1, id_2)) {
        //                         continue;
        //                     }
        //
        //                     let idx_1 = partition_membership[id_1];
        //                     let idx_2 = partition_membership[id_2];
        //
        //                     if !intra_graphs[idx_1].1.contains_key(id_1) {
        //                         intra_graphs[idx_1].add_node(*id_1);
        //                     }
        //
        //                     if !intra_graphs[idx_2].1.contains_key(id_2) {
        //                         intra_graphs[idx_2].add_node(*id_2);
        //                     }
        //
        //                     if idx_1 == idx_2 {
        //                         intra_graphs[idx_1].add_edge(*id_1, *id_2, *dist);
        //                     } else {
        //                         new_inter_edges.push(((idx_1, *id_1), (idx_2, *id_2), *dist));
        //                     }
        //
        //                     inserted_edges.insert((id_1, id_2));
        //                     inserted_edges.insert((id_2, id_1));
        //                 }
        //             }
        //         });
        // }

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
    use std::{marker::PhantomData, str::FromStr};

    use uuid::Uuid;

    use super::*;
    use crate::{db::component::partition::VectorEntry, vector::Vector};

    fn split_partition_test(
        vectors: Vec<VectorEntry<f32, Vector<f32, 2>>>,
        edges: Vec<(f32, Uuid, Uuid)>,
    ) {
        // Initialize a sample partition and graphs
        let mut partition: Partition<f32, Vector<f32, 2>, 20, 10> = Partition::new();
        partition.size = vectors.len();
        vectors
            .iter()
            .enumerate()
            .for_each(|(idx, vector)| partition.vectors[idx] = Some(vector.clone()));

        let mut intra_graph = IntraPartitionGraph::<f32>::new(PartitionId(partition.id));

        vectors.iter().map(|x| VectorId(x.id)).for_each(|id| {
            intra_graph.add_node(id);
        });
        edges
            .iter()
            .map(|(weight, id_1, id_2)| (weight, VectorId(*id_1), VectorId(*id_2)))
            .for_each(|(weight, id_1, id_2)| {
                intra_graph.add_edge(id_1, id_2, *weight);
            });
        let mut inter_graph = InterPartitionGraph::<f32>::new();
        inter_graph.add_node(intra_graph.2);

        // Call the function under test
        let result = split_partition(&mut partition, &mut intra_graph, &mut inter_graph);

        // Check the result
        match result {
            Ok(splits) => {
                // 1. Ensure no duplicate vectors across partitions
                let mut all_vectors = vec![];
                for (split_partition, _) in splits.iter() {
                    all_vectors.extend(
                        split_partition
                            .vectors
                            .iter()
                            .take(split_partition.size)
                            .map(|x| x.unwrap()),
                    );
                }
                let unique_vectors: HashSet<_> = all_vectors.iter().collect();
                assert_eq!(
                    all_vectors.len(),
                    unique_vectors.len(),
                    "Duplicate vectors found across splits"
                );

                // 2. Ensure no duplicate vectors in the same split_partition
                for (split_partition, _) in splits.iter() {
                    let mut seen_vectors = HashSet::new();
                    for vector in split_partition.vectors.iter().take(split_partition.size) {
                        if let Some(vector) = vector {
                            assert!(
                                seen_vectors.insert(VectorId(vector.id)),
                                "Duplicate vector found within a single partition"
                            );
                        }
                    }
                }

                // 3. Ensure graph-partition consistency
                for (split_partition, split_graph) in splits.iter() {
                    let partition_vectors: HashSet<_> = split_partition
                        .vectors
                        .iter()
                        .take(split_partition.size)
                        .map(|x| x.unwrap())
                        .map(|x| VectorId(x.id))
                        .collect();
                    let graph_vectors: HashSet<_> =
                        split_graph.1.iter().map(|(id, _)| *id).collect();

                    assert_eq!(
                        partition_vectors, graph_vectors,
                        "Mismatch between partition and graph vectors"
                    );
                }

                // 4. Ensure no data loss: vectors in target should remain after splitting
                let target_vectors: HashSet<_> = vectors.into_iter().collect();
                let result_vectors: HashSet<_> = all_vectors.into_iter().collect();

                assert_eq!(
                    target_vectors, result_vectors,
                    "Some vectors from the target were dropped after splitting"
                );
            }
            Err(err) => {
                panic!("split_partition failed with error: {:?}", err);
            }
        }
    }

    #[test]
    fn test_split_partition_correctness() {
        let vectors: Vec<VectorEntry<f32, Vector<f32, 2>>> = vec![
            VectorEntry {
                vector: Vector([0.11623396, 0.12840234]),
                id: Uuid::from_str("3ae497e1-744e-4630-b5fb-d668e61b8a99").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.05831603, 0.12463821]),
                id: Uuid::from_str("ea413c57-d043-4979-8cdb-d042b48d32c1").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.091962166, 0.10388949]),
                id: Uuid::from_str("11ac1bdf-fe20-4a92-a163-e00ac9ad88ee").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.058563404, 0.08736718]),
                id: Uuid::from_str("8ad73200-1af9-48ad-a5ce-a25352326ac2").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.08708069, 0.121189125]),
                id: Uuid::from_str("385c832b-155c-4a12-9ad7-45721097805d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.08421735, 0.12917832]),
                id: Uuid::from_str("a31b1590-7687-400a-8a51-6cbfedab298e").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10285964, 0.09018607]),
                id: Uuid::from_str("732adbda-6fe1-431b-9ef9-a63a853463ef").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10480461, 0.10610912]),
                id: Uuid::from_str("c6a10eeb-2d58-4e7b-8b3f-262a72455d44").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.092545256, 0.05741372]),
                id: Uuid::from_str("be7cc8fc-25ce-4c30-8212-e4eb99b73ac5").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10123189, 0.09770964]),
                id: Uuid::from_str("641098dc-0b36-4fd9-b016-c7836fa22289").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.084239475, 0.052746795]),
                id: Uuid::from_str("d1249738-373e-4d71-977d-bcf7019828c6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10068026, 0.10186246]),
                id: Uuid::from_str("d108073c-8052-4e88-bca9-5704513621c0").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.099185295, 0.100453876]),
                id: Uuid::from_str("2139155d-fc09-4964-ba3d-071fd79be71c").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.12748502, 0.124801084]),
                id: Uuid::from_str("b217c232-f636-4464-baee-f6e4a980b67b").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.14168441, 0.07358162]),
                id: Uuid::from_str("fbb18999-c858-41cf-919a-fa24d4c6db2e").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.13251953, 0.09862726]),
                id: Uuid::from_str("fb5bb61d-0b49-4299-a92a-539daf754b43").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10723352, 0.08500998]),
                id: Uuid::from_str("d122fe3b-85e1-4d1a-8d3c-52cec4dd2ab0").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8811717, 0.05688962]),
                id: Uuid::from_str("34bbf9b3-597f-4a39-bfc3-0988c4e01802").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.96600795, 0.1730615]),
                id: Uuid::from_str("be48aa32-5611-4291-b768-5260bc47c4d6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.94200873, 0.12781328]),
                id: Uuid::from_str("43a57d74-e4d8-4ddf-895b-a5f9e11ff216").unwrap(),
                _phantom_data: PhantomData,
            },
        ];
        let edges = vec![
            (
                7.202588e-05,
                Uuid::from_str("a31b1590-7687-400a-8a51-6cbfedab298e").unwrap(),
                Uuid::from_str("385c832b-155c-4a12-9ad7-45721097805d").unwrap(),
            ),
            (
                9.0766174e-05,
                Uuid::from_str("d1249738-373e-4d71-977d-bcf7019828c6").unwrap(),
                Uuid::from_str("be7cc8fc-25ce-4c30-8212-e4eb99b73ac5").unwrap(),
            ),
            (
                0.00013955543,
                Uuid::from_str("b217c232-f636-4464-baee-f6e4a980b67b").unwrap(),
                Uuid::from_str("3ae497e1-744e-4630-b5fb-d668e61b8a99").unwrap(),
            ),
            (
                0.002623364,
                Uuid::from_str("43a57d74-e4d8-4ddf-895b-a5f9e11ff216").unwrap(),
                Uuid::from_str("be48aa32-5611-4291-b768-5260bc47c4d6").unwrap(),
            ),
        ];

        split_partition_test(vectors, edges);
    }
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
