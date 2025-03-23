use std::{
    array,
    cmp::{Ordering, Reverse},
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem,
    ops::{Index, IndexMut},
};

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

pub struct PartitionSubSet<
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
    fn from(_value: PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
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
pub trait SplitStrategy<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>
{
    fn split<'a>(
        partition: &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        graph: &'a IntraPartitionGraph<A>,
    ) -> [PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>; SPLITS];
}

pub struct KMean;

impl<
        A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > SplitStrategy<A, B, PARTITION_CAP, VECTOR_CAP> for KMean
{
    fn split<'a>(
        target: &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        _graph: &'a IntraPartitionGraph<A>,
    ) -> [PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>; SPLITS] {
        let centroid = target.centroid();
        let mut distances_1 = target
            .iter()
            .map(|vector| B::dist(&centroid, &vector.vector))
            .enumerate()
            .map(|(index, dist)| KeyValuePair(index, Reverse(dist)))
            .collect::<Vec<KeyValuePair<usize, A>>>();

        if distances_1.is_empty() {
            todo!()
        }

        distances_1.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Ordering::Equal));

        // maybe try distance matrix to maximize distance between starting vectors
        for i1 in 0..distances_1.len() {
            let ref_vec_1 = distances_1[i1].0;

            let mut distances_2 = {
                let vec_1 = target[ref_vec_1];
                target
                    .iter()
                    .skip(i1)
                    .map(|vector| B::dist(&vec_1.vector, &vector.vector))
                    .enumerate()
                    .map(|(index, dist)| KeyValuePair(index, Reverse(dist)))
                    .collect::<Vec<KeyValuePair<usize, A>>>()
            };
            distances_2.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Ordering::Equal));

            if distances_2.is_empty() {
                todo!()
            }

            for i2 in 0..distances_2.len() {
                let ref_vec_2 = distances_2[i2].0;
                // if i1 == i2 {
                //     panic!("End of list - attempted all possible solutions")
                // }
                let mut prev_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = array::from_fn(|_| PartitionSubSet::new(&target));
                let mut new_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = [
                    {
                        let mut tmp = PartitionSubSet::new(&target);

                        tmp.add(ref_vec_1).unwrap();

                        tmp
                    },
                    {
                        let mut tmp = PartitionSubSet::new(&target);

                        tmp.add(ref_vec_2).unwrap();

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
                                    x1.partial_cmp(x2).unwrap_or(Ordering::Equal)
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

                    return new_partitions;
                };
            }
        }

        todo!();
    }
}

pub struct BFS;

impl<
        A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > SplitStrategy<A, B, PARTITION_CAP, VECTOR_CAP> for BFS
{
    fn split<'a>(
        target: &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        graph: &'a IntraPartitionGraph<A>,
    ) -> [PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>; SPLITS] {
        let vec_to_idx: HashMap<VectorId, usize> = target
            .iter()
            .enumerate()
            .map(|(idx, id)| (VectorId(id.id), idx))
            .collect();

        let centroid = target.centroid();
        let mut distances_1 = target
            .iter()
            .map(|vector| B::dist(&centroid, &vector.vector))
            .enumerate()
            .map(|(index, dist)| KeyValuePair(index, Reverse(dist)))
            .collect::<Vec<KeyValuePair<usize, A>>>();

        if distances_1.is_empty() {
            todo!()
        }

        distances_1.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Ordering::Equal));

        // maybe try distance matrix to maximize distance between starting vectors
        for i1 in 0..distances_1.len() {
            let ref_vec_1 = distances_1[i1].0;

            let mut distances_2 = {
                let vec_1 = target[ref_vec_1];
                target
                    .iter()
                    .skip(i1)
                    .map(|vector| B::dist(&vec_1.vector, &vector.vector))
                    .enumerate()
                    .map(|(index, dist)| KeyValuePair(index, Reverse(dist)))
                    .collect::<Vec<KeyValuePair<usize, A>>>()
            };
            distances_2.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Ordering::Equal));

            if distances_2.is_empty() {
                todo!()
            }

            for i2 in 0..distances_2.len() {
                let mut visited = HashSet::new();

                let mut partition_1 = PartitionSubSet::new(&target);
                let mut stack_1 = vec![VectorId(target[distances_1[i1].0].id)];
                let mut partition_2 = PartitionSubSet::new(&target);
                let mut stack_2 = vec![VectorId(target[distances_2[i2].0].id)];

                while stack_1.len() + stack_2.len() > 0 {
                    'stack_1: {
                        if let Some(id) = stack_1.pop() {
                            if visited.contains(&id) {
                                break 'stack_1;
                            }
                            graph
                                .0
                                .neighbors(graph.1[&id])
                                .map(|node_idx| graph.0.node_weight(node_idx).unwrap())
                                .filter(|vec_id| !visited.contains(*vec_id))
                                .for_each(|node_id| {
                                    stack_1.push(*node_id);
                                });

                            visited.insert(id);
                            let _ = partition_1.add(vec_to_idx[&id]);
                        }
                    }
                    'stack_2: {
                        if let Some(id) = stack_2.pop() {
                            if visited.contains(&id) {
                                break 'stack_2;
                            }
                            graph
                                .0
                                .neighbors(graph.1[&id])
                                .map(|node_idx| graph.0.node_weight(node_idx).unwrap())
                                .filter(|vec_id| !visited.contains(*vec_id))
                                .for_each(|node_id| {
                                    stack_2.push(*node_id);
                                });

                            visited.insert(id);
                            let _ = partition_2.add(vec_to_idx[&id]);
                        }
                    }
                }

                partition_1.id = target.id;

                return [partition_1, partition_2];
                // do breath first search from both starting points to create two near
            }
        }

        todo!();
    }
}

pub struct FirstTreeSplitStrategy;

impl<
        A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
        B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > SplitStrategy<A, B, PARTITION_CAP, VECTOR_CAP> for FirstTreeSplitStrategy
{
    fn split<'a>(
        partition: &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        graph: &'a IntraPartitionGraph<A>,
    ) -> [PartitionSubSet<'a, A, B, PARTITION_CAP, VECTOR_CAP>; SPLITS] {
        let mut visited_nodes: HashSet<&VectorId> = HashSet::new();
        let mut first_tree_subset = PartitionSubSet::new(partition);
        let mut remaining_points_subset = PartitionSubSet::new(partition);

        // Helper to find the first connected component (tree)
        let mut found_first_tree = false;

        for node in graph.1.keys() {
            if visited_nodes.contains(node) {
                continue;
            }
            let mut stack = vec![node];
            let mut current_tree_nodes = vec![];

            while let Some(current_node) = stack.pop() {
                if visited_nodes.insert(current_node) {
                    current_tree_nodes.push(current_node);

                    // Add neighbors for DFS traversal
                    for neighbor in graph.0.neighbors(graph.1[current_node]) {
                        if !visited_nodes.contains(graph.0.node_weight(neighbor).unwrap()) {
                            stack.push(graph.0.node_weight(neighbor).unwrap());
                        }
                    }
                }
            }

            // Populate either the first tree or remaining points
            if !found_first_tree {
                for vec_id in current_tree_nodes {
                    let partition_index = partition
                        .iter()
                        .position(|v| v.id == **vec_id)
                        .expect("Node not found in partition");
                    first_tree_subset.add(partition_index).unwrap();
                }
                found_first_tree = true;
            } else {
                for node_index in current_tree_nodes {
                    let partition_index = partition
                        .iter()
                        .position(|v| v.id == **node_index)
                        .expect("Node not found in partition");
                    remaining_points_subset.add(partition_index).unwrap();
                }
            }
        }

        first_tree_subset.id = partition.id.clone();

        [first_tree_subset, remaining_points_subset]
    }
}

// testable
pub fn split_partition<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
    S: SplitStrategy<A, B, PARTITION_CAP, VECTOR_CAP>,
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

    let new_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>; SPLITS] =
        match target.size == SPLITS {
            true => {
                let mut new_partitions: [PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>;
                    SPLITS] = array::from_fn(|_| PartitionSubSet::new(&target));

                target.iter().enumerate().for_each(|(idx, _)| {
                    let _ = new_partitions[idx].add(idx).unwrap();
                });

                new_partitions
            }
            false => S::split(&target, &intra_graph),
        };

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
                    // Need to redo this for better updating
                    ((true, Some(0) | None), (false, _)) => None,
                    ((true | false, Some(n)), (false, _)) => {
                        Some((idx, (*dist, (intra_graphs[*n].2, id_1.1), *id_2)))
                    }
                    // ((false, Some(n)), _) => {
                    //     Some((idx, (*dist, (intra_graphs[*n].2, id_1.1), *id_2))) // maybe need to verify
                    // }
                    ((false, _), (true, Some(0) | None)) => None,
                    ((false, _), (true | false, Some(n))) => {
                        Some((idx, (*dist, *id_1, (intra_graphs[*n].2, id_2.1))))
                    }

                    edge_case => panic!("Edge case not accounted for {edge_case:?}"),
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
        array::from_fn(|i1| (new_partitions[i1], intra_graphs[i1].clone()));

    let i1 = result.len() - 1;
    result.swap(0, i1);

    Ok(result)
}

pub fn calculate_number_of_trees<A: Field<A> + Debug>(graph: &IntraPartitionGraph<A>) -> usize {
    let mut visited: HashSet<&VectorId> = HashSet::new();
    let mut tree_count = 0;

    for node in graph.1.keys() {
        if !visited.contains(node) {
            // Start DFS from this node
            tree_count += 1;
            let mut stack = vec![node];

            while let Some(current) = stack.pop() {
                if visited.insert(current) {
                    for neighbor in graph.0.neighbors(graph.1[&current]) {
                        if !visited.contains(graph.0.node_weight(neighbor).unwrap()) {
                            stack.push(graph.0.node_weight(neighbor).unwrap());
                        }
                    }
                }
            }
        }
    }

    tree_count
}
pub fn split_partition_into_trees<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    partition: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    graph: &mut IntraPartitionGraph<A>,
    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<
    Vec<(
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        IntraPartitionGraph<A>,
    )>,
    PartitionErr,
> {
    let mut result: Vec<(
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        IntraPartitionGraph<A>,
    )> = vec![(partition.clone(), graph.clone())];
    let mut i1 = 0;

    while i1 != result.len() {
        let (partition, graph) = result.get_mut(i1).unwrap();

        if calculate_number_of_trees(&graph) < 2 {
            i1 += 1;
            continue;
        }

        let [pair_1, pair_2] =
            split_partition::<A, B, FirstTreeSplitStrategy, PARTITION_CAP, VECTOR_CAP>(
                partition,
                graph,
                inter_graph,
            )?;

        *partition = pair_2.0;
        *graph = pair_2.1;

        result.push(pair_1);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, str::FromStr};

    use uuid::Uuid;

    use super::*;
    use crate::{db::component::partition::VectorEntry, vector::Vector};

    fn split_partition_test<S: SplitStrategy<f32, Vector<f32, 2>, 20, 10>>(
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

        let original_graph = intra_graph.clone();
        edges
            .iter()
            .map(|(weight, id_1, id_2)| (weight, VectorId(*id_1), VectorId(*id_2)))
            .for_each(|(weight, id_1, id_2)| {
                intra_graph.add_edge(id_1, id_2, *weight);
            });
        let mut inter_graph = InterPartitionGraph::<f32>::new();
        inter_graph.add_node(intra_graph.2);

        // Call the function under test
        let result = split_partition::<f32, Vector<f32, 2>, S, 20, 10>(
            &mut partition,
            &mut intra_graph,
            &mut inter_graph,
        );

        // Check the result
        match result {
            Ok(splits) => {
                // 1. Ensure no duplicate vectors across partitions
                let mut all_vectors = vec![];
                {
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
                }
                // 2. Ensure no duplicate vectors in the same split_partition
                {
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
                }

                // 3. Ensure graph-partition consistency
                {
                    for (split_partition, split_graph) in splits.iter() {
                        let partition_vectors: HashSet<_> =
                            split_partition.iter().map(|x| VectorId(x.id)).collect();
                        let graph_vectors: HashSet<_> =
                            split_graph.1.iter().map(|(id, _)| *id).collect();

                        assert_eq!(
                            partition_vectors, graph_vectors,
                            "Mismatch between partition and graph vectors"
                        );
                    }
                }
                // 4. Ensure no data loss: vectors in target should remain after splitting
                {
                    let target_vectors: HashSet<_> = vectors.into_iter().collect();
                    let result_vectors: HashSet<_> = all_vectors.into_iter().collect();

                    assert_eq!(
                        target_vectors, result_vectors,
                        "Some vectors from the target were dropped after splitting"
                    );
                }
                // 5. Ensure all edges and vertices are preserved
                {
                    let [(_, new_graph_1), (_, new_graph_2)] = splits;
                    original_graph.0
                        .edge_indices()
                        .map(|edge_index| (
                            original_graph.0.edge_endpoints(edge_index).unwrap(),
                            original_graph.0.edge_weight(edge_index).unwrap()
                        ))
                        .map(|((start, end), dist)| (
                            original_graph.0.node_weight(start).unwrap(),
                            original_graph.0.node_weight(end).unwrap(),
                            dist,
                        ))
                        .for_each(|(start, end, dist)| {
                            let in_graph_1 = new_graph_1.0.edge_indices()
                                .map(|edge_index| (
                                    new_graph_1.0.edge_endpoints(edge_index).unwrap(),
                                    new_graph_1.0.edge_weight(edge_index).unwrap()
                                ))
                                .map(|((start, end), dist)| (
                                    new_graph_1.0.node_weight(start).unwrap(),
                                    new_graph_1.0.node_weight(end).unwrap(),
                                    dist,
                                ))
                                .any(|(start_1, end_1, dist_1)| {
                                    (start_1 == start && end_1 == end && dist_1 == dist) || (end_1 == start && start_1 == end && dist_1 == dist)
                                });

                            let in_graph_2 = new_graph_2.0.edge_indices()
                                .map(|edge_index| (
                                    new_graph_1.0.edge_endpoints(edge_index).unwrap(),
                                    new_graph_1.0.edge_weight(edge_index).unwrap()
                                ))
                                .map(|((start, end), dist)| (
                                    new_graph_1.0.node_weight(start).unwrap(),
                                    new_graph_1.0.node_weight(end).unwrap(),
                                    dist,
                                ))
                                .any(|(start_1, end_1, dist_1)|{
                                    (start_1 == start && end_1 == end && dist_1 == dist) || (end_1 == start && start_1 == end && dist_1 == dist)
                                });

                            let between = inter_graph.0.edge_indices()
                                .map(|edge_index| (
                                    inter_graph.0.edge_weight(edge_index).unwrap()
                                ))
                                .any(|(dist_1, (_, start_1), (_, end_1))| {
                                    (start_1 == start && end_1 == end && dist_1 == dist) || (end_1 == start && start_1 == end && dist_1 == dist)
                                });

                            match (in_graph_1, in_graph_2, between) {
                                (true, false, false) | (false, true, false) | (false, false, true) => assert!(true),
                                (false, false, false) => assert!(false, "Edge has been dropped: ({dist:?}, {start:?}, {end:?})\nOriginal:\n{original_graph:?}\nGraph_1:\n{new_graph_1:?}\nGraph_2:\n{new_graph_1:?}\nInter_graph:\n{inter_graph:?}"),
                                _ => assert!(false, "Duplicate edge: ({dist:?}, {start:?}, {end:?})\nOriginal:\n{original_graph:?}\nGraph_1:\n{new_graph_1:?}\nGraph_2:\n{new_graph_1:?}\nInter_graph:\n{inter_graph:?}")
                            }
                        });
                }
            }
            Err(err) => {
                panic!("split_partition failed with error: {:?}", err);
            }
        }
    }

    #[test]
    fn test_knn_split() {
        let vectors = vec![
            VectorEntry {
                vector: Vector([0.92162, 0.15653355]),
                id: Uuid::from_str("1926289c-4b15-4683-93e0-733a1cff8d0d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8563366, 0.06344742]),
                id: Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.87333435, 0.10792838]),
                id: Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.904121, 0.14064054]),
                id: Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8634496, 0.048130006]),
                id: Uuid::from_str("de60cba1-ae9b-4303-b213-cdaabd6d14bf").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.704498, 0.6998113]),
                id: Uuid::from_str("5b3f398a-b640-4eb8-b468-cba3b1ade5d9").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5392826, 0.6072107]),
                id: Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5484414, 0.5518184]),
                id: Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.45432913, 0.55870813]),
                id: Uuid::from_str("581b542f-1645-483c-b4b5-5bfcb62ce08d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4408836, 0.5238235]),
                id: Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5484165, 0.56746066]),
                id: Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10779287, 0.11290348]),
                id: Uuid::from_str("77dae981-1a88-4907-8471-4f0ac020faad").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5125989, 0.53425103]),
                id: Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4696728, 0.52770525]),
                id: Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.46052048, 0.5586407]),
                id: Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4390052, 0.50848174]),
                id: Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.50311416, 0.6384853]),
                id: Uuid::from_str("589ed7b2-846d-4f91-a07a-d631bcd41148").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.44561064, 0.29694903]),
                id: Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.2946079, 0.3537509]),
                id: Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.106322125, 0.11829137]),
                id: Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
                _phantom_data: PhantomData,
            },
        ];

        let edges = vec![
            (
                0.005585020099079774,
                Uuid::from_str("77dae981-1a88-4907-8471-4f0ac020faad").unwrap(),
                Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
            ),
            (
                0.006191717179216102,
                Uuid::from_str("581b542f-1645-483c-b4b5-5bfcb62ce08d").unwrap(),
                Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
            ),
            (
                0.015456325127843276,
                Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
            ),
            (
                0.015642279818415126,
                Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
            ),
            (
                0.016888396627489435,
                Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                Uuid::from_str("de60cba1-ae9b-4303-b213-cdaabd6d14bf").unwrap(),
            ),
            (
                0.023639009451753703,
                Uuid::from_str("1926289c-4b15-4683-93e0-733a1cff8d0d").unwrap(),
                Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
            ),
            (
                0.029049716344613435,
                Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
            ),
            (
                0.032260921066902266,
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
                Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
            ),
            (
                0.03991612825621872,
                Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
            ),
            (
                0.04078595112549913,
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
            ),
            (
                0.04342231335406252,
                Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
            ),
            (
                0.04492107779303715,
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
            ),
            (
                0.04761805652884314,
                Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
            ),
            (
                0.04781481629363016,
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                Uuid::from_str("589ed7b2-846d-4f91-a07a-d631bcd41148").unwrap(),
            ),
            (
                0.16133282345203193,
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
                Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
            ),
            (
                0.189396408407129,
                Uuid::from_str("5b3f398a-b640-4eb8-b468-cba3b1ade5d9").unwrap(),
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
            ),
            (
                0.21163581747317184,
                Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
            ),
            (
                0.3014842008052354,
                Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
                Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
            ),
            (
                0.467628461732802,
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
            ),
        ];

        split_partition_test::<KMean>(vectors.clone(), edges.clone());
    }

    #[test]
    fn test_bfs_split() {
        let vectors = vec![
            VectorEntry {
                vector: Vector([0.92162, 0.15653355]),
                id: Uuid::from_str("1926289c-4b15-4683-93e0-733a1cff8d0d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8563366, 0.06344742]),
                id: Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.87333435, 0.10792838]),
                id: Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.904121, 0.14064054]),
                id: Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8634496, 0.048130006]),
                id: Uuid::from_str("de60cba1-ae9b-4303-b213-cdaabd6d14bf").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.704498, 0.6998113]),
                id: Uuid::from_str("5b3f398a-b640-4eb8-b468-cba3b1ade5d9").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5392826, 0.6072107]),
                id: Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5484414, 0.5518184]),
                id: Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.45432913, 0.55870813]),
                id: Uuid::from_str("581b542f-1645-483c-b4b5-5bfcb62ce08d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4408836, 0.5238235]),
                id: Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5484165, 0.56746066]),
                id: Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10779287, 0.11290348]),
                id: Uuid::from_str("77dae981-1a88-4907-8471-4f0ac020faad").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5125989, 0.53425103]),
                id: Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4696728, 0.52770525]),
                id: Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.46052048, 0.5586407]),
                id: Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4390052, 0.50848174]),
                id: Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.50311416, 0.6384853]),
                id: Uuid::from_str("589ed7b2-846d-4f91-a07a-d631bcd41148").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.44561064, 0.29694903]),
                id: Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.2946079, 0.3537509]),
                id: Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.106322125, 0.11829137]),
                id: Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
                _phantom_data: PhantomData,
            },
        ];

        let edges = vec![
            (
                0.005585020099079774,
                Uuid::from_str("77dae981-1a88-4907-8471-4f0ac020faad").unwrap(),
                Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
            ),
            (
                0.006191717179216102,
                Uuid::from_str("581b542f-1645-483c-b4b5-5bfcb62ce08d").unwrap(),
                Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
            ),
            (
                0.015456325127843276,
                Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
            ),
            (
                0.015642279818415126,
                Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
            ),
            (
                0.016888396627489435,
                Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                Uuid::from_str("de60cba1-ae9b-4303-b213-cdaabd6d14bf").unwrap(),
            ),
            (
                0.023639009451753703,
                Uuid::from_str("1926289c-4b15-4683-93e0-733a1cff8d0d").unwrap(),
                Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
            ),
            (
                0.029049716344613435,
                Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
            ),
            (
                0.032260921066902266,
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
                Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
            ),
            (
                0.03991612825621872,
                Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
            ),
            (
                0.04078595112549913,
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
            ),
            (
                0.04342231335406252,
                Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
            ),
            (
                0.04492107779303715,
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
            ),
            (
                0.04761805652884314,
                Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
            ),
            (
                0.04781481629363016,
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                Uuid::from_str("589ed7b2-846d-4f91-a07a-d631bcd41148").unwrap(),
            ),
            (
                0.16133282345203193,
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
                Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
            ),
            (
                0.189396408407129,
                Uuid::from_str("5b3f398a-b640-4eb8-b468-cba3b1ade5d9").unwrap(),
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
            ),
            (
                0.21163581747317184,
                Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
            ),
            (
                0.3014842008052354,
                Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
                Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
            ),
            (
                0.467628461732802,
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
            ),
        ];

        split_partition_test::<BFS>(vectors.clone(), edges.clone());
    }

    #[test]
    fn test_tree_split_1() {
        let vectors = vec![
            VectorEntry {
                vector: Vector([0.92162, 0.15653355]),
                id: Uuid::from_str("1926289c-4b15-4683-93e0-733a1cff8d0d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8563366, 0.06344742]),
                id: Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.87333435, 0.10792838]),
                id: Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.904121, 0.14064054]),
                id: Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.8634496, 0.048130006]),
                id: Uuid::from_str("de60cba1-ae9b-4303-b213-cdaabd6d14bf").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.704498, 0.6998113]),
                id: Uuid::from_str("5b3f398a-b640-4eb8-b468-cba3b1ade5d9").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5392826, 0.6072107]),
                id: Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5484414, 0.5518184]),
                id: Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.45432913, 0.55870813]),
                id: Uuid::from_str("581b542f-1645-483c-b4b5-5bfcb62ce08d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4408836, 0.5238235]),
                id: Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5484165, 0.56746066]),
                id: Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.10779287, 0.11290348]),
                id: Uuid::from_str("77dae981-1a88-4907-8471-4f0ac020faad").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5125989, 0.53425103]),
                id: Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4696728, 0.52770525]),
                id: Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.46052048, 0.5586407]),
                id: Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4390052, 0.50848174]),
                id: Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.50311416, 0.6384853]),
                id: Uuid::from_str("589ed7b2-846d-4f91-a07a-d631bcd41148").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.44561064, 0.29694903]),
                id: Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.2946079, 0.3537509]),
                id: Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.106322125, 0.11829137]),
                id: Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
                _phantom_data: PhantomData,
            },
        ];

        let edges = vec![
            (
                0.005585020099079774,
                Uuid::from_str("77dae981-1a88-4907-8471-4f0ac020faad").unwrap(),
                Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
            ),
            (
                0.006191717179216102,
                Uuid::from_str("581b542f-1645-483c-b4b5-5bfcb62ce08d").unwrap(),
                Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
            ),
            (
                0.015456325127843276,
                Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
            ),
            (
                0.015642279818415126,
                Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
            ),
            (
                0.016888396627489435,
                Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                Uuid::from_str("de60cba1-ae9b-4303-b213-cdaabd6d14bf").unwrap(),
            ),
            (
                0.023639009451753703,
                Uuid::from_str("1926289c-4b15-4683-93e0-733a1cff8d0d").unwrap(),
                Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
            ),
            (
                0.029049716344613435,
                Uuid::from_str("c90e7bc5-e439-4561-9d9a-6f4ff103a2f7").unwrap(),
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
            ),
            (
                0.032260921066902266,
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
                Uuid::from_str("bd169baf-a42a-40e2-839f-e5e6217acbcb").unwrap(),
            ),
            (
                0.03991612825621872,
                Uuid::from_str("35928ca9-372e-433c-a70a-9223b4e83c00").unwrap(),
                Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
            ),
            (
                0.04078595112549913,
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                Uuid::from_str("bb055eb8-56ec-4933-88d0-b69dfb1223b6").unwrap(),
            ),
            (
                0.04342231335406252,
                Uuid::from_str("47ce4c45-d6de-4de2-bd63-90b29ff6c786").unwrap(),
                Uuid::from_str("9a1a07cc-315c-4ba1-ab98-81a9dd7be601").unwrap(),
            ),
            (
                0.04492107779303715,
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                Uuid::from_str("577c353a-fb29-442c-8127-524ff1680efc").unwrap(),
            ),
            (
                0.04761805652884314,
                Uuid::from_str("c070aa1f-7ec6-447b-a4ee-a7fb062950df").unwrap(),
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
            ),
            (
                0.04781481629363016,
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
                Uuid::from_str("589ed7b2-846d-4f91-a07a-d631bcd41148").unwrap(),
            ),
            (
                0.16133282345203193,
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
                Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
            ),
            (
                0.189396408407129,
                Uuid::from_str("5b3f398a-b640-4eb8-b468-cba3b1ade5d9").unwrap(),
                Uuid::from_str("7086aba4-5d2e-43e6-8e84-6653da6a5eb0").unwrap(),
            ),
            (
                0.21163581747317184,
                Uuid::from_str("b7ea8823-4827-4729-92c5-30753fc32b72").unwrap(),
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
            ),
            (
                0.3014842008052354,
                Uuid::from_str("c376134c-1429-4e6b-a355-d8c3ebb50511").unwrap(),
                Uuid::from_str("f98a5f12-9f9e-435d-911c-fb5658d0bb96").unwrap(),
            ),
            (
                0.467628461732802,
                Uuid::from_str("bab0862b-33d5-4e8d-a9b2-0e13510e8eb7").unwrap(),
                Uuid::from_str("0798dbd3-3522-4f14-89e0-4bc3818d825e").unwrap(),
            ),
        ];

        split_partition_test::<FirstTreeSplitStrategy>(vectors.clone(), edges.clone());
    }
    #[test]
    fn test_tree_split_2() {
        let vectors = vec![
            VectorEntry {
                vector: Vector([0.54795116, 0.50905]),
                id: Uuid::from_str("42aac5e0-9ea2-4a10-ad4c-9c652854d420").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4877212, 0.5060333]),
                id: Uuid::from_str("518fc04d-5b29-4157-b9af-b5ff8bb33472").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.49342033, 0.5013394]),
                id: Uuid::from_str("2663e716-2be9-46a5-96b1-bbee971195bb").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4696443, 0.5229496]),
                id: Uuid::from_str("625a14d2-3018-44f1-9a2a-a988c5cf5146").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.53619784, 0.5049916]),
                id: Uuid::from_str("3fde01a3-0233-445f-85d7-2215ca139dec").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.49201742, 0.51932555]),
                id: Uuid::from_str("65d610a9-c8d7-41df-82b2-ebbb43e118f6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5015184, 0.4676091]),
                id: Uuid::from_str("9b7a0705-7503-4ff0-ab3e-0515b0a585ab").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5012751, 0.49511862]),
                id: Uuid::from_str("63716bf9-f6d4-4d64-a749-3cf92c1bf35c").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5001987, 0.50035655]),
                id: Uuid::from_str("fa68019a-fabf-4318-958c-5b90412b7197").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5122232, 0.48015934]),
                id: Uuid::from_str("082c4b1f-3b55-425f-8589-2e4a7fd9b2e6").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.45651713, 0.48801365]),
                id: Uuid::from_str("cdc6b0ae-4df9-4003-8d8d-094048cf209b").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5361129, 0.48330697]),
                id: Uuid::from_str("1865dedc-5bfe-4cd6-90f0-5051932542f4").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.4996799, 0.50941885]),
                id: Uuid::from_str("e14a420b-602d-476c-8746-524265c0861d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.47749156, 0.455703]),
                id: Uuid::from_str("76069018-edc1-403c-9795-efba004fce1d").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.46677235, 0.52104586]),
                id: Uuid::from_str("e5ae870a-8ea2-407f-8ace-56469febcb95").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5325789, 0.49239337]),
                id: Uuid::from_str("c2c28637-f27a-48a0-9903-b5bdefbf95da").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.52878046, 0.4843407]),
                id: Uuid::from_str("691f2996-810b-41ac-baed-c10431171d7b").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.49987164, 0.49994764]),
                id: Uuid::from_str("c25c97ab-3b0e-41c3-aab2-8de84dac9fee").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.50576365, 0.45795757]),
                id: Uuid::from_str("4c5731f7-61a6-4191-b3a2-fd288d15efec").unwrap(),
                _phantom_data: PhantomData,
            },
            VectorEntry {
                vector: Vector([0.5150571, 0.5078569]),
                id: Uuid::from_str("c36e1ec4-50a5-4b61-a431-314471787136").unwrap(),
                _phantom_data: PhantomData,
            },
        ];

        let edges = vec![
            (
                5.4512995e-05,
                Uuid::from_str("2663e716-2be9-46a5-96b1-bbee971195bb").unwrap(),
                Uuid::from_str("518fc04d-5b29-4157-b9af-b5ff8bb33472").unwrap(),
            ),
            (
                0.00015461132,
                Uuid::from_str("3fde01a3-0233-445f-85d7-2215ca139dec").unwrap(),
                Uuid::from_str("42aac5e0-9ea2-4a10-ad4c-9c652854d420").unwrap(),
            ),
            (
                8.239428e-05,
                Uuid::from_str("e14a420b-602d-476c-8746-524265c0861d").unwrap(),
                Uuid::from_str("fa68019a-fabf-4318-958c-5b90412b7197").unwrap(),
            ),
            (
                2.5289186e-05,
                Uuid::from_str("c25c97ab-3b0e-41c3-aab2-8de84dac9fee").unwrap(),
                Uuid::from_str("63716bf9-f6d4-4d64-a749-3cf92c1bf35c").unwrap(),
            ),
            (
                0.0005136896,
                Uuid::from_str("65d610a9-c8d7-41df-82b2-ebbb43e118f6").unwrap(),
                Uuid::from_str("625a14d2-3018-44f1-9a2a-a988c5cf5146").unwrap(),
            ),
            (
                0.00034364,
                Uuid::from_str("082c4b1f-3b55-425f-8589-2e4a7fd9b2e6").unwrap(),
                Uuid::from_str("63716bf9-f6d4-4d64-a749-3cf92c1bf35c").unwrap(),
            ),
            (
                0.00027210018,
                Uuid::from_str("082c4b1f-3b55-425f-8589-2e4a7fd9b2e6").unwrap(),
                Uuid::from_str("9b7a0705-7503-4ff0-ab3e-0515b0a585ab").unwrap(),
            ),
            (
                2.7419532e-07,
                Uuid::from_str("c25c97ab-3b0e-41c3-aab2-8de84dac9fee").unwrap(),
                Uuid::from_str("fa68019a-fabf-4318-958c-5b90412b7197").unwrap(),
            ),
            (
                0.0001718119,
                Uuid::from_str("c2c28637-f27a-48a0-9903-b5bdefbf95da").unwrap(),
                Uuid::from_str("3fde01a3-0233-445f-85d7-2215ca139dec").unwrap(),
            ),
            (
                1.1872278e-05,
                Uuid::from_str("e5ae870a-8ea2-407f-8ace-56469febcb95").unwrap(),
                Uuid::from_str("625a14d2-3018-44f1-9a2a-a988c5cf5146").unwrap(),
            ),
            (
                7.927364e-05,
                Uuid::from_str("691f2996-810b-41ac-baed-c10431171d7b").unwrap(),
                Uuid::from_str("c2c28637-f27a-48a0-9903-b5bdefbf95da").unwrap(),
            ),
            (
                0.00015685642,
                Uuid::from_str("e14a420b-602d-476c-8746-524265c0861d").unwrap(),
                Uuid::from_str("65d610a9-c8d7-41df-82b2-ebbb43e118f6").unwrap(),
            ),
            (
                0.00071904616,
                Uuid::from_str("76069018-edc1-403c-9795-efba004fce1d").unwrap(),
                Uuid::from_str("9b7a0705-7503-4ff0-ab3e-0515b0a585ab").unwrap(),
            ),
            (
                0.0011962963,
                Uuid::from_str("e5ae870a-8ea2-407f-8ace-56469febcb95").unwrap(),
                Uuid::from_str("cdc6b0ae-4df9-4003-8d8d-094048cf209b").unwrap(),
            ),
            (
                5.4833323e-05,
                Uuid::from_str("691f2996-810b-41ac-baed-c10431171d7b").unwrap(),
                Uuid::from_str("1865dedc-5bfe-4cd6-90f0-5051932542f4").unwrap(),
            ),
            (
                0.0002916271,
                Uuid::from_str("691f2996-810b-41ac-baed-c10431171d7b").unwrap(),
                Uuid::from_str("082c4b1f-3b55-425f-8589-2e4a7fd9b2e6").unwrap(),
            ),
            (
                4.355632e-05,
                Uuid::from_str("c25c97ab-3b0e-41c3-aab2-8de84dac9fee").unwrap(),
                Uuid::from_str("2663e716-2be9-46a5-96b1-bbee971195bb").unwrap(),
            ),
            (
                0.000111174166,
                Uuid::from_str("4c5731f7-61a6-4191-b3a2-fd288d15efec").unwrap(),
                Uuid::from_str("9b7a0705-7503-4ff0-ab3e-0515b0a585ab").unwrap(),
            ),
            (
                0.00023889774,
                Uuid::from_str("c36e1ec4-50a5-4b61-a431-314471787136").unwrap(),
                Uuid::from_str("e14a420b-602d-476c-8746-524265c0861d").unwrap(),
            ),
        ];

        split_partition_test::<FirstTreeSplitStrategy>(vectors.clone(), edges.clone());
    }
}
