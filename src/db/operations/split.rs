use std::{
    array,
    cmp::Reverse,
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

use super::LoadedPartitions;

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

#[derive(Debug)]
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

// testable
pub fn split_partition<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>> + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    target: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    intra_graph: &mut IntraPartitionGraph<A>,

    splits: usize,

    inter_graph: &mut InterPartitionGraph<A>,
) -> Result<
    Vec<(
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        IntraPartitionGraph<A>,
    )>,
    PartitionErr,
> {
    if target.size < splits {
        return Err(PartitionErr::InsufficientSizeForSplits);
    }

    if target.size == splits {
        // simple hard coded solution
        // each vector in target becomes it's own partition
        todo!()
    }

    let mut prev_partitions: Vec<PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>> =
        (0..splits).map(|_| PartitionSubSet::new(&target)).collect();
    let mut new_partitions = {
        // Get closet vector to centeroid
        let mut start_points: Vec<PartitionSubSet<'_, A, B, PARTITION_CAP, VECTOR_CAP>> =
            (0..splits).map(|_| PartitionSubSet::new(&target)).collect(); // Vec::with_capacity(splits);

        let centroid = target.centroid();

        let mut distance = target
            .iter()
            .map(|vector| B::dist(&centroid, &vector.vector))
            .enumerate()
            .map(|(index, dist)| KeyValuePair(index, Reverse(dist)))
            .collect::<Vec<KeyValuePair<usize, A>>>();

        if distance.len() == 0 {
            todo!()
        }

        make_heap(&mut distance);

        start_points.iter_mut().for_each(|subset| {
            pop_heap(&mut distance);
            let KeyValuePair(ref_index, Reverse(_dist)) = distance.pop().unwrap();

            if let Err(err) = subset.add(ref_index) {
                todo!()
            }
        });

        start_points
    };

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
                    .min_by(|(_, x1), (_, x2)| x1.partial_cmp(x2).unwrap())
                    .unwrap();

                index
            })
            .enumerate()
            .for_each(|(vector_index, partition_index)| {
                new_partitions[partition_index].add(vector_index).unwrap();
            });
    }

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
    let (intra_graphs, new_inter_edges) = {
        let mut intra_graphs: Vec<IntraPartitionGraph<A>> = new_partitions
            .iter()
            .map(|x| IntraPartitionGraph::new(PartitionId(x.id)))
            .collect();
        // (0..(splits)).map(|_| IntraPartitionGraph::new()).collect();

        let mut new_inter_edges: Vec<((usize, VectorId), (usize, VectorId), A)> = Vec::new();

        let mut not_visited_nodes: HashSet<VectorId> =
            target.iter().map(|vector| VectorId(vector.id)).collect();

        let mut visit_stack: Vec<VectorId> = Vec::new();

        visit_stack.push(VectorId(target[new_partitions[0][0]].id));
        let mut not_visited_nodes_size = not_visited_nodes.len();
        while not_visited_nodes_size > 0 {
            // println!("not_visited_nodes: {:?}", not_visited_nodes);
            // println!("visit_stack: {:?}", visit_stack);
           
            let current_node = match visit_stack.pop() {
                Some(node) => node,
                None => {
                    let vector_id = not_visited_nodes.iter().next().unwrap().clone();
                    visit_stack.push(vector_id);
                    // not_visited_nodes.remove(&vector_id);
                    continue;
                }
            };

            if !not_visited_nodes.remove(&current_node) {
                // already visited current node
                continue;
            }


            let current_partition_index = partition_membership[&current_node];

            // println!("current node: Partition_index({current_partition_index}) :- {current_node:?}\n\n");

            if !intra_graphs[current_partition_index]
                .1
                .contains_key(&current_node)
            {
                intra_graphs[current_partition_index].add_node(current_node);
            }

            intra_graph
                .0
                .edges(intra_graph.1[&current_node])
                .map(|edge| {
                    (
                        {
                            let source = intra_graph.0.node_weight(edge.source()).unwrap();
                            let target = intra_graph.0.node_weight(edge.target()).unwrap();
                            match source == &current_node {
                                true => *target,
                                false => *source,
                            }
                        },
                        edge.weight(),
                    )
                })
                .filter(|(id, _dist)| not_visited_nodes.contains(id))
                .for_each(|(other_node, dist)| {
                    // Add id into new intra_graphs
                    let other_partition_index = partition_membership[&other_node];

                    if other_partition_index == current_partition_index {
                        if !intra_graphs[other_partition_index]
                            .1
                            .contains_key(&other_node)
                        {
                            intra_graphs[other_partition_index].add_node(other_node);
                        }

                        intra_graphs[other_partition_index].add_edge(
                            current_node,
                            other_node,
                            *dist,
                        );
                    } else {
                        new_inter_edges.push((
                            (other_partition_index, other_node),
                            (current_partition_index, current_node),
                            *dist,
                        ));
                    }

                    visit_stack.push(other_node);
                });

            not_visited_nodes_size = not_visited_nodes.len()
        }

        (intra_graphs, new_inter_edges)
    };

    let new_partitions: Vec<Partition<A, B, PARTITION_CAP, VECTOR_CAP>> = {
        let mut tmp: Vec<Partition<A, B, PARTITION_CAP, VECTOR_CAP>> =
            new_partitions.iter().map(|x| Partition::from(x)).collect();

        let _ = mem::replace(&mut tmp[0].id, target.id.clone());

        // mem::swap(target, &mut tmp[0]);

        tmp
    };
    {
        
        new_partitions.iter().skip(1).for_each(|partition| {
            inter_graph.add_node(PartitionId(partition.id));
        });

        // println!("{:#?}", inter_graph.1);
        // println!("{:#?}", new_inter_edges);

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
    }

    // Note: new_partitions[0].id = target.id -> therefore should replace target after split_target call
    let mut result: Vec<_> = new_partitions
    .into_iter()
    .zip(intra_graphs.into_iter())
    .collect();

    let i1= result.len()-1;
    result.swap(0, i1);
    
    Ok(result)
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

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::{
        db::{component::partition::VectorEntry, operations::add::add_into},
        vector::Vector,
    };

    #[test]
    fn basic_split() {
        // Setup mock data
        type TestField = f32; // Example field type
        type TestVector = Vector<f32, 2>; // Example vector type

        let mut inter_graph = InterPartitionGraph::new();

        let mut partition = Partition::<TestField, TestVector, 500, 500>::new();
        let mut intra_graph = IntraPartitionGraph::new(PartitionId(partition.id));
        inter_graph.add_node(PartitionId(partition.id));

        assert_eq!(inter_graph.1.len(), 1);

        let splits = 2; // Example split count

        // initialize partitions
        let expected_partitions = vec![
            vec![
                VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4()),
                VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4()),
            ],
            vec![
                VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4()),
                VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4()),
            ],
        ];
        let expected_centroids = vec![Vector::splat(1.5), Vector::splat(-1.5)];

        expected_partitions
            .iter()
            .map(|x| x.iter())
            .flatten()
            .for_each(|vector| {
                let result = add_into(
                    &mut partition,
                    vector.clone(),
                    &mut intra_graph,
                    &mut inter_graph,
                    &mut [],
                );
                assert!(result.is_ok());
            });

        // Call split_partition
        let result = split_partition(&mut partition, &mut intra_graph, splits, &mut inter_graph);
        // Validate results
        assert!(result.is_ok());
        let new_partitions = result.unwrap();
        assert_eq!(new_partitions.len(), splits);

        new_partitions.iter().for_each(|(x, _)| {
            println!("{:?}", x);
        });

        let result = new_partitions.iter().all(|(actual_partition, actual_graph)| {
            expected_partitions
                .iter()
                .any(|expected_partition| partition_check(&expected_partition, actual_partition))
                && expected_centroids
                    .iter()
                    .any(|expected_centroid| centroid_check(expected_centroid, actual_partition))
        });
        assert!(result);

        assert_eq!(inter_graph.1.len(), 2);
        assert_eq!(inter_graph.0.edges(inter_graph.1[&PartitionId(partition.id)]).count(), 1);
        // inter_graph.0.edges(inter_graph.1[&PartitionId(partition.id)])
        //     .map(|x|x.source())
        assert_eq!(inter_graph.0.edges(inter_graph.1[&PartitionId(new_partitions[0].0.id)]).count(), 1);
        // assert_eq!(inter_graph.1.len(), 2);
    }

    fn partition_check<
        A: Field<A> + PartialEq + Copy,
        B: VectorSpace<A> + Copy + PartialEq,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    >(
        search_vectors: &[VectorEntry<A, B>],
        partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    ) -> bool {
        search_vectors.iter().all(|search_vector| {
            partition.iter().any(|vector| {
                vector.id == search_vector.id && vector.vector == search_vector.vector
            })
        })
    }
    fn centroid_check<
        A: Field<A> + PartialEq + Copy,
        B: VectorSpace<A> + Copy + PartialEq,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    >(
        centroid: &B,
        partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    ) -> bool {
        &partition.centroid() == centroid
    }

    #[test]
    fn invalid_split() {
        // Setup mock data
        type TestField = f32; // Example field type
        type TestVector = Vector<f32, 2>; // Example vector type

        let mut inter_graph = InterPartitionGraph::new();

        let mut partition = Partition::<TestField, TestVector, 500, 500>::new();
        let mut intra_graph = IntraPartitionGraph::new(PartitionId(partition.id));
        inter_graph.add_node(PartitionId(partition.id));

        assert_eq!(inter_graph.1.len(), 1);

        let splits = 5; // Example split count

        // initialize partitions
        let expected_partitions = vec![
            vec![
                VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4()),
                VectorEntry::from_uuid(Vector::splat(2.), Uuid::new_v4()),
            ],
            vec![
                VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4()),
                VectorEntry::from_uuid(Vector::splat(-2.), Uuid::new_v4()),
            ],
        ];

        expected_partitions
            .iter()
            .map(|x| x.iter())
            .flatten()
            .for_each(|vector| {
                let result = add_into(
                    &mut partition,
                    vector.clone(),
                    &mut intra_graph,
                    &mut inter_graph,
                    &mut [],
                );
                assert!(result.is_ok());
            });

        // Call split_partition
        let result = split_partition(&mut partition, &mut intra_graph, splits, &mut inter_graph);
        // Validate results
        assert!(result.is_err());
    }
}
