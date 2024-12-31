use std::{
    any::Any,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use petgraph::{
    adj::NodeIndex,
    algo::min_spanning_tree,
    csr::DefaultIx,
    data::DataMap,
    graph::EdgeIndex,
    visit::{Data, EdgeRef, IntoEdgeReferences},
};
use tokio::{
    sync::{Mutex, RwLock},
    task::JoinSet,
};
use uuid::Uuid;

use crate::{
    db::partition,
    vector::{Field, VectorSerial, VectorSpace},
};

use super::{
    component::data_buffer::DataBuffer, meta::Meta, InterPartitionGraph, IntraPartitionGraph,
    LoadedPartitions, Partition, PartitionErr, PartitionId, VectorEntry, VectorId,
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

pub fn add_into<
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
    A: PartialOrd,
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
    // target.size += 1;

    target.centroid = B::add(&target.centroid, &value.vector);

    // insert node into a minimum spanning tree
    'add_edges: {
        let value_node_index = intra_graph.add_node(VectorId(value.id));
        intra_graph.1.insert(VectorId(value.id), value_node_index);
        let partition_splits = {
            let mut tmp = vec![target.size];

            neighbors
                .iter()
                .map(|x| &x.0.size)
                .for_each(|x| tmp.push(*x));

            tmp
        };

        if partition_splits.len() == 1 && partition_splits[0] == 0 {
            break 'add_edges;
        }

        let dist = {
            if cfg!(feature = "gpu_processing") {
                todo!()
            } else {
                target
                    .iter()
                    .map(|x| (target.id, x))
                    .chain(
                        neighbors
                            .iter()
                            .map(|partition| partition.0.iter().map(|vec| (partition.0.id, vec)))
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
                .min_by(|(_, (_, x1)), (_, (_, x2))| x1.partial_cmp(x2).unwrap_or(Ordering::Equal))
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
                    .filter(|edge| {
                        edge.weight().1 .1 == closet_id.1 || edge.weight().2 .1 == closet_id.1
                    })
                    .map(|edge| {
                        let edge_id = edge.id();
                        let (edge_dist, vector_1, vector_2) = edge.weight();

                        let other_vector = match target.id == *vector_1.0 {
                            true => vector_2,
                            false => vector_1,
                        };

                        (edge_id, edge_dist, (other_vector.0, other_vector.1))
                    })
                    .filter(|(_, _, id)| dist.contains_key(id))
                    .map(|(edge_id, edge_dist, (other_partition, other_vector))| {
                        EdgeType::InterEdge(
                            *edge_dist,
                            dist[&(other_partition, other_vector)].1,
                            edge_id,
                            closet_id.1,
                            other_vector,
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
                    .or_else(|| Some(&neighbors[partition_splits.len() - 1 - 1]))
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
                    .filter(|edge| {
                        edge.weight().1 .1 == closet_id.1 || edge.weight().2 .1 == closet_id.1
                    })
                    .map(|edge| {
                        let edge_id = edge.id();
                        let (edge_dist, vector_1, vector_2) = edge.weight();

                        let other_vector = match target.id == *vector_1.0 {
                            true => vector_2,
                            false => vector_1,
                        };

                        (edge_id, edge_dist, (other_vector.0, other_vector.1))
                    })
                    .filter(|(_, _, id)| dist.contains_key(id))
                    .map(|(edge_id, edge_dist, (other_partition, other_vector))| {
                        EdgeType::InterEdge(
                            *edge_dist,
                            dist[&(other_partition, other_vector)].1,
                            edge_id,
                            closet_id.1,
                            other_vector,
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
                intra_graph
                    .0
                    .add_edge(intra_graph.1[vector_id], value_node_index, *dist);
                intra_graph.0.remove_edge(*edge_index);
                // todo!()
            }
            EdgeType::NeighborInternal(_, dist, edge_index, _, vector_id, partition_id) => {
                let (_, intra_graph) = &mut neighbors[id_to_graph[&**partition_id]];
                intra_graph.0.remove_edge(*edge_index);

                inter_graph.add_edge(
                    PartitionId(target.id),
                    *partition_id,
                    (
                        *dist,
                        (PartitionId(target.id), *vector_id),
                        (*partition_id, VectorId(value.id)),
                    ),
                );
            }
            EdgeType::InterEdge(_, dist, edge_index, _, vector_id, partition_id) => {
                inter_graph.add_edge(
                    PartitionId(target.id),
                    *partition_id,
                    (
                        *dist,
                        (PartitionId(target.id), *vector_id),
                        (*partition_id, VectorId(value.id)),
                    ),
                );
                inter_graph.0.remove_edge(*edge_index);
            }
        });

        match index < partition_splits[0] {
            true => {
                // new intra_edge
                intra_graph
                    .0
                    .add_edge(intra_graph.1[&closet_id.1], value_node_index, closet_dist);
            }
            false => {
                inter_graph.add_edge(
                    closet_id.0,
                    PartitionId(target.id),
                    (
                        closet_dist,
                        closet_id,
                        (PartitionId(target.id), VectorId(value.id)),
                    ),
                );
            }
        };
    }
    target.size += 1;

    Ok(())
}

pub async fn add<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    value: VectorEntry<A, B>,

    inter_graph: &mut RwLock<InterPartitionGraph<A>>,

    partition_buffer: &mut RwLock<
        DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, MAX_LOADED>,
    >,
    min_spanning_tree_buffer: &mut RwLock<DataBuffer<IntraPartitionGraph<A>, MAX_LOADED>>,

    meta_data: &mut RwLock<HashMap<Uuid, RwLock<Meta<A, B>>>>,
) -> Result<(), PartitionErr>
where
    A: PartialOrd + Ord,
    VectorSerial<A>: From<B>,
    A: for<'a> rkyv::Serialize<
        rancor::Strategy<
            rkyv::ser::Serializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::ser::sharing::Share,
            >,
            rancor::Error,
        >,
    >,
{
    let meta_data = &mut *meta_data.write().await;
    let mut partition_buffer = partition_buffer.write().await;
    let min_spanning_tree_buffer = &mut *min_spanning_tree_buffer.write().await;
    let inter_graph = &mut *inter_graph.write().await;

    // find closet id
    let closet_id = {
        let mut meta_data = meta_data.iter();
        let mut closest = {
            let (_, data) = meta_data.next().unwrap();

            let Meta { id, centroid, .. } = &*data.read().await;

            (*id, B::dist(centroid, &value.vector))
        };

        for (_, data) in meta_data {
            let Meta { id, centroid, .. } = &*data.read().await;

            let dist = B::dist(centroid, &value.vector);

            if closest.1 < dist {
                closest = (*id, dist)
            }
        }

        closest.0
    };
    // getting neighbor ids
    let neighbor_ids: Vec<PartitionId> = inter_graph
        .0
        .edges(inter_graph.1[&closet_id])
        .map(|edge| edge.weight())
        .map(|edge| [edge.1 .0, edge.2 .0])
        .flatten()
        .collect::<HashSet<PartitionId>>()
        .drain()
        .collect();

    // load partitions
    let partition_buffer = &mut *partition_buffer;

    let required_ids: Vec<&PartitionId> = [&closet_id]
        .into_iter()
        .chain(neighbor_ids.iter())
        .collect();

    // Allocate enough space by unloading least used resources
    {
        let required_ids: HashSet<&PartitionId> = required_ids.iter().map(|x| *x).collect();

        let not_loaded: usize = required_ids
            .iter()
            .filter(|x| !partition_buffer.contains(**x))
            .map(|x| *x)
            .count();

        if let Some(i1) = not_loaded.checked_sub(MAX_LOADED) {
            // unload i1 elements
            let unload_indexes: Vec<(usize, Uuid)> = partition_buffer
                .least_used_iter()
                .unwrap()
                .filter(|(index, id)| !required_ids.contains(&PartitionId(*id)))
                .take(i1)
                .collect();

            // sorting?
            for (index, id) in unload_indexes {
                partition_buffer.unload(&id);
            }
        }
    }

    let size = required_ids.len();
    let mut partitions = Vec::with_capacity(size);
    for id in required_ids {
        partitions.push(partition_buffer.access(&*id).await.unwrap());
    }

    let target_partition = partitions.remove(0);
    let mut target_partition = target_partition.write().await;

    let mut neighbor_partitions = Vec::with_capacity(size - 1);

    for partition in partitions.iter_mut().skip(1) {
        neighbor_partitions.push(partition.read().await);
    }
    let neighbor_partitions = neighbor_partitions;
    drop(partition_buffer);
    drop(size);

    // load min_spanning trees
    let min_spanning_trees = &mut *min_spanning_tree_buffer;

    let required_ids: Vec<&PartitionId> = [&closet_id]
        .into_iter()
        .chain(neighbor_ids.iter())
        .collect();

    // Allocate enough space by unloading least used resources
    {
        let required_ids: HashSet<&PartitionId> = required_ids.iter().map(|x| *x).collect();

        let not_loaded: usize = required_ids
            .iter()
            .filter(|x| !min_spanning_trees.contains(**x))
            .map(|x| *x)
            .count();

        if let Some(i1) = not_loaded.checked_sub(MAX_LOADED) {
            // unload i1 elements
            let unload_indexes: Vec<(usize, Uuid)> = min_spanning_trees
                .least_used_iter()
                .unwrap()
                .filter(|(index, id)| !required_ids.contains(&PartitionId(*id)))
                .take(i1)
                .collect();

            // sorting?
            for (index, id) in unload_indexes {
                min_spanning_trees.unload(&id);
            }
        }
    }

    let size = required_ids.len();
    let mut trees = Vec::with_capacity(size);
    for id in required_ids {
        trees.push(min_spanning_trees.access(&*id).await.unwrap());
    }

    let target_tree = trees.remove(0);
    let mut target_tree = target_tree.write().await;

    let mut neighbor_trees = Vec::with_capacity(size - 1);

    for tree in trees.iter_mut().skip(1) {
        neighbor_trees.push(tree.write().await);
    }
    let mut neighbor_trees = neighbor_trees;
    drop(min_spanning_trees);
    drop(min_spanning_tree_buffer);
    drop(size);

    add_into(
        &mut *target_partition,
        value,
        &mut *target_tree,
        inter_graph,
        &mut neighbor_partitions
            .iter()
            .map(|x| &**x)
            .map(|x| x)
            .zip(neighbor_trees.iter_mut().map(|x| &mut **x))
            .map(|(x, y)| (x, y))
            .collect::<Vec<(
                &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
                &mut IntraPartitionGraph<A>,
            )>>(),
    )
}

macro_rules! load_partitions {
    (
        $partition_buffer:expr,
        $inter_graph:expr,
        $closet_id:expr,
        $max_loaded:expr
    ) => {
        let neighbor_ids: Vec<PartitionId> = $inter_graph
            .0
            .edges($inter_graph.1[&$closet_id])
            .map(|edge| edge.weight())
            .map(|edge| [edge.1 .0, edge.2 .0])
            .flatten()
            .collect::<HashSet<PartitionId>>()
            .drain()
            .collect();

        let partition_buffer = &mut *$partition_buffer;

        let required_ids: Vec<&PartitionId> = [&$closet_id]
            .into_iter()
            .chain(neighbor_ids.iter())
            .collect();

        {
            let required_ids_set: HashSet<&PartitionId> = required_ids.iter().copied().collect();

            let not_loaded = required_ids_set
                .iter()
                .filter(|id| !partition_buffer.contains(***id))
                .count();

            if let Some(to_unload) = not_loaded.checked_sub($max_loaded) {
                let unload_indexes: Vec<(usize, Uuid)> = partition_buffer
                    .least_used_iter()
                    .unwrap()
                    .filter(|(_, id)| !required_ids_set.contains(&PartitionId(*id)))
                    .take(to_unload)
                    .collect();

                for (_, id) in unload_indexes {
                    partition_buffer.unload(&id);
                }
            }
        }

        let mut partitions = Vec::with_capacity(required_ids.len());
        for id in required_ids {
            partitions.push(partition_buffer.access(id).await.unwrap());
        }

        let target_partition = partitions.remove(0);
        let target_partition = target_partition.write().await;

        let mut neighbor_partitions = Vec::with_capacity(partitions.len());
        for partition in partitions {
            neighbor_partitions.push(partition.read().await);
        }

        (target_partition, neighbor_partitions)
    };
}

#[cfg(test)]
mod test {
    use petgraph::visit::EdgeRef;
    use rkyv::vec;
    use uuid::Uuid;

    use crate::{
        db::partition::{
            self, add::add_into, InterPartitionGraph, IntraPartitionGraph, Partition, PartitionId,
            VectorEntry, VectorId,
        },
        vector::{self, Vector, VectorSpace},
    };

    // test all cases
    // best Case:
    //  - insert into partition
    //  - Check vector in partition
    //  - Check if MST is correct structure (add one edge)
    #[test]
    fn insert_one_vertex() {
        let mut inter_graph = InterPartitionGraph::new();
        let mut partition = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph = IntraPartitionGraph::new();

        inter_graph.add_node(PartitionId(partition.id));

        // Insert into partition
        let vector = VectorEntry::from_uuid(Vector::splat(1.), Uuid::new_v4());

        let result = add_into(
            &mut partition,
            vector.clone(),
            &mut intra_graph,
            &mut inter_graph,
            &mut [],
        );

        assert!(result.is_ok());

        assert!(partition
            .vectors
            .iter()
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .any(|x| x.id == vector.id && x.vector == vector.vector));

        assert!(intra_graph.1.contains_key(&VectorId(vector.id)));

        assert!(inter_graph
            .0
            .node_weight(intra_graph.1[&VectorId(vector.id)])
            .is_some());

        assert_eq!(partition.centroid(), Vector::splat(1.));
    }

    #[test]
    fn insert_multiple_vertex() {
        let mut inter_graph = InterPartitionGraph::new();
        let mut partition = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph = IntraPartitionGraph::new();

        inter_graph.add_node(PartitionId(partition.id));

        // Insert into partition
        // TODO - replace with random inserts and check MST
        let vectors = (0..=10)
            .map(|i| VectorEntry::from_uuid(Vector::splat(i as f32), Uuid::new_v4()))
            .collect::<Vec<VectorEntry<f32, Vector<f32, 2>>>>();

        vectors.iter().for_each(|vector| {
            let result = add_into(
                &mut partition,
                vector.clone(),
                &mut intra_graph,
                &mut inter_graph,
                &mut [],
            );

            assert!(result.is_ok());
        });

        vectors.iter().for_each(|v| {
            assert!(partition
                .vectors
                .iter()
                .filter(|x| x.is_some())
                .map(|x| x.unwrap())
                .any(|x| x.vector == v.vector && x.id == v.id));
        });

        let (mut prev, mut cur, mut next) = (
            vectors.iter(),
            vectors.iter().skip(1),
            vectors.iter().skip(2),
        );

        prev.zip(cur)
            .zip(next)
            .map(|((prev, cur), next)| (prev, cur, next))
            .for_each(|(prev, cur, next)| {
                // check cur in prev edges
                intra_graph
                    .0
                    .edges(intra_graph.1[&VectorId(prev.id)])
                    .map(|edge| (edge.weight(), edge.target(), edge.source()))
                    .filter(|(_dist, source, target)| {
                        intra_graph.0.node_weight(*source).unwrap() == &VectorId(cur.id)
                            || intra_graph.0.node_weight(*target).unwrap() == &VectorId(cur.id)
                    })
                    .map(|(dist, source, target)| {
                        (dist, {
                            if intra_graph.0.node_weight(source).unwrap() == &VectorId(prev.id) {
                                intra_graph.0.node_weight(target).unwrap()
                            } else {
                                intra_graph.0.node_weight(source).unwrap()
                            }
                        })
                    })
                    .for_each(|(dist, other)| {
                        assert_eq!(dist, &Vector::dist(&prev.vector, &cur.vector));
                        assert_eq!(other, &VectorId(cur.id));
                    });
                // check prev and next in cur edges
                intra_graph
                    .0
                    .edges(intra_graph.1[&VectorId(cur.id)])
                    .map(|edge| (edge.weight(), edge.target(), edge.source()))
                    .for_each(|(dist, source, target)| {
                        let other = {
                            if intra_graph.0.node_weight(source).unwrap() == &VectorId(cur.id) {
                                intra_graph.0.node_weight(target).unwrap()
                            } else {
                                intra_graph.0.node_weight(source).unwrap()
                            }
                        };

                        assert!(other == &VectorId(prev.id) || other == &VectorId(next.id));

                        if other == &VectorId(prev.id) {
                            assert_eq!(dist, &Vector::dist(&prev.vector, &cur.vector));
                        } else {
                            assert_eq!(dist, &Vector::dist(&next.vector, &cur.vector));
                        }
                    });

                // check cur in next edges
                intra_graph
                    .0
                    .edges(intra_graph.1[&VectorId(next.id)])
                    .map(|edge| (edge.weight(), edge.target(), edge.source()))
                    .filter(|(_dist, source, target)| {
                        intra_graph.0.node_weight(*source).unwrap() == &VectorId(cur.id)
                            || intra_graph.0.node_weight(*target).unwrap() == &VectorId(cur.id)
                    })
                    .map(|(dist, source, target)| {
                        (dist, {
                            if intra_graph.0.node_weight(source).unwrap() == &VectorId(next.id) {
                                intra_graph.0.node_weight(target).unwrap()
                            } else {
                                intra_graph.0.node_weight(source).unwrap()
                            }
                        })
                    })
                    .for_each(|(dist, other)| {
                        assert_eq!(dist, &Vector::dist(&next.vector, &cur.vector));
                        assert_eq!(other, &VectorId(cur.id));
                    });
            });

        assert_eq!(partition.centroid(), Vector::splat(5.));
    }

    // foreign case 1:
    //  - nearest vector is in target partition
    //  - but must check neighbor

    #[test]
    fn foreign_neighbor_1() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new();
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new();
        inter_graph.add_node(PartitionId(partition_2.id));

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
        let vector = VectorEntry::from_uuid(Vector::splat(-1.), Uuid::new_v4());
        let result = add_into(
            &mut partition_2,
            vector.clone(),
            &mut intra_graph_2,
            &mut inter_graph,
            &mut [(&partition_1, &mut intra_graph_1)],
        );
        assert!(result.is_ok());

        {
            let size = inter_graph
                .0
                .edges(inter_graph.1[&PartitionId(partition_1.id)])
                .count();
            assert_eq!(1, size);

            let size = inter_graph
                .0
                .edges(inter_graph.1[&PartitionId(partition_2.id)])
                .count();
            assert_eq!(1, size);
        }

        inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(partition_1.id)])
            .map(|edge| (edge.weight(), edge.source(), edge.target()))
            .for_each(|((dist, vec_id_1, vec_id_2), _source, _target)| {
                let vec_1 = {
                    match intra_graph_1.1.contains_key(&vec_id_1.1) {
                        true => partition_1
                            .iter()
                            .filter(|vertex| vertex.id == *vec_id_1.1)
                            .next()
                            .unwrap(),
                        false => {
                            assert!(false);
                            panic!()
                        }
                    }
                };
                let vec_2 = {
                    match intra_graph_2.1.contains_key(&vec_id_2.1) {
                        true => partition_2
                            .iter()
                            .filter(|vertex| vertex.id == *vec_id_2.1)
                            .next()
                            .unwrap(),
                        false => {
                            assert!(false);
                            panic!()
                        }
                    }
                };

                assert_eq!(dist, &Vector::dist(&vec_1.vector, &vec_2.vector));
            });

        inter_graph
            .0
            .edges(inter_graph.1[&PartitionId(partition_2.id)])
            .map(|edge| (edge.weight(), edge.source(), edge.target()))
            .for_each(|((dist, vec_id_1, vec_id_2), _source, _target)| {
                let vec_1 = {
                    match intra_graph_1.1.contains_key(&vec_id_1.1) {
                        true => partition_1
                            .iter()
                            .filter(|vertex| vertex.id == *vec_id_1.1)
                            .next()
                            .unwrap(),
                        false => {
                            assert!(false);
                            panic!()
                        }
                    }
                };
                let vec_2 = {
                    match intra_graph_2.1.contains_key(&vec_id_2.1) {
                        true => partition_2
                            .iter()
                            .filter(|vertex| vertex.id == *vec_id_2.1)
                            .next()
                            .unwrap(),
                        false => {
                            assert!(false);
                            panic!()
                        }
                    }
                };

                assert_eq!(dist, &Vector::dist(&vec_1.vector, &vec_2.vector));
            });
    }

    // foreign case 2:
    //  - nearest vector is in target partition
    //  - but must check neighbor
    //  - Must replace inter_partition graph
    //  - Must replace intra_partition graph

    #[test]
    fn foreign_neighbor_2() {
        let mut inter_graph = InterPartitionGraph::new();

        let mut partition_1 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_1 = IntraPartitionGraph::new();
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new();
        inter_graph.add_node(PartitionId(partition_2.id));

        // Insert into partition
        {
            let vectors = (0..=10)
                .map(|i| VectorEntry::from_uuid(Vector([i as f32, -0.5]), Uuid::new_v4()))
                .collect::<Vec<VectorEntry<f32, Vector<f32, 2>>>>();

            vectors.iter().for_each(|vector| {
                let result = add_into(
                    &mut partition_1,
                    vector.clone(),
                    &mut intra_graph_1,
                    &mut inter_graph,
                    &mut [],
                );

                assert!(result.is_ok());
            });
        }
        {
            let vectors = (0..=10)
                .map(|i| VectorEntry::from_uuid(Vector([i as f32, 0.5]), Uuid::new_v4()))
                .collect::<Vec<VectorEntry<f32, Vector<f32, 2>>>>();

            vectors.iter().for_each(|vector| {
                let result = add_into(
                    &mut partition_2,
                    vector.clone(),
                    &mut intra_graph_2,
                    &mut inter_graph,
                    &mut [],
                );

                assert!(result.is_ok());
            });
        }
        {
            inter_graph.add_edge(
                PartitionId(partition_1.id),
                PartitionId(partition_2.id),
                (
                    0.5,
                    (
                        PartitionId(partition_1.id),
                        VectorId(
                            partition_1
                                .vectors
                                .iter()
                                .skip(4)
                                .next()
                                .unwrap()
                                .unwrap()
                                .id,
                        ),
                    ),
                    (
                        PartitionId(partition_2.id),
                        VectorId(
                            partition_2
                                .vectors
                                .iter()
                                .skip(4)
                                .next()
                                .unwrap()
                                .unwrap()
                                .id,
                        ),
                    ),
                ),
            );
        }

        let new_vector = VectorEntry::from_uuid(Vector([4., 0.]), Uuid::new_v4());
        let result = add_into(
            &mut partition_1,
            new_vector.clone(),
            &mut intra_graph_1,
            &mut inter_graph,
            &mut [(&partition_2, &mut intra_graph_2)],
        );
        assert!(result.is_ok());

        assert!(intra_graph_1.1.contains_key(&VectorId(new_vector.id)));

        {
            let edge = intra_graph_1
                .0
                .edges(intra_graph_1.1[&VectorId(new_vector.id)])
                .next()
                .unwrap();

            let (weight, source, target) = (
                edge.weight(),
                intra_graph_1.0.node_weight(edge.source()).unwrap(),
                intra_graph_1.0.node_weight(edge.target()).unwrap(),
            );

            assert_eq!(weight, &0.25);

            let other_vector_id = match **source == new_vector.id {
                true => target,
                false => source,
            };

            assert_eq!(
                *other_vector_id,
                VectorId(
                    partition_1
                        .vectors
                        .iter()
                        .skip(4)
                        .next()
                        .unwrap()
                        .unwrap()
                        .id
                )
            );
        };

        {
            let edge = inter_graph
                .0
                .edges(inter_graph.1[&PartitionId(partition_1.id)])
                .next()
                .unwrap();

            let ((dist, source_vec_id, target_vec_id), source, target) = (
                edge.weight(),
                intra_graph_1.0.node_weight(edge.source()).unwrap(),
                intra_graph_1.0.node_weight(edge.target()).unwrap(),
            );

            assert_eq!(dist, &0.25);

            let (partition_1_vec, partition_2_vec) = match **source == partition_1.id {
                true => (target, source),
                false => (source, target),
            };

            assert!(intra_graph_1.1.contains_key(partition_1_vec));
            let vec_1 = partition_1
                .iter()
                .filter(|vec| vec.id == **partition_1_vec)
                .next();

            assert!(vec_1.is_some());
            let vec_1 = vec_1.unwrap().vector;
            assert_eq!(vec_1, new_vector.vector);

            // check if partition_1_vec == Vector([4., 0])
            assert!(intra_graph_2.1.contains_key(partition_2_vec));
            let vec_2 = partition_2
                .iter()
                .filter(|vec| vec.id == **partition_2_vec)
                .next();

            assert!(vec_2.is_some());
            let vec_2 = vec_2.unwrap().vector;
            assert_eq!(
                vec_2,
                partition_2
                    .vectors
                    .iter()
                    .skip(4)
                    .next()
                    .unwrap()
                    .unwrap()
                    .vector
            );
        };
    }

    // foreign case 3:
    //  - nearest vector is in neighbor partition
    //  - Must replace inter_partition graph (new edge)
}
