use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem,
    sync::Arc,
};

use petgraph::{csr::DefaultIx, graph::EdgeIndex, visit::EdgeRef};

use tokio::sync::RwLock;
use tracing::{event, Level};
use uuid::Uuid;

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    rancor,
    rend::u32_le,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, DeserializeUnsized,
};

use crate::{
    db::{
        component::{
            data_buffer::{BufferError, DataBuffer},
            graph::{GraphSerial, IntraPartitionGraph},
            ids::{PartitionId, VectorId},
            meta::Meta,
            partition::{
                ArchivedVectorEntrySerial, Partition, PartitionErr, PartitionSerial, VectorEntry,
                VectorEntrySerial,
            },
        },
        operations::split::split_partition,
    },
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;

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

fn find_closet_vector<
    A: PartialEq + Clone + Copy + Field<A> + PartialOrd,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    vector: VectorEntry<A, B>,
    partitions: &[&Partition<A, B, PARTITION_CAP, VECTOR_CAP>],

    dist_map: &mut HashMap<(PartitionId, VectorId), A>,
) -> ((PartitionId, VectorId), A) {
    if cfg!(feature = "gpu_processing") {
        todo!()
    } else {
        partitions
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| ((PartitionId(x.id), VectorId(y.id)), y.vector))
            })
            .flatten()
            .map(|(id, vec)| {
                let dist = B::dist(&vector.vector, &vec);

                dist_map.insert(id, dist);

                (id, dist)
            })
            .min_by(
                |(id_1, dist_1), (id_2, dist_2)| match dist_1.partial_cmp(dist_2) {
                    Some(x) => x,
                    None => Ordering::Equal,
                },
            )
            .unwrap()
    }
}

fn add_into_partition<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + From<VectorSerial<A>>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    vector: VectorEntry<A, B>,
    partition: &mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
) -> Result<(), PartitionErr> {
    if partition.size + 1 >= PARTITION_CAP {
        return Err(PartitionErr::Overflow);
    };

    partition.vectors[partition.size] = Some(vector);

    partition.centroid = B::add(&partition.centroid, &vector.vector);

    partition.size = partition.size + 1;

    Ok(())
}

fn group_edges<A: PartialEq + Clone + Copy + Field<A>>(
    edges: Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
) -> HashMap<
    PartitionId,
    Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
> {
    let mut partition_map = HashMap::new();

    for edge in edges {
        let (idx, dist, (partition_id_1, vector_id_1), (partition_id_2, vector_id_2)) = edge;

        partition_map
            .entry(partition_id_2)
            .or_insert_with(Vec::new)
            .push((
                idx,
                dist,
                (partition_id_1, vector_id_1),
                (partition_id_2, vector_id_2),
            ));
    }

    partition_map
}

// fn update_inter_graph<A: PartialEq + Clone + Copy + Field<A> + PartialOrd>(
//     vector_id: VectorId,
//     source_partition_id: PartitionId,
//
//     check_edges: &[(A, (PartitionId, VectorId), (PartitionId, VectorId))],
//
//     inter_graph: &mut InterPartitionGraph<A>,
//
//     dist_map: HashMap<(PartitionId, VectorId), A>,
// ) -> Result<(), PartitionErr> {
//     // might be unnessary as we can assume all inserted values will from the same
//     let mut partition_groups: HashMap<
//         &PartitionId,
//         Vec<(&A, (&PartitionId, &VectorId), (&PartitionId, &VectorId))>,
//     > = HashMap::new();
//
//     check_edges.iter().for_each(|(d, (p1, v1), (p2, v2))| {
//         match partition_groups.contains_key(p1) {
//             true => {
//                 partition_groups
//                     .get_mut(p1)
//                     .expect("Key should exist")
//                     .push((d, (p1, v1), (p2, v2)));
//             }
//             false => {
//                 partition_groups.insert(p1, vec![(d, (p1, v1), (p2, v2))]);
//             }
//         }
// });
//
//     for (partition, edge_value) in partition_groups.into_iter() {
//         let replace_edges: Vec<(EdgeIndex, (PartitionId, VectorId))> = inter_graph
//             .0
//             .edges(inter_graph.1[&partition_id])
//             .map(|edge| (edge.id(), edge.weight()))
//             .filter_map(|(edge_index, (dist, id_1, id_2))| {
//                 edge_value.iter().any(|(dist, tmp_id_1, tmp_id_2)| {
//                     (id_1 == tmp_id_1 && id_2 == tmp_id_2) || (id_2 == tmp_id_1 && id_1 == tmp_id_2)
//                 });
//                 // match (id_1 == &(partition_id, target_vector_id), id_1.1 == target_vector_id, id_2.1 == target_vector_id) {
//                 //     (true, true, false) => Some((edge_index, (dist, id_2))),
//                 //     (false, false, true) => Some((edge_index, (dist, id_1))),
//
//                 //     (_, true, true) => panic!(),
//
//                 //     _ => None
//                 // }
//
//                 todo!()
//             })
//             .filter(|(_, (dist, id))| dist < &&dist_map[*id])
//             .map(|(edge_id, (_, id))| (edge_id, *id))
//             .collect();
//     }
//
//     // let replace_edges: Vec<(EdgeIndex, (PartitionId, VectorId))> = inter_graph.0.edges(inter_graph.1[&partition_id])
//     //     .map(|edge|(edge.id(), edge.weight()))
//     //     .filter_map(|(edge_index, (dist, id_1,id_2))| {
//     //         match (id_1 == &(partition_id, target_vector_id), id_1.1 == target_vector_id, id_2.1 == target_vector_id) {
//     //             (true, true, false) => Some((edge_index, (dist, id_2))),
//     //             (false, false, true) => Some((edge_index, (dist, id_1))),
//
//     //             (_, true, true) => panic!(),
//
//     //             _ => None
//     //         }
//     //     })
//     //     .filter(|(_, (dist, id))| {
//     //         dist < &&dist_map[*id]
//     //     })
//     //     .map(|(edge_id, (_, id))| (edge_id, *id))
//     //     .collect();
//
//     // // add new edges
//     // replace_edges.iter()
//     //     .map(|(_, (target_id))| (
//     //         dist_map[target_id],
//     //         *target_id,
//     //         (source_partition_id, vector_id),
//     //     ))
//     //     .for_each(|(dist, id_1, id_2)| {
//     //         inter_graph.add_edge(
//     //             id_1.0,
//     //             id_2.0,
//     //             (dist, id_1, id_2)
//     //         );
//     //     });
//
//     // // remove edges
//     // replace_edges.into_iter()
//     //     .map(|(edge_index, _)| edge_index)
//     //     .for_each(|edge_index| {
//     //         inter_graph.0.remove_edge(edge_index);
//     //     });
//
//     // Ok(())
//     todo!()
// }

fn update_local_min_span_tree<A: PartialEq + Clone + Copy + Field<A> + PartialOrd>(
    vector_id: VectorId,

    partition_id: PartitionId,

    target_vector_id: VectorId,

    inter_graph: &InterPartitionGraph<A>, // Used to determine all foreign edges that need to be updated
    min_spanning_graph: &mut IntraPartitionGraph<A>,

    dist_map: HashMap<VectorId, A>,
) -> Result<
    Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
    PartitionErr,
> where IntraPartitionGraph<A>: Debug {
    let node_index = min_spanning_graph.add_node(vector_id);

    // check if new inter_graph to replace intra_edges
    let replace_edges: Vec<(EdgeIndex, VectorId)> = min_spanning_graph
        .0
        .edges(match min_spanning_graph.1.get(&target_vector_id) {
            Some(val) => *val,
            None => {
                panic!("Failed to extract {target_vector_id:?} from {min_spanning_graph:#?} ")
            }
        })
        .map(|edge| {
            (
                edge.id(),
                min_spanning_graph.0.node_weight(edge.target()).unwrap(),
                min_spanning_graph.0.node_weight(edge.source()).unwrap(),
                edge.weight(),
            )
        })
        .map(
            |(edge_index, target, source, dist)| match target == &target_vector_id {
                true => (edge_index, source, dist),
                false => (edge_index, target, dist),
            },
        )
        .filter(|(_, id, dist)| dist < &&dist_map[*id])
        .map(|(edge_index, id, _)| (edge_index, *id))
        .collect();

    // add new edges
    replace_edges
        .iter()
        .map(|(_, target_id)| (dist_map[target_id], *target_id, vector_id))
        .for_each(|(dist, id_1, id_2)| {
            min_spanning_graph.add_edge(id_1, id_2, dist);
        });

    // remove edges
    replace_edges
        .into_iter()
        .map(|(edge_index, _)| edge_index)
        .for_each(|edge_index| {
            min_spanning_graph.0.remove_edge(edge_index);
        });

    min_spanning_graph.0.add_edge(
        node_index,
        min_spanning_graph.1[&target_vector_id],
        dist_map[&target_vector_id],
    );

    let replace_inter_edges: Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )> = inter_graph
        .0
        .edges(inter_graph.1[&partition_id])
        .map(|edge| (edge.id(), edge.weight()))
        .filter_map(|(idx, (dist, id_1, id_2))| {
            match (
                id_1 == &(partition_id, target_vector_id),
                id_2 == &(partition_id, target_vector_id),
            ) {
                (true, false) => Some((idx, *dist, *id_1, *id_2)),
                (false, true) => Some((idx, *dist, *id_2, *id_1)),
                _ => None,
            }
        })
        .collect();

    Ok(replace_inter_edges)
}

fn update_foreign_min_span_tree<A: PartialEq + Clone + Copy + Field<A> + PartialOrd>(
    source_id: PartitionId,
    vector_id: VectorId,

    target_partition_id: PartitionId,
    target_vector_id: VectorId,

    inter_graph: &mut InterPartitionGraph<A>, // Used to determine all foreign edges that need to be updated
    min_spanning_graph: &mut IntraPartitionGraph<A>,

    dist_map: HashMap<VectorId, A>,
) -> Result<
    Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )>,
    PartitionErr,
> where IntraPartitionGraph<A>: Debug {
    let replace_inter_edges: Vec<(
        EdgeIndex,
        A,
        (PartitionId, VectorId),
        (PartitionId, VectorId),
    )> = inter_graph
        .0
        .edges(inter_graph.1[&target_partition_id])
        .map(|edge| (edge.id(), edge.weight()))
        .filter_map(|(idx, (dist, id_1, id_2))| {
            match (
                id_1 == &(target_partition_id, target_vector_id),
                id_2 == &(target_partition_id, target_vector_id),
            ) {
                // (target_id, other)
                (true, false) => Some((idx, *dist, *id_1, *id_2)),
                (false, true) => Some((idx, *dist, *id_2, *id_1)),
                _ => None,
            }
        })
        .collect();

    // check if new inter_graph to replace intra_edges
    let replace_edges: Vec<(EdgeIndex, VectorId)> = min_spanning_graph
        .0
        .edges(
            match min_spanning_graph.1.get(&target_vector_id) {
                Some(val) => *val,
                None => {
                    panic!("Failed to extract {target_vector_id:?} from {min_spanning_graph:#?} ")
                }
            }
        )
        .map(|edge| {
            (
                edge.id(),
                min_spanning_graph.0.node_weight(edge.target()).unwrap(),
                min_spanning_graph.0.node_weight(edge.source()).unwrap(),
                edge.weight(),
            )
        })
        .map(
            |(edge_index, target, source, dist)| match target == &target_vector_id {
                true => (edge_index, source, dist),
                false => (edge_index, target, dist),
            },
        )
        .filter(|(_, id, dist)| dist < &&dist_map[*id])
        .map(|(edge_index, id, _)| (edge_index, *id))
        .collect();

    // add new edges
    replace_edges
        .iter()
        .map(|(_, target_id)| (dist_map[target_id], *target_id))
        .for_each(|(dist, id_1)| {
            inter_graph.add_edge(
                source_id,
                target_partition_id,
                (dist, (target_partition_id, id_1), (source_id, vector_id)),
            );
        });

    // remove edges
    replace_edges
        .into_iter()
        .map(|(edge_index, _)| edge_index)
        .for_each(|edge_index| {
            min_spanning_graph.0.remove_edge(edge_index);
        });

    Ok(replace_inter_edges)
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

macro_rules! resolve_buffer {
    ($buffer:expr, $id:expr) => {{
        match $buffer.access(&$id).await {
            Ok(partition) => partition,
            Err(_) => {
                let mut least_used = $buffer.least_used_iter().await.unwrap();

                loop {
                    event!(Level::DEBUG, "Attempt get least used");
                    let Some(next_unload) = least_used.next() else {
                        
                        event!(Level::DEBUG, "Restarting least used iter");
                        least_used = $buffer.least_used_iter().await.unwrap();
                        continue;
                    };

                    event!(Level::DEBUG, "Filtering values any value that is equal to load goal");
                    if $id == PartitionId(next_unload.1) {
                        continue;
                    }
                    
                    event!(Level::DEBUG, "Attempt to unload({:?}) & load({:?})", next_unload.1, $id);
                    if let Err(err) = $buffer.unload_and_load(&next_unload.1, &$id).await {
                        event!(Level::DEBUG, "Err({err:?})");
                        continue;
                    };

                    
                    event!(Level::DEBUG, "Break loop and return");
                    break $buffer.access(&$id).await.unwrap();
                }
            }
        }
    }};
}

pub async fn add<
    A: PartialEq
        + PartialOrd
        + Clone
        + Copy
        + Field<A>
        + for<'a> rkyv::Serialize<
            rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rancor::Error,
            >,
        > + Debug,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + From<VectorSerial<A>> + Debug,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    value: VectorEntry<A, B>,

    inter_graph: Arc<RwLock<InterPartitionGraph<A>>>,

    partition_buffer: Arc<
        RwLock<
            DataBuffer<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionSerial<A>, MAX_LOADED>,
        >,
    >,
    min_spanning_tree_buffer: Arc<
        RwLock<DataBuffer<IntraPartitionGraph<A>, GraphSerial<A>, MAX_LOADED>>,
    >,

    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
) -> Result<(), PartitionErr>
where
    VectorSerial<A>: From<B>,
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<A>]:
        DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
{
    event!(Level::INFO, "🔥 Adding vector: {value:?}");

    let meta_data = &mut *meta_data.write().await;
    event!(Level::DEBUG, "🔒 Locked `meta_data`");

    let mut w_partition_buffer = partition_buffer.write().await;
    event!(Level::DEBUG, "🔒 Locked `partition_buffer`");

    let mut w_min_spanning_tree_buffer = min_spanning_tree_buffer.write().await;
    event!(Level::DEBUG, "🔒 Locked `min_spanning_tree_buffer`");

    let inter_graph = &mut *inter_graph.write().await;
    event!(Level::DEBUG, "🔒 Locked `inter_graph`");

    // find closet id
    let (mut closet_partition_id, closet_size) = {
        event!(Level::INFO, "✨ Finding closest partition");
        let mut meta_data = meta_data.iter();
        let mut closet_size = 0;
        let mut closest = {
            let (_, data) = meta_data.next().unwrap();

            let Meta {
                id, centroid, size, ..
            } = &*data.read().await;

            closet_size = *size;

            (*id, B::dist(centroid, &value.vector))
        };

        for (_, data) in meta_data {
            let Meta {
                id, centroid, size, ..
            } = &*data.read().await;

            let dist = B::dist(centroid, &value.vector);

            if closest.1 < dist {
                closet_size = *size;

                closest = (*id, dist)
            }
        }

        (closest.0, closet_size)
    };
    event!(
        Level::INFO,
        "💎 Closest partition: {closet_partition_id:?} - {closet_size:?}"
    );

    if closet_size == 0 {
        let partition_buffer = &mut *w_partition_buffer;

        let partition = resolve_buffer!(partition_buffer, closet_partition_id);

        let Some(partition) = &mut *partition.write().await else {
            todo!()
        };
        add_into_partition(value.clone(), partition)
            .expect("Unable to insert value into empty partition");

        let tree_buffer = &mut *w_min_spanning_tree_buffer;

        let tree = resolve_buffer!(tree_buffer, closet_partition_id);

        let Some(tree) = &mut *tree.write().await else {
            todo!()
        };

        tree.add_node(VectorId(value.id));

        let meta = &mut *meta_data[&*closet_partition_id].write().await;

        meta.size = partition.size;
        meta.centroid = partition.centroid();

        return Ok(());
    }

    // getting neighbor ids
    let neighbor_ids: Vec<PartitionId> = inter_graph
        .0
        .edges(inter_graph.1[&closet_partition_id])
        .map(|edge| edge.weight())
        .map(|edge| (edge.1 .0, edge.2 .0))
        .map(
            |(partition_id_1, partition_id_2)| match partition_id_1 == closet_partition_id {
                true => partition_id_2,
                false => partition_id_1,
            },
        )
        // .flatten()
        .collect::<HashSet<PartitionId>>()
        .drain()
        // .filter()
        .collect();
    event!(Level::INFO, "🤝 Neighbors: {neighbor_ids:?}");

    // get closet vectors in partition & generate dist_map
    let mut dist_map = HashMap::new();

    let required_ids: Vec<&PartitionId> = [&closet_partition_id]
        .into_iter()
        .chain(neighbor_ids.iter())
        .collect();

    let mut closet_vector_id = {
        let partition_buffer = &mut *w_partition_buffer;

        event!(Level::DEBUG, "📦 Finding closet vector");
        let mut required_partitions: HashSet<&PartitionId> =
            required_ids.clone().into_iter().collect();

        let mut closet_id = None;
        let mut closet_dist = None;

        loop {
            let mut acquired_partitions = Vec::new();
            event!(Level::DEBUG, "Attempt to acquired required ids that aren't loaded");
            for id in required_partitions.iter() {
                // Replace with try access and/or batch access
                let Ok(partition) = partition_buffer.access(*id).await else {
                    continue;
                };

                acquired_partitions.push(partition);
            }

            let mut acquired_partitions_locks = Vec::new();
            {
                for partition in acquired_partitions.iter() {
                    if let Ok(lock) = partition.try_read() {
                        acquired_partitions_locks.push(lock);
                    }
                }
            }
            

            let mut partitions = Vec::new();
            {
                for partition in acquired_partitions_locks.iter() {
                    if let Some(inner) = &**partition {
                        partitions.push(inner);
                    }
                }
            }
            // possible future optimization to remove size of dist map
            let (id, dist) = find_closet_vector(value, partitions.as_slice(), &mut dist_map);

            (closet_id, closet_dist) = match closet_id {
                None => (Some(id), Some(dist)),
                Some(_current_id) => {
                    if dist < closet_dist.unwrap() {
                        (Some(id), Some(dist))
                    } else {
                        (closet_id, closet_dist)
                    }
                }
            };

            // let mut remove_index: Vec<_> =
            partitions
                .iter()
                .map(|partition| PartitionId(partition.id))
                .for_each(|id| {
                    required_partitions.remove(&id);
                });

            if required_partitions.len() == 0 {
                break;
            }
            partitions = Vec::new();
            acquired_partitions_locks = Vec::new();
            acquired_partitions = Vec::new();
            // while let Some(_) = partitions.pop() {};
            // while let Some(_) = acquired_partitions_locks.pop() {};
            // while let Some(_) = acquired_partitions.pop() {};

            //unload and swap
            let mut least_used = partition_buffer
                .least_used_iter()
                .await
                .unwrap()
                .filter(|(_index, id)| !required_partitions.contains(&PartitionId(*id)));
            // .collect();

            for id in required_partitions.clone() {
                match partition_buffer.load(&*id).await {
                    Ok(_) => {
                        event!(Level::DEBUG, "📦 Load buffer space");
                        // partitions.push(partition_buffer.access(&*id).await.unwrap());
                    }
                    Err(BufferError::OutOfSpace) => {
                        event!(Level::DEBUG, "📦 Unload and Load buffer space");
                        partition_buffer
                            .unload_and_load(&least_used.next().unwrap().1, &*id)
                            .await
                            .unwrap();
                        // partitions.push(partition_buffer.access(&*id).await.unwrap());
                    }
                    Err(BufferError::FileNotFound) => {
                        todo!()
                    }
                    Err(_) => todo!(),
                };
            }
        }

        
        event!(Level::DEBUG, "📦 Found closet vector");
        closet_id.unwrap()
    };

    // insert into partition
    {
        let partition_buffer = &mut *w_partition_buffer;

        let partition = resolve_buffer!(partition_buffer, closet_partition_id);

        let Some(partition) = &mut *partition.write().await else {
            todo!()
        };

        if closet_size + 1 >= PARTITION_CAP {
            // pre_mature split operation so don't have to do it after add_into_partition fails
        }

        let result = add_into_partition(value.clone(), partition);

        'add_into_partition: {
            event!(Level::DEBUG, "Add result: {result:?}");
            if let Err(_) = result {
                event!(Level::DEBUG, "Splitting partition");

                let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

                let min_span_tree = resolve_buffer!(min_span_tree_buffer, closet_partition_id);
                let min_span_tree = &mut *min_span_tree.write().await;
                let Some(min_span_tree) = min_span_tree else {
                    todo!()
                };

                let Ok(mut new_splits) = split_partition(partition, min_span_tree, 2, inter_graph)
                else {
                    panic!()
                };

                event!(Level::DEBUG, "New split size: {}", new_splits.len());

                event!(Level::DEBUG, "OG graph: {min_span_tree:#?}");


                {
                    let (new_partition, new_graph) = new_splits.pop().unwrap();

                    *partition = new_partition;
                    
                    *min_span_tree = new_graph;
                    event!(Level::DEBUG, "partition - 1: {partition:?}");
                    event!(Level::DEBUG, "New graph - 1: {min_span_tree:#?}");
                }

                let (mut new_partition, new_graph) = new_splits.pop().unwrap();
                event!(Level::DEBUG, "partition - 2: {new_partition:?}");
                event!(Level::DEBUG, "New graph - 2: {new_graph:#?}");
                event!(Level::DEBUG, "inter: {inter_graph:#?}");
                

                //add into partition or new_partition
                new_partition.iter()
                    .map(|vector| VectorId(vector.id))
                    .for_each(|vector_id| {
                        dist_map.insert(
                            (PartitionId(new_partition.id), vector_id),
                            dist_map[&(PartitionId(partition.id), vector_id)]
                        );
                        dist_map.remove(&(PartitionId(partition.id), vector_id));
                    });

                let dist_old = B::dist(&value.vector, &partition.centroid());
                let dist_new = B::dist(&value.vector, &new_partition.centroid());

                event!(
                    Level::DEBUG,
                    "closet_vector_id = {closet_vector_id:?}"
                );
                event!(
                    Level::DEBUG,
                    target_partition_id = %partition.id,
                    target_partition_size = partition.size,
                    new_partition_id = %new_partition.id,
                    new_partition_size = new_partition.size,
                    "Calculated distances for new and target partitions"
                );

                if dist_old < dist_new {
                    add_into_partition(value, partition)
                        .expect("Partition should now have space but an error occurred");
                } else {
                    add_into_partition(value, &mut new_partition)
                        .expect("Partition should now have space but an error occurred");
                    let new_id = (PartitionId(new_partition.id), closet_vector_id.1);
                    event!(
                        Level::DEBUG,
                        "closet_vector_id:{closet_vector_id:?}\tcloset_partition_id:{closet_partition_id:?}\tChanged closet id from = {closet_vector_id:?} -> {new_id:?}"
                    );

                    if closet_vector_id.0 == closet_partition_id {
                        closet_partition_id = PartitionId(new_partition.id);
                        closet_vector_id = new_id;
                    }
                }

                // if closet_vector_id.0 == closet_partition_id {
                //     if let Err(_) = partition_buffer.push(new_partition.clone()).await {
                //         event!(
                //             Level::WARN,
                //             "Partition buffer full, unloading least-used partition"
                //         );
                //         let mut iter = partition_buffer.least_used_iter().await.unwrap();
                //         loop {
                //             let (_index, id) = match iter.next() {
                //                 Some(val) => val,
                //                 None => {
                //                     iter = partition_buffer.least_used_iter().await.unwrap();
// 
                //                     continue;
                //                 }
                //             };
// 
                //             match partition_buffer.unload_and_push(&id, new_partition.clone()).await {
                //                 Ok(_) => {
                //                     break;
                //                 }
                //                 Err(err) => {
                //                     event!(
                //                         Level::WARN,
                //                         "Buffer error in attempt to insert new split partition := {err:?}"
                //                     );
                //                 }
                //             }
                //         }
                //     }
                //     if let Err(_) = min_span_tree_buffer.push(new_graph.clone()).await {
                //         event!(
                //             Level::WARN,
                //             "tree buffer full, unloading least-used tree"
                //         );
                //         let mut iter = min_span_tree_buffer.least_used_iter().await.unwrap();
                //         loop {
                //             let (_index, id) = match iter.next() {
                //                 Some(val) => val,
                //                 None => {
                //                     iter = min_span_tree_buffer.least_used_iter().await.unwrap();
// 
                //                     continue;
                //                 }
                //             };
// 
                //             match min_span_tree_buffer.unload_and_push(&id, new_graph.clone()).await {
                //                 Ok(_) => {
                //                     break;
                //                 }
                //                 Err(err) => {
                //                     event!(
                //                         Level::WARN,
                //                         "Buffer error in attempt to insert new split partition := {err:?}"
                //                     );
                //                 }
                //             }
                //         }
                //     }
// 
                    // 
// 
                //     break 'add_into_partition;
                // }
                meta_data.insert(
                    new_partition.id,
                    Arc::new(RwLock::new(Meta::new(
                        PartitionId(new_partition.id),
                        new_partition.size,
                        new_partition.centroid(),
                    ))),
                );
                if let Err(_) = partition_buffer.push(new_partition.clone()).await {
                    event!(
                        Level::WARN,
                        "Partition buffer full, unloading least-used partition"
                    );
                    let (_index, id) = partition_buffer
                        .least_used_iter()
                        .await
                        .unwrap()
                        .next()
                        .unwrap();

                    partition_buffer
                        .unload_and_push(&id, new_partition)
                        .await
                        .unwrap();
                }
                if let Err(_) = min_span_tree_buffer.push(new_graph.clone()).await {
                    event!(
                        Level::WARN,
                        "Partition buffer full, unloading least-used partition"
                    );
                    let (_index, id) = min_span_tree_buffer
                        .least_used_iter()
                        .await
                        .unwrap()
                        .next()
                        .unwrap();

                    min_span_tree_buffer
                        .unload_and_push(&id, new_graph)
                        .await
                        .unwrap();
                }
            };
        }

        // update original partition
        {
            let target_meta = &mut *meta_data[&*closet_partition_id].write().await;

            target_meta.size = partition.size;
            target_meta.centroid = partition.centroid();
        }
    }

    //update min_spanning
    {
        let min_span_tree_buffer = &mut *w_min_spanning_tree_buffer;

        let min_span_tree = resolve_buffer!(min_span_tree_buffer, closet_vector_id.0);
        let Some(min_span_tree) = &mut *min_span_tree.write().await else {
            todo!();
        };

        let update_inter_graph_edges = match closet_vector_id.0 == closet_partition_id {
            true => {
                update_local_min_span_tree(
                    VectorId(value.id),
                    closet_vector_id.0,
                    closet_vector_id.1,
                    &inter_graph,
                    min_span_tree,
                    // filter and collection operation is unnecessary
                    dist_map
                        .iter()
                        .filter(|((partition_id, vector_id), _dist)| {
                            partition_id == &closet_vector_id.0
                        })
                        .map(|((partition_id, vector_id), dist)| (*vector_id, *dist))
                        .collect(),
                )
            }
            false => {
                update_foreign_min_span_tree(
                    closet_partition_id,
                    VectorId(value.id),
                    closet_vector_id.0,
                    closet_vector_id.1,
                    inter_graph,
                    min_span_tree,
                    // filter and collection operation is unnecessary
                    dist_map
                        .iter()
                        .filter(|((partition_id, vector_id), _dist)| {
                            partition_id == &closet_vector_id.0
                        })
                        .map(|((partition_id, vector_id), dist)| (*vector_id, *dist))
                        .collect(),
                )
            }
        }
        .unwrap();

        // update_inter_graph.iter()
        //     .filter(|(dist, _, _)| )

        // Vec<(A, (PartitionId, VectorId), (PartitionId, VectorId))>

        {
            let partition_buffer = &mut *w_partition_buffer;

            let mut edges_partitions = group_edges(update_inter_graph_edges);
            loop {
                let mut remove_key = Vec::new();
                edges_partitions.iter().for_each(|(partition, edges)| {
                    if !dist_map.contains_key(&edges[0].3) {
                        return;
                    }

                    edges
                        .iter()
                        .filter(|(_, dist, _, id)| dist < &dist_map[id])
                        .for_each(|(idx, _, _, id_2)| {
                            inter_graph.add_edge(
                                id_2.0,
                                closet_vector_id.0,
                                (dist_map[id_2], *id_2, closet_vector_id),
                            );

                            inter_graph.0.remove_edge(*idx);
                        });

                    remove_key.push(*partition);
                });

                remove_key.into_iter().for_each(|key| {
                    edges_partitions.remove_entry(&key);
                });

                if edges_partitions.len() <= 0 {
                    break;
                }

                event!(Level::DEBUG, "remaining keys: {edges_partitions:?}");

                let partition_id = edges_partitions.iter().map(|(key, _)| *key).next().unwrap();

                event!(Level::DEBUG, "Load: {partition_id:?}");

                let partition = resolve_buffer!(partition_buffer, partition_id);
                event!(Level::DEBUG, "Just resolved buffer");

                let Some(partition) = &*partition.read().await else {
                    todo!()
                };

                partition
                    .iter()
                    .map(|y| ((PartitionId(partition.id), VectorId(y.id)), y.vector))
                    .for_each(|(id, vec)| {
                        let dist = B::dist(&value.vector, &vec);

                        dist_map.insert(id, dist);
                    });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use petgraph::visit::EdgeRef;
    use uuid::Uuid;

    use crate::{
        db::{
            self,
            component::{
                graph::{InterPartitionGraph, IntraPartitionGraph},
                ids::{PartitionId, VectorId},
                partition::{Partition, VectorEntry},
            },
            operations::add::add_into,
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
        let mut intra_graph = IntraPartitionGraph::new(PartitionId(partition.id));

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
        let mut intra_graph = IntraPartitionGraph::new(PartitionId(partition.id));

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
        let mut intra_graph_1 = IntraPartitionGraph::new(PartitionId(partition_1.id));
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new(PartitionId(partition_2.id));
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
        let mut intra_graph_1 = IntraPartitionGraph::new(PartitionId(partition_1.id));
        inter_graph.add_node(PartitionId(partition_1.id));

        let mut partition_2 = Partition::<f32, Vector<f32, 2>, 500, 500>::new();
        let mut intra_graph_2 = IntraPartitionGraph::new(PartitionId(partition_2.id));
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
