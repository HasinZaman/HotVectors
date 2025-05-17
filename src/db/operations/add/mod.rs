use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    sync::Arc,
};

use spade::{DelaunayTriangulation, HasPosition, Triangulation};
use tokio::sync::RwLock;
use uuid::Uuid;

use rkyv::rancor;

#[cfg(feature = "benchmark")]
use crate::db::component::benchmark::Benchmark;
use crate::{
    db::component::{
        ids::{PartitionId, VectorId},
        meta::Meta,
    },
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::InterPartitionGraph;

pub mod batch;
pub mod single;

// local

async fn create_local_meta<A: Field<A>, B: VectorSpace<A> + Clone>(
    meta_data: &HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,
    acquired_partitions: &HashSet<PartitionId>,
) -> HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>
where
    Meta<A, B>: Clone,
{
    let mut local_meta_data: HashMap<Uuid, Arc<RwLock<Meta<A, B>>>> = HashMap::new();
    for partition_id in acquired_partitions.iter() {
        let data: &Meta<A, B> = &*meta_data[&**partition_id].read().await;

        // RwLock may not be required as no data needs to be synced over multiple threads
        // potentially be required in the future to multi-thread mst updating
        local_meta_data.insert(**partition_id, Arc::new(RwLock::new(data.clone())));
    }
    local_meta_data
}

fn create_local_inter_graph<
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
        > + Debug
        + Extremes,
>(
    acquired_partitions: &HashSet<PartitionId>,
    inter_graph: &InterPartitionGraph<A>,
) -> InterPartitionGraph<A> {
    let mut local_inter_graph: InterPartitionGraph<A> = InterPartitionGraph::new();
    for partition_id in acquired_partitions.iter() {
        local_inter_graph.add_node(*partition_id);
    }
    let mut inserted_edges: HashSet<((PartitionId, VectorId), (PartitionId, VectorId))> =
        HashSet::new();
    for partition_id in acquired_partitions.iter() {
        for edge_ref in inter_graph.0.edges(inter_graph.1[partition_id]) {
            let (weight, start, end) = edge_ref.weight();

            if inserted_edges.contains(&(*start, *end)) || inserted_edges.contains(&(*end, *start))
            {
                continue;
            }
            if !acquired_partitions.contains(&start.0) {
                continue;
            }
            if !acquired_partitions.contains(&end.0) {
                continue;
            }

            inserted_edges.insert((*start, *end));
            inserted_edges.insert((*end, *start));

            local_inter_graph.add_edge(start.0, end.0, (*weight, *start, *end));
        }
    }
    local_inter_graph
}

// neighbors
pub(super) struct DelaunayVertex<A: Into<f32>, B: HasPosition<Scalar = f32>> {
    id: PartitionId,
    vertex: B,
    _phantom: PhantomData<A>,
}
impl<A: Debug + Into<f32>, B: HasPosition<Scalar = f32>> HasPosition for DelaunayVertex<A, B> {
    type Scalar = f32;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        self.vertex.position()
    }
}

pub(super) async fn get_neighbors<
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
        > + Debug
        + Extremes,
    B: VectorSpace<A>
        + Sized
        + Clone
        + Copy
        + PartialEq
        + Extremes
        + From<VectorSerial<A>>
        + Debug
        + HasPosition<Scalar = f32>,
    const VECTOR_CAP: usize,
>(
    inter_graph: &InterPartitionGraph<A>,
    closet_partition_id: PartitionId,
    meta_data: &HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>,
) -> Vec<PartitionId>
where
    f32: From<A>,
{
    let graph_neighbors = inter_graph
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
        .collect();

    if inter_graph.1.len() < 3 {
        return graph_neighbors;
    }

    let mut triangulation: DelaunayTriangulation<DelaunayVertex<A, B>> =
        DelaunayTriangulation::new();

    if VECTOR_CAP > 2 {
        // let data: Vec<Vec<f32>> = Vec::new();

        // for (id, meta_data) in meta_data {
        //     let meta_data = &*meta_data.read().await;

        //     let centroid = meta_data.centroid;

        //     let vertex = DelaunayVertex {
        //         id: PartitionId(*id),
        //         vertex: centroid,
        //         _phantom: PhantomData::<A>,
        //     };
        //     data.append(vertex.into());
        // }

        // let model = umap(data);

        // let transformed = model.transform(data);

        // for data in transformed {
        //     let _ = triangulation.insert(data);
        // }
        todo!()
    } else {
        for (id, meta) in meta_data {
            let meta = &*meta.read().await;

            let centroid = meta.centroid;

            let vertex = DelaunayVertex {
                id: PartitionId(*id),
                vertex: centroid,
                _phantom: PhantomData::<A>,
            };
            let _ = triangulation.insert(vertex);
        }
    }

    triangulation
        .inner_faces()
        .filter(|face_handle| {
            let vertices = face_handle.vertices();

            vertices
                .iter()
                .any(|vertex| vertex.data().id == closet_partition_id)
        })
        .flat_map(|face_handle| {
            face_handle
                .adjacent_edges()
                .iter()
                .map(|edge_handle| [edge_handle.from().data().id, edge_handle.to().data().id])
                .filter(|[from, to]| from == &closet_partition_id || to == &closet_partition_id)
                .flatten()
                .collect::<Vec<_>>()
        })
        .chain(graph_neighbors.into_iter())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn get_required_partitions<
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
        > + Debug
        + Extremes,
>(
    sources: &[PartitionId],
    sink: &PartitionId,
    inter_graph: &InterPartitionGraph<A>,
) -> HashSet<PartitionId> {
    let mut required_partitions = HashSet::new();

    for source in sources {
        if source == sink {
            continue;
        }

        let (_, path) = inter_graph
            .find_trail(*source, *sink)
            .expect("Err in finding trail")
            .expect(&format!(
                "Couldn't find a trail from {:?} -> {:?}",
                source, sink
            ));

        path.iter().for_each(|((partition_id, _), ..)| {
            required_partitions.insert(*partition_id);
        });
    }
    required_partitions.insert(*sink);

    required_partitions
}

// find better name
pub(super) fn expand<
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
        > + Debug
        + Extremes,
>(
    sources: &[PartitionId],
    inter_graph: &InterPartitionGraph<A>,
) -> HashSet<PartitionId> {
    let mut partitions: HashSet<PartitionId> = sources.iter().map(|x| *x).collect();

    for source in sources {
        for edge_ref in inter_graph.0.edges(inter_graph.1[source]) {
            let (_, (partition_id_1, _), (partition_id_2, _)) = edge_ref.weight();

            partitions.insert(*partition_id_1);
            partitions.insert(*partition_id_2);
        }
    }

    partitions
}
