use std::{cmp::Ordering, collections::HashMap, fmt::Debug, str::FromStr};

use petgraph::{
    csr::DefaultIx, graph::NodeIndex, prelude::StableGraph, visit::EdgeRef, Undirected,
};
use rancor::Strategy;
use uuid::Uuid;

use crate::{
    db::component::ids::{PartitionId, VectorId},
    vector::Field,
};
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    from_bytes,
    rend::u32_le,
    string::ArchivedString,
    to_bytes,
    tuple::{ArchivedTuple2, ArchivedTuple3},
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, DeserializeUnsized, Serialize,
};

use super::serial::FileExtension;

#[derive(Debug, Clone)]
pub struct InterPartitionGraph<A: Field<A>>(
    pub  StableGraph<
        PartitionId,
        (A, (PartitionId, VectorId), (PartitionId, VectorId)),
        Undirected,
        DefaultIx,
    >,
    pub HashMap<PartitionId, NodeIndex<DefaultIx>>,
);

impl<A: Field<A> + Clone + Copy> InterPartitionGraph<A> {
    pub fn new() -> Self {
        Self(StableGraph::default(), HashMap::new())
    }
    pub async fn load(path: &str) -> Self
    where
        A: Archive,
        [ArchivedTuple3<
            u32_le,
            u32_le,
            ArchivedTuple3<
                <A as Archive>::Archived,
                ArchivedTuple2<ArchivedString, ArchivedString>,
                ArchivedTuple2<ArchivedString, ArchivedString>,
            >,
        >]: DeserializeUnsized<
            [(
                usize,
                usize,
                (
                    A,
                    (std::string::String, std::string::String),
                    (std::string::String, std::string::String),
                ),
            )],
            Strategy<Pool, rancor::Error>,
        >,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    {
        let bytes = tokio::fs::read(&path).await.unwrap();

        from_bytes::<GraphSerial<(A, (String, String), (String, String))>, rancor::Error>(&bytes)
            .unwrap()
            .into()
    }

    pub async fn save(&self, dir: &str, name: &str)
    where
        A: Archive
            + for<'a> rkyv::Serialize<
                rancor::Strategy<
                    rkyv::ser::Serializer<
                        rkyv::util::AlignedVec,
                        rkyv::ser::allocator::ArenaHandle<'a>,
                        rkyv::ser::sharing::Share,
                    >,
                    rancor::Error,
                >,
            >,
        [ArchivedTuple3<
            u32_le,
            u32_le,
            ArchivedTuple3<
                <A as Archive>::Archived,
                ArchivedTuple2<ArchivedString, ArchivedString>,
                ArchivedTuple2<ArchivedString, ArchivedString>,
            >,
        >]: DeserializeUnsized<
            [(
                usize,
                usize,
                (
                    A,
                    (std::string::String, std::string::String),
                    (std::string::String, std::string::String),
                ),
            )],
            Strategy<Pool, rancor::Error>,
        >,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    {
        let serial_data: InterGraphSerial<A> = self.clone().into();

        let bytes = to_bytes::<rancor::Error>(&serial_data).unwrap();

        tokio::fs::write(
            &format!("{dir}//{name}.{}", InterGraphSerial::<A>::extension()),
            bytes.as_slice(),
        )
        .await
        .unwrap();
    }

    pub fn add_node(&mut self, node: PartitionId) -> NodeIndex {
        let idx = self.0.add_node(node);
        self.1.insert(node, idx);
        idx
    }

    pub fn remove_node(&mut self, node: PartitionId) {
        if let Some(idx) = self.1.remove(&node) {
            self.0.remove_node(idx);
        }
    }

    pub fn add_edge(
        &mut self,
        node_1: PartitionId,
        node_2: PartitionId,
        weight: (A, (PartitionId, VectorId), (PartitionId, VectorId)),
    ) {
        let idx_1 = self.1[&node_1];
        let idx_2 = self.1[&node_2];
        self.0.add_edge(idx_1, idx_2, weight);
    }

    pub fn remove_edge(
        &mut self,
        node_1: (PartitionId, VectorId),
        node_2: (PartitionId, VectorId),
    ) -> Result<(), ()> {
        let Some((_, edge_idx)) = self
            .0
            .edges(
                *self
                    .1
                    .get(&node_1.0)
                    .unwrap_or_else(|| self.1.get(&node_2.0).unwrap()),
            )
            .map(|edge_ref| (edge_ref.weight(), edge_ref.id()))
            .filter(|((_, id_1, id_2), _)| {
                (id_1 == &node_1 && id_2 == &node_2) || (id_2 == &node_1 && id_1 == &node_2)
            })
            .next()
        else {
            return Err(());
        };

        self.0.remove_edge(edge_idx);

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct IntraPartitionGraph<A: Field<A> + Debug>(
    pub StableGraph<VectorId, A, Undirected, DefaultIx>,
    pub HashMap<VectorId, NodeIndex<DefaultIx>>,
    pub PartitionId,
);

impl<A: Field<A> + Clone + Copy + Debug> IntraPartitionGraph<A> {
    pub fn new(id: PartitionId) -> Self {
        Self(StableGraph::default(), HashMap::new(), id)
    }
    pub async fn load(path: &str) -> Self
    where
        A: Archive + Clone + Copy,
        [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
            DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    {
        let bytes = tokio::fs::read(&path).await.unwrap();

        from_bytes::<GraphSerial<A>, rancor::Error>(&bytes)
            .unwrap()
            .into()
    }

    pub fn add_node(&mut self, node: VectorId) -> NodeIndex {
        let idx = self.0.add_node(node);
        self.1.insert(node, idx);
        idx
    }

    pub fn add_edge(&mut self, node_1: VectorId, node_2: VectorId, weight: A) {
        let idx_1 = *self
            .1
            .get(&node_1)
            .expect(&format!("Failed to extract {node_1:?} from {self:#?}"));
        let idx_2 = *self
            .1
            .get(&node_2)
            .expect(&format!("Failed to extract {node_2:?} from {self:#?}"));
        self.0.add_edge(idx_1, idx_2, weight);
    }

    pub fn remove_edge(&mut self, node_1: VectorId, node_2: VectorId) -> Result<(), ()> {
        let Some((_, _, edge_idx)) = self
            .0
            .edges(
                *self
                    .1
                    .get(&node_1)
                    .unwrap_or_else(|| self.1.get(&node_2).unwrap()),
            ) // should return error
            .map(|edge_ref| ((edge_ref.source(), edge_ref.target()), edge_ref.id()))
            .map(|((source, target), e_idx)| {
                (
                    *self.0.node_weight(source).unwrap(),
                    *self.0.node_weight(target).unwrap(),
                    e_idx,
                )
            })
            .filter(|(id_1, id_2, _)| {
                (id_1 == &node_1 && id_2 == &node_2) || (id_2 == &node_1 && id_1 == &node_2)
            })
            .next()
        else {
            return Err(());
        };

        self.0.remove_edge(edge_idx);

        Ok(())
    }

    pub fn smallest_edge(&self) -> Option<(VectorId, VectorId, A)>
    where
        A: PartialOrd,
    {
        self.0
            .edge_indices()
            .map(|e_idx| {
                (
                    self.0.edge_endpoints(e_idx).unwrap(),
                    self.0.edge_weight(e_idx).unwrap(),
                )
            })
            .map(|((source, target), dist)| {
                (
                    *self.0.node_weight(source).unwrap(),
                    *self.0.node_weight(target).unwrap(),
                    *dist,
                )
            })
            .min_by(|(_, _, dist_1), (_, _, dist_2)| {
                dist_1.partial_cmp(dist_2).unwrap_or(Ordering::Equal)
            })
    }
    pub fn largest_edge(&self) -> Option<(VectorId, VectorId, A)>
    where
        A: PartialOrd,
    {
        self.0
            .edge_indices()
            .map(|e_idx| {
                (
                    self.0.edge_endpoints(e_idx).unwrap(),
                    self.0.edge_weight(e_idx).unwrap(),
                )
            })
            .map(|((source, target), dist)| {
                (
                    *self.0.node_weight(source).unwrap(),
                    *self.0.node_weight(target).unwrap(),
                    *dist,
                )
            })
            .max_by(|(_, _, dist_1), (_, _, dist_2)| {
                dist_1.partial_cmp(dist_2).unwrap_or(Ordering::Equal)
            })
    }
}

impl<'a, A: Field<A> + Debug> Into<Uuid> for &'a IntraPartitionGraph<A> {
    fn into(self) -> Uuid {
        *self.2
    }
}

#[derive(Archive, Serialize, Deserialize)]
pub struct GraphSerial<A> {
    ids: Vec<String>,
    connections: Vec<(usize, usize, A)>,
    id: Option<String>,
}

impl<A> FileExtension for GraphSerial<A> {
    fn extension() -> &'static str {
        "graph"
    }
}

impl<A: Field<A> + Clone + Copy + Debug> From<IntraPartitionGraph<A>> for GraphSerial<A> {
    fn from(value: IntraPartitionGraph<A>) -> Self {
        let id_to_index_map: HashMap<VectorId, usize> = value
            .1
            .iter()
            .map(|(id, _)| id)
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();

        GraphSerial {
            ids: value
                .1
                .iter()
                .map(|(id, _)| id)
                .map(|x| x.to_string())
                .collect(),
            connections: value
                .0
                .edge_indices()
                .map(|index| {
                    (
                        value.0.edge_endpoints(index).unwrap(),
                        value.0.edge_weight(index).unwrap(),
                    )
                })
                .map(|((start, end), weight)| {
                    (
                        id_to_index_map[value.0.node_weight(start).unwrap()],
                        id_to_index_map[value.0.node_weight(end).unwrap()],
                        *weight,
                    )
                })
                .collect(),
            id: Some((*value.2).to_string()),
        }
    }
}

impl<A: Field<A> + Clone + Copy + Debug> From<GraphSerial<A>> for IntraPartitionGraph<A> {
    fn from(value: GraphSerial<A>) -> Self {
        let mut graph: StableGraph<VectorId, A, Undirected> = StableGraph::default();
        let mut uuid_to_index = HashMap::new();

        value
            .ids
            .iter()
            .map(|id| VectorId(Uuid::from_str(id).unwrap()))
            .for_each(|id| {
                let idx = graph.add_node(id);

                uuid_to_index.insert(id, idx);
            });

        value
            .connections
            .iter()
            .map(|(start_index, end_index, weight)| {
                (
                    uuid_to_index[&VectorId(Uuid::from_str(&value.ids[*start_index]).unwrap())],
                    uuid_to_index[&VectorId(Uuid::from_str(&value.ids[*end_index]).unwrap())],
                    weight,
                )
            })
            .for_each(|(id1, id2, weight)| {
                graph.add_edge(id1, id2, *weight);
            });

        Self(
            graph,
            uuid_to_index,
            PartitionId(Uuid::from_str(&value.id.unwrap()).unwrap()),
        )
    }
}

pub type InterGraphSerial<A> = GraphSerial<(A, (String, String), (String, String))>;
impl<A: Field<A> + Clone + Copy> From<InterPartitionGraph<A>>
    for GraphSerial<(A, (String, String), (String, String))>
{
    fn from(value: InterPartitionGraph<A>) -> Self {
        let id_to_index_map: HashMap<PartitionId, usize> = value
            .1
            .iter()
            .map(|(id, _)| id)
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();

        GraphSerial {
            ids: value
                .1
                .iter()
                .map(|(id, _)| id)
                .map(|x| x.to_string())
                .collect(),
            connections: value
                .0
                .edge_indices()
                .map(|index| {
                    (
                        value.0.edge_endpoints(index).unwrap(),
                        value.0.edge_weight(index).unwrap(),
                    )
                })
                .map(
                    |((start, end), (dist, (partition_1, vector_1), (partition_2, vector_2)))| {
                        (
                            id_to_index_map[value.0.node_weight(start).unwrap()],
                            id_to_index_map[value.0.node_weight(end).unwrap()],
                            (
                                *dist,
                                (partition_1.to_string(), vector_1.to_string()),
                                (partition_2.to_string(), vector_2.to_string()),
                            ),
                        )
                    },
                )
                .collect(),
            id: None,
        }
    }
}

impl<A: Field<A> + Clone + Copy> From<GraphSerial<(A, (String, String), (String, String))>>
    for InterPartitionGraph<A>
{
    fn from(value: GraphSerial<(A, (String, String), (String, String))>) -> Self {
        let mut graph = StableGraph::default();
        let mut uuid_to_index = HashMap::new();

        value
            .ids
            .iter()
            .map(|id| PartitionId(Uuid::from_str(id).unwrap()))
            .for_each(|id| {
                let idx = graph.add_node(id);

                uuid_to_index.insert(id, idx);
            });

        value
            .connections
            .iter()
            .map(|(start_index, end_index, weight)| {
                (
                    uuid_to_index[&PartitionId(Uuid::from_str(&value.ids[*start_index]).unwrap())],
                    uuid_to_index[&PartitionId(Uuid::from_str(&value.ids[*end_index]).unwrap())],
                    weight,
                )
            })
            .for_each(
                |(id1, id2, (dist, (partition_1, vector_1), (partition_2, vector_2)))| {
                    graph.add_edge(
                        id1,
                        id2,
                        (
                            *dist,
                            (
                                PartitionId(Uuid::from_str(&partition_1).unwrap()),
                                VectorId(Uuid::from_str(&vector_1).unwrap()),
                            ),
                            (
                                PartitionId(Uuid::from_str(&partition_2).unwrap()),
                                VectorId(Uuid::from_str(&vector_2).unwrap()),
                            ),
                        ),
                    );
                },
            );

        Self(graph, uuid_to_index)
    }
}
