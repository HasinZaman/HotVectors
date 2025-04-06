use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    str::FromStr,
};

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

use super::{ids::ClusterId, serial::FileExtension};

#[derive(Debug, Clone)]
pub struct InterClusterGraph<A: Field<A>>(
    pub  StableGraph<
        ClusterId,
        (A, (ClusterId, VectorId), (ClusterId, VectorId)),
        Undirected,
        DefaultIx,
    >,
    pub HashMap<ClusterId, NodeIndex<DefaultIx>>,
);

impl<A: Field<A> + Clone + Copy> InterClusterGraph<A> {
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

    pub fn add_node(&mut self, node: ClusterId) -> NodeIndex {
        let idx = self.0.add_node(node);
        self.1.insert(node, idx);
        idx
    }

    pub fn remove_node(&mut self, node: ClusterId) {
        if let Some(idx) = self.1.remove(&node) {
            self.0.remove_node(idx);
        }
    }

    pub fn add_edge(
        &mut self,
        node_1: ClusterId,
        node_2: ClusterId,
        weight: (A, (ClusterId, VectorId), (ClusterId, VectorId)),
    ) {
        let idx_1 = self.1[&node_1];
        let idx_2 = self.1[&node_2];
        self.0.add_edge(idx_1, idx_2, weight);
    }

    pub fn remove_edge(
        &mut self,
        node_1: (ClusterId, VectorId),
        node_2: (ClusterId, VectorId),
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
        let idx_1 = *self
            .1
            .get(&node_1)
            .expect(&format!("Expect {node_1:?} to be in {:?}", self.1));
        let idx_2 = *self
            .1
            .get(&node_2)
            .expect(&format!("Expect {node_2:?} to be in {:?}", self.1));
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

    pub fn find_walk(
        &self,
        from: PartitionId,
        to: PartitionId,
    ) -> Result<Option<(A, Vec<(PartitionId, VectorId)>)>, ()> {
        if !self.1.contains_key(&from) {
            return Err(());
        }

        if !self.1.contains_key(&to) {
            return Err(());
        }

        let mut visited_partitions: HashSet<PartitionId> = HashSet::new();
        visited_partitions.insert(from);

        let mut search_queue: VecDeque<(A, Vec<(PartitionId, VectorId)>)> = self
            .0
            .edges(self.1[&from])
            // .filter(|edge_ref| {
            //     !visited_partitions.contains(&self.0.node_weight(edge_ref.source()).unwrap())
            // })
            // .filter(|edge_ref| {
            //     !visited_partitions.contains(&self.0.node_weight(edge_ref.target()).unwrap())
            // })
            .map(|edge_ref| {
                let (weight, (partition_id_1, vector_id_1), (partition_id_2, vector_id_2)) =
                    edge_ref.weight();
                let path = match partition_id_1 == &from {
                    true => vec![
                        (*partition_id_1, *vector_id_1),
                        (*partition_id_2, *vector_id_2),
                    ],
                    false => vec![
                        (*partition_id_2, *vector_id_2),
                        (*partition_id_1, *vector_id_1),
                    ],
                };
                (weight.clone(), path)
            })
            .collect();

        while let Some((current_weight, current_path)) = search_queue.pop_front() {
            let (partition_id, _vector_id) = current_path.last().unwrap();

            if partition_id == &to {
                return Ok(Some((current_weight, current_path)));
            }

            self.0
                .edges(self.1[&partition_id])
                .filter(|edge_ref| {
                    !visited_partitions.contains(&self.0.node_weight(edge_ref.source()).unwrap())
                })
                .filter(|edge_ref| {
                    !visited_partitions.contains(&self.0.node_weight(edge_ref.target()).unwrap())
                })
                .for_each(|edge_ref| {
                    let (weight, (partition_id_1, vector_id_1), (partition_id_2, vector_id_2)) =
                        edge_ref.weight();

                    let next_node = match partition_id_1 == partition_id {
                        true => (*partition_id_2, *vector_id_2),
                        false => (*partition_id_1, *vector_id_1),
                    };

                    let mut new_path = current_path.clone();
                    new_path.push(next_node);
                    let new_weight = A::add(&current_weight, weight);

                    search_queue.push_back((new_weight, new_path));
                });
            visited_partitions.insert(*partition_id);
        }

        Ok(None)
    }

    pub fn find_trail(
        &self,
        from: PartitionId,
        to: PartitionId,
    ) -> Result<
        Option<(
            A,
            Vec<((PartitionId, VectorId), (PartitionId, VectorId), A)>,
        )>,
        (),
    > {
        if !self.1.contains_key(&from) || !self.1.contains_key(&to) {
            return Err(());
        }

        let mut visited_partitions: HashSet<PartitionId> = HashSet::new();
        visited_partitions.insert(from);

        let mut search_queue: VecDeque<(
            A,
            Vec<((PartitionId, VectorId), (PartitionId, VectorId), A)>,
        )> = self
            .0
            .edges(self.1[&from])
            .map(|edge_ref| {
                let (weight, node_1, node_2) = edge_ref.weight();
                let edge = match node_1.0 == from {
                    true => (*node_1, *node_2, weight.clone()),
                    false => (*node_2, *node_1, weight.clone()),
                };
                (weight.clone(), vec![edge])
            })
            .collect();

        while let Some((current_weight, current_trail)) = search_queue.pop_front() {
            let (last_partition, _last_vector) = current_trail.last().unwrap().1;

            if last_partition == to {
                return Ok(Some((current_weight, current_trail)));
            }

            self.0
                .edges(self.1[&last_partition])
                .filter(|edge_ref| {
                    !visited_partitions.contains(&self.0.node_weight(edge_ref.source()).unwrap())
                })
                .filter(|edge_ref| {
                    !visited_partitions.contains(&self.0.node_weight(edge_ref.target()).unwrap())
                })
                .for_each(|edge_ref| {
                    let (weight, node_1, node_2) = edge_ref.weight();

                    let new_edge = match node_1.0 == last_partition {
                        true => (*node_1, *node_2, weight.clone()),
                        false => (*node_2, *node_1, weight.clone()),
                    };
                    let mut new_trail = current_trail.clone();
                    new_trail.push(new_edge);

                    let new_weight = A::add(&current_weight, weight);
                    search_queue.push_back((new_weight, new_trail));
                });
            visited_partitions.insert(last_partition);
        }

        Ok(None)
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

    pub fn find_walk(
        &self,
        from: VectorId,
        to: VectorId,
    ) -> Result<Option<(A, Vec<VectorId>)>, ()> {
        if !self.1.contains_key(&from) {
            return Err(());
        }

        if !self.1.contains_key(&to) {
            return Err(());
        }

        let mut visited_vectors: HashSet<VectorId> = HashSet::new();
        visited_vectors.insert(from);

        let mut search_queue: VecDeque<(Vec<VectorId>, A)> = self
            .0
            .edges(self.1[&from])
            // .filter(|edge_ref| {
            //     !visited_vectors.contains(&self.0.node_weight(edge_ref.source()).unwrap())
            // })
            // .filter(|edge_ref| {
            //     !visited_vectors.contains(&self.0.node_weight(edge_ref.target()).unwrap())
            // })
            .map(|edge_ref| {
                let weight = edge_ref.weight().clone();
                let (vector_id_1, vector_id_2) = (
                    self.0.node_weight(edge_ref.source()).unwrap(),
                    self.0.node_weight(edge_ref.target()).unwrap(),
                );
                let path = match vector_id_1 == &from {
                    true => vec![*vector_id_1, *vector_id_2],
                    false => vec![*vector_id_2, *vector_id_1],
                };
                (path, weight)
            })
            .collect();

        while !search_queue.is_empty() {
            let (current_path, current_weight) = search_queue.pop_front().unwrap();

            let vector_id = current_path.last().unwrap();

            if vector_id == &to {
                return Ok(Some((current_weight, current_path)));
            }

            self.0
                .edges(self.1[&vector_id])
                // .filter(|edge_ref| {
                //     !visited_vectors.contains(&self.0.node_weight(edge_ref.source()).unwrap())
                // })
                // .filter(|edge_ref| {
                //     !visited_vectors.contains(&self.0.node_weight(edge_ref.target()).unwrap())
                // })
                .for_each(|edge_ref| {
                    let weight = edge_ref.weight();
                    let (vector_id_1, vector_id_2) = (
                        self.0.node_weight(edge_ref.source()).unwrap(),
                        self.0.node_weight(edge_ref.target()).unwrap(),
                    );
                    let next_node = match vector_id_1 == &from {
                        true => *vector_id_1,
                        false => *vector_id_2,
                    };

                    let mut new_path = current_path.clone();
                    new_path.push(next_node);
                    let new_weight = A::add(&current_weight, weight);

                    search_queue.push_back((new_path, new_weight));
                });

            visited_vectors.insert(*vector_id);
        }

        Ok(None)
    }

    pub fn find_trail(
        &self,
        from: VectorId,
        to: VectorId,
    ) -> Result<Option<(A, Vec<(VectorId, VectorId, A)>)>, ()> {
        if !self.1.contains_key(&from) || !self.1.contains_key(&to) {
            return Err(());
        }

        let mut visited_vectors: HashSet<VectorId> = HashSet::new();
        visited_vectors.insert(from);

        let mut search_queue: VecDeque<(A, Vec<(VectorId, VectorId, A)>)> = self
            .0
            .edges(self.1[&from])
            .map(|edge_ref| {
                let weight = edge_ref.weight().clone();
                let (vector_id_1, vector_id_2) = (
                    *self.0.node_weight(edge_ref.source()).unwrap(),
                    *self.0.node_weight(edge_ref.target()).unwrap(),
                );

                let edge = match vector_id_1 == from {
                    true => (vector_id_1, vector_id_2, weight.clone()),
                    false => (vector_id_2, vector_id_1, weight.clone()),
                };
                (weight, vec![edge])
            })
            .collect();

        while let Some((current_weight, current_trail)) = search_queue.pop_front() {
            let (_, last_vector, _) = current_trail.last().unwrap();

            if last_vector == &to {
                return Ok(Some((current_weight, current_trail)));
            }

            self.0
                .edges(self.1[&last_vector])
                .filter(|edge_ref| {
                    !visited_vectors.contains(&self.0.node_weight(edge_ref.source()).unwrap())
                })
                .filter(|edge_ref| {
                    !visited_vectors.contains(&self.0.node_weight(edge_ref.target()).unwrap())
                })
                .for_each(|edge_ref| {
                    let weight = edge_ref.weight();
                    let (vector_id_1, vector_id_2) = (
                        *self.0.node_weight(edge_ref.source()).unwrap(),
                        *self.0.node_weight(edge_ref.target()).unwrap(),
                    );

                    let next_node = if vector_id_1 == *last_vector {
                        vector_id_2
                    } else {
                        vector_id_1
                    };

                    let new_edge = (*last_vector, next_node, weight.clone());
                    let mut new_trail = current_trail.clone();
                    new_trail.push(new_edge);

                    let new_weight = A::add(&current_weight, weight);
                    search_queue.push_back((new_weight, new_trail));
                });
            visited_vectors.insert(*last_vector);
        }

        Ok(None)
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

impl<A: Field<A> + Clone + Copy> From<InterClusterGraph<A>>
    for GraphSerial<(A, (String, String), (String, String))>
{
    fn from(value: InterClusterGraph<A>) -> Self {
        let id_to_index_map: HashMap<ClusterId, usize> = value
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

impl<A: Field<A> + Clone + Copy> From<GraphSerial<(A, (String, String), (String, String))>>
    for InterClusterGraph<A>
{
    fn from(value: GraphSerial<(A, (String, String), (String, String))>) -> Self {
        let mut graph = StableGraph::default();
        let mut uuid_to_index = HashMap::new();

        value
            .ids
            .iter()
            .map(|id| ClusterId(Uuid::from_str(id).unwrap()))
            .for_each(|id| {
                let idx = graph.add_node(id);

                uuid_to_index.insert(id, idx);
            });

        value
            .connections
            .iter()
            .map(|(start_index, end_index, weight)| {
                (
                    uuid_to_index[&ClusterId(Uuid::from_str(&value.ids[*start_index]).unwrap())],
                    uuid_to_index[&ClusterId(Uuid::from_str(&value.ids[*end_index]).unwrap())],
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
                                ClusterId(Uuid::from_str(&partition_1).unwrap()),
                                VectorId(Uuid::from_str(&vector_1).unwrap()),
                            ),
                            (
                                ClusterId(Uuid::from_str(&partition_2).unwrap()),
                                VectorId(Uuid::from_str(&vector_2).unwrap()),
                            ),
                        ),
                    );
                },
            );

        Self(graph, uuid_to_index)
    }
}
