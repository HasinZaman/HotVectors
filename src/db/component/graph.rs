use std::{
    cmp::{Ordering, Reverse},
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    str::FromStr,
};

use petgraph::{
    csr::DefaultIx,
    graph::NodeIndex,
    prelude::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
    Undirected,
};
use rancor::Strategy;
use uuid::Uuid;

use crate::{
    db::component::ids::{PartitionId, VectorId},
    vector::{Extremes, Field},
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
            // println!("Current trail ({last_partition:?}, {_last_vector:?})");

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

            self.0.edges(self.1[&vector_id]).for_each(|edge_ref| {
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

    pub fn find_walk_with_hints(
        &self,
        from: VectorId,
        to: VectorId,
        hint: HashMap<VectorId, A>,
    ) -> Result<Option<(A, Vec<VectorId>)>, ()> {
        todo!();
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

            self.0.edges(self.1[&vector_id]).for_each(|edge_ref| {
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

    pub fn find_trail_with_hints(
        &self,
        from: VectorId,
        to: VectorId,
        hint: &HashMap<VectorId, A>,
    ) -> Result<Option<(A, Vec<(VectorId, VectorId, A)>)>, ()>
    where
        A: Extremes + PartialOrd,
    {
        if !self.1.contains_key(&from) || !self.1.contains_key(&to) {
            return Err(());
        }

        let mut visited_vectors: HashSet<VectorId> = HashSet::new();
        visited_vectors.insert(from);

        let mut search_queue: Vec<Reverse<(A, A, Vec<(VectorId, VectorId, A)>)>> = self
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

                let next_node_weight = hint[&edge.1];

                Reverse((next_node_weight, weight, vec![edge]))
            })
            .collect();

        heapify::make_heap(&mut search_queue);

        heapify::pop_heap(&mut search_queue);
        while let Some(Reverse((_, current_weight, current_trail))) = search_queue.pop() {
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

                    search_queue.push(Reverse((hint[&vector_id_2], new_weight, new_trail)));

                    heapify::push_heap(&mut search_queue);
                });
            visited_vectors.insert(*last_vector);

            heapify::pop_heap(&mut search_queue);
        }

        Ok(None)
    }

    pub fn dijkstra_weights(&self, start: VectorId) -> Result<HashMap<VectorId, A>, ()>
    where
        A: PartialOrd,
    {
        let &start_idx = self.1.get(&start).ok_or(())?;

        let mut dist: HashMap<NodeIndex<_>, A> = HashMap::new();
        let mut prev: HashMap<NodeIndex<_>, NodeIndex<_>> = HashMap::new();
        let mut heap = Vec::new();

        dist.insert(start_idx, A::additive_identity());
        heap.push(Reverse((A::additive_identity(), start_idx)));

        while let Some(Reverse((cost_u, u))) = heap.pop() {
            if cost_u > dist[&u] {
                continue; // stale entry
            }

            for edge in self.0.edges(u) {
                let v = if edge.source() == u {
                    edge.target()
                } else {
                    edge.source()
                };
                let edge_weight = edge.weight().clone();
                let new_cost = A::add(&cost_u, &edge_weight);

                if dist.get(&v).map_or(true, |old_cost| new_cost < *old_cost) {
                    dist.insert(v, new_cost.clone());
                    prev.insert(v, u);
                    heap.push(Reverse((new_cost, v)));
                    heapify::push_heap(&mut heap);
                }
            }
        }

        // Now convert to output format: HashMap<VectorId, (cost, Option<prev VectorId>)>
        let mut result = HashMap::new();
        for (&node_idx, cost) in &dist {
            let vector_id = *self.0.node_weight(node_idx).unwrap();

            let prev_vector = prev
                .get(&node_idx)
                .map(|&pidx| *self.0.node_weight(pidx).unwrap());

            result.insert(vector_id, cost.clone());
        }

        Ok(result)
    }
}

pub trait UpdateTree<Strategy, A: Field<A> + Clone + Copy + Debug> {
    fn update(
        &mut self,
        new_vector: VectorId,
        new_edges: &[(A, VectorId)],
    ) -> Result<Vec<(A, VectorId, VectorId)>, ()>;
    fn batch_update(
        &mut self,
        new_vectors: &[VectorId],
        new_edges: &[(A, VectorId, VectorId)],
    ) -> Result<Vec<(A, VectorId, VectorId)>, ()>;
}

pub struct ReConstruct;

struct DSU {
    parent: HashMap<VectorId, VectorId>,
}
impl DSU {
    fn new(nodes: &[VectorId]) -> Self {
        Self {
            parent: nodes.iter().map(|&v| (v, v)).collect(),
        }
    }
    fn find(&mut self, x: VectorId) -> VectorId {
        if self.parent[&x] != x {
            let root = self.find(self.parent[&x]);
            self.parent.insert(x, root);
        }
        self.parent[&x]
    }
    fn union(&mut self, x: VectorId, y: VectorId) {
        let px = self.find(x);
        let py = self.find(y);
        if px != py {
            self.parent.insert(px, py);
        }
    }
}

impl<A: Field<A> + Clone + Copy + Debug + PartialOrd> UpdateTree<ReConstruct, A>
    for IntraPartitionGraph<A>
{
    fn update(
        &mut self,
        new_vector: VectorId,
        new_edges: &[(A, VectorId)],
    ) -> Result<Vec<(A, VectorId, VectorId)>, ()> {
        let IntraPartitionGraph(ref mut graph, ref mut node_map, _) = self;

        let old_edges: HashSet<(VectorId, VectorId)> = graph
            .edge_references()
            .map(|e| {
                let a = *graph.node_weight(e.source()).unwrap();
                let b = *graph.node_weight(e.target()).unwrap();
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();

        let mut all_vertices: Vec<VectorId> = graph.node_weights().cloned().collect();
        all_vertices.push(new_vector);

        let mut edges: Vec<(A, VectorId, VectorId)> = graph
            .edge_references()
            .map(|e| {
                let a = *graph.node_weight(e.source()).unwrap();
                let b = *graph.node_weight(e.target()).unwrap();
                (*e.weight(), a, b)
            })
            .collect();

        for &(weight, other) in new_edges {
            edges.push((weight, new_vector, other));
        }

        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Build new MST graph
        let mut new_graph: StableGraph<VectorId, A, Undirected> = StableGraph::default();
        let mut new_node_map: HashMap<VectorId, NodeIndex> = HashMap::new();

        for &v in &all_vertices {
            let idx = new_graph.add_node(v);
            new_node_map.insert(v, idx);
        }

        let mut dsu = DSU::new(&all_vertices);
        let mut inserted_edges: Vec<(A, VectorId, VectorId)> = Vec::new();
        let mut new_edges_only: Vec<(A, VectorId, VectorId)> = Vec::new();

        for (weight, u, v) in edges {
            if dsu.find(u) != dsu.find(v) {
                dsu.union(u, v);
                let u_idx = new_node_map[&u];
                let v_idx = new_node_map[&v];
                new_graph.add_edge(u_idx, v_idx, weight);

                inserted_edges.push((weight, u, v));

                // Check if edge didn't exist in old graph
                let (min, max) = if u < v { (u, v) } else { (v, u) };
                if !old_edges.contains(&(min, max)) {
                    new_edges_only.push((weight, u, v));
                }
            }
        }

        // Replace original graph
        *graph = new_graph;
        *node_map = new_node_map;

        Ok(new_edges_only)
    }

    fn batch_update(
        &mut self,
        new_vectors: &[VectorId],
        new_edges: &[(A, VectorId, VectorId)],
    ) -> Result<Vec<(A, VectorId, VectorId)>, ()> {
        let IntraPartitionGraph(ref mut graph, ref mut node_map, _) = self;

        let old_edges: HashSet<(VectorId, VectorId)> = graph
            .edge_references()
            .map(|e| {
                let a = *graph.node_weight(e.source()).unwrap();
                let b = *graph.node_weight(e.target()).unwrap();
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();

        let mut all_vertices: Vec<VectorId> = graph.node_weights().cloned().collect();
        for new_vector in new_vectors {
            all_vertices.push(*new_vector);
        }

        let mut edges: Vec<(A, VectorId, VectorId)> = graph
            .edge_references()
            .map(|e| {
                let a = *graph.node_weight(e.source()).unwrap();
                let b = *graph.node_weight(e.target()).unwrap();
                (*e.weight(), a, b)
            })
            .collect();

        edges.extend(new_edges);

        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Build new MST graph
        let mut new_graph: StableGraph<VectorId, A, Undirected> = StableGraph::default();
        let mut new_node_map: HashMap<VectorId, NodeIndex> = HashMap::new();

        for &v in &all_vertices {
            let idx = new_graph.add_node(v);
            new_node_map.insert(v, idx);
        }

        let mut dsu = DSU::new(&all_vertices);
        let mut inserted_edges: Vec<(A, VectorId, VectorId)> = Vec::new();
        let mut new_edges_only: Vec<(A, VectorId, VectorId)> = Vec::new();

        for (weight, u, v) in edges {
            if dsu.find(u) != dsu.find(v) {
                dsu.union(u, v);
                let u_idx = new_node_map[&u];
                let v_idx = new_node_map[&v];
                new_graph.add_edge(u_idx, v_idx, weight);

                inserted_edges.push((weight, u, v));

                // Check if edge didn't exist in old graph
                let (min, max) = if u < v { (u, v) } else { (v, u) };
                if !old_edges.contains(&(min, max)) {
                    new_edges_only.push((weight, u, v));
                }
            }
        }

        // Replace original graph
        *graph = new_graph;
        *node_map = new_node_map;

        Ok(new_edges_only)
    }
}

// {
//     #[cfg(feature = "benchmark")]
//     let _child_benchmark =
//         Benchmark::spawn_child("Updating local graph".to_string(), &_child_benchmark);
//
//     let min_span_tree = resolve_buffer!(ACCESS, min_span_tree_buffer, closet_partition_id);
//
//     let Some(min_span_tree) = &mut *min_span_tree.write().await else {
//         todo!()
//     };
//
//     let vector_iter = min_span_tree.1.iter().map(|(x, _)| *x).collect::<Vec<_>>();
//     let mut vector_iter = vector_iter.iter();
//
//     // first edge
//     min_span_tree.add_node(VectorId(value.id));
//
//     min_span_tree.add_node(VectorId(value.id));
//     match vector_iter.next() {
//         Some(vector_id) => {
//             min_span_tree.add_edge(
//                 VectorId(value.id),
//                 *vector_id,
//                 *dist_map.get(vector_id).expect(""),
//             );
//         }
//         None => todo!(),
//     }
//
//     //  -> replace loop with while hashset.is_empty() & for loop for a trail
//     // while iterating through trail (add and remove edges based on deleting or adding edges)
//     //  This would reduce # of times trails are calculated
//     //      Current O(vectors*(BFS)) = O(|V|*|V|*|E|))
//     //                               = O(|V|*|V|^2)
//     //                               = O(|V|^3)
//     //      Proposed solution: O(|T|*(BFS)) = O(|T|*|V|*|E|))
//     //                                      = O(|T|*|V|*|V|))
//     //                                      = O(|T|*|V|^2))
//     //      note: |T| := # of searched trails
//     //      V(T1) U...U V(Tn) = V (The set of all searched vectors := vectors)
//     //      |T| < |V|, therefore O(|T|*(BFS)) < O(vectors*(BFS))
//     for vector_id in vector_iter {
//         let weight = *dist_map.get(vector_id).expect("");
//         event!(Level::DEBUG, "{:?} -> {:?}", vector_id, VectorId(value.id));
//         let Ok(path) = min_span_tree.find_trail(*vector_id, VectorId(value.id)) else {
//             event!(
//                 Level::DEBUG,
//                 "Failed to find trail:\n{min_span_tree:#?}\n{:?}\n{:?}",
//                 vector_id,
//                 VectorId(value.id)
//             );
//             todo!()
//         };
//         let Some((_, path)) = path else { todo!() };
//
//         let (max_vector_id_1, max_vector_id_2, max_weight) = path.into_iter().fold(
//             (VectorId(Uuid::nil()), VectorId(Uuid::nil()), A::min()),
//             |(acc_id_1, acc_id_2, acc_weight), (next_id_1, next_id_2, next_weight)| {
//                 match next_weight.partial_cmp(&acc_weight) {
//                     Some(Ordering::Greater) => (next_id_1, next_id_2, next_weight),
//                     _ => (acc_id_1, acc_id_2, acc_weight),
//                 }
//             },
//         );
//
//         if weight >= max_weight {
//             continue;
//         };
//
//         let _ = min_span_tree
//             .remove_edge(max_vector_id_1, max_vector_id_2)
//             .unwrap();
//
//         let _ = remove_cluster_edge(cluster_sets, max_vector_id_1, max_vector_id_2);
//
//         min_span_tree.add_edge(VectorId(value.id), *vector_id, weight);
//
//         update_cluster(cluster_sets, &weight, VectorId(value.id), *vector_id).await;
//     }
// }

impl<'a, A: Field<A> + Debug> Into<Uuid> for &'a IntraPartitionGraph<A> {
    fn into(self) -> Uuid {
        *self.2
    }
}

#[derive(Archive, Serialize, Deserialize, Debug)]
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
