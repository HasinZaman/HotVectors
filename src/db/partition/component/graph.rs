use std::{collections::HashMap, hash::Hash};

use petgraph::{csr::DefaultIx, graph::NodeIndex, prelude::StableGraph, Undirected};
use uuid::Uuid;

use crate::{
    db::partition::{PartitionId, VectorId},
    vector::Field,
};

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

impl<A: Field<A>> InterPartitionGraph<A> {
    pub fn new() -> Self {
        Self(StableGraph::default(), HashMap::new())
    }
    pub fn load() -> Self {
        todo!()
    }
    pub fn add_node(&mut self, node: PartitionId) -> NodeIndex {
        let idx = self.0.add_node(node);
        self.1.insert(node, idx);
        idx
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
}

#[derive(Debug, Clone)]
pub struct IntraPartitionGraph<A: Field<A>>(
    pub StableGraph<VectorId, A, Undirected, DefaultIx>,
    pub HashMap<VectorId, NodeIndex<DefaultIx>>,
    pub PartitionId,
);

impl<A: Field<A>> IntraPartitionGraph<A> {
    pub fn new(id: PartitionId) -> Self {
        Self(StableGraph::default(), HashMap::new(), id)
    }
    pub fn load() -> Self {
        todo!()
    }
    pub fn add_node(&mut self, node: VectorId) -> NodeIndex {
        let idx = self.0.add_node(node);
        self.1.insert(node, idx);
        idx
    }

    pub fn add_edge(&mut self, node_1: VectorId, node_2: VectorId, weight: A) {
        let idx_1 = self.1[&node_1];
        let idx_2 = self.1[&node_2];
        self.0.add_edge(idx_1, idx_2, weight);
    }
}

impl<'a, A: Field<A>> Into<Uuid> for &'a IntraPartitionGraph<A> {
    fn into(self) -> Uuid {
        *self.2
    }
}
