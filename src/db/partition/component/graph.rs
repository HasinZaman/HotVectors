use std::{collections::HashMap, hash::Hash};

use petgraph::{csr::DefaultIx, graph::NodeIndex, prelude::StableGraph, Undirected};
use uuid::Uuid;

use crate::{
    db::partition::{PartitionId, VectorId},
    vector::Field,
};
#[derive(Debug, Clone)]
pub struct PartitionGraph<I: Clone + Copy + Hash + PartialEq + Eq, V>(
    pub StableGraph<I, V, Undirected, DefaultIx>,
    pub HashMap<I, NodeIndex<DefaultIx>>,
    pub PartitionId,
);

impl<I: Clone + Copy + Hash + PartialEq + Eq, V> PartitionGraph<I, V> {
    pub fn new() -> Self {
        Self(StableGraph::default(), HashMap::new(), todo!())
    }
    pub fn load() -> Self {
        todo!()
    }
    pub fn add_node(&mut self, node: I) -> NodeIndex {
        let idx = self.0.add_node(node);
        // maybe do this last
        self.1.insert(node, idx);

        idx
    }

    pub fn add_edge(&mut self, node_1: I, node_2: I, weight: V) {
        let idx_1 = self.1[&node_1];
        let idx_2 = self.1[&node_2];

        self.0.add_edge(idx_1, idx_2, weight);
    }
}

impl<'a, I: Clone + Copy + Hash + PartialEq + Eq, V> Into<Uuid> for &'a PartitionGraph<I, V> {
    fn into(self) -> Uuid {
        *self.2
    }
}

pub type InterPartitionGraph<A: Field<A>> =
    PartitionGraph<PartitionId, (A, (PartitionId, VectorId), (PartitionId, VectorId))>;
pub type IntraPartitionGraph<A: Field<A>> = PartitionGraph<VectorId, A>;
