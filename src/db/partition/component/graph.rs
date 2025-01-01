use std::{collections::HashMap, hash::Hash, str::FromStr};

use petgraph::{csr::DefaultIx, graph::NodeIndex, prelude::StableGraph, Undirected};
use uuid::Uuid;

use crate::{
    db::partition::{PartitionId, VectorId},
    vector::Field,
};
use rkyv::{Archive, Deserialize, Serialize};

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

#[derive(Archive, Serialize, Deserialize)]
pub struct GraphSerial<A> {
    ids: Vec<String>,
    connections: Vec<(usize, usize, A)>,
}

impl<A> FileExtension for GraphSerial<A> {
    fn extension() -> &'static str {
        "graph"
    }
}

impl<A: Field<A> + Clone + Copy> From<IntraPartitionGraph<A>> for GraphSerial<A> {
    fn from(value: IntraPartitionGraph<A>) -> Self {
        GraphSerial {
            ids: todo!(),
            connections: todo!(), // ids: value
                                  //     .0
                                  //     .raw_nodes()
                                  //     .iter()
                                  //     .map(|node| node.weight.to_string())
                                  //     .collect::<Vec<String>>(),
                                  // connections: value
                                  //     .0
                                  //     .raw_edges()
                                  //     .iter()
                                  //     .map(|edge| (edge.source().index(), edge.target().index(), edge.weight))
                                  //     .collect::<Vec<(usize, usize, A)>>(),
        }
    }
}

impl<A: Field<A> + Clone + Copy> From<GraphSerial<A>> for IntraPartitionGraph<A> {
    fn from(value: GraphSerial<A>) -> Self {
        let mut graph: StableGraph<Uuid, A, Undirected> = todo!();
        let mut uuid_to_index = HashMap::new();

        value
            .ids
            .iter()
            .map(|id| Uuid::from_str(id).unwrap())
            .for_each(|id| {
                let idx = graph.add_node(id);

                uuid_to_index.insert(id, idx);
            });

        value
            .connections
            .iter()
            .map(|(i1, i2, weight)| {
                (
                    uuid_to_index[&Uuid::from_str(&value.ids[*i1]).unwrap()],
                    uuid_to_index[&Uuid::from_str(&value.ids[*i2]).unwrap()],
                    weight,
                )
            })
            .for_each(|(id1, id2, weight)| {
                graph.add_edge(id1, id2, *weight);
            });

        todo!()
        // IntraPartitionGraph(todo!(), todo!())
    }
}
