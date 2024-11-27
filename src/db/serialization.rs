use std::{array, collections::HashMap, str::FromStr};

use petgraph::{Graph, Undirected};
use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

use crate::vector::{Field, VectorSerial, VectorSpace};

use super::partition::{IntraPartitionGraph, Partition, VectorEntry};

#[derive(Archive, Debug, Serialize, Deserialize)]
pub struct PartitionSerial<A: Clone + Copy> {
    vectors: Vec<VectorEntrySerial<A>>,
    centroid: VectorSerial<A>,
    id: String,
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + Into<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<Partition<A, B, PARTITION_CAP, VECTOR_CAP>> for PartitionSerial<A>
where
    VectorEntry<A, B>: Into<VectorEntrySerial<A>>,
{
    fn from(value: Partition<A, B, PARTITION_CAP, VECTOR_CAP>) -> Self {
        PartitionSerial {
            vectors: value
                .vectors
                .iter()
                .filter(|x| x.is_some())
                .map(|x| x.unwrap())
                .map(|x| Into::<VectorEntrySerial<A>>::into(x))
                .collect::<Vec<VectorEntrySerial<A>>>(),
            centroid: value.centroid.into(),
            id: value.id.to_string(),
        }
    }
}
impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy + From<VectorSerial<A>>,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > From<PartitionSerial<A>> for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
where
    VectorEntrySerial<A>: Into<VectorEntry<A, B>>,
{
    fn from(value: PartitionSerial<A>) -> Self {
        let mut iter = value
            .vectors
            .iter()
            .map(|x| Into::<VectorEntry<A, B>>::into(x.clone())); //TODO!() derive from a reference

        Partition {
            size: value.vectors.len(),
            vectors: array::from_fn(|_| iter.next()),
            centroid: value.centroid.into(),
            id: Uuid::from_str(&value.id).unwrap(),
        }
    }
}

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
pub struct VectorEntrySerial<A: Clone + Copy> {
    vector: VectorSerial<A>,
    id: String,
}

impl<A: Clone + Copy + Field<A>, B: Clone + Copy + Into<VectorSerial<A>> + VectorSpace<A>>
    From<VectorEntry<A, B>> for VectorEntrySerial<A>
{
    fn from(value: VectorEntry<A, B>) -> Self {
        VectorEntrySerial {
            vector: value.vector.into(),
            id: value.id.to_string(),
        }
    }
}
impl<
        A: Clone + Copy + Field<A>,
        B: PartialEq + Clone + Copy + From<VectorSerial<A>> + VectorSpace<A>,
    > From<VectorEntrySerial<A>> for VectorEntry<A, B>
{
    fn from(value: VectorEntrySerial<A>) -> Self {
        VectorEntry::from_str_id(value.vector.into(), &value.id)
        //  {
        //     vector: value.vector.into(),
        //     id: Uuid::from_str(&value.id).unwrap(),
        //     _phantom_data: PhantomData,
        // }
    }
}

#[derive(Archive, Serialize, Deserialize)]
pub struct PartitionGraphSerial<A> {
    ids: Vec<String>,
    connections: Vec<(usize, usize, A)>,
}

impl<A: Field<A> + Clone + Copy> From<IntraPartitionGraph<A>> for PartitionGraphSerial<A> {
    fn from(value: IntraPartitionGraph<A>) -> Self {
        PartitionGraphSerial {
            ids: value
                .0
                .raw_nodes()
                .iter()
                .map(|node| node.weight.to_string())
                .collect::<Vec<String>>(),
            connections: value
                .0
                .raw_edges()
                .iter()
                .map(|edge| (edge.source().index(), edge.target().index(), edge.weight))
                .collect::<Vec<(usize, usize, A)>>(),
        }
    }
}

impl<A: Field<A> + Clone + Copy> From<PartitionGraphSerial<A>> for IntraPartitionGraph<A> {
    fn from(value: PartitionGraphSerial<A>) -> Self {
        let mut graph: Graph<Uuid, A, Undirected> = Graph::<Uuid, A, Undirected>::new_undirected();
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

        IntraPartitionGraph::new(graph)
    }
}
