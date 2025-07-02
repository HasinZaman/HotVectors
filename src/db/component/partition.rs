use std::{array, hash::Hash, marker::PhantomData, ops::Index, str::FromStr};

use rkyv::{Archive, Deserialize, Serialize};

use sled::Db;
use uuid::Uuid;

use crate::{
    db::component::ids::{PartitionId, VectorId},
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

use super::serial::FileExtension;

#[derive(Debug)]
pub enum PartitionErr {
    Overflow,
    VectorNotFound,
    PartitionEmpty,
    InsufficientSizeForSplits,
}

#[derive(Debug, Clone)]
pub struct PartitionMembership(String, pub Db);
impl PartitionMembership {
    pub fn new(dir: String) -> Self {
        let db = sled::open(dir.clone()).expect("Failed to open sled DB");
        Self(dir, db)
    }

    pub fn assign(&mut self, vec_id: VectorId, partition_id: PartitionId) {
        let key = (*vec_id).as_bytes(); // &[u8; 16]
        let value = (*partition_id).as_bytes(); // &[u8; 16]
        self.1.insert(key, value).expect("DB insert failed");
    }

    pub fn get_partition_id(&self, vec_id: VectorId) -> Option<PartitionId> {
        let key = (*vec_id).as_bytes();
        self.1.get(key).ok().flatten().map(|ivec| {
            let uuid = Uuid::from_slice(&ivec).expect("Invalid UUID in DB");
            PartitionId(uuid)
        })
    }

    #[inline]
    pub fn update_membership<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    >(
        &mut self,
        partition: &Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    ) {
        for VectorEntry { id, .. } in partition.iter() {
            self.assign(VectorId(*id), PartitionId(partition.id));
        }
    }

    pub fn flush(&mut self, flush_db: &PartitionMembership) {
        for result in flush_db.1.iter() {
            let (key, value) = result.expect("Error reading flush_db");

            let vec_id = VectorId(Uuid::from_slice(&key).expect("Invalid UUID in key"));
            let partition_id =
                PartitionId(Uuid::from_slice(&value).expect("Invalid UUID in value"));

            self.assign(vec_id, partition_id);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Partition<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
> {
    pub size: usize,

    pub vectors: Box<[Option<VectorEntry<A, B>>; PARTITION_CAP]>,
    pub centroid: B,

    pub id: Uuid,
}

impl<
        'a,
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Into<Uuid> for &'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    fn into(self) -> Uuid {
        self.id
    }
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    pub fn new() -> Self
    where
        B: Extremes,
    {
        Partition {
            size: 0usize,
            vectors: Box::new([None; PARTITION_CAP]),
            centroid: B::additive_identity(),
            id: Uuid::new_v4(),
        }
    }

    pub fn centroid(&self) -> B {
        B::scalar_mult(
            &self.centroid,
            &A::div(&A::multiplicative_identity(), &A::from_usize(self.size)),
        )
    }

    // pub fn remove_by_vector(&mut self, value: B) -> Result<VectorEntry<A, B>, PartitionErr>
    // where
    //     B: PartialEq,
    // {
    //     if self.size == 0 {
    //         return Err(PartitionErr::PartitionEmpty);
    //     }

    //     let index = {
    //         let mut index = 0;
    //         let mut found = false;
    //         for i in 0..self.size {
    //             if self.vectors[i].unwrap().vector == value {
    //                 found = true;
    //                 index = i;

    //                 break;
    //             }
    //         }

    //         if !found {
    //             return Err(PartitionErr::VectorNotFound);
    //         }

    //         index
    //     };

    //     let mut removed_vec = None;

    //     mem::swap(&mut removed_vec, &mut self.vectors[index]);

    //     for i in index..self.size {
    //         match i + 1 == self.size {
    //             true => {
    //                 self.vectors[i] = None;
    //             }
    //             false => {
    //                 self.vectors[i] = self.vectors[i + 1];
    //             }
    //         }
    //     }
    //     self.size -= 1;

    //     Ok(removed_vec.unwrap())
    // }
    // pub fn remove_by_id(&mut self, id: Uuid) -> Result<VectorEntry<A, B>, PartitionErr> {
    //     if self.size == 0 {
    //         return Err(PartitionErr::PartitionEmpty);
    //     }

    //     let index = {
    //         let mut index = 0;
    //         let mut found = false;
    //         for i in 0..self.size {
    //             if self.vectors[i].unwrap().id == id {
    //                 found = true;
    //                 index = i;

    //                 break;
    //             }
    //         }

    //         if !found {
    //             return Err(PartitionErr::VectorNotFound);
    //         }

    //         index
    //     };

    //     let mut removed_vec = None;

    //     mem::swap(&mut removed_vec, &mut self.vectors[index]);

    //     for i in index..self.size {
    //         match i + 1 == self.size {
    //             true => {
    //                 self.vectors[i] = None;
    //             }
    //             false => {
    //                 self.vectors[i] = self.vectors[i + 1];
    //             }
    //         }
    //     }
    //     self.size -= 1;

    //     Ok(removed_vec.unwrap())
    // }

    // pub fn pop(&mut self) -> Result<VectorEntry<A, B>, PartitionErr> {
    //     if self.size == 0 {
    //         return Err(PartitionErr::PartitionEmpty);
    //     }

    //     let index = self.size - 1;

    //     let mut removed_vec = None;

    //     mem::swap(&mut removed_vec, &mut self.vectors[index]);

    //     for i in index..self.size {
    //         match i + 1 == self.size {
    //             true => {
    //                 self.vectors[i] = None;
    //             }
    //             false => {
    //                 self.vectors[i] = self.vectors[i + 1];
    //             }
    //         }
    //     }
    //     self.size -= 1;

    //     Ok(removed_vec.unwrap())
    // }

    // pub fn remove_by_func(
    //     &mut self,
    //     func: fn(&VectorEntry<A, B>) -> bool,
    // ) -> Result<(), PartitionErr> {
    //     if self.size == 0 {
    //         return Ok(());
    //     }

    //     todo!();
    // }

    // pub fn split(
    //     &mut self,
    //     graph_1: &mut IntraPartitionGraph<A>,
    //     inter_graph: &mut InterPartitionGraph<A>,
    // ) -> Result<
    //     (
    //         Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    //         IntraPartitionGraph<A>,
    //     ),
    //     PartitionErr,
    // >
    // where
    //     A: PartialOrd + Ord,
    //     B: Extremes,
    // {
    //     if self.size == 0 {
    //         return Err(PartitionErr::PartitionEmpty);
    //     }

    //     let mut new_partition = Partition::new();
    //     let mut new_graph = IntraPartitionGraph::new(PartitionId(new_partition.id));

    //     todo!();
    //     // let _ = new_partition.add(self.pop().unwrap(), &mut new_graph);

    //     if cfg!(feature = "gpu_processing") {
    //         todo!()
    //     } else {
    //         let traitors = (0..self.size)
    //             .map(|i| {
    //                 (
    //                     i,
    //                     B::dist(&self.vectors[i].unwrap().vector, &self.centroid),
    //                     B::dist(&self.vectors[i].unwrap().vector, &new_partition.centroid),
    //                 )
    //             })
    //             .filter(|(_i, old_dist, new_dist)| new_dist < old_dist)
    //             .map(|(i, _old_dist, _new_dist)| i)
    //             .collect::<Vec<usize>>();

    //         // move values from graph old -> new
    //         todo!();

    //         traitors
    //             .iter()
    //             .enumerate()
    //             .map(|(i1, i2)| (i1, i2 + 1))
    //             .for_each(|(i1, i2)| {
    //                 mem::swap(&mut self.vectors[i1], &mut new_partition.vectors[i2]);
    //             });
    //         new_partition.size = 1 + traitors.len();

    //         self.fix_holes();

    //         self.size = self.size - traitors.len();
    //     }

    //     Ok((new_partition, new_graph))
    // }

    // fn fix_holes(&mut self) {
    //     for i1 in 0..self.size {
    //         if self.vectors[i1].is_some() {
    //             continue;
    //         }

    //         let mut next_pos = i1;

    //         while self.vectors[next_pos].is_none() && next_pos < self.size {
    //             next_pos += 1;
    //         }

    //         if next_pos >= self.size {
    //             break;
    //         }
    //     }
    // }

    pub fn iter(&self) -> Box<dyn Iterator<Item = &VectorEntry<A, B>> + '_> {
        Box::new(self.vectors.iter().take(self.size).map(vec_unwrap))
    }
}

fn vec_unwrap<'a, A: Field<A>, B: VectorSpace<A>>(
    vector: &'a Option<VectorEntry<A, B>>,
) -> &'a VectorEntry<A, B> {
    match vector {
        Some(vector) => vector,
        None => panic!(),
    }
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Index<usize> for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    type Output = VectorEntry<A, B>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.size);

        match self.vectors.get(index) {
            Some(Some(vector)) => vector,
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorEntry<A: Field<A>, B: VectorSpace<A> + Sized> {
    pub vector: B,
    pub id: Uuid,

    pub _phantom_data: PhantomData<A>,
}

impl<A: Field<A>, B: VectorSpace<A> + Sized> VectorEntry<A, B> {
    pub fn from_str_id(vector: B, id: &str) -> Self {
        Self {
            vector: vector,
            id: Uuid::from_str(id).unwrap(),
            _phantom_data: PhantomData,
        }
    }
    pub fn from_uuid(vector: B, id: Uuid) -> Self {
        Self {
            vector: vector,
            id: id,
            _phantom_data: PhantomData,
        }
    }
}

impl<A: Field<A>, B: VectorSpace<A> + Sized> PartialEq for VectorEntry<A, B> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<A: Field<A>, B: VectorSpace<A> + Sized> Eq for VectorEntry<A, B> {}
impl<A: Field<A>, B: VectorSpace<A> + Sized> Hash for VectorEntry<A, B> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Archive, Debug, Serialize, Deserialize)]
pub struct PartitionSerial<A: Clone + Copy> {
    vectors: Vec<VectorEntrySerial<A>>,
    centroid: VectorSerial<A>,
    id: String,
}

impl<A: PartialEq + Clone + Copy + Field<A>> FileExtension for PartitionSerial<A> {
    fn extension() -> &'static str {
        "partition"
    }
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
            vectors: Box::new(array::from_fn(|_| iter.next())),
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
