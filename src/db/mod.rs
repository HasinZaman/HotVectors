use std::{collections::HashMap, marker::PhantomData, mem, ops::Index};

use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSpace};

mod serialization;

pub struct LoadedPartitions<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
> {
    partitions: Vec<Partition<A, B, PARTITION_CAP, VECTOR_CAP>>,
    stats: Vec<(u64,)>,
    hash_map: HashMap<Uuid, usize>,
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
        const MAX_LOADED: usize,
    > LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>
{
    pub fn new() -> Self {
        LoadedPartitions {
            partitions: Vec::new(),
            stats: Vec::new(),

            hash_map: HashMap::new(),
        }
    }
    pub async fn load(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
    pub fn unload_by_id(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
    pub fn unload_by_least_used(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
    pub fn unload_by_youngest(&mut self, id: Uuid) -> Result<(), ()> {
        todo!()
    }
}

#[derive(Debug)]
pub enum PartitionErr {
    Overflow,
    VectorNotFound,
    PartitionEmpty,
}

#[derive(Debug, Clone, Copy)]
pub struct Partition<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
> {
    size: usize,

    vectors: [Option<VectorEntry<A, B, VECTOR_CAP>>; PARTITION_CAP],
    centroid: B,

    id: Uuid,
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
            vectors: [None; PARTITION_CAP],
            centroid: B::additive_identity(),
            id: Uuid::new_v4(),
        }
    }

    pub fn add(&mut self, value: VectorEntry<A, B, VECTOR_CAP>) -> Result<(), PartitionErr> {
        if self.size + 1 >= PARTITION_CAP {
            return Err(PartitionErr::Overflow);
        };

        self.vectors[self.size] = Some(value);
        self.size += 1;

        self.centroid = B::add(&self.centroid, &value.vector);

        Ok(())
    }

    pub fn remove_by_vector(
        &mut self,
        value: B,
    ) -> Result<VectorEntry<A, B, VECTOR_CAP>, PartitionErr>
    where
        B: PartialEq,
    {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let index = {
            let mut index = 0;
            let mut found = false;
            for i in 0..self.size {
                if self.vectors[i].unwrap().vector == value {
                    found = true;
                    index = i;

                    break;
                }
            }

            if !found {
                return Err(PartitionErr::VectorNotFound);
            }

            index
        };

        let mut removed_vec = None;

        mem::swap(&mut removed_vec, &mut self.vectors[index]);

        for i in index..self.size {
            match i + 1 == self.size {
                true => {
                    self.vectors[i] = None;
                }
                false => {
                    self.vectors[i] = self.vectors[i + 1];
                }
            }
        }
        self.size -= 1;

        Ok(removed_vec.unwrap())
    }
    pub fn remove_by_id(
        &mut self,
        id: Uuid,
    ) -> Result<VectorEntry<A, B, VECTOR_CAP>, PartitionErr> {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let index = {
            let mut index = 0;
            let mut found = false;
            for i in 0..self.size {
                if self.vectors[i].unwrap().id == id {
                    found = true;
                    index = i;

                    break;
                }
            }

            if !found {
                return Err(PartitionErr::VectorNotFound);
            }

            index
        };

        let mut removed_vec = None;

        mem::swap(&mut removed_vec, &mut self.vectors[index]);

        for i in index..self.size {
            match i + 1 == self.size {
                true => {
                    self.vectors[i] = None;
                }
                false => {
                    self.vectors[i] = self.vectors[i + 1];
                }
            }
        }
        self.size -= 1;

        Ok(removed_vec.unwrap())
    }

    pub fn pop(&mut self) -> Result<VectorEntry<A, B, VECTOR_CAP>, PartitionErr> {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let index = self.size - 1;

        let mut removed_vec = None;

        mem::swap(&mut removed_vec, &mut self.vectors[index]);

        for i in index..self.size {
            match i + 1 == self.size {
                true => {
                    self.vectors[i] = None;
                }
                false => {
                    self.vectors[i] = self.vectors[i + 1];
                }
            }
        }
        self.size -= 1;

        Ok(removed_vec.unwrap())
    }

    pub fn remove_by_func(
        &mut self,
        func: fn(&VectorEntry<A, B, VECTOR_CAP>) -> bool,
    ) -> Result<(), PartitionErr> {
        if self.size == 0 {
            return Ok(());
        }

        todo!();
    }

    pub fn split(&mut self) -> Result<Partition<A, B, PARTITION_CAP, VECTOR_CAP>, PartitionErr>
    where
        B: Extremes,
    {
        if self.size == 0 {
            return Err(PartitionErr::PartitionEmpty);
        }

        let mut new_partition = Partition::new();

        new_partition.add(self.pop().unwrap()).unwrap();

        Ok(new_partition)
    }

    pub fn merge(&mut self, other: Self) -> Result<(), PartitionErr> {
        if self.size + other.size >= VECTOR_CAP {
            return Err(PartitionErr::Overflow);
        }

        (0..other.size)
            .map(|i| (i, other.vectors[i]))
            .for_each(|(i, x)| {
                self.centroid = B::add(&self.centroid, &x.unwrap().vector);

                let _ = mem::replace(&mut self.vectors[self.size + i], x);
            });

        self.size += other.size;

        Ok(())
    }
}

impl<
        A: PartialEq + Clone + Copy + Field<A>,
        B: VectorSpace<A> + Sized + Clone + Copy,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
    > Index<usize> for Partition<A, B, PARTITION_CAP, VECTOR_CAP>
{
    type Output = Option<VectorEntry<A, B, VECTOR_CAP>>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index >= self.size);

        &self.vectors[index]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorEntry<A: Field<A>, B: VectorSpace<A> + Sized, const CAP: usize> {
    vector: B,
    id: Uuid,

    _phantom_data: PhantomData<A>,
}
