use std::{
    array, collections::HashMap, error::Error, fmt::Display, marker::PhantomData, mem::{self, swap}, sync::Arc
};

use heapify::{make_heap_with, pop_heap_with};
use tokio::sync::RwLock;
use uuid::Uuid;

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    from_bytes, rancor,
    rend::u32_le,
    to_bytes,
    tuple::ArchivedTuple3,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, DeserializeUnsized,
};

use super::{serial::FileExtension};

#[derive(PartialEq, Eq, Debug)]
pub enum BufferError{
    OutOfSpace,
    DataNotFound
}

impl From<rancor::Error> for BufferError {
    fn from(value: rancor::Error) -> Self {
        todo!()
    }
}

impl From<std::io::Error> for BufferError {
    fn from(value: std::io::Error) -> Self {
        todo!()
    }
}

impl Display for BufferError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
impl Error for BufferError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        todo!()
    }

    fn description(&self) -> &str {
        todo!()
    }

    fn cause(&self) -> Option<&dyn Error> {
        todo!()
    }
}

pub struct LeastUsedIterator(Vec<(usize, usize, Uuid)>);

impl LeastUsedIterator {
    pub fn new(mut data: Vec<(usize, usize, Uuid)>) -> Self {
        make_heap_with(&mut data, |(score_1, _, _), (score_2, _, _)| {
            score_1.partial_cmp(score_2)
        });

        LeastUsedIterator(data)
    }
}

impl Iterator for LeastUsedIterator {
    type Item = (usize, Uuid);

    fn next(&mut self) -> Option<Self::Item> {
        pop_heap_with(&mut self.0, |(score_1, _, _), (score_2, _, _)| {
            score_1.partial_cmp(score_2)
        });

        self.0.pop().map(|(_, index, id)| (index, id))
    }
}

pub struct DataBuffer<
    A: Clone + Sized + Into<B>,
    B: Into<A> + FileExtension + Archive,
    const CAP: usize,
> where
    for<'a> &'a A: Into<Uuid>,
    for<'a> <B as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <B as Archive>::Archived: Deserialize<B, Strategy<Pool, rancor::Error>>,
    B: for<'a> rkyv::Serialize<
        rancor::Strategy<
            rkyv::ser::Serializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::ser::sharing::Share,
            >,
            rancor::Error,
        >,
    >,
{
    source: String,

    buffer: [Arc<RwLock<Option<A>>>; CAP],
    used_stack: [RwLock<Option<(usize, usize)>>; CAP],
    pub buffer_size: RwLock<usize>,

    empty_index_stack: RwLock<[Option<usize>; CAP]>,
    empty_index_size: RwLock<usize>,

    pub index_map: RwLock<HashMap<Uuid, usize>>,

    _phantom_data: PhantomData<B>,
}

impl<A: Clone + Sized + Into<B>, B: Into<A> + FileExtension + Archive, const CAP: usize>
    DataBuffer<A, B, CAP>
where
    for<'a> &'a A: Into<Uuid>,
    for<'a> <B as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <B as Archive>::Archived: Deserialize<B, Strategy<Pool, rancor::Error>>,
    B: for<'a> rkyv::Serialize<
        rancor::Strategy<
            rkyv::ser::Serializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::ser::sharing::Share,
            >,
            rancor::Error,
        >,
    >,
{
    pub fn new(source: String) -> Self {
        Self {
            source,
            buffer: array::from_fn(|_| Arc::new(RwLock::new(None))),
            used_stack: array::from_fn(|_| RwLock::new(None)),
            buffer_size: RwLock::new(0),
            empty_index_stack: RwLock::new(array::from_fn(|i| Some(CAP - i))),
            empty_index_size: RwLock::new(CAP),
            index_map: RwLock::new(HashMap::new()),
            _phantom_data: PhantomData,
        }
    }

    // access data from buffer
    pub fn try_access(&mut self, id: &Uuid) -> Result<RwLock<A>, BufferError> {
        todo!()
    }

    pub async fn max_try_access<const MAX: usize>(
        &mut self,
        id: &Uuid,
    ) -> Result<RwLock<A>, BufferError> {
        todo!()
    }

    pub async fn access(&mut self, id: &Uuid) -> Result<Arc<RwLock<Option<A>>>, BufferError> {
        let index_map = &*self.index_map.read().await;

        let index = match index_map.get(id) {
            Some(index) => *index,
            None => return Err(BufferError::DataNotFound),
        };

        let Some((_prev, cur)) = &mut *self.used_stack[index].write().await else {
            todo!()
        };

        let _ = mem::replace(cur, *cur + 1);

        Ok(self.buffer[index].clone())
    }

    // find least used data
    fn least_used(&self) -> Option<usize> {
        self.used_stack
            .iter()
            .map(|x| x.try_read())
            .enumerate()
            .filter(|(_index, x)| x.is_ok())
            .map(|(index, x)| (index, x.unwrap()))
            .filter(|(_index, x)| x.is_some())
            .map(|(index, x)| (index, x.unwrap()))
            .min_by(|(_, (acc_prev, acc_cur)), (_, (next_prev, next_cur))| {
                (next_prev + next_cur)
                    .partial_cmp(&(acc_prev + acc_cur))
                    .unwrap()
            })
            .map(|(x, _)| x)
    }

    pub async fn least_used_iter(&self) -> Option<LeastUsedIterator> {
        let data: Vec<(usize, usize, Uuid)> = self
            .index_map
            .read()
            .await
            .iter()
            .map(|(id, index)| (*index, self.used_stack[*index].try_read(), *id))
            .filter(|(_, used, _)| used.is_ok())
            .map(|(index, used, id)| (index, used.unwrap().unwrap(), id))
            .map(|(index, (prev_use, cur_use), id)| (index, prev_use + cur_use, id))
            .collect();

        if data.len() == 0 {
            return None;
        }

        Some(LeastUsedIterator::new(data))
    }

    pub fn de_increment(&mut self) {
        self.used_stack
            .iter_mut()
            .map(|x| x.try_write())
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap())
            .filter(|x| x.is_some())
            .for_each(|mut x| {
                let Some((prev, cur)) = &mut *x else { todo!() };
                swap(cur, prev);
                swap(cur, &mut 0);
            });
    }

    // insert values into the data_buffer
    pub async fn push(&mut self, value: A) -> Result<(), BufferError> {
        let buffer_size = &mut *self.buffer_size.write().await;

        if *buffer_size + 1 >= CAP {
            return Err(BufferError::OutOfSpace);
        };

        let index = *buffer_size;

        // check if enough space
        let used_stack = &mut *self.used_stack[index].write().await;

        let buffer = &mut *self.buffer[index].write().await;

        let _empty_index_stack = &mut *self.empty_index_stack.write().await;
        let empty_index_size = &mut *self.empty_index_size.write().await;

        let mut index_map = self.index_map.write().await;

        // insert values
        index_map.insert((&value).into(), index);
        let _ = mem::replace(used_stack, Some((0, 1)));
        let _ = mem::replace(buffer, Some(value));
        let _ = mem::replace(buffer_size, *buffer_size + 1);

        // add to free spot index (should be a min heap to ensure continuous blocks in the buffer)
        let _ = mem::replace(buffer_size, *buffer_size + 1);
        // empty_index_stack[*empty_index_size] = Some(index); don't need to replace with None - as remove and unload operations will write over values
        let _ = mem::replace(empty_index_size, *empty_index_size - 1);

        Ok(())
    }

    pub async fn load(&mut self, id: &Uuid) -> Result<(), BufferError> {
        let buffer_size = &mut *self.buffer_size.write().await;

        if *buffer_size + 1 >= CAP {
            return Err(BufferError::OutOfSpace);
        }

        let empty_index_stack = &mut *self.empty_index_stack.write().await;
        let empty_index_size = &mut *self.empty_index_size.write().await;

        let index = empty_index_stack[*empty_index_size].unwrap();

        let buffer = &mut *self.buffer[index].write().await;
        let used_stack = &mut *self.used_stack[index].write().await;

        let index_map = &mut *self.index_map.write().await;

        //load file
        let value: A = {
            let bytes = tokio::fs::read(&format!(
                "{}//{}.{}",
                self.source,
                id.to_string(),
                B::extension()
            ))
            .await?;

            from_bytes::<B, rancor::Error>(&bytes)?.into()
        };

        // insert value

        index_map.insert((&value).into(), index);
        let _ = mem::replace(used_stack, Some((0, 1)));
        let _ = mem::replace(buffer, Some(value));
        let _ = mem::replace(buffer_size, *buffer_size + 1);

        // add to free spot index (should be a min heap to ensure continuous blocks in the buffer)
        let _ = mem::replace(buffer_size, *buffer_size + 1);
        // empty_index_stack[*empty_index_size] = Some(index); don't need to replace with None - as remove and unload operations will write over values
        let _ = mem::replace(empty_index_size, *empty_index_size - 1);

        Ok(())
    }

    // remove values from the data_buffer
    pub async fn remove(&mut self, id: &Uuid) -> Result<(), BufferError> {
        // get access to all locks
        let mut index_map = self.index_map.write().await;

        let index = match index_map.get(id) {
            Some(index) => *index,
            None => {
                return Err(BufferError::DataNotFound)
            },
        };
        let used_stack = &mut *self.used_stack[index].write().await;

        let buffer_size = &mut *self.buffer_size.write().await;
        let buffer = &mut *self.buffer[index].write().await;

        let empty_index_stack = &mut *self.empty_index_stack.write().await;
        let empty_index_size = &mut *self.empty_index_size.write().await;

        // drop data
        let _ = mem::replace(used_stack, None);
        let _ = mem::replace(buffer, None);
        let _ = mem::replace(buffer_size, *buffer_size - 1);

        // add to free spot index (should be a min heap to ensure continuous blocks in the buffer)
        empty_index_stack[*empty_index_size] = Some(index);
        let _ = mem::replace(empty_index_size, *empty_index_size + 1);

        // removed id from index map
        index_map.remove(id);

        Ok(())
    }

    pub async fn unload(&mut self, id: &Uuid) -> Result<(), BufferError> {
        // get access to all locks
        let mut index_map = self.index_map.write().await;

        let index = match index_map.get(id) {
            Some(index) => *index,
            None => {
                return Err(BufferError::DataNotFound)
            },
        };
        let used_stack = &mut *self.used_stack[index].write().await;

        let buffer_size = &mut *self.buffer_size.write().await;
        let buffer = &mut *self.buffer[index].write().await;

        let empty_index_stack = &mut *self.empty_index_stack.write().await;
        let empty_index_size = &mut *self.empty_index_size.write().await;

        // drop data
        let _ = mem::replace(used_stack, None);
        let Some(value) = mem::replace(buffer, None) else {
            todo!()
        };
        let _ = mem::replace(buffer_size, *buffer_size - 1);

        // add to free spot index (should be a min heap to ensure continuous blocks in the buffer)
        empty_index_stack[*empty_index_size] = Some(index);
        let _ = mem::replace(empty_index_size, *empty_index_size + 1);

        // removed id from index map
        index_map.remove(id);

        // Export value
        let serial_value: B = value.into();

        let bytes = to_bytes::<rancor::Error>(&serial_value)?;

        tokio::fs::write(
            &format!("{}//{}.{}", self.source, id.to_string(), B::extension()),
            bytes.as_slice(),
        )
        .await?;

        Ok(())
    }
}
