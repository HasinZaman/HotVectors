use std::{
    array,
    cmp::Ordering,
    collections::HashMap,
    error::Error,
    fmt::Display,
    marker::PhantomData,
    mem::{self, swap},
    sync::Arc,
};

use heapify::{make_heap_with, pop_heap_with};
use tokio::{
    runtime::Runtime,
    sync::RwLock,
    task::{JoinHandle, JoinSet},
};
use tracing::{event, Level};
use uuid::Uuid;

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    from_bytes, rancor, to_bytes,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize,
};

use super::serial::FileExtension;

#[derive(PartialEq, Eq, Debug)]
pub enum BufferError {
    OutOfSpace,
    DataNotFound,

    FileNotFound,

    FailedLockAccess,
}

impl From<rancor::Error> for BufferError {
    fn from(value: rancor::Error) -> Self {
        todo!()
    }
}

impl From<std::io::Error> for BufferError {
    fn from(value: std::io::Error) -> Self {
        println!("err: {value}");

        match value.kind() {
            std::io::ErrorKind::NotFound => Self::FileNotFound,
            std::io::ErrorKind::PermissionDenied => todo!(),
            std::io::ErrorKind::ConnectionRefused => todo!(),
            std::io::ErrorKind::ConnectionReset => todo!(),
            std::io::ErrorKind::ConnectionAborted => todo!(),
            std::io::ErrorKind::NotConnected => todo!(),
            std::io::ErrorKind::AddrInUse => todo!(),
            std::io::ErrorKind::AddrNotAvailable => todo!(),
            std::io::ErrorKind::BrokenPipe => todo!(),
            std::io::ErrorKind::AlreadyExists => todo!(),
            std::io::ErrorKind::WouldBlock => todo!(),
            std::io::ErrorKind::InvalidInput => todo!(),
            std::io::ErrorKind::InvalidData => todo!(),
            std::io::ErrorKind::TimedOut => todo!(),
            std::io::ErrorKind::WriteZero => todo!(),
            std::io::ErrorKind::Interrupted => todo!(),
            std::io::ErrorKind::Unsupported => todo!(),
            std::io::ErrorKind::UnexpectedEof => todo!(),
            std::io::ErrorKind::OutOfMemory => todo!(),
            std::io::ErrorKind::Other => todo!(),

            _ => todo!(),
        }
        // value.downcast()

        // Have to look every implementation of this from
    }
}

impl From<tokio::sync::TryLockError> for BufferError {
    fn from(value: tokio::sync::TryLockError) -> Self {
        Self::FailedLockAccess
    }
}

impl Display for BufferError {
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

    pub async fn save(&self) {
        let iter = self.index_map.read().await.iter();

        for (id, index) in &*self.index_map.read().await {
            let index = *index;

            let Some(data) = &*self.buffer[index].read().await else {
                todo!()
            };

            let path = &format!("{}//{}.{}", self.source, id.to_string(), B::extension());
            event!(Level::INFO, "saving:{path}");

            let serial_value: B = data.clone().into();

            let bytes = to_bytes::<rancor::Error>(&serial_value).unwrap();

            tokio::fs::write(path, bytes.as_slice()).await.unwrap();
        }
    }

    pub async fn save_pooling(&self, runtime: &Runtime) {
        // let save_threads: Vec<JoinHandle<()>> = self.index_map
        //     .read()
        //     .await
        //     .iter()
        //     .map(|(_, index)| {
        //         let index = *index;

        //         let data = self.buffer[index].clone();
        //         runtime.spawn(async move{

        //             self.buffer[index]
        //                 .read();
        //         })
        //     })
        //     .collect();

        todo!()
    }

    // access data from buffer
    pub fn try_access(&mut self, id: &Uuid) -> Result<Arc<RwLock<Option<A>>>, BufferError> {
        let index_map = &*self.index_map.try_read()?;

        let index = match index_map.get(id) {
            Some(index) => *index,
            None => return Err(BufferError::DataNotFound),
        };

        let Some((_prev, cur)) = &mut *self.used_stack[index].try_write()? else {
            todo!()
        };

        let _ = mem::replace(cur, *cur + 1);

        Ok(self.buffer[index].clone())
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
                    .unwrap_or(Ordering::Equal)
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

        if *buffer_size == CAP {
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

        // add to free spot index (should be a min heap to ensure continuous blocks in the buffer)
        let _ = mem::replace(buffer_size, *buffer_size + 1);
        // empty_index_stack[*empty_index_size] = Some(index); don't need to replace with None - as remove and unload operations will write over values
        let _ = mem::replace(empty_index_size, *empty_index_size - 1);

        Ok(())
    }

    pub async fn load(&mut self, id: &Uuid) -> Result<(), BufferError> {
        let index_map = &mut *self.index_map.write().await;

        if index_map.contains_key(id) {
            return Ok(());
        }

        let buffer_size = &mut *self.buffer_size.write().await;

        if *buffer_size == CAP {
            return Err(BufferError::OutOfSpace);
        }

        let empty_index_stack = &mut *self.empty_index_stack.write().await;
        let empty_index_size = &mut *self.empty_index_size.write().await;

        let index = empty_index_stack[*empty_index_size].unwrap();

        let buffer = &mut *self.buffer[index].write().await;
        let used_stack = &mut *self.used_stack[index].write().await;

        //load file
        let value: A = {
            let file_path = &format!("{}//{}.{}", self.source, id.to_string(), B::extension());

            event!(Level::DEBUG, "Load {file_path}");

            let bytes = tokio::fs::read(&file_path).await?;

            from_bytes::<B, rancor::Error>(&bytes)?.into()
        };

        // insert value

        index_map.insert((&value).into(), index);
        let _ = mem::replace(used_stack, Some((0, 1)));
        let _ = mem::replace(buffer, Some(value));

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
            None => return Err(BufferError::DataNotFound),
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
        event!(Level::INFO, "unload :- id {}", id);
        // get access to all locks
        let mut index_map = self.index_map.write().await;

        let index = match index_map.get(id) {
            Some(index) => *index,
            None => return Err(BufferError::DataNotFound),
        };

        event!(Level::INFO, "unload :- index {}", index);
        let used_stack = &mut *self.used_stack[index].write().await;

        let buffer_size = &mut *self.buffer_size.write().await;
        let buffer = &mut *self.buffer[index].write().await;

        let empty_index_stack = &mut *self.empty_index_stack.write().await;
        let empty_index_size = &mut *self.empty_index_size.write().await;

        // drop data
        event!(Level::DEBUG, "Replacing value @ index with None");
        let _ = mem::replace(used_stack, None);
        let Some(value) = mem::replace(buffer, None) else {
            todo!()
        };
        let _ = mem::replace(buffer_size, *buffer_size - 1);

        // add to free spot index (should be a min heap to ensure continuous blocks in the buffer)
        empty_index_stack[*empty_index_size] = Some(index);
        let _ = mem::replace(empty_index_size, *empty_index_size + 1);
        event!(Level::DEBUG, "Empty index stack :- {:?}", empty_index_stack);

        // removed id from index map
        index_map.remove(id);

        // Export value
        let path = &format!("{}//{}.{}", self.source, id.to_string(), B::extension());
        event!(Level::INFO, "saving: {path}");

        let serial_value: B = value.into();

        let bytes = to_bytes::<rancor::Error>(&serial_value)?;

        tokio::fs::write(path, bytes.as_slice()).await?;

        Ok(())
    }

    pub async fn unload_and_load(
        &mut self,
        unload_id: &Uuid,
        load_id: &Uuid,
    ) -> Result<(), BufferError> {
        event!(Level::INFO, "Unloading and load");
        let mut index_map = self.index_map.write().await;
        event!(Level::DEBUG, "Acquired index_map");

        let index = match index_map.get(unload_id) {
            Some(index) => *index,
            None => return Err(BufferError::DataNotFound),
        };
        event!(Level::DEBUG, "Acquired index");

        let used_stack = &mut *self.used_stack[index].write().await;
        event!(Level::DEBUG, "Acquired used_stack");

        let buffer = &mut *self.buffer[index].write().await;
        event!(Level::DEBUG, "Acquired buffer");

        //load file
        let load_value: A = {
            let file_path = &format!(
                "{}//{}.{}",
                self.source,
                load_id.to_string(),
                B::extension()
            );

            event!(Level::DEBUG, "Load {file_path}");

            let bytes = tokio::fs::read(file_path).await?;

            from_bytes::<B, rancor::Error>(&bytes)?.into()
        };
        event!(Level::DEBUG, "Loaded file");

        // drop data
        *used_stack = Some((0, 1));
        event!(Level::DEBUG, "Updated used_stack");

        index_map.insert((&load_value).into(), index);
        event!(Level::DEBUG, "Updated index_map");

        let Some(unload_value) = mem::replace(buffer, Some(load_value)) else {
            todo!()
        };
        event!(Level::DEBUG, "Swapped unload_value with load_value");

        // removed id from index map
        index_map.remove(unload_id);

        // Export value
        {
            let path = &format!(
                "{}//{}.{}",
                self.source,
                unload_id.to_string(),
                B::extension()
            );
            event!(Level::INFO, "saving: {path}");
            let serial_value: B = unload_value.into();

            let bytes = to_bytes::<rancor::Error>(&serial_value)?;

            tokio::fs::write(path, bytes.as_slice()).await?;
        }
        Ok(())
    }
    pub async fn unload_and_push(
        &mut self,
        unload_id: &Uuid,
        push_value: A,
    ) -> Result<(), BufferError> {
        event!(Level::INFO, "Unloading and Pushing");
        let mut index_map = self.index_map.write().await;
        event!(Level::DEBUG, "Acquired index_map");

        let index = match index_map.get(unload_id) {
            Some(index) => *index,
            None => return Err(BufferError::DataNotFound),
        };
        let used_stack = &mut *self.used_stack[index].write().await;
        event!(Level::DEBUG, "Acquired used_stack");

        let buffer = &mut *self.buffer[index].write().await;
        event!(Level::DEBUG, "Acquired buffer");

        // drop data
        *used_stack = Some((0, 1));

        index_map.insert((&push_value).into(), index);

        let Some(unload_value) = mem::replace(buffer, Some(push_value)) else {
            todo!()
        };

        // removed id from index map
        index_map.remove(unload_id);

        // Export value
        let path = &format!(
            "{}//{}.{}",
            self.source,
            unload_id.to_string(),
            B::extension()
        );
        event!(Level::INFO, "saving: {path}");

        let serial_value: B = unload_value.into();

        let bytes = to_bytes::<rancor::Error>(&serial_value)?;

        tokio::fs::write(path, bytes.as_slice()).await?;

        Ok(())
    }
}
