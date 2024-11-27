// pub struct LogBuffer();

use uuid::Uuid;

use crate::vector::{Field, VectorSpace};

use super::AtomicCmd;
#[derive(Debug)]
pub enum State<A> {
    Start(A),
    End(A),
}
pub struct Log<A: Field<A>, B: VectorSpace<A>, const MEMORY_BUFF: usize> {
    buffer: [State<AtomicCmd<A, B>>; MEMORY_BUFF],
    index: usize,
}

impl<A: Field<A>, B: VectorSpace<A>, const MEMORY_BUFF: usize> Log<A, B, MEMORY_BUFF> {
    pub fn flush(&mut self) {
        todo!()
    }

    pub fn load(&mut self) {
        todo!()
    }

    pub fn push(&mut self, cmd: State<AtomicCmd<A, B>>) {
        todo!()
    }
    //move buffer
}
