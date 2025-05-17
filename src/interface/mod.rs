use std::sync::Arc;

use tokio::sync::mpsc::Sender;

use crate::{
    db::{Cmd, Response},
    vector::{Field, VectorSpace},
};

// #[cfg(feature = "rest")]
// pub mod rest;
pub mod rest_api;

pub mod conn_pool;

pub struct HotRequest<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Sized + Send + Sync> {
    sender: Arc<Sender<(Cmd<A, B>, Sender<Response<A>>)>>,
}

// pub type InputLoop = fn(Sender<(Cmd<A, B>, Sender<Response<A>>)) -> !;
