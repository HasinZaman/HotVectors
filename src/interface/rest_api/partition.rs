use axum::{extract::State, routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};
use tokio::sync::mpsc::channel;

use crate::{
    db::{AtomicCmd, Cmd, Response, Success},
    vector::{Field, VectorSpace},
};

use super::{AddRoute, HotRequest};

#[derive(Serialize, Deserialize)]
pub(super) struct RequestedPartitions {
    pub ids: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    count: usize,
}

async fn partition_meta_data<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Sized + Send + Sync>(
    State(state): State<Arc<HotRequest<A, B>>>,
) -> Json<String>
where
    f32: From<A>,
{
    println!("META REQUEST");

    let (tx, mut rx) = channel(64);

    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::GetMetaData {
                transaction_id: None,
            }),
            tx,
        ))
        .await;

    let mut data: Vec<(String, usize, Vec<f32>)> = Vec::new();

    while let Some(Response::Success(meta_data)) = rx.recv().await {
        let Success::MetaData(id, size, vector_serial) = meta_data else {
            panic!("")
        };

        data.push((
            (*id).to_string(),
            size,
            vector_serial.0.into_iter().map(|x| f32::from(x)).collect(),
        ));
    }

    Json(format!("{:?}", data))
}

pub(super) struct PartitionRoutes;

impl<
        A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
        B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
    > AddRoute<PartitionRoutes> for Router<Arc<HotRequest<A, B>>>
where
    <B as TryFrom<Vec<f32>>>::Error: std::fmt::Debug,
    f32: From<A>,
{
    fn add_routes(self) -> Self {
        self.route("/metadata", get(partition_meta_data::<A, B>))
    }
}
