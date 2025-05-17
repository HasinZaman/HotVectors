use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, str::FromStr, sync::Arc};
use tokio::sync::mpsc::channel;
use uuid::Uuid;

use crate::{
    db::{component::ids::PartitionId, AtomicCmd, Cmd, Response, Success},
    interface::HotRequest,
    vector::{Field, VectorSerial, VectorSpace},
};

use super::{partition::RequestedPartitions, AddRoute};

#[derive(Serialize, Deserialize)]
pub struct VectorData<T> {
    vector: Vec<T>,
}
// read vectors

#[derive(Serialize, Deserialize)]
pub(super) struct FromPartitionPayload(Vec<(String, Vec<(String, Vec<f32>)>)>);

pub(super) async fn from_partitions<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(RequestedPartitions { ids }): Json<RequestedPartitions>,
) -> Json<FromPartitionPayload>
where
    f32: From<A>,
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    println!("partition vector REQUEST");

    let (tx, mut rx) = channel(64);

    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::Partitions {
                ids: ids
                    .into_iter()
                    .map(|x| PartitionId(Uuid::from_str(&x).unwrap()))
                    .collect(),
            }),
            tx,
        ))
        .await;

    let mut json_data: Vec<(String, Vec<(String, Vec<f32>)>)> = Vec::new();

    while let Some(Response::Success(data)) = rx.recv().await {
        match data {
            Success::Partition(partition_id) => {
                json_data.push(((*partition_id).to_string(), Vec::new()));
            }
            Success::Vector(vector_id, VectorSerial(vector)) => {
                json_data
                    .last_mut()
                    .expect("Must provide partition Id before streaming vectors")
                    .1
                    .push((
                        (*vector_id).to_string(),
                        vector.into_iter().map(|x| f32::from(x)).collect(),
                    ));
            }
            _ => panic!(""),
        };
    }

    Json(FromPartitionPayload(json_data))
}

#[derive(Serialize, Deserialize)]
pub(super) struct KNNPayload<A> {
    vector: Vec<A>,
    k: usize,
}
pub(super) async fn knn<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(KNNPayload { vector, k }): Json<KNNPayload<f32>>,
) -> Json<String>
where
    f32: From<A>,
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    println!("KNN REQUEST");

    let (tx, mut rx) = channel(64);

    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::Knn {
                vector: B::try_from(vector).unwrap(),
                k: k,
                transaction_id: None,
            }),
            tx,
        ))
        .await;

    let mut data: Vec<(String, A)> = Vec::new();

    while let Some(Response::Success(knn_data)) = rx.recv().await {
        let Success::Knn(vec_id, dist) = knn_data else {
            panic!("")
        };

        data.push(((*vec_id).to_string(), dist));
    }

    Json(format!("{:?}", data))
}

// vector insertions
#[derive(Serialize, Deserialize)]
pub struct InsertPayload {
    vector_id: String,
}

pub(super) async fn insert<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(payload): Json<VectorData<f32>>,
) -> Json<InsertPayload>
where
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    let vector = payload.vector;

    println!("INSERT VECTOR");
    println!("{:?}", vector);

    let (tx, mut rx) = channel(2);

    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::InsertVector {
                vector: B::try_from(vector).unwrap(),
                transaction_id: None,
            }),
            tx,
        ))
        .await;

    let inserted_id = match rx.recv().await {
        Some(Response::Success(Success::Vector(id, _))) => id,
        None => {
            todo!()
        }
        _ => {
            todo!()
        }
    };
    match rx.recv().await {
        Some(Response::Done) => Json(InsertPayload {
            vector_id: inserted_id.0.to_string(),
        }),
        None => {
            todo!()
        }
        _ => {
            todo!()
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct BatchInsertPayload {
    vector_ids: Vec<String>,
}
pub(super) async fn batch_insert<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(payload): Json<Vec<VectorData<f32>>>,
) -> Json<BatchInsertPayload>
where
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    let mut data_recvs = Vec::with_capacity(payload.len());
    let mut results = Vec::with_capacity(payload.len());
    println!("INSERT BATCH VECTOR");

    let (tx, rx) = channel(payload.len() + 1);

    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::BatchInsertVectors {
                vectors: payload
                    .into_iter()
                    .map(|load| B::try_from(load.vector).unwrap())
                    .collect(),
                transaction_id: None,
            }),
            tx.clone(),
        ))
        .await;

    data_recvs.push(rx);

    for mut rx in data_recvs {
        let inserted_id = match rx.recv().await {
            Some(Response::Success(Success::Vector(id, _))) => id,
            None => {
                todo!()
            }
            bad => {
                println!("{bad:?}");
                todo!()
            }
        };
        // rx.recv().await;
        results.push(inserted_id.0.to_string());
    }

    return Json(BatchInsertPayload {
        vector_ids: results,
    });
}

pub(super) struct VectorRoutes;

impl<
        A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
        B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
    > AddRoute<VectorRoutes> for Router<Arc<HotRequest<A, B>>>
where
    <B as TryFrom<Vec<f32>>>::Error: std::fmt::Debug,
    f32: From<A>,
{
    fn add_routes(self) -> Self {
        self.route("/vector/insert", post(insert::<A, B>))
            .route("/vector/insert_batch", post(batch_insert::<A, B>))
            .route("/vector/from_partitions", get(from_partitions::<A, B>))
            .route("/vector/knn", get(knn::<A, B>))
    }
}
