use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::{array, fmt::Debug, sync::Arc};
use tokio::sync::mpsc::{channel, Sender};

use crate::{
    db::{AtomicCmd, Cmd, Response, Success},
    vector::{Field, Vector, VectorSpace},
};

impl<A: PartialEq + Clone + Copy + From<f32>, const CAP: usize> TryFrom<Vec<f32>>
    for Vector<A, CAP>
{
    type Error = ();

    fn try_from(value: Vec<f32>) -> Result<Self, Self::Error> {
        if value.len() != CAP {
            return Err(());
        }

        Ok(Vector(array::from_fn(move |index| value[index].into())))
    }
}

#[derive(Serialize, Deserialize)]
struct VectorData<T> {
    vector: Vec<T>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    count: usize,
}

struct AppState<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Sized + Send + Sync> {
    sender: Arc<Sender<(Cmd<A, B>, Sender<Response<A>>)>>,
}

async fn insert_vector_route<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<AppState<A, B>>>,
    Json(payload): Json<VectorData<f32>>,
) -> Json<String>
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

    match rx.recv().await {
        Some(Response::Done) => Json("Vector received".to_string()),
        None => {
            todo!()
        }
        _ => {
            todo!()
        }
    }
}

async fn insert_batch_vector_route<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<AppState<A, B>>>,
    Json(payload): Json<Vec<VectorData<f32>>>,
) -> Json<String>
where
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    let (tx, mut rx) = channel(2);

    let mut results = Vec::with_capacity(payload.len());
    println!("INSERT BATCH VECTOR");
    for vector in payload {
        let vector = vector.vector;
        println!("{:?}", vector);

        let _ = state
            .sender
            .send((
                Cmd::Atomic(AtomicCmd::InsertVector {
                    vector: B::try_from(vector).unwrap(),
                    transaction_id: None,
                }),
                tx.clone(),
            ))
            .await;

        results.push(rx.recv().await);
    }

    match results.iter().all(|x| x.is_some()) {
        true => Json("Vector received".to_string()),
        false => {
            todo!()
        }
    }
}

async fn metadata_route<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Sized + Send + Sync>(
    State(state): State<Arc<AppState<A, B>>>,
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

#[derive(Serialize, Deserialize)]
struct KNNPayload<A> {
    vector: Vec<A>,
    k: usize,
}
async fn knn_route<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<AppState<A, B>>>,
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

        data.push((
            (*vec_id).to_string(),
            dist,
        ));
    }

    Json(format!("{:?}", data))
}

pub async fn input_loop<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    sender: Sender<(Cmd<A, B>, Sender<Response<A>>)>,
) -> !
where
    <B as TryFrom<Vec<f32>>>::Error: Debug,
    f32: From<A>,
{
    let shared_state = Arc::new(AppState {
        sender: Arc::new(sender.clone()),
    });

    let app = Router::new()
        .route("/insert", post(insert_vector_route::<A, B>))
        .route("/insert_batch", post(insert_batch_vector_route::<A, B>))
        .route("/metadata", post(metadata_route::<A, B>))
        .route("/knn", post(knn_route::<A, B>))
        .with_state(shared_state);

    // Start the Axum server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
    panic!("Server stopped unexpectedly");
}
