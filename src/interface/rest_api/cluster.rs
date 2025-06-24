use std::{collections::HashMap, f32, fmt::Debug, sync::Arc};

use axum::{
    extract::State,
    routing::{get, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::channel;

use crate::{
    db::{AtomicCmd, Cmd, Response, Success},
    vector::{Field, VectorSpace},
};

use super::{AddRoute, HotRequest};

#[derive(Serialize, Deserialize)]
struct ClusterData {
    threshold: f32,
}

async fn create_cluster<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + From<f32> + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(ClusterData { threshold }): Json<ClusterData>,
) -> Json<bool> {
    let (tx, mut rx) = channel(64);
    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::CreateCluster {
                threshold: threshold.into(),
            }),
            tx,
        ))
        .await;

    match rx.recv().await {
        Some(Response::Done) => return Json(true),
        _ => return Json(false),
    }
}

#[derive(Serialize, Deserialize)]
struct ClusterPayload {
    clusters: HashMap<String, Vec<String>>,
}

async fn get_cluster<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + From<f32> + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(ClusterData { threshold }): Json<ClusterData>,
) -> Json<ClusterPayload> {
    // let (tx, mut rx) = channel(64);
    todo!();
    // let _ = state
    //     .sender
    //     .send((
    //         Cmd::Atomic(AtomicCmd::GetClusters {
    //             threshold: threshold.into(),
    //         }),
    //         tx,
    //     ))
    //     .await;

    // let mut payload = ClusterPayload {
    //     clusters: HashMap::new(),
    // };

    // let mut key = String::new();

    // while let Some(data) = rx.recv().await {
    //     match data {
    //         Response::Success(Success::Cluster(cluster_id)) => {
    //             key = cluster_id.0.into();
    //             payload.clusters.insert(cluster_id.0.into(), Vec::new());
    //         }
    //         Response::Success(Success::Vector(vector_id, _)) => {
    //             payload
    //                 .clusters
    //                 .get_mut(&key)
    //                 .unwrap()
    //                 .push(vector_id.0.into());
    //         }
    //         Response::Success(_) => {
    //             panic!()
    //         }
    //         Response::Fail => {
    //             todo!()
    //         }
    //         Response::Done => {
    //             break;
    //         }
    //     }
    // }

    // Json(payload)
}

pub(super) struct ClusterRoutes;

impl<
        A: Field<A> + Clone + Copy + Sized + Send + Sync + From<f32> + Debug + 'static,
        B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
    > AddRoute<ClusterRoutes> for Router<Arc<HotRequest<A, B>>>
where
    <B as TryFrom<Vec<f32>>>::Error: std::fmt::Debug,
    f32: From<A>,
{
    fn add_routes(self) -> Self {
        self.route("/cluster/new", put(create_cluster::<A, B>))
            .route("/cluster", get(get_cluster::<A, B>))

        // .route(
        //     "/graph/inter/by_partitions",
        //     get(inter_from_partitions::<A, B>),
        // )
    }
}
