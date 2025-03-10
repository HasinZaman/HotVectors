use axum::{extract::State, routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, str::FromStr, sync::Arc};
use tokio::sync::mpsc::channel;
use uuid::Uuid;

use crate::{
    db::{
        component::ids::{PartitionId, VectorId},
        AtomicCmd, Cmd, Response, Success,
    },
    vector::{Field, VectorSpace},
};

use super::{partition::RequestedPartitions, AddRoute, HotRequest};

#[derive(Serialize, Deserialize)]
struct PartitionEdgesPayload(Vec<(String, String, f32)>);

async fn intra_from_partitions<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(RequestedPartitions { ids }): Json<RequestedPartitions>,
) -> Json<PartitionEdgesPayload>
where
    f32: From<A>,
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    let mut json_data: Vec<(String, String, f32)> = Vec::new();
    for id in ids {
        let (tx, mut rx) = channel(64);

        let _ = state
            .sender
            .send((
                Cmd::Atomic(AtomicCmd::GetPartitionGraph {
                    transaction_id: None,
                    partition_id: PartitionId(Uuid::from_str(&id).unwrap()),
                }),
                tx,
            ))
            .await;

        while let Some(Response::Success(data)) = rx.recv().await {
            match data {
                Success::Edge(VectorId(source), VectorId(target), dist) => {
                    json_data.push(((source).to_string(), (target).to_string(), f32::from(dist)));
                }
                _ => panic!(""),
            };
        }
    }

    Json(PartitionEdgesPayload(json_data))
}

async fn inter_from_partitions<
    A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
    B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
>(
    State(state): State<Arc<HotRequest<A, B>>>,
    Json(RequestedPartitions { ids }): Json<RequestedPartitions>,
) -> Json<PartitionEdgesPayload>
where
    f32: From<A>,
    <B as TryFrom<Vec<f32>>>::Error: Debug,
{
    let mut json_data: Vec<(String, String, f32)> = Vec::new();
    for id in ids {
        let (tx, mut rx) = channel(64);

        let _ = state
            .sender
            .send((
                Cmd::Atomic(AtomicCmd::GetInterPartitionGraph {
                    transaction_id: None,
                    partition_id: PartitionId(Uuid::from_str(&id).unwrap()),
                }),
                tx,
            ))
            .await;

        while let Some(Response::Success(data)) = rx.recv().await {
            match data {
                Success::InterEdge((_, VectorId(source)), (_, VectorId(target)), dist) => {
                    json_data.push(((source).to_string(), (target).to_string(), f32::from(dist)));
                }
                _ => panic!(""),
            };
        }
    }

    Json(PartitionEdgesPayload(json_data))
}

pub(super) struct GraphRoutes;

impl<
        A: Field<A> + Clone + Copy + Sized + Send + Sync + Debug + 'static,
        B: VectorSpace<A> + Sized + Send + Sync + 'static + TryFrom<Vec<f32>>,
    > AddRoute<GraphRoutes> for Router<Arc<HotRequest<A, B>>>
where
    <B as TryFrom<Vec<f32>>>::Error: std::fmt::Debug,
    f32: From<A>,
{
    fn add_routes(self) -> Self {
        self.route(
            "/graph/intra/by_partitions",
            get(intra_from_partitions::<A, B>),
        )
        .route(
            "/graph/inter/by_partitions",
            get(inter_from_partitions::<A, B>),
        )
    }
}
