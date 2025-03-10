use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use graph::GraphRoutes;
use partition::PartitionRoutes;
use serde::{Deserialize, Serialize};
use std::{array, fmt::Debug, str::FromStr, sync::Arc};
use tokio::sync::mpsc::{channel, Sender};
use uuid::Uuid;
use vector::VectorRoutes;

use crate::{
    db::{
        component::ids::{PartitionId, VectorId},
        AtomicCmd, Cmd, Response, Success,
    },
    vector::{Field, Vector, VectorSerial, VectorSpace},
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

pub mod graph;
pub mod partition;
pub mod vector;

pub(crate) struct HotRequest<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Sized + Send + Sync> {
    sender: Arc<Sender<(Cmd<A, B>, Sender<Response<A>>)>>,
}

pub(crate) trait AddRoute<F> {
    fn add_routes(self) -> Self;
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
    let shared_state = Arc::new(HotRequest {
        sender: Arc::new(sender.clone()),
    });

    let app: Router<Arc<HotRequest<A, B>>> = Router::new();

    let app = AddRoute::<GraphRoutes>::add_routes(app);
    let app = AddRoute::<VectorRoutes>::add_routes(app);
    let app = AddRoute::<PartitionRoutes>::add_routes(app);

    let app = app.with_state(shared_state);

    // Start the Axum server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
    panic!("Server stopped unexpectedly");
}
