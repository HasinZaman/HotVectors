use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use tokio::sync::{mpsc::Sender, RwLock};
use uuid::Uuid;

use crate::{
    db::{component::meta::Meta, Response, Success},
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

pub mod knn;

pub async fn stream_meta_data<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy + PartialEq + Extremes + Into<VectorSerial<A>>,
>(
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<A, B>>>>>>,
    sender: &Sender<Response<A>>,
) -> Result<(), ()> {
    let meta_data = &*meta_data.read().await;

    let mut visited = HashSet::new();

    while visited.len() != meta_data.len() {
        let iter: Vec<_> = meta_data
            .iter()
            .filter(|(id, _)| !visited.contains(*id))
            // .map(|(id, data)| (id, data.try_read()))
            // .filter(|(_, data)| data.is_ok())
            // .map(|(id, data)| (id, data.unwrap()))
            .collect();

        for (id, data) in iter {
            let data = &*data.read().await;

            let _ = sender
                .send(Response::Success(Success::MetaData(
                    id.to_string(),
                    data.size,
                    data.centroid.clone().into(),
                )))
                .await;

            visited.insert(*id);
        }
    }

    Ok(())
}
