use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use tokio::sync::{mpsc::Sender, RwLock};
use uuid::Uuid;

use crate::{
    db::{
        component::{
            cluster::ClusterSet,
            ids::{ClusterId, PartitionId, VectorId},
            meta::Meta,
        },
        Response, Success,
    },
    vector::{Extremes, VectorSerial, VectorSpace},
};

pub fn stream_cluster_meta<
    V: VectorSpace<f32>
        + Sized
        + Clone
        + Copy
        + Send
        + Sync
        + Sized
        + From<VectorSerial<f32>>
        + Extremes
        + PartialEq
        + 'static
        + Debug
        + From<Vec<f32>>
        + Into<Vec<f32>>,
>(
    // resources
    cluster_sets: Arc<RwLock<Vec<ClusterSet<f32>>>>,

    //input
    cluster_id: ClusterId,
    threshold: f32,

    // output
    tx: Sender<Response<f32>>,
) where
    VectorSerial<f32>: From<V>,
{
    tokio::spawn(async move {
        let cluster_sets = &*cluster_sets.read().await;
        match cluster_sets.binary_search_by(|x| {
            x.threshold
                .partial_cmp(&threshold)
                .unwrap_or(Ordering::Equal) // == Ordering::Equal
        }) {
            Ok(pos) => {
                let cluster_set = &cluster_sets[pos];

                if cluster_id.0 == Uuid::nil() {
                    for cluster_id in cluster_set.get_clusters() {
                        cluster_meta::<V>(&tx, cluster_set, cluster_id).await;
                    }

                    let _ = tx.send(Response::Done).await;
                } else {
                    cluster_meta::<V>(&tx, cluster_set, cluster_id).await;

                    let _ = tx.send(Response::Done).await;
                }
            }
            Err(_) => {
                let _ = tx.send(Response::Fail).await;
            }
        }
    });
}

async fn cluster_meta<
    V: VectorSpace<f32>
        + Sized
        + Clone
        + Copy
        + Send
        + Sync
        + Sized
        + From<VectorSerial<f32>>
        + Extremes
        + PartialEq
        + 'static
        + Debug
        + From<Vec<f32>>
        + Into<Vec<f32>>,
>(
    tx: &Sender<Response<f32>>,
    cluster_set: &ClusterSet<f32>,
    cluster_id: ClusterId,
) where
    VectorSerial<f32>: From<V>,
{
    // cluster id
    let _ = tx
        .send(Response::Success(Success::Cluster(cluster_id)))
        .await;
    // get all vectors in cluster
    let vector_ids = cluster_set.get_cluster_members(cluster_id).unwrap();
    // send cluster side
    let _ = tx
        .send(Response::Success(Success::UInt(vector_ids.len())))
        .await;
    // send vector_ids
    for vec_id in vector_ids {
        let _ = tx
            .send(Response::Success(Success::Vector(
                vec_id,
                VectorSerial(Vec::new()),
            )))
            .await;
    }
    // send cluster centroid
    let _ = tx
        .send(Response::Success(Success::Vector(
            VectorId(Uuid::nil()),
            VectorSerial(Vec::new()),
        )))
        .await;
}

pub fn stream_partition_meta<
    V: VectorSpace<f32>
        + Sized
        + Clone
        + Copy
        + Send
        + Sync
        + Sized
        + From<VectorSerial<f32>>
        + Extremes
        + PartialEq
        + 'static
        + Debug
        + From<Vec<f32>>
        + Into<Vec<f32>>,
>(
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<f32, V>>>>>>,
    partition_id: PartitionId,
    tx: Sender<Response<f32>>,
) where
    VectorSerial<f32>: From<V>,
{
    if partition_id.0 == Uuid::nil() {
        tokio::spawn(async move {
            let meta_data = &*meta_data.read().await;

            let mut visited = HashSet::new();

            while visited.len() != meta_data.len() {
                let iter: Vec<_> = meta_data
                    .iter()
                    .filter(|(id, _)| !visited.contains(*id))
                    .collect();

                for (id, data) in iter {
                    partition_meta(&tx, data).await;

                    visited.insert(*id);
                }
            }

            let _ = tx.send(Response::Done).await;
        });
    } else {
        tokio::spawn(async move {
            let meta_data = &*meta_data.read().await;

            let Some(data) = meta_data.get(&partition_id.0) else {
                todo!()
            };

            partition_meta(&tx, data).await;

            let _ = tx.send(Response::Done).await;
        });
    }
}

#[inline]
async fn partition_meta<
    V: VectorSpace<f32>
        + Sized
        + Clone
        + Copy
        + Send
        + Sync
        + Sized
        + From<VectorSerial<f32>>
        + Extremes
        + PartialEq
        + 'static
        + Debug
        + From<Vec<f32>>
        + Into<Vec<f32>>,
>(
    tx: &Sender<Response<f32>>,
    data: &Arc<RwLock<Meta<f32, V>>>,
) where
    VectorSerial<f32>: From<V>,
{
    let data = &*data.read().await;
    let _ = tx
        .send(Response::Success(Success::Partition(data.id)))
        .await;

    let _ = tx.send(Response::Success(Success::UInt(data.size))).await;

    let _ = tx
        .send(Response::Success(Success::Vector(
            VectorId(Uuid::nil()),
            VectorSerial::from(data.centroid.clone()),
        )))
        .await;
}
