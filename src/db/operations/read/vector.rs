use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    sync::Arc,
};

use burn::{prelude::Backend, tensor::backend::AutodiffBackend};
use rkyv::vec;
use tokio::sync::{
    mpsc::{self, Receiver, Sender},
    oneshot, RwLock,
};
use uuid::Uuid;

use tracing::{event, Level};

use crate::{
    db::{
        banker::{AccessMode, AccessResponse, BankerMessage},
        component::{
            cluster::ClusterSet,
            data_buffer::{DataBuffer, Global},
            ids::{ClusterId, PartitionId, VectorId},
            partition::{Partition, PartitionMembership, PartitionSerial, VectorEntry},
            umap::InferenceError,
        },
        ProjectionMode, Response, Source, Success,
    },
    resolve_buffer,
    vector::{Extremes, Vector, VectorSerial, VectorSpace},
};

pub fn get_vectors<
    B: Backend + AutodiffBackend + Send + Sync,
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
        + Into<Vec<f32>>
        + for<'a> From<&'a [f32]>,
    LV: VectorSpace<f32>
        + Clone
        + Copy
        + Send
        + Sync
        + for<'a> From<&'a [f32]>
        + From<Vec<f32>>
        + Into<Vec<f32>>
        + From<VectorSerial<f32>>
        + PartialEq
        + Extremes
        + Debug
        + 'static,
    // K: Send + Sync + Clone + Debug + 'static + From<usize>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    // const VECTOR_CAP_2: usize,
    const MAX_LOADED: usize,
>(
    // resources
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<f32, V, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<f32>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
    cluster_sets: Arc<RwLock<Vec<ClusterSet<f32>>>>,
    partition_membership: Arc<RwLock<PartitionMembership>>,
    access_tx: Sender<BankerMessage>,
    projection_tx: Sender<(V, oneshot::Sender<Result<LV, InferenceError>>)>,

    // input
    transaction_id: Option<Uuid>,
    source: Source<f32>,
    dim_projection: Option<usize>,
    projection_mode: ProjectionMode,

    // output
    tx: Sender<Response<f32>>,
) where
    VectorSerial<f32>: From<V> + From<LV>,
    Vector<f32, VECTOR_CAP>: From<V>,
    // Vector<f32, VECTOR_CAP_2>: From<LV>,
    Vector<f32, 2>: From<V>,
{
    let (vector_tx, vector_rx) = mpsc::channel(64);
    let _vector_loader_thread = match source {
        Source::VectorId(vector_id) => tokio::spawn(async move {
            stream_vector_by_id(
                transaction_id,
                vector_tx,
                vector_id,
                access_tx,
                partition_buffer,
                partition_membership,
            )
            .await;
        }),
        Source::PartitionId(partition_id) => tokio::spawn(async move {
            stream_vectors_from_partition(
                transaction_id,
                vector_tx,
                partition_id,
                access_tx,
                partition_buffer,
            )
            .await;
        }),
        Source::ClusterId(cluster_id, threshold) => tokio::spawn(async move {
            stream_vectors_from_cluster(
                vector_tx,
                cluster_id,
                threshold,
                cluster_sets,
                access_tx,
                partition_buffer,
                partition_membership,
            )
            .await;
        }),
    };
    // project points if required
    let _vector_tx_thread = match dim_projection {
        Some(dim) => tokio::spawn(async move {
            let (dim_lowered_tx, dim_lowered_rx) = mpsc::channel(64);

            let dim_lower_thread = tokio::spawn(async move {
                stream_lower_dim_vector(vector_rx, projection_tx, dim_lowered_tx).await;
            });

            let stream_thread = stream_projected_vectors::<LV>(tx, projection_mode, dim_lowered_rx).await;

            // let _ = tokio::join!(dim_lower_thread, stream_thread);
        }),
        None => tokio::spawn(async move {
            stream_projected_vectors::<V>(tx, projection_mode, vector_rx).await;
        }),
    };
}

pub async fn stream_vectors_from_cluster<
    // B: Backend + AutodiffBackend + Send + Sync,
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
    // LV: VectorSpace<f32> + Send + Sync + for<'a> From<&'a [f32]> + From<Vec<f32>> + Into<Vec<f32>>,
    // K: Send + Sync + Clone + Debug + 'static + From<usize>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    // output
    vector_tx: Sender<Option<VectorEntry<f32, V>>>,

    // input
    cluster_id: ClusterId,
    threshold: f32,

    // resources
    cluster_sets: Arc<RwLock<Vec<ClusterSet<f32>>>>,
    access_tx: Sender<BankerMessage>,
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<f32, V, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<f32>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
    partition_membership: Arc<RwLock<PartitionMembership>>,
) where
    VectorSerial<f32>: From<V>,
    Vector<f32, VECTOR_CAP>: From<V>,
    Vector<f32, 2>: From<V>,
{
    let vectors = {
        let cluster_sets = &*cluster_sets.read().await;

        match cluster_sets.binary_search_by(|x| {
            x.threshold
                .partial_cmp(&threshold)
                .unwrap_or(Ordering::Equal) // == Ordering::Equal
        }) {
            Ok(pos) => {
                let cluster_set = &cluster_sets[pos];

                cluster_set.get_cluster_members(cluster_id).unwrap()
            }
            Err(_) => {
                let _ = vector_tx.send(None).await;
                // let _ = tx.send(Response::Fail).await;
                return;
            }
        }
    };
    let transaction_id = Uuid::new_v4();
    let required_partitions: HashMap<PartitionId, HashSet<VectorId>> = loop {
        let partition_membership = &*partition_membership.read().await;

        let mut partition_vectors_pairs: HashMap<PartitionId, HashSet<VectorId>> = HashMap::new();

        for vector_id in &vectors {
            let partition_id: PartitionId =
                partition_membership.get_partition_id(*vector_id).unwrap();

            partition_vectors_pairs
                .entry(partition_id)
                .or_default()
                .insert(*vector_id);
        }

        let (tx, rx) = oneshot::channel();

        let _ = access_tx
            .send(BankerMessage::RequestAccess {
                transaction_id: transaction_id,
                partitions: partition_vectors_pairs
                    .keys()
                    .map(|id| (*id, AccessMode::Read))
                    .collect(),
                respond_to: tx,
            })
            .await;

        match rx.await {
            Ok(AccessResponse::Granted) => break partition_vectors_pairs,
            _ => {}
        }
    };
    for (partition_id, required_vectors) in required_partitions.into_iter() {
        {
            let partition_buffer = &mut *partition_buffer.write().await;
            let partition = resolve_buffer!(ACCESS, partition_buffer, partition_id);

            let Some(partition) = &*partition.read().await else {
                todo!()
            };

            for i in 0..partition.size {
                let Some(vector_entry) = partition.vectors[i] else {
                    todo!()
                };

                if !required_vectors.contains(&VectorId(vector_entry.id)) {
                    continue;
                }

                let _ = vector_tx.send(Some(vector_entry.clone())).await;
            }
        }

        
        // TODO - drop partition before releasing?
        let _ = access_tx
            .send(
                BankerMessage::ReleaseAccess{
                    transaction_id: transaction_id,
                    partitions: vec![partition_id]
            })
            .await;

    }


    let _ = vector_tx.send(None).await;
}

pub async fn stream_vectors_from_partition<
    // B: Backend + AutodiffBackend + Send + Sync,
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
    // LV: VectorSpace<f32> + Send + Sync + for<'a> From<&'a [f32]> + From<Vec<f32>> + Into<Vec<f32>>,
    // K: Send + Sync + Clone + Debug + 'static + From<usize>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    // transaction id
    transaction_id: Option<Uuid>,

    // output
    vector_tx: Sender<Option<VectorEntry<f32, V>>>,

    // input
    partition_id: PartitionId,

    // required resources
    access_tx: Sender<BankerMessage>,
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<f32, V, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<f32>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
) where
    VectorSerial<f32>: From<V>,
    Vector<f32, VECTOR_CAP>: From<V>,
    Vector<f32, 2>: From<V>,
{
    let transaction_id = transaction_id.unwrap_or_else(Uuid::new_v4);

    // Request access to the partition
    let granted_partition = loop {
        let (tx, rx) = oneshot::channel();

        access_tx
            .send(BankerMessage::RequestAccess {
                transaction_id,
                partitions: vec![(partition_id, AccessMode::Read)],
                respond_to: tx,
            })
            .await
            .expect("failed to send access request");

        match rx.await {
            Ok(AccessResponse::Granted) => break partition_id,
            _ => {
                // Retry until access is granted
                // Optionally add delay or backoff
            }
        }
    };

    // Load the partition
    let partition_buffer = &mut *partition_buffer.write().await;
    let partition = resolve_buffer!(ACCESS, partition_buffer, granted_partition);

    let Some(partition) = &*partition.read().await else {
        todo!("Partition buffer resolve failed");
    };

    // Send all vectors in the partition

    for i in 0..partition.size {
        let Some(vector_entry) = partition.vectors[i] else {
            todo!()
        };

        let _ = vector_tx.send(Some(vector_entry.clone())).await;
    }
    
    // TODO - drop partition before releasing?
    let _ = access_tx
        .send(
            BankerMessage::ReleaseAccess{
                transaction_id: transaction_id,
                partitions: vec![partition_id]
        })
        .await;

    let _ = vector_tx.send(None).await;
}

pub async fn stream_vector_by_id<
    // B: Backend + AutodiffBackend + Send + Sync,
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
    // LV: VectorSpace<f32> + Send + Sync + for<'a> From<&'a [f32]> + From<Vec<f32>> + Into<Vec<f32>>,
    // K: Send + Sync + Clone + Debug + 'static + From<usize>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    // transaction id
    transaction_id: Option<Uuid>,

    // output
    vector_tx: Sender<Option<VectorEntry<f32, V>>>,

    // input
    vector_id: VectorId,

    // required resources
    access_tx: Sender<BankerMessage>,
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<f32, V, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<f32>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
    partition_membership: Arc<RwLock<PartitionMembership>>,
) where
    VectorSerial<f32>: From<V>,
    Vector<f32, VECTOR_CAP>: From<V>,
    Vector<f32, 2>: From<V>,
{
    let transaction_id = transaction_id.unwrap_or_else(Uuid::new_v4);

    let partition_id = loop {
        let partition_membership = &*partition_membership.read().await;

        let partition_id: PartitionId = partition_membership.get_partition_id(vector_id).unwrap();

        let (tx, rx) = oneshot::channel();

        let _ = access_tx
            .send(BankerMessage::RequestAccess {
                transaction_id: transaction_id,
                partitions: vec![(partition_id, AccessMode::Read)],
                respond_to: tx,
            })
            .await;

        match rx.await {
            Ok(AccessResponse::Granted) => break partition_id,
            _ => {}
        }
    };

    let partition_buffer = &mut *partition_buffer.write().await;
    let partition = resolve_buffer!(ACCESS, partition_buffer, partition_id);

    let Some(partition) = &*partition.read().await else {
        todo!()
    };

    for i in 0..partition.size {
        let Some(vector_entry) = partition.vectors[i] else {
            todo!()
        };

        if vector_id != VectorId(vector_entry.id) {
            continue;
        }

        let _ = vector_tx.send(Some(vector_entry.clone())).await;

        break;
    }
    
    // TODO - drop partition before releasing?
    let _ = access_tx
        .send(
            BankerMessage::ReleaseAccess{
                transaction_id: transaction_id,
                partitions: vec![partition_id]
        })
        .await;


    let _ = vector_tx.send(None).await;
}

pub async fn stream_lower_dim_vector<
    HD: VectorSpace<f32> + Debug + Clone + Copy + Send + Sync + 'static,
    LD: VectorSpace<f32> + Debug + Clone + Send + Sync + 'static,
>(
    mut vector_rx: Receiver<Option<VectorEntry<f32, HD>>>,
    projection_tx: Sender<(HD, oneshot::Sender<Result<LD, InferenceError>>)>,
    tx: Sender<Option<VectorEntry<f32, LD>>>,
) {
    while let Some(vector) = vector_rx.recv().await.unwrap() {
        let new_vector = loop {
            let (lower_dim_tx, lower_dim_rx) = oneshot::channel();

            let _ = projection_tx.send((vector.vector, lower_dim_tx)).await;
            
            match lower_dim_rx.await.unwrap() {
                Ok(vector) => break vector,
                Err(_) => todo!(),
            }
        };


        let _ = tx.send(Some(VectorEntry {
            vector: new_vector,
            id: vector.id,
            _phantom_data: PhantomData,
        })).await;
    }
    let _ = tx.send(None).await;
}

pub async fn stream_projected_vectors<
    V: VectorSpace<f32> + Sized + Clone + Copy + Send + Sync + 'static + Debug,
>(
    tx: Sender<Response<f32>>,
    projection_mode: ProjectionMode,
    mut vector_rx: Receiver<Option<VectorEntry<f32, V>>>,
) where
    VectorSerial<f32>: From<V>,
{
    match projection_mode {
        ProjectionMode::Default => {
            while let Some(vector_entry) = vector_rx.recv().await.unwrap() {
                let _ = tx
                    .send(Response::Success(Success::Vector(
                        VectorId(vector_entry.id),
                        VectorSerial::<f32>::from(vector_entry.vector),
                    )))
                    .await;
            }
        }
        ProjectionMode::IdOnly => {
            while let Some(vector_entry) = vector_rx.recv().await.unwrap() {
                let _ = tx
                    .send(Response::Success(Success::Vector(
                        VectorId(vector_entry.id),
                        VectorSerial::<f32>::from(vector_entry.vector),
                    )))
                    .await;
            }
        }
        ProjectionMode::VectorOnly => {
            while let Some(vector_entry) = vector_rx.recv().await.unwrap() {
                let _ = tx
                    .send(Response::Success(Success::Vector(
                        VectorId(Uuid::nil()),
                        VectorSerial::<f32>::from(vector_entry.vector),
                    )))
                    .await;
            }
        }
    }
    let _ = tx.send(Response::Done).await;
}
