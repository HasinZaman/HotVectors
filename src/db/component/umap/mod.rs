use core::panic;
use std::{
    array,
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use tracing::{event, Level};

use burn::{
    backend::Autodiff,
    config::Config,
    data::dataloader::batcher::Batcher,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, Optimizer},
    prelude::Backend,
    record::RecorderError,
    tensor::{backend::AutodiffBackend, Tensor, TensorData},
    train::{RegressionOutput, TrainOutput, TrainStep},
};
use futures::FutureExt;
use petgraph::{csr::DefaultIx, graph::NodeIndex, prelude::StableGraph, Undirected};
use rand::{rng, seq::SliceRandom};
use tokio::{
    sync::{mpsc, oneshot, Mutex, Notify, RwLock},
    task::{yield_now, JoinHandle},
};
use uuid::Uuid;

use crate::{
    db::component::{
        data_buffer::{DataBuffer, Global},
        ids::{PartitionId, VectorId},
        meta::Meta,
        partition::{Partition, PartitionMembership, PartitionSerial, VectorEntry}, // umap::incremental_model::{NoStrategy, UMapStrategy},
    },
    resolve_buffer,
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

pub mod model;

#[derive(Debug)]
pub enum InferenceError {
    ModelNotInitialized,
}

pub fn model_inference_loop<
    B: Backend + AutodiffBackend,
    U: ParamUMap<B>
        + Module<B>
        + for<'a> TrainStep<
            (
                UMapBatch<B, VectorId, HD>,
                &'a FuzzyNeighborGraph<f32, HD, VectorId>,
                (f32, f32),
                PhantomData<(LD, HD)>,
            ),
            RegressionOutput<B>,
        > + AutodiffModule<B>
        + Send,
    HD: VectorSpace<f32>
        + Extremes
        + Debug
        + Clone
        + Copy
        + Into<Vec<f32>>
        + From<Vec<f32>>
        + PartialEq
        + Send
        + Sync
        + From<VectorSerial<f32>>
        + for<'a> From<&'a [f32]>
        + 'static,
    LD: VectorSpace<f32>
        + Extremes
        + Debug
        + Clone
        + From<Vec<f32>>
        + Send
        + Sync
        + for<'a> From<&'a [f32]>
        + 'static,
    const PROJECTIONS: usize,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
    const MAX_CAP: usize,
>(
    // data required for training
    meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<f32, HD>>>>>>,
    partition_buffer: Arc<
        RwLock<
            DataBuffer<
                Partition<f32, HD, PARTITION_CAP, VECTOR_CAP>,
                PartitionSerial<f32>,
                Global,
                MAX_LOADED,
            >,
        >,
    >,
    partition_membership: Arc<RwLock<PartitionMembership>>,
) -> (
    (JoinHandle<()>, mpsc::Sender<VectorEntry<f32, HD>>),
    (
        JoinHandle<()>,
        mpsc::Sender<(HD, oneshot::Sender<Result<LD, InferenceError>>)>,
    ),
)
where
    VectorSerial<f32>: From<HD>,
    for<'a> &'a Partition<f32, HD, PARTITION_CAP, VECTOR_CAP>: Into<Uuid>,
{
    let force_train: Arc<[Notify; PROJECTIONS]> = Arc::new(array::from_fn(|_| Notify::new()));
    let updated_flags: Arc<[AtomicBool; PROJECTIONS]> =
        Arc::new(array::from_fn(|_| AtomicBool::new(false)));

    let batch_size: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));

    // input conns
    let (training_tx, mut training_rx) = mpsc::channel(64);
    let (inference_tx, mut inference_rx) = mpsc::channel(64);

    //
    let _training_thread: JoinHandle<()> = tokio::spawn({
        let force_train = force_train.clone();
        let updated_flags = updated_flags.clone();
        let batch_size = batch_size.clone();
        async move {
            let device = <Autodiff<B> as Backend>::Device::default();

            let mut models_trained: [bool; PROJECTIONS] = array::from_fn(|_| false);

            let mut model = U::new(&device);
            let mut buffer: Vec<VectorEntry<f32, HD>> = Vec::new();

            let config: UMapTrainingConfig = UMapTrainingConfig {
                optimizer: AdamConfig::new(),
                attractive_size: 16,
                repulsive_size: 8,
                epoch: 2,
                seed: 42,
                learning_rate: 0.01,
            };

            loop {
                let mut train_idx: Option<usize> = None;
                tokio::select! {
                    maybe_data = training_rx.recv() => {
                        match maybe_data {
                            Some(data) => {
                                buffer.push(data);

                                let batch_size = &mut *batch_size.lock().await;
                                *batch_size = *batch_size + 1;
                                println!("GOT DATA TO TRAIN - {:?}", batch_size);

                            },
                            None => {
                                println!("[] Data channel closed");
                                todo!();
                            }
                        }
                    }
                    idx = notified(&*force_train) => {
                        train_idx = Some(idx);
                        println!("[{idx}] Early training triggered");
                    }
                };

                if let Some(idx) = train_idx {
                    println!("TRAINING - forced");
                    // create dataset/batches
                    let batches: Vec<Vec<PartitionId>> = partition_batches::<HD, MAX_CAP>(
                        buffer
                            .iter()
                            .map(|vector_entry| VectorId(vector_entry.id))
                            .collect(),
                        &*partition_membership.read().await,
                        &*meta_data.read().await,
                    )
                    .await;

                    println!("Required partitions for updating u-map {batches:#?}");
                    // epochs
                    let epochs = match models_trained[idx] {
                        false => 3,
                        true => 1,
                    };
                    for _i in 0..epochs {
                        println!("Epoch {_i:}");
                        for batch in &batches {
                            println!("batch {batch:?}");
                            // load data
                            let training_data = {
                                let mut training_data = Vec::new();

                                let partition_buffer = &mut *partition_buffer.write().await;

                                for partition_id in batch {
                                    let partition =
                                        resolve_buffer!(ACCESS, partition_buffer, *partition_id);

                                    let Some(partition) = &*partition.read().await else {
                                        todo!();
                                    };

                                    for vector_entry in partition.iter() {
                                        training_data
                                            .push((VectorId(vector_entry.id), vector_entry.vector));
                                    }
                                }

                                training_data
                            };

                            // train model
                            model =
                                train_umap::<B, VectorId, U, HD, LD>(model, training_data, &config);
                        }
                    }

                    models_trained[idx] = true;

                    println!("Trying to save");
                    let model_path = format!("model");
                    if let Err(e) = model.save(&model_path) {
                        eprintln!("[{idx}] Save failed: {e:?}");
                    } else {
                        updated_flags[idx].store(true, Ordering::Relaxed);
                    }
                    buffer.clear();

                    let batch_size = &mut *batch_size.lock().await;
                    *batch_size = 0;
                } else if buffer.len() > 64 {
                    println!("TRAINING - buffer overflow");
                    let batches: Vec<Vec<PartitionId>> = partition_batches::<HD, MAX_CAP>(
                        buffer
                            .iter()
                            .map(|vector_entry| VectorId(vector_entry.id))
                            .collect(),
                        &*partition_membership.read().await,
                        &*meta_data.read().await,
                    )
                    .await;

                    println!("Required partitions for updating u-map {batches:#?}");
                    for i in 0..PROJECTIONS {
                        let epochs = match models_trained[i] {
                            false => 3,
                            true => 1,
                        };
                        for _i in 0..epochs {
                            println!("Epoch {_i:}");
                            // create dataset/batches
                            for batch in &batches {
                                println!("batch {batch:?}");
                                // load data
                                let training_data = {
                                    let mut training_data = Vec::new();

                                    let partition_buffer = &mut partition_buffer.write().await;

                                    for partition_id in batch {
                                        let partition = resolve_buffer!(
                                            ACCESS,
                                            partition_buffer,
                                            *partition_id
                                        );

                                        let Some(partition) = &*partition.read().await else {
                                            todo!();
                                        };

                                        for vector_entry in partition.iter() {
                                            training_data.push((
                                                VectorId(vector_entry.id),
                                                vector_entry.vector,
                                            ));
                                        }
                                    }

                                    training_data
                                };

                                // train model
                                model = train_umap::<B, VectorId, U, HD, LD>(
                                    model,
                                    training_data,
                                    &config,
                                );
                            }
                        }

                        models_trained[i] = true;
                        println!("Trying to save");
                        let model_path = format!("model");
                        if let Err(e) = model.save(&model_path) {
                            eprintln!("[{i}] Save failed: {e:?}");
                        } else {
                            updated_flags[i].store(true, Ordering::Relaxed);
                        }
                    }
                    buffer.clear();

                    let batch_size = &mut *batch_size.lock().await;
                    *batch_size = 0;
                }
            }
        }
    });

    let _inference_thread: JoinHandle<()> = tokio::spawn({
        async move {
            let device = <Autodiff<B> as Backend>::Device::default();
            let mut models: [Option<U>; PROJECTIONS] = array::from_fn(|_| None);

            loop {
                tokio::select! {
                    data = inference_rx.recv() => {
                        let Some((vector, tx)) = data else { todo!() };
                        model_inference::<B, U, HD, LD, PROJECTIONS>(
                            vector,
                            tx,
                            &mut models,
                            &batch_size,
                            &force_train,
                            &updated_flags
                        ).await;
                    }
                    idx = model_updated(&updated_flags)=> {
                        let model_path = format!("model");

                        match U::load(&device, &model_path) {
                            Ok(new_model) => {
                                models[0] = Some(new_model);
                                updated_flags[idx].store(false, Ordering::Relaxed);
                                println!("[{idx}] Model loaded for inference");
                            }
                            Err(e) => {
                                eprintln!("Failed to load model from {model_path}: {e:?}");
                            }
                        }
                    }
                }
            }
        }
    });

    (
        (_training_thread, training_tx),
        (_inference_thread, inference_tx),
    )
}

async fn model_inference<
    B: Backend + AutodiffBackend,
    U: ParamUMap<B>,
    HD: VectorSpace<f32> + Extremes + Debug + Clone + Into<Vec<f32>> + Send + Sync + 'static,
    LD: VectorSpace<f32> + Extremes + Debug + Clone + From<Vec<f32>> + Send + Sync + 'static,
    const PROJECTIONS: usize,
>(
    vector: HD,
    tx: oneshot::Sender<Result<LD, InferenceError>>,
    models: &mut [Option<U>; PROJECTIONS],
    batch_size: &Arc<Mutex<usize>>,
    force_train: &Arc<[Notify; PROJECTIONS]>,
    updated_flags: &Arc<[AtomicBool; PROJECTIONS]>,
) {
    if let Some(ref model) = models[0] {
        // Already loaded
        let result = model.forward::<HD, LD>(vector);
        let _ = tx.send(Ok(result));
    } else {
        // Model not loaded
        let current_batch = *batch_size.lock().await;
        if current_batch > 5 {
            println!(
                "[Inference] Forcing training due to missing model and batch_size={current_batch}"
            );

            // Trigger training
            force_train[0].notify_one();

            // Wait for model to be trained
            model_updated(updated_flags).await;

            // Try loading the model
            let device = <Autodiff<B> as Backend>::Device::default();
            let model_path = "model".to_string();
            match U::load(&device, &model_path) {
                Ok(new_model) => {
                    models[0] = Some(new_model);
                    updated_flags[0].store(false, Ordering::Relaxed);
                    println!("[Inference] Model loaded after training");
                    // Now retry inference
                    if let Some(ref model) = models[0] {
                        let result = model.forward::<HD, LD>(vector);
                        let _ = tx.send(Ok(result));
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("[Inference] Model failed to load after forced training: {e:?}");
                }
            }
        }

        let _ = tx.send(Err(InferenceError::ModelNotInitialized));
    }
}

async fn model_updated<const PROJECTIONS: usize>(
    update_bool: &Arc<[AtomicBool; PROJECTIONS]>,
) -> usize {
    loop {
        for i in 0..PROJECTIONS {
            if update_bool[i].load(Ordering::Relaxed) {
                return i;
            }
        }

        yield_now().await;
    }
}

async fn notified(notifies: &[Notify]) -> usize {
    loop {
        for (i, notify) in notifies.iter().enumerate() {
            let notified = notify.notified();
            tokio::pin!(notified);

            // Check if it's ready immediately
            if notified.as_mut().now_or_never().is_some() {
                return i;
            }
        }

        // Yield to avoid busy-waiting
        yield_now().await;
    }
}

pub async fn partition_batches<HD: VectorSpace<f32> + Clone, const MAX_BATCH: usize>(
    buffer: Vec<VectorId>,
    partition_membership: &PartitionMembership,
    meta_data: &HashMap<Uuid, Arc<RwLock<Meta<f32, HD>>>>,
) -> Vec<Vec<PartitionId>> {
    let mut required_partitions: HashSet<PartitionId> = buffer
        .into_iter()
        .filter_map(|vec_id| partition_membership.get_partition_id(vec_id))
        .collect();

    // Extract sizes for required partitions
    let mut partition_sizes: HashMap<PartitionId, usize> = HashMap::new();
    {
        for partition_id in &required_partitions {
            if let Some(data) = meta_data.get(partition_id) {
                let data = data.read().await;
                partition_sizes.insert(*partition_id, data.size);
            }
        }
    }

    // check if size is smaller the MAX_BATCH -> then return batch
    let total_size: usize = partition_sizes.iter().map(|(_,x)| *x).sum();
    if required_partitions.len() == meta_data.len() && total_size < MAX_BATCH{
        println!("Base case 1");
        return vec![required_partitions.into_iter().collect()]
    }

    if required_partitions.len() < meta_data.len() && total_size < MAX_BATCH {
        println!("Base case 2");
        let available: Vec<PartitionId> = {
            let mut rng = rng();

            let mut available: Vec<PartitionId> = meta_data
                .keys()
                .filter(|partition_id| !required_partitions.contains(&PartitionId(**partition_id)))
                .map(|partition_id| PartitionId(*partition_id))
                .collect();
            
            available.shuffle(&mut rng);

            available
            
        };

        let mut batch: Vec<PartitionId> = required_partitions.iter().cloned().collect();
        let mut current_size = total_size;
        for partition_id in available {
            let mut size = partition_sizes.get(&partition_id).cloned();

            if size.is_none() {
                if let Some(meta) = meta_data.get(&partition_id) {
                    let meta = meta.read().await;
                    size = Some(meta.size);
                }
            }

            if let Some(size) = size {
                if current_size + size <= MAX_BATCH {
                    batch.push(partition_id);
                    current_size += size;
                }
            }

            if current_size >= MAX_BATCH {
                break;
            }
        }

        return vec![batch];
    }

    if required_partitions.len() < meta_data.len() {
        let add_count = (required_partitions.len() + 3) / 4;

        let available: Vec<PartitionId> = {
            let mut rng = rng();

            let mut available: Vec<PartitionId> = meta_data
                .keys()
                .filter(|pid| !required_partitions.contains(&PartitionId(**pid)))
                .map(|pid| PartitionId(*pid))
                .collect();

            available.shuffle(&mut rng);

            available
        };

        for pid in available.into_iter().take(add_count) {
            required_partitions.insert(pid);

            // also fetch and cache its size if not already in partition_sizes
            if !partition_sizes.contains_key(&pid) {
                if let Some(meta) = meta_data.get(&pid) {
                    let meta = meta.read().await;
                    partition_sizes.insert(pid, meta.size);
                }
            }
        }

        println!(
            "Added {add_count} random partitions to improve training quality (now {} required)",
            required_partitions.len()
        );
    }
    
    // Initialize usage count: how many batches each partition has been in
    let mut usage_counts: HashMap<PartitionId, usize> =
        required_partitions.iter()
            .map(|&partition_id| (partition_id, 0))
            .collect();

    let mut batches = Vec::new();


    println!("Attempting to generate partitions for batches");
    while !required_partitions.is_empty() {
        // println!("batches:{batches:?}");
        // println!("required_partitions:{required_partitions:?}");

        let mut batch = Vec::new();
        let mut current_size = 0;

        // Group partitions by usage count
        let mut usage_buckets: HashMap<usize, Vec<PartitionId>> = HashMap::new();
        for &partition_id in &required_partitions {
            let count = usage_counts.get(&partition_id).cloned().unwrap_or(0);
            usage_buckets.entry(count).or_default().push(partition_id);
        }


        // println!("usage_buckets: {usage_buckets:?}");
        // Sort usage levels and shuffle within groups
        let mut sorted_partitions = Vec::new();
        let mut thread_rng = rng();
        let mut usage_levels: Vec<_> = usage_buckets.keys().cloned().collect();
        usage_levels.sort();

        for level in usage_levels {
            let mut group = usage_buckets.remove(&level).unwrap();
            group.shuffle(&mut thread_rng);
            sorted_partitions.extend(group);
        }

        // println!("sorted_partitions: {sorted_partitions:?}");
        // println!("current_size: {current_size:?}");

        // Fill the batch with shuffled, least-used partitions
        for partition_id in sorted_partitions {
            let size = *partition_sizes.get(&partition_id).unwrap_or(&0);
            // println!("size: {current_size} + {size} ({}) <= {MAX_BATCH}", current_size + size);
            if current_size + size <= MAX_BATCH {
                batch.push(partition_id);
                current_size += size;
                *usage_counts.entry(partition_id).or_default() += 1;
            }
            if current_size >= MAX_BATCH {
                break;
            }
        }

        batches.push(batch.clone());

        // Remove partitions from required set if they're balanced
        for pid in &batch {
            let max_usage = *usage_counts.values().max().unwrap_or(&1);
            let min_usage = *usage_counts.values().min().unwrap_or(&0);
            
            println!("{pid:?}:{max_usage:} - {min_usage:} <= 1");
            if max_usage - min_usage <= 1 {
                required_partitions.remove(pid);
            }
        }
    }

    batches
}

pub trait ParamUMap<B: Backend> {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

    fn new(device: &B::Device) -> Self;

    fn forward_tensor(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;

    fn forward<
        HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + Into<Vec<f32>>,
        LD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + From<Vec<f32>>,
    >(
        &self,
        input: HD,
    ) -> LD
    where
        B: AutodiffBackend;

    fn save(&self, base_path: &str) -> Result<(), RecorderError>;

    fn load(device: &B::Device, base_path: &str) -> Result<Self, RecorderError>
    where
        Self: Sized;
}

#[derive(Debug, Clone)]
pub struct FuzzyNeighborGraph<
    A: Debug + Clone + Field<A>,
    B: Clone + VectorSpace<A> + Extremes,
    ID: Clone + Copy + Hash + PartialEq,
>(
    pub StableGraph<ID, A, Undirected, DefaultIx>,
    pub HashMap<ID, NodeIndex<DefaultIx>>,
    PhantomData<B>,
);

impl<B: Clone + VectorSpace<f32> + Extremes, ID: Clone + Copy + Hash + PartialEq + Eq>
    FuzzyNeighborGraph<f32, B, ID>
// should make more generic
where
    f32: Debug + Clone + Field<f32> + Extremes + PartialOrd,
{
    pub fn higher_dim<const K: usize>(points: &Vec<(ID, B)>) -> Self {
        // get similarity matrix
        // println!("Calculating similarity matrix... {:?}", points.len());
        let similarity_matrix = {
            // shit code can be improved
            let mut similarity_matrix: Vec<f32> = Vec::with_capacity(points.len() * points.len());
            for _ in 0..(points.len() * points.len()) {
                similarity_matrix.push(0.);
            }

            for i in 0..points.len() {
                for j in i..points.len() {
                    let dist = B::dist(&points[i].1, &points[j].1);

                    similarity_matrix[i * points.len() + j] = dist.clone();
                    similarity_matrix[j * points.len() + i] = dist;
                }
            }

            similarity_matrix
        };

        // knn on points using similarity matrix
        let k_neighbors = {
            let mut k_neighbors = Vec::with_capacity(points.len());
            for i in 0..points.len() {
                let mut distances: Vec<_> = similarity_matrix
                    [i * points.len()..(i + 1) * points.len()]
                    .iter()
                    .enumerate()
                    .collect();

                heapify::make_heap_with(&mut distances, |(_, x), (_, y)| (**x).partial_cmp(*y));

                let mut neighbor_index = Vec::with_capacity(K);
                for _ in 0..K {
                    heapify::pop_heap_with(&mut distances, |(_, x), (_, y)| (**x).partial_cmp(*y));

                    match distances.pop() {
                        Some((idx, _)) => neighbor_index.push(idx),
                        None => break,
                    }
                }

                k_neighbors.push(neighbor_index);
            }

            k_neighbors
        };

        // calculate weight
        let weights = {
            let target = (K as f32).ln();

            let mut weights = Vec::with_capacity(points.len());
            for _ in 0..(points.len() * points.len()) {
                weights.push(0.);
            }
            let mut rhos = vec![0.; points.len()];
            let mut sigmas = vec![0.; points.len()];

            for i in 0..points.len() {
                let dists: Vec<f32> = k_neighbors[i]
                    .iter()
                    .map(|&j| similarity_matrix[i * points.len() + j].clone())
                    .collect();

                // rho is the distance to the nearest neighbor
                if !dists.is_empty() {
                    // first value is min value
                    rhos[i] = dists[0].clone();
                }

                // binary search for sigma
                let mut sigma = 1.;
                for _ in 0..10 {
                    let mut sum_probs = 0.;

                    for d in &dists {
                        let dist = {
                            let dist = d - &rhos[i];

                            match 0. < dist {
                                true => dist,
                                false => 0.,
                            }
                        };

                        let prob: f32 = -dist / sigma;
                        let prob: f32 = prob.exp();
                        sum_probs = sum_probs + prob;
                    }

                    if (sum_probs.clone() - target.clone()).abs() < 1e-5 {
                        break;
                    }

                    if sum_probs > target {
                        sigma = sigma * 1.1;
                    } else {
                        sigma = sigma / 1.1;
                    }
                }

                sigmas[i] = sigma;

                for &j in &k_neighbors[i] {
                    let dist = similarity_matrix[i * points.len() + j].clone();
                    let adjusted = {
                        let val = dist - rhos[i];

                        match 0. < val {
                            true => val,
                            false => 0.,
                        }
                    };

                    let weight = (-adjusted / sigma.clone()).exp();

                    weights[i * points.len() + j] = weight;
                }
            }

            weights
        };

        let edge_weights = {
            let mut edge_weights = Vec::with_capacity(points.len() * points.len());
            for _ in 0..(points.len() * points.len()) {
                edge_weights.push(0.);
            }

            for i in 0..points.len() {
                for &j in &k_neighbors[i] {
                    let w_ij = weights[i * points.len() + j].clone();
                    let w_ji = weights[j * points.len() + i].clone();
                    let sym = (w_ij + w_ji) - w_ij * w_ji;
                    edge_weights[i * points.len() + j] = sym;
                    edge_weights[j * points.len() + i] = sym;
                }
            }

            edge_weights
        };

        let mut graph: StableGraph<ID, f32, Undirected, DefaultIx> =
            StableGraph::<ID, f32, Undirected, DefaultIx>::default();
        let mut id_to_idx: HashMap<ID, NodeIndex<DefaultIx>> = HashMap::new();

        for (node_id, _) in points.iter() {
            let idx: NodeIndex<DefaultIx> = graph.add_node(*node_id);
            id_to_idx.insert(*node_id, idx);
        }

        for i in 0..points.len() {
            for j in 0..i {
                graph.add_edge(
                    id_to_idx[&points[i].0],
                    id_to_idx[&points[j].0],
                    edge_weights[i * points.len() + j],
                );
            }
        }

        Self(graph, id_to_idx, PhantomData)
    }
}

struct UMapBatcher<
    ID: Copy + Clone + Hash + PartialEq + Eq + PartialOrd + Ord,
    HD: Clone + VectorSpace<f32> + Extremes,
> {
    attractive_size: usize,
    max_repulsive_size: usize,

    input_dim: usize,

    seed: u64,

    points: HashMap<ID, HD>,
    higher_dim_graph: FuzzyNeighborGraph<f32, HD, ID>,

    visited: HashMap<ID, usize>,
}

impl<
        ID: Copy + Clone + Hash + PartialEq + Eq + PartialOrd + Ord + Debug,
        HD: Clone + VectorSpace<f32> + Extremes + Into<Vec<f32>>,
    > UMapBatcher<ID, HD>
{
    pub fn new(
        attractive_size: usize,
        max_repulsive_size: usize,
        input_dim: usize,
        seed: u64,
        points: HashMap<ID, HD>,
        higher_dim_graph: FuzzyNeighborGraph<f32, HD, ID>,
    ) -> Self {
        Self {
            attractive_size,
            max_repulsive_size,
            input_dim,
            seed,
            points,
            higher_dim_graph,
            visited: HashMap::new(),
        }
    }

    pub fn empty(&self) -> bool {
        // checks if all visited points have been processed by getting the intersection of higher_dim_graph.1.keys
        self.visited
            .keys()
            .all(|id| !self.higher_dim_graph.1.contains_key(id))
    }

    pub fn reset(&mut self) {
        self.visited.clear();
    }

    pub fn batch<B: Backend>(&mut self, device: &B::Device) -> UMapBatch<B, ID, HD> {
        // select random points from higher_dim_graph.1
        let attractive_points: Vec<ID> = {
            // get all keys from higher_dim_graph.1
            let keys: Vec<ID> = self.higher_dim_graph.1.keys().cloned().collect();
            // filter out visited points
            let unvisited: Vec<ID> = keys
                .into_iter()
                .filter(|id| !self.visited.contains_key(id))
                .collect();
            // shuffle unvisited points
            let mut rng = rand::rng();
            let mut shuffled = unvisited.clone();
            shuffled.shuffle(&mut rng);
            // take attractive_size points
            let points: Vec<ID> = shuffled.into_iter().take(self.attractive_size).collect();

            // mark points as visited
            for point in &points {
                *self.visited.entry(*point).or_insert(0) += 1;
            }

            // get all neighbors of the selected points
            let mut neighbors = Vec::new();
            for point in &points {
                if let Some(&node_index) = self.higher_dim_graph.1.get(point) {
                    for neighbor in self.higher_dim_graph.0.neighbors(node_index) {
                        let neighbor_id = self.higher_dim_graph.0[neighbor];
                        neighbors.push(neighbor_id);
                    }
                }
            }

            // remove duplicates from neighbors
            neighbors.sort();
            neighbors.dedup();

            neighbors
        };

        // select random repulsive points that are not in attractive_points
        let repulsive_points: Vec<ID> = {
            // get all keys from higher_dim_graph.1
            let keys: Vec<ID> = self.higher_dim_graph.1.keys().cloned().collect();
            // filter out visited points and attractive points
            let unvisited: Vec<ID> = keys
                .into_iter()
                .filter(|id| !self.visited.contains_key(id) && !attractive_points.contains(id))
                .collect();
            // shuffle unvisited points
            let mut rng = rand::rng();
            let mut shuffled = unvisited.clone();
            shuffled.shuffle(&mut rng);
            // take max_repulsive_size points
            let points: Vec<ID> = shuffled.into_iter().take(self.max_repulsive_size).collect();

            // mark points as visited
            for point in &points {
                *self.visited.entry(*point).or_insert(0) += 1;
            }

            points
        };

        // create batch from attractive and repulsive points
        let mut items: Vec<(ID, HD)> =
            Vec::with_capacity(attractive_points.len() + repulsive_points.len());
        for point in attractive_points {
            if let Some(&node_index) = self.higher_dim_graph.1.get(&point) {
                if let Some(_) = self.higher_dim_graph.0.node_weight(node_index) {
                    // check if point is in points
                    if let Some(value) = self.points.get(&point) {
                        items.push((point, value.clone()));
                    } else {
                        panic!("Point not found in points: {:?}", point);
                    }
                }
            }
        }
        for point in repulsive_points {
            if let Some(&node_index) = self.higher_dim_graph.1.get(&point) {
                if let Some(_) = self.higher_dim_graph.0.node_weight(node_index) {
                    // check if point is in points
                    if let Some(value) = self.points.get(&point) {
                        items.push((point, value.clone()));
                    } else {
                        panic!("Point not found in points: {:?}", point);
                    }
                }
            }
        }

        // create UMapBatch from items
        UMapBatch::<B, ID, HD>::new(self.input_dim, items, device)
    }
}

#[derive(Debug, Clone)]
pub struct UMapBatch<B: Backend, ID: Copy + Hash + PartialEq + Eq, V> {
    sources: Vec<ID>,
    points: Tensor<B, 2>,
    _phantom: PhantomData<V>,
}

impl<
        B: Backend,
        ID: Copy + Hash + PartialEq + Eq + PartialOrd + Ord,
        V: Clone + Into<Vec<f32>>,
    > UMapBatch<B, ID, V>
{
    pub fn new(input_dim: usize, items: Vec<(ID, V)>, device: &<B as Backend>::Device) -> Self {
        let sources: Vec<ID> = items.iter().map(|(id, _)| *id).collect();
        let points = TensorData::new(
            items
                .iter()
                .map(|(_, v)| {
                    let v: Vec<f32> = v.clone().into();

                    v
                })
                .flatten()
                .collect::<Vec<_>>(),
            vec![items.len(), input_dim],
        );

        let points: Tensor<B, 2> = Tensor::from_data(points, device);

        UMapBatch {
            sources,
            points,
            _phantom: PhantomData,
        }
    }
}

// impl<
//         B: Backend,
//         ID: Copy + Hash + PartialEq + Eq + PartialOrd + Ord,
//         V,
//     > Default for UMapBatch<B, ID, V>
// {
//     fn default() -> Self {
//         UMapBatch {
//             sources: Vec::new(),
//             points: Tensor::from_data(
//                 TensorData::new(Vec::<f32>::new(), vec![0, 1]),
//                 &B::Device::default(),
//             ),
//             _phantom: PhantomData,
//         }
//     }
// }

impl<
        B: Backend,
        ID: Copy + Hash + PartialEq + Eq + PartialOrd + Ord + Send + Sync + Debug,
        V: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + Into<Vec<f32>>,
    > Batcher<B, (ID, V), UMapBatch<B, ID, V>> for UMapBatch<B, ID, V>
{
    fn batch(&self, items: Vec<(ID, V)>, device: &<B as Backend>::Device) -> UMapBatch<B, ID, V> {
        let input_dim: usize = items
            .iter()
            .map(|(_, v)| v.clone())
            .map(|v| {
                let v: Vec<f32> = v.into();

                v
            })
            .next()
            .unwrap()
            .len();

        let sources: Vec<ID> = items.iter().map(|(id, _)| *id).collect();
        let points = TensorData::new(
            items
                .iter()
                .map(|(_, v)| {
                    let v: Vec<f32> = v.clone().into();

                    v
                })
                .flatten()
                .collect::<Vec<_>>(),
            vec![items.len(), input_dim],
        );

        let points: Tensor<B, 2> = Tensor::from_data(points, device);

        UMapBatch {
            sources,
            points,
            _phantom: PhantomData,
        }
    }
}

#[derive(Config)]
pub struct UMapTrainingConfig {
    pub optimizer: AdamConfig,

    #[config(default = 100)]
    pub attractive_size: usize,
    #[config(default = 100)]
    pub repulsive_size: usize,

    #[config(default = 4)]
    pub epoch: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train_umap<
    B: Backend + AutodiffBackend,
    ID: Copy + Clone + Hash + PartialEq + Eq + PartialOrd + Ord + Debug,
    M: Module<B>
        + for<'a> TrainStep<
            (
                UMapBatch<B, ID, HD>,
                &'a FuzzyNeighborGraph<f32, HD, ID>,
                (f32, f32),
                PhantomData<(LD, HD)>,
            ),
            RegressionOutput<B>,
        > + ParamUMap<B>
        + AutodiffModule<B>,
    // input and output vector types
    HD: VectorSpace<f32>
        + Extremes
        + Clone
        + Debug
        + Send
        + Sync
        + From<Vec<f32>>
        + for<'a> From<&'a [f32]>
        + Into<Vec<f32>>,
    LD: VectorSpace<f32>
        + Extremes
        + Clone
        + Debug
        + Send
        + Sync
        + From<Vec<f32>>
        // + for<'a> From<&'a [f32; OUTPUT_DIM]>
        + for<'a> From<&'a [f32]>,
    // const OUTPUT_DIM: usize,
>(
    mut model: M,
    data: Vec<(ID, HD)>,
    config: &UMapTrainingConfig,
) -> M
//where for<'a> &'a [f32]: Into<HD> + Into<LD>,
{
    let device = B::Device::default();

    // Create the higher-dimensional graph
    let higher_dim_graph = FuzzyNeighborGraph::<f32, HD, ID>::higher_dim::<3>(&data);

    // create optimizer (Adam)
    let mut optimizer = config.optimizer.init::<B, M>();

    // let mut optimizer = Adam::new(&config.optimizer);

    // lower-dimensional parameters
    let a = 1.0; // UMAP attractive parameter (tune or get from config)
    let b = 1.5; // UMAP repulsive parameter (tune or get from config)

    // prepare batches
    let mut batcher = UMapBatcher::<ID, HD>::new(
        config.attractive_size,
        config.repulsive_size,
        model.input_dim(),
        config.seed,
        data.iter().cloned().collect(),
        higher_dim_graph.clone(),
    );

    for epoch in 0..config.epoch {
        batcher.reset();
        while !batcher.empty() {
            let batch = batcher.batch(&device);

            // Run train step on batch
            let TrainOutput { grads, item } = TrainStep::step(
                &model,
                (
                    batch.clone(),
                    &higher_dim_graph,
                    (a, b),
                    PhantomData::<(LD, HD)>,
                ),
            );

            // Update model with gradients from train step
            model = optimizer.step(config.learning_rate, model, grads);

            println!("Epoch {}: loss = {:?}", epoch, item.loss);
        }
    }

    model
}

// pub struct NoUMap<B, HV, LV> {
//     _phantom: PhantomData<(B, HV, LV)>,
// }

// impl<B, HV, LV> Default for NoUMap<B, HV, LV> {
//     fn default() -> Self {
//         Self {
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<
//         B: Backend + AutodiffBackend + Send + Sync,
//         HV: VectorSpace<f32>
//             + Copy
//             + Clone
//             + Send
//             + Sync
//             + Debug
//             + PartialEq
//             + From<VectorSerial<f32>>
//             + Into<Vec<f32>>,
//         LV: VectorSpace<f32> + Send + Sync + From<Vec<f32>> + for<'a> From<&'a [f32]> + Into<Vec<f32>>,
//         S: Send + Sync,
//         const PARTITION_CAP: usize,
//         const VECTOR_CAP: usize,
//         const MAX_LOADED: usize,
//     > UMapStrategy<HV, LV, S, NoStrategy, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>
//     for NoUMap<B, HV, LV>
// {
//     async fn project(&mut self, _model_key: NoStrategy, _new_value: VectorEntry<f32, HV>) -> LV {
//         panic!()
//     }

//     async fn update<const EPOCHS: usize>(
//         &mut self,
//         _model_key: NoStrategy,
//         _inter_graph: Arc<RwLock<IntraPartitionGraph<f32>>>,
//         _partition_buffer: Arc<
//             RwLock<
//                 DataBuffer<
//                     Partition<f32, HV, PARTITION_CAP, VECTOR_CAP>,
//                     PartitionSerial<f32>,
//                     S,
//                     MAX_LOADED,
//                 >,
//             >,
//         >,
//         _partition_membership: Arc<RwLock<PartitionMembership>>,
//     ) where
//         for<'a> &'a Partition<f32, HV, PARTITION_CAP, VECTOR_CAP>: Into<Uuid>,
//         VectorSerial<f32>: From<HV> + From<LV>,
//     {
//         panic!()
//     }

//     async fn save(&self, _model_key: NoStrategy) {
//         panic!()
//     }
// }
