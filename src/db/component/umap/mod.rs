use core::panic;
use std::{collections::HashMap, fmt::Debug, hash::Hash, marker::PhantomData};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, Optimizer},
    prelude::Backend,
    record::RecorderError,
    tensor::{backend::AutodiffBackend, Tensor, TensorData},
    train::{RegressionOutput, TrainOutput, TrainStep},
};
use petgraph::{csr::DefaultIx, graph::NodeIndex, prelude::StableGraph, Undirected};
use rand::seq::SliceRandom;

use crate::vector::{Extremes, Field, VectorSpace};

pub mod dynamic_model;
pub mod incremental_model;
pub mod model;

pub trait ParamUMap<B: Backend> {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

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

    async fn save(&self, base_path: &str) -> Result<(), RecorderError>;

    async fn load(device: &B::Device, base_path: &str) -> Result<Self, RecorderError>
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
        println!("Calculating similarity matrix... {:?}", points.len());
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
    config: UMapTrainingConfig,
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
