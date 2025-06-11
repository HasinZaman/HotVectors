use core::panic;
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{
        Linear, LinearConfig, Relu,
    },
    optim::{AdamConfig, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor, TensorData},
    train::{
        RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use petgraph::{
    csr::DefaultIx,
    graph::NodeIndex,
    prelude::StableGraph,
    Undirected,
};
use rand::seq::SliceRandom;

use crate::
    vector::{Extremes, Field, VectorSpace}
;

use super::ids::PartitionId;

#[derive(Module, Debug)]
pub struct Model<B: Backend, const INPUT_DIM: usize, const LAYERS: usize, const OUTPUT_DIM: usize> {
    #[module]
    layers: [Linear<B>; LAYERS],
    activation: Relu,
}
impl<B: Backend, const INPUT_DIM: usize, const LAYERS: usize, const OUTPUT_DIM: usize>
    Model<B, INPUT_DIM, LAYERS, OUTPUT_DIM>
{
    pub fn new<const HIDDEN_DIM: usize>(device: &B::Device) -> Self {
        assert!(
            LAYERS >= 2,
            "At least 2 layers required: one hidden and one output"
        );

        let layers: [Linear<B>; LAYERS] = core::array::from_fn(|i| {
            if i == 0 {
                LinearConfig::new(INPUT_DIM, HIDDEN_DIM)
                    .with_bias(true)
                    .init(device)
            } else if i == LAYERS - 1 {
                LinearConfig::new(HIDDEN_DIM, OUTPUT_DIM)
                    .with_bias(true)
                    .init(device)
            } else {
                LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM)
                    .with_bias(true)
                    .init(device)
            }
        });

        Self {
            layers,
            activation: Relu::new(),
        }
    }

    pub fn forward_tensor(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < LAYERS - 1 {
                x = self.activation.forward(x);
            }
        }
        x
    }

    pub fn forward<
        HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + Into<Vec<f32>>,
        LD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + From<Vec<f32>>,
    >(
        &self,
        input: HD,
    ) -> LD
    where
        B: AutodiffBackend,
    {
        let point = TensorData::new(
            {
                let item: Vec<f32> = input.into();

                item
            },
            vec![1, INPUT_DIM],
        );

        let input_tensor: Tensor<B, 2> = Tensor::from_data(point, &B::Device::default());

        // let input_tensor: Tensor<B, 2> = Tensor::from_data(point, &B::Device::default());
        let output_tensor: Tensor<B, 2> = self.forward_tensor(input_tensor);

        // convert output tensor to LD
        let output_data: TensorData = output_tensor.to_data();
        let output_flat: Vec<f32> = output_data.to_vec().unwrap();

        LD::from(output_flat)
    }

    // pub fn forward_batch_tensor(
    //     &self,
    //     batch: UMapBatch<B, { INPUT_DIM }>,
    // ) -> Tensor<B, 2> {
    //     self.forward_tensor(batch.points)
    // }
}

impl<
        // 'a,
        B: AutodiffBackend,
        HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync,
        LD: VectorSpace<f32>
            + Extremes
            + Clone
            + Debug
            + Send
            + Sync
            + for<'a> From<&'a [f32; OUTPUT_DIM]>,
        const INPUT_DIM: usize,
        const OUTPUT_DIM: usize,
        const LAYERS: usize,
    >
    TrainStep<
        (
            UMapBatch<B, HD, INPUT_DIM>,
            &FuzzyNeighborGraph<f32, HD, PartitionId>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
        RegressionOutput<B>,
    > for Model<B, INPUT_DIM, LAYERS, OUTPUT_DIM>
where
    Vec<f32>: Into<HD> + Into<LD>,
    for<'a> &'a [f32]: Into<HD> + Into<LD>,
    for<'a> &'a [f32; OUTPUT_DIM]: Into<LD>,
{
    fn step(
        &self,
        (batch, higher_dim_graph, (a, b), _): (
            UMapBatch<B, HD, INPUT_DIM>,
            &FuzzyNeighborGraph<f32, HD, PartitionId>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
    ) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_tensor(batch.points.clone());

        let tensor_data = output.to_data();
        let flat: Vec<f32> = tensor_data.to_vec().unwrap();

        // let shape = &tensor_data.shape;
        let cols = tensor_data.shape[0];
        let rows = tensor_data.shape[1];

        // Now reshape
        let lower_dim_points: Vec<LD> = flat
            .chunks_exact(cols)
            .map(|chunk| {
                let arr: &[f32; OUTPUT_DIM] = chunk
                    .try_into()
                    .expect("Chunk size does not match OUTPUT_DIM");
                arr
            })
            .map(|arr| LD::from(arr))
            .collect();
        let ids: Vec<PartitionId> = (0..rows).map(|i| batch.sources[i]).collect();

        let mut total_cost = 0.;
        // should be replaced by searching all attractive edges and repulsive edges
        for z_1 in 0..rows {
            for z_2 in 0..z_1 {
                let id_1 = ids[z_1];
                let v_1 = &lower_dim_points[z_1];

                let id_2 = ids[z_2];
                let v_2 = &lower_dim_points[z_2];

                let dist: f32 = LD::dist(&v_1, &v_2);
                let weight: f32 = 1. / (1. + a * f32::powf(dist, 2. * b));

                let cost = {
                    let higher_weight = higher_dim_graph
                        .0
                        .find_edge(higher_dim_graph.1[&id_1], higher_dim_graph.1[&id_2])
                        .map_or(0., |e| *higher_dim_graph.0.edge_weight(e).unwrap_or(&0.));

                    let lower_weight = weight;

                    higher_weight * f32::log10(higher_weight / lower_weight)
                        + (1. - higher_weight)
                            * f32::log10((1. - higher_weight) / (1. - lower_weight))
                };

                total_cost += cost;
            }
        }
        total_cost *= -2.0;

        let device = (&batch.points).device();

        let loss_tensor: Tensor<B, 1> = Tensor::from_data([[total_cost]], &device);

        let regression_output = RegressionOutput {
            loss: loss_tensor.clone().reshape([1]),
            output: output.clone(),
            targets: output, // dummy targets matching output shape
        };

        let grads = loss_tensor.backward();

        TrainOutput::new(self, grads, regression_output)
    }
}

impl<
        // 'a,
        B: AutodiffBackend,
        HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync,
        LD: VectorSpace<f32>
            + Extremes
            + Clone
            + Debug
            + Send
            + Sync
            + for<'a> From<&'a [f32; OUTPUT_DIM]>,
        const INPUT_DIM: usize,
        const OUTPUT_DIM: usize,
        const LAYERS: usize,
    >
    ValidStep<
        (
            UMapBatch<B, HD, INPUT_DIM>,
            &FuzzyNeighborGraph<f32, HD, PartitionId>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
        RegressionOutput<B>,
    > for Model<B, INPUT_DIM, LAYERS, OUTPUT_DIM>
where
    Vec<f32>: Into<HD> + Into<LD>,
    for<'a> &'a [f32]: Into<HD> + Into<LD>,
    for<'a> &'a [f32; OUTPUT_DIM]: Into<LD>,
{
    fn step(
        &self,
        (batch, higher_dim_graph, (a, b), _): (
            UMapBatch<B, HD, INPUT_DIM>,
            &FuzzyNeighborGraph<f32, HD, PartitionId>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
    ) -> RegressionOutput<B> {
        let output = self.forward_tensor(batch.points.clone());

        let tensor_data = output.to_data();
        let flat: Vec<f32> = tensor_data.to_vec().unwrap();

        // let shape = &tensor_data.shape;
        let cols = tensor_data.shape[0];
        let rows = tensor_data.shape[1];

        // Now reshape
        let lower_dim_points: Vec<LD> = flat
            .chunks_exact(cols)
            .map(|chunk| {
                let arr: &[f32; OUTPUT_DIM] = chunk
                    .try_into()
                    .expect("Chunk size does not match OUTPUT_DIM");
                arr
            })
            .map(|arr| LD::from(arr))
            .collect();
        let ids: Vec<PartitionId> = (0..rows).map(|i| batch.sources[i]).collect();

        let mut total_cost = 0.;
        // should be replaced by searching all attractive edges and repulsive edges
        for z_1 in 0..rows {
            for z_2 in 0..z_1 {
                let id_1 = ids[z_1];
                let v_1 = &lower_dim_points[z_1];

                let id_2 = ids[z_2];
                let v_2 = &lower_dim_points[z_2];

                let dist: f32 = LD::dist(&v_1, &v_2);
                let weight: f32 = 1. / (1. + a * f32::powf(dist, 2. * b));

                let cost = {
                    let higher_weight = higher_dim_graph
                        .0
                        .find_edge(higher_dim_graph.1[&id_1], higher_dim_graph.1[&id_2])
                        .map_or(0., |e| *higher_dim_graph.0.edge_weight(e).unwrap_or(&0.));

                    let lower_weight = weight;

                    higher_weight * f32::log10(higher_weight / lower_weight)
                        + (1. - higher_weight)
                            * f32::log10((1. - higher_weight) / (1. - lower_weight))
                };

                total_cost += cost;
            }
        }
        total_cost *= -2.0;

        let device = (&batch.points).device();

        let loss_tensor: Tensor<B, 1> = Tensor::from_data([[total_cost]], &device);

        let regression_output = RegressionOutput {
            loss: loss_tensor.clone().reshape([1]),
            output: output.clone(),
            targets: output, // dummy targets matching output shape
        };

        regression_output
    }
}
#[derive(Debug, Clone)]
struct FuzzyNeighborGraph<
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

    // pub fn lower_dim(points: &Vec<(ID, B)>, a: f32, b: f32) -> Self {
    //     let mut graph: StableGraph<ID, f32, Undirected> = StableGraph::default();
    //     let mut id_to_idx: HashMap<ID, NodeIndex<DefaultIx>> = HashMap::new();

    //     for (node_id, _) in points.iter() {
    //         let idx: NodeIndex<DefaultIx> = graph.add_node(*node_id);
    //         id_to_idx.insert(*node_id, idx);
    //     }

    //     for i in 0..points.len() {
    //         for j in 0..i {
    //             let dist: f32 = B::dist(&points[i].1, &points[j].1);

    //             let weight: f32 = 1. / (1. + a * f32::powf(dist, 2. * b));

    //             graph.add_edge(id_to_idx[&points[i].0], id_to_idx[&points[j].0], weight);
    //         }
    //     }

    //     Self(graph, id_to_idx, PhantomData)
    // }

    // pub fn cost(higher_dim: &Self, lower_dim: &Self, id_1: ID, id_2: ID) -> f32 {
    //     let idx_1 = higher_dim.1[&id_1];
    //     let idx_2 = higher_dim.1[&id_2];

    //     let weight_higher = higher_dim
    //         .0
    //         .find_edge(idx_1, idx_2)
    //         .map_or(0., |e| *higher_dim.0.edge_weight(e).unwrap_or(&0.));

    //     let idx_1 = lower_dim.1[&id_1];
    //     let idx_2 = lower_dim.1[&id_2];

    //     let weight_lower = lower_dim
    //         .0
    //         .find_edge(idx_1, idx_2)
    //         .map_or(0., |e| *lower_dim.0.edge_weight(e).unwrap_or(&0.));

    //     weight_higher * f32::log10(weight_higher / weight_lower)
    //         + (1. - weight_higher) * f32::log10((1. - weight_higher) / (1. - weight_lower))
    // }
}

struct UMapBatcher<HD: Clone + VectorSpace<f32> + Extremes, const INPUT_DIM: usize> {
    attractive_size: usize,
    max_repulsive_size: usize,

    seed: u64,

    points: HashMap<PartitionId, HD>,
    higher_dim_graph: FuzzyNeighborGraph<f32, HD, PartitionId>,

    visited: HashMap<PartitionId, usize>,
}

impl<HD: Clone + VectorSpace<f32> + Extremes + Into<Vec<f32>>, const INPUT_DIM: usize>
    UMapBatcher<HD, INPUT_DIM>
{
    pub fn new(
        attractive_size: usize,
        max_repulsive_size: usize,
        seed: u64,
        points: HashMap<PartitionId, HD>,
        higher_dim_graph: FuzzyNeighborGraph<f32, HD, PartitionId>,
    ) -> Self {
        Self {
            attractive_size,
            max_repulsive_size,
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

    pub fn batch<B: Backend>(&mut self, device: &B::Device) -> UMapBatch<B, HD, INPUT_DIM> {
        // select random points from higher_dim_graph.1
        let attractive_points: Vec<PartitionId> = {
            // get all keys from higher_dim_graph.1
            let keys: Vec<PartitionId> = self.higher_dim_graph.1.keys().cloned().collect();
            // filter out visited points
            let unvisited: Vec<PartitionId> = keys
                .into_iter()
                .filter(|id| !self.visited.contains_key(id))
                .collect();
            // shuffle unvisited points
            let mut rng = rand::rng();
            let mut shuffled = unvisited.clone();
            shuffled.shuffle(&mut rng);
            // take attractive_size points
            let points: Vec<PartitionId> =
                shuffled.into_iter().take(self.attractive_size).collect();

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
        let repulsive_points: Vec<PartitionId> = {
            // get all keys from higher_dim_graph.1
            let keys: Vec<PartitionId> = self.higher_dim_graph.1.keys().cloned().collect();
            // filter out visited points and attractive points
            let unvisited: Vec<PartitionId> = keys
                .into_iter()
                .filter(|id| !self.visited.contains_key(id) && !attractive_points.contains(id))
                .collect();
            // shuffle unvisited points
            let mut rng = rand::rng();
            let mut shuffled = unvisited.clone();
            shuffled.shuffle(&mut rng);
            // take max_repulsive_size points
            let points: Vec<PartitionId> =
                shuffled.into_iter().take(self.max_repulsive_size).collect();

            // mark points as visited
            for point in &points {
                *self.visited.entry(*point).or_insert(0) += 1;
            }

            points
        };

        // create batch from attractive and repulsive points
        let mut items: Vec<(PartitionId, HD)> =
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
        UMapBatch::<B, HD, INPUT_DIM>::new(items, device)
    }
}

// impl<
//         B: Backend,
//         HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync,
//         const INPUT_DIM: usize,
//     > Batcher<B, UMapBatcher<HD, INPUT_DIM>, UMapBatch<B, HD, INPUT_DIM>>
//     for UMapBatcher<HD, INPUT_DIM>
// where
//     Vec<f32>: From<HD>,
// {
//     fn batch(
//         &self,
//         items: UMapBatcher<HD, INPUT_DIM>,
//         device: &<B as Backend>::Device,
//     ) -> UMapBatch<B, HD, INPUT_DIM> {
//         let sources: Vec<PartitionId> = items.iter().map(|(id, _)| *id).collect();
//         let points = TensorData::new(
//             items
//                 .iter()
//                 .map(|(_, v)| Vec::<f32>::from(v.clone()))
//                 .flatten()
//                 .collect::<Vec<_>>(),
//             vec![items.len(), INPUT_DIM],
//         );

//         let points: Tensor<B, 2> = Tensor::from_data(points, device);

//         UMapBatch {
//             sources,
//             points,
//             _phantom: PhantomData,
//         }
//     }
// }

#[derive(Debug, Clone)]
struct UMapBatch<B: Backend, V, const INPUT_DIM: usize> {
    sources: Vec<PartitionId>,
    points: Tensor<B, 2>,
    _phantom: PhantomData<V>,
}

impl<B: Backend, V: Clone + Into<Vec<f32>>, const INPUT_DIM: usize> UMapBatch<B, V, INPUT_DIM> {
    pub fn new(items: Vec<(PartitionId, V)>, device: &<B as Backend>::Device) -> Self {
        let sources: Vec<PartitionId> = items.iter().map(|(id, _)| *id).collect();
        let points = TensorData::new(
            items
                .iter()
                .map(|(_, v)| {
                    let v: Vec<f32> = v.clone().into();

                    v
                })
                .flatten()
                .collect::<Vec<_>>(),
            vec![items.len(), INPUT_DIM],
        );

        let points: Tensor<B, 2> = Tensor::from_data(points, device);

        UMapBatch {
            sources,
            points,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, V, const INPUT_DIM: usize> Default for UMapBatch<B, V, INPUT_DIM> {
    fn default() -> Self {
        UMapBatch {
            sources: Vec::new(),
            points: Tensor::from_data(
                TensorData::new(Vec::<f32>::new(), vec![0, INPUT_DIM]),
                &B::Device::default(),
            ),
            _phantom: PhantomData,
        }
    }
}

impl<
        B: Backend,
        V: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + Into<Vec<f32>>,
        const INPUT_DIM: usize,
    > Batcher<B, (PartitionId, V), UMapBatch<B, V, INPUT_DIM>> for UMapBatch<B, V, INPUT_DIM>
{
    fn batch(
        &self,
        items: Vec<(PartitionId, V)>,
        device: &<B as Backend>::Device,
    ) -> UMapBatch<B, V, INPUT_DIM> {
        let sources: Vec<PartitionId> = items.iter().map(|(id, _)| *id).collect();
        let points = TensorData::new(
            items
                .iter()
                .map(|(_, v)| {
                    let v: Vec<f32> = v.clone().into();

                    v
                })
                .flatten()
                .collect::<Vec<_>>(),
            vec![items.len(), INPUT_DIM],
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
    B: Backend,
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
        + for<'a> From<&'a [f32; OUTPUT_DIM]>
        + for<'a> From<&'a [f32]>,
    const LAYERS: usize,
    const INPUT_DIM: usize,
    const HIDDEN_DIM: usize,
    const OUTPUT_DIM: usize,
>(
    data: Vec<(PartitionId, HD)>,
    config: UMapTrainingConfig,
) -> Model<B, INPUT_DIM, LAYERS, OUTPUT_DIM>
where
    B: AutodiffBackend,
    // for<'a> &'a [f32]: Into<HD> + Into<LD>,
{
    let device = B::Device::default();
    let mut model = Model::<B, INPUT_DIM, LAYERS, OUTPUT_DIM>::new::<HIDDEN_DIM>(&device);

    // Create the higher-dimensional graph
    let higher_dim_graph = FuzzyNeighborGraph::<f32, HD, PartitionId>::higher_dim::<3>(&data);

    // create optimizer (Adam)
    let mut optimizer = config.optimizer.init();

    // let mut optimizer = Adam::new(&config.optimizer);

    // lower-dimensional parameters
    let a = 1.0; // UMAP attractive parameter (tune or get from config)
    let b = 1.5; // UMAP repulsive parameter (tune or get from config)

    // prepare batches
    let mut batcher = UMapBatcher::<HD, INPUT_DIM>::new(
        config.attractive_size,
        config.repulsive_size,
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
// pub fn p_umap_model<B: Backend, const LAYERS: usize>() -> Model<B, LAYERS> {
//     todo!()
// }
