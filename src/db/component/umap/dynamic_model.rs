use std::{fmt::Debug, hash::Hash, marker::PhantomData};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    record::RecorderError,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::AutodiffBackend, Tensor, TensorData},
    train::{RegressionOutput, TrainOutput, TrainStep},
};
use rkyv::{to_bytes, Archive, Deserialize, Serialize};

use crate::{
    db::component::{
        serial::FileExtension,
        umap::{FuzzyNeighborGraph, ParamUMap, UMapBatch},
    },
    vector::{Extremes, VectorSpace},
};

#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct DynamicModelConfig {
    input_dim: usize,
    output_dim: usize,
}

impl FileExtension for DynamicModelConfig {
    fn extension() -> &'static str {
        "config"
    }
}

#[derive(Module, Debug)]
pub struct DynamicModel<B: Backend, const LAYERS: usize> {
    input_dim: usize,
    output_dim: usize,

    #[module]
    layers: [Linear<B>; LAYERS],
    activation: Relu,
}

impl<B: Backend, const LAYERS: usize> DynamicModel<B, LAYERS> {
    pub fn new(device: &B::Device, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        assert!(
            LAYERS >= 2,
            "At least 2 layers required: one hidden and one output"
        );

        let layers: [Linear<B>; LAYERS] = core::array::from_fn(|i| {
            if i == 0 {
                LinearConfig::new(input_dim, hidden_dim)
                    .with_bias(true)
                    .init(device)
            } else if i == LAYERS - 1 {
                LinearConfig::new(hidden_dim, output_dim)
                    .with_bias(true)
                    .init(device)
            } else {
                LinearConfig::new(hidden_dim, hidden_dim)
                    .with_bias(true)
                    .init(device)
            }
        });

        Self {
            input_dim,
            output_dim,
            layers,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend, const LAYERS: usize> ParamUMap<B> for DynamicModel<B, LAYERS> {
    fn forward_tensor(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < LAYERS - 1 {
                x = self.activation.forward(x);
            }
        }
        x
    }

    fn forward<
        HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + Into<Vec<f32>>,
        LD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + From<Vec<f32>>,
    >(
        &self,
        input: HD,
    ) -> LD
    where
        B: AutodiffBackend,
    {
        let flat: Vec<f32> = input.into();
        assert_eq!(
            flat.len(),
            self.input_dim,
            "Input vector size ({}) does not match model input_dim ({})",
            flat.len(),
            self.input_dim
        );

        let point = TensorData::new(flat, vec![1, self.input_dim]);
        let input_tensor: Tensor<B, 2> = Tensor::from_data(point, &B::Device::default());

        let output_tensor: Tensor<B, 2> = self.forward_tensor(input_tensor);
        let output_data: TensorData = output_tensor.to_data();
        let output_flat: Vec<f32> = output_data.to_vec().unwrap();

        LD::from(output_flat)
    }

    async fn save(&self, base_path: &str) -> Result<(), RecorderError> {
        let model_config: DynamicModelConfig = DynamicModelConfig {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
        };

        let bytes = to_bytes::<rancor::Error>(&model_config).unwrap();

        tokio::fs::write(
            &format!("{base_path}//settings.{}", DynamicModelConfig::extension()),
            bytes.as_slice(),
        )
        .await
        .unwrap();

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        self.clone().save_file(base_path, &recorder)
    }

    async fn load(device: &B::Device, base_path: &str) -> Result<Self, RecorderError> {
        let config_path = format!("{base_path}//settings.{}", DynamicModelConfig::extension());
        let config_bytes = tokio::fs::read(&config_path).await.unwrap();
        let model_config: DynamicModelConfig =
            rkyv::from_bytes::<DynamicModelConfig, rancor::Error>(&config_bytes).unwrap();

        let mut model = Self::new(
            device,
            model_config.input_dim,
            model_config.output_dim, // hidden_dim is not used in this context
            model_config.output_dim,
        );

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model = model.load_file(base_path, &recorder, device)?;
        Ok(model)
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl<
        B: AutodiffBackend,
        ID: Clone + Copy + Hash + PartialEq + Eq,
        HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync,
        LD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync + From<Vec<f32>>,
        const LAYERS: usize,
    >
    TrainStep<
        (
            UMapBatch<B, ID, HD>,
            &FuzzyNeighborGraph<f32, HD, ID>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
        RegressionOutput<B>,
    > for DynamicModel<B, LAYERS>
where
    Vec<f32>: Into<HD> + Into<LD>,
    for<'a> &'a [f32]: Into<HD> + Into<LD>,
{
    fn step(
        &self,
        (batch, higher_dim_graph, (a, b), _): (
            UMapBatch<B, ID, HD>,
            &FuzzyNeighborGraph<f32, HD, ID>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
    ) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_tensor(batch.points.clone());

        let tensor_data = output.to_data();
        let flat: Vec<f32> = tensor_data.to_vec().unwrap();

        let cols = tensor_data.shape[0]; // batch size
        let rows = tensor_data.shape[1]; // output_dim

        assert_eq!(
            rows, self.output_dim,
            "Mismatch: tensor row count {} vs output_dim {}",
            rows, self.output_dim
        );

        let lower_dim_points: Vec<LD> = flat
            .chunks_exact(self.output_dim)
            .map(|chunk| LD::from(chunk.to_vec()))
            .collect();

        let ids: Vec<ID> = (0..cols).map(|i| batch.sources[i]).collect();

        let mut total_cost = 0.0;

        for z_1 in 0..cols {
            for z_2 in 0..z_1 {
                let id_1 = ids[z_1];
                let v_1 = &lower_dim_points[z_1];

                let id_2 = ids[z_2];
                let v_2 = &lower_dim_points[z_2];

                let dist = LD::dist(v_1, v_2);
                let weight = 1. / (1. + a * dist.powf(2. * b));

                let cost = {
                    let higher_weight = higher_dim_graph
                        .0
                        .find_edge(higher_dim_graph.1[&id_1], higher_dim_graph.1[&id_2])
                        .map_or(0., |e| *higher_dim_graph.0.edge_weight(e).unwrap_or(&0.));

                    let lower_weight = weight;

                    higher_weight * (higher_weight / lower_weight).log10()
                        + (1. - higher_weight)
                            * ((1. - higher_weight) / (1. - lower_weight)).log10()
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
            targets: output,
        };

        let grads = loss_tensor.backward();

        TrainOutput::new(self, grads, regression_output)
    }
}

// impl<
//         // 'a,
//         B: AutodiffBackend,
//         HD: VectorSpace<f32> + Extremes + Clone + Debug + Send + Sync,
//         LD: VectorSpace<f32>
//             + Extremes
//             + Clone
//             + Debug
//             + Send
//             + Sync
//             + for<'a> From<&'a [f32; OUTPUT_DIM]>,
//         const INPUT_DIM: usize,
//         const LAYERS: usize,
//     >
//     ValidStep<
//         (
//             UMapBatch<B, HD, INPUT_DIM>,
//             &FuzzyNeighborGraph<f32, HD, PartitionId>,
//             (f32, f32),
//             PhantomData<(LD, HD)>,
//         ),
//         RegressionOutput<B>,
//     > for DynamicModel<B, LAYERS>
// where
//     Vec<f32>: Into<HD> + Into<LD>,
//     for<'a> &'a [f32]: Into<HD> + Into<LD>,
//     for<'a> &'a [f32; OUTPUT_DIM]: Into<LD>,
// {
//     fn step(
//         &self,
//         (batch, higher_dim_graph, (a, b), _): (
//             UMapBatch<B, HD, INPUT_DIM>,
//             &FuzzyNeighborGraph<f32, HD, PartitionId>,
//             (f32, f32),
//             PhantomData<(LD, HD)>,
//         ),
//     ) -> RegressionOutput<B> {
//         let output = self.forward_tensor(batch.points.clone());
//
//         let tensor_data = output.to_data();
//         let flat: Vec<f32> = tensor_data.to_vec().unwrap();
//
//         // let shape = &tensor_data.shape;
//         let cols = tensor_data.shape[0];
//         let rows = tensor_data.shape[1];
//
//         // Now reshape
//         let lower_dim_points: Vec<LD> = flat
//             .chunks_exact(cols)
//             .map(|chunk| {
//                 let arr: &[f32; OUTPUT_DIM] = chunk
//                     .try_into()
//                     .expect("Chunk size does not match OUTPUT_DIM");
//                 arr
//             })
//             .map(|arr| LD::from(arr))
//             .collect();
//         let ids: Vec<PartitionId> = (0..rows).map(|i| batch.sources[i]).collect();
//
//         let mut total_cost = 0.;
//         // should be replaced by searching all attractive edges and repulsive edges
//         for z_1 in 0..rows {
//             for z_2 in 0..z_1 {
//                 let id_1 = ids[z_1];
//                 let v_1 = &lower_dim_points[z_1];
//
//                 let id_2 = ids[z_2];
//                 let v_2 = &lower_dim_points[z_2];
//
//                 let dist: f32 = LD::dist(&v_1, &v_2);
//                 let weight: f32 = 1. / (1. + a * f32::powf(dist, 2. * b));
//
//                 let cost = {
//                     let higher_weight = higher_dim_graph
//                         .0
//                         .find_edge(higher_dim_graph.1[&id_1], higher_dim_graph.1[&id_2])
//                         .map_or(0., |e| *higher_dim_graph.0.edge_weight(e).unwrap_or(&0.));
//
//                     let lower_weight = weight;
//
//                     higher_weight * f32::log10(higher_weight / lower_weight)
//                         + (1. - higher_weight)
//                             * f32::log10((1. - higher_weight) / (1. - lower_weight))
//                 };
//
//                 total_cost += cost;
//             }
//         }
//         total_cost *= -2.0;
//
//         let device = (&batch.points).device();
//
//         let loss_tensor: Tensor<B, 1> = Tensor::from_data([[total_cost]], &device);
//
//         let regression_output = RegressionOutput {
//             loss: loss_tensor.clone().reshape([1]),
//             output: output.clone(),
//             targets: output, // dummy targets matching output shape
//         };
//
//         regression_output
//     }
// }
