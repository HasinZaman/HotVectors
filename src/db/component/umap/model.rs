use std::{fmt::Debug, hash::Hash, marker::PhantomData};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    record::RecorderError,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::AutodiffBackend, Tensor, TensorData},
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    db::component::umap::{FuzzyNeighborGraph, ParamUMap, UMapBatch},
    vector::{Extremes, VectorSpace},
};

#[derive(Module, Debug)]
pub struct Model<
    B: Backend,
    const INPUT_DIM: usize,
    const LAYERS: usize,
    const HIDDEN_DIM: usize,
    const OUTPUT_DIM: usize,
> {
    #[module]
    layers: [Linear<B>; LAYERS],
    activation: Relu,
}
impl<
        B: Backend,
        const INPUT_DIM: usize,
        const LAYERS: usize,
        const HIDDEN_DIM: usize,
        const OUTPUT_DIM: usize,
    > Model<B, INPUT_DIM, LAYERS, HIDDEN_DIM, OUTPUT_DIM>
{
    pub fn new(device: &B::Device) -> Self {
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
}

impl<
        B: Backend,
        const INPUT_DIM: usize,
        const LAYERS: usize,
        const HIDDEN_DIM: usize,
        const OUTPUT_DIM: usize,
    > ParamUMap<B> for Model<B, INPUT_DIM, LAYERS, HIDDEN_DIM, OUTPUT_DIM>
{
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

    async fn save(&self, base_path: &str) -> Result<(), RecorderError> {
        let recorder: NamedMpkFileRecorder<FullPrecisionSettings> =
            NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.clone().save_file(base_path, &recorder)
    }

    async fn load(device: &B::Device, base_path: &str) -> Result<Self, RecorderError> {
        let recorder: NamedMpkFileRecorder<FullPrecisionSettings> =
            NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let mut model: Model<B, INPUT_DIM, LAYERS, HIDDEN_DIM, OUTPUT_DIM> = Self::new(device);
        model = model.load_file(base_path, &recorder, device)?;
        Ok(model)
    }

    fn input_dim(&self) -> usize {
        INPUT_DIM
    }

    fn output_dim(&self) -> usize {
        OUTPUT_DIM
    }
}

impl<
        // 'a,
        B: AutodiffBackend,
        ID: Clone + Copy + Hash + PartialEq + Eq,
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
        const HIDDEN_DIM: usize,
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
    > for Model<B, INPUT_DIM, LAYERS, HIDDEN_DIM, OUTPUT_DIM>
where
    Vec<f32>: Into<HD> + Into<LD>,
    for<'a> &'a [f32]: Into<HD> + Into<LD>,
    for<'a> &'a [f32; OUTPUT_DIM]: Into<LD>,
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
        let ids: Vec<ID> = (0..rows).map(|i| batch.sources[i]).collect();

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
        ID: Clone + Copy + Hash + PartialEq + Eq,
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
        const HIDDEN_DIM: usize,
        const LAYERS: usize,
    >
    ValidStep<
        (
            UMapBatch<B, ID, HD>,
            &FuzzyNeighborGraph<f32, HD, ID>,
            (f32, f32),
            PhantomData<(LD, HD)>,
        ),
        RegressionOutput<B>,
    > for Model<B, INPUT_DIM, LAYERS, HIDDEN_DIM, OUTPUT_DIM>
where
    Vec<f32>: Into<HD> + Into<LD>,
    for<'a> &'a [f32]: Into<HD> + Into<LD>,
    for<'a> &'a [f32; OUTPUT_DIM]: Into<LD>,
{
    fn step(
        &self,
        (batch, higher_dim_graph, (a, b), _): (
            UMapBatch<B, ID, HD>,
            &FuzzyNeighborGraph<f32, HD, ID>,
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
        let ids: Vec<ID> = (0..rows).map(|i| batch.sources[i]).collect();

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

// pub fn p_umap_model<B: Backend, const LAYERS: usize>() -> Model<B, LAYERS> {
//     todo!()
// }
