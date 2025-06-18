use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    sync::Arc,
};

use burn::{
    backend::Autodiff, module::AutodiffModule, optim::AdamConfig, prelude::Backend, tensor::backend::AutodiffBackend, train::{RegressionOutput, TrainStep}
};
use rancor::{Error, Strategy};
use rkyv::{
    bytecheck::CheckBytes, de::Pool, ser::{allocator::ArenaHandle, sharing::Share, Serializer}, util::AlignedVec, validation::{archive::ArchiveValidator, shared::SharedValidator, Validator}, Archive, DeserializeUnsized, Serialize
};
use sled::Db;
use tokio::sync::RwLock;
use uuid::Uuid;

use tracing::{event, Level};

use crate::{
    db::component::{
        data_buffer::{DataBuffer, Global},
        graph::IntraPartitionGraph,
        ids::{PartitionId, VectorId},
        partition::{ArchivedVectorEntrySerial, Partition, PartitionMembership, PartitionSerial, VectorEntry, VectorEntrySerial},
        umap::{train_umap, FuzzyNeighborGraph, ParamUMap, UMapBatch, UMapTrainingConfig},
    },
    resolve_buffer,
    vector::{Extremes, Field, VectorSerial, VectorSpace},
};

pub struct IncrementalUMap<
    B: Backend + AutodiffBackend,
    M: ParamUMap<B> + Sized + AutodiffModule<B>,
    HV: VectorSpace<f32> + Debug + Clone + Copy + PartialEq + Extremes + From<Vec<f32>> + for<'a> From<&'a [f32]>,
    LV: VectorSpace<f32> + Debug + Clone + Copy + From<Vec<f32>> + for<'a> From<&'a [f32]>,
> {
    dir: String,

    // training data
    attractive_size: usize,
    repulsive_size: usize,

    // model
    model: M,

    // cache
    // Future Optimization (use DB in-order to reduce amount of data in memory)
    trained: HashSet<Uuid>,
    dirty_vecs: Vec<VectorEntry<f32, HV>>,
    cache: HashMap<Uuid, LV>,

    _phantom: PhantomData<B>,
}

impl<
        B: Backend + AutodiffBackend,
        M: ParamUMap<B>
            + AutodiffModule<B>
            + Sized
            + Clone
            + for<'a> TrainStep<
                (
                    UMapBatch<B, VectorId, HV>,
                    &'a FuzzyNeighborGraph<f32, HV, VectorId>,
                    (f32, f32),
                    PhantomData<(LV, HV)>,
                ),
                RegressionOutput<B>,
            >,
        HV: VectorSpace<f32> + Debug + Clone + Copy + Extremes + PartialEq + From<VectorSerial<f32>> + From<Vec<f32>> + for<'a> From<&'a [f32]> + Send + Sync,
        LV: VectorSpace<f32> + Extremes + Clone + Copy + Debug + Send + Sync + From<Vec<f32>> + for<'a> From<&'a [f32]>,
    > IncrementalUMap<B, M, HV, LV>
{
    pub fn new(dir: String, attractive_size: usize, repulsive_size: usize, model: M) -> Self {
        Self {
            dir,
            attractive_size,
            repulsive_size,
            model,
            trained: HashSet::new(),
            dirty_vecs: Vec::new(),
            cache: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    pub fn project(&mut self, new_value: VectorEntry<f32, HV>) -> LV where Vec<f32>: From<HV> + From<LV>, {
        let uuid = new_value.id;

        if let Some(cached) = self.cache.get(&uuid) {
            return cached.clone();
        }

        if self.trained.len() == 0 {
            return LV::additive_identity();
        };

        let projected: LV = self.model.forward::<HV, LV>(new_value.vector.clone());

        if self.trained.contains(&uuid) {
            let projected: LV = self.model.forward::<HV, LV>(new_value.vector.clone());

            return projected;
        }

        self.cache.insert(uuid, projected.clone());

        self.dirty_vecs.push(new_value);

        projected
    }

    pub async fn update<
        S,
        const EPOCHS: usize,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
        const MAX_LOADED: usize,
    >(
        &mut self,
        inter_graph: Arc<RwLock<IntraPartitionGraph<f32>>>,
        partition_buffer: Arc<
            RwLock<
                DataBuffer<
                    Partition<f32, HV, PARTITION_CAP, VECTOR_CAP>,
                    PartitionSerial<f32>,
                    S,
                    MAX_LOADED,
                >,
            >,
        >,

        partition_membership: Arc<RwLock<PartitionMembership>>,
    ) where
        for<'a> &'a Partition<f32, HV, PARTITION_CAP, VECTOR_CAP>: Into<Uuid>,
        Vec<f32>: From<HV> + From<LV>,
        VectorSerial<f32>: From<HV> + From<LV>
    {
        // Acquire locks in correct order
        let inter_graph = inter_graph.read().await;
        let partition_buffer = &mut *partition_buffer.write().await;

        let partitions_to_train: HashSet<PartitionId> = {
            let mut partitions_to_train: HashSet<PartitionId> = HashSet::new();
            let partition_membership = partition_membership.read().await;

            for vector_entry in &self.dirty_vecs {
                let vec_id = VectorId(vector_entry.id);
                if let Some(partition_id) = partition_membership.get_partition_id(vec_id) {
                    partitions_to_train.insert(partition_id);
                } else {
                    todo!()
                    // tracing::warn!("No partition found in membership DB for vector id {:?}", vec_id);
                }
            }

            partitions_to_train
        };
        // get all the vectors
        let mut vectors: Vec<(VectorId, HV)> = Vec::new();

        for partitions_to_train in &partitions_to_train {
            let partition = resolve_buffer!(ACCESS, partition_buffer, *partitions_to_train);

            let Some(partition) = &mut *partition.try_write().unwrap() else {
                todo!()
            };

            partition.iter().for_each(|VectorEntry { id, vector, .. }| {
                vectors.push((VectorId(*id), *vector));
            });
        }

        let model = self.model.clone();

        self.model = train_umap::<B, VectorId, M, HV, LV>(
            model,
            vectors,
            UMapTrainingConfig {
                optimizer: AdamConfig::new(),
                attractive_size: 16,
                repulsive_size: 8,
                epoch: EPOCHS,
                seed: 42,
                learning_rate: 0.01,
            },
        );
        // Locks auto-release at end of scope
    }
}

// pub struct IncrementalConfig {

// }

// todo
// umap model struct
// pipe that
