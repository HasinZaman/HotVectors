use std::{collections::{HashMap, HashSet}, marker::PhantomData, sync::Arc};

use burn::prelude::Backend;
use sled::Db;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{db::component::{data_buffer::{DataBuffer, Global}, graph::IntraPartitionGraph, meta::Meta, partition::{Partition, PartitionSerial, VectorEntry, VectorEntrySerial}, umap::ParamUMap}, vector::{Extremes, Field, VectorSpace}};


pub struct IncrementalUMap<B: Backend, M: ParamUMap<B> + Sized, F: Field<F>, HV: VectorSpace<F>, LV: VectorSpace<F>> {
    dir: String,

    // training data
    attractive_size: usize,
    repulsive_size: usize,

    // model
    model: M,

    // cache
    // Future Optimization (use DB in-order to reduce amount of data in memory)
    trained: HashSet<Uuid>,
    dirty_vecs: Vec<VectorEntry<F, HV>>,
    cache: HashMap<Uuid, LV>,

    _phantom: PhantomData<B>
}

impl<B: Backend, M: ParamUMap<B> + Sized, F: Field<F>, HV: VectorSpace<F>, LV: VectorSpace<F> + Extremes> IncrementalUMap<B, M, F, HV, LV> {
    pub fn new(
        dir: String,
        attractive_size: usize,
        repulsive_size: usize,
        model: M,
    ) -> Self {
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

    pub fn project(&mut self, new_value: VectorEntry<F, HV>) -> LV {
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

    pub async fn train_model<
        V: VectorSpace<F> + Clone + Debug + Extremes + 'static,
        const PARTITION_CAP: usize,
        const VECTOR_CAP: usize,
        const MAX_LOADED: usize,
    >(
        &self,
        meta_data: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Meta<F, V>>>>>>,
        inter_graph: Arc<RwLock<IntraPartitionGraph<F>>>,
        partition_buffer: Arc<RwLock<DataBuffer<
            Partition<F, V, PARTITION_CAP, VECTOR_CAP>,
            PartitionSerial<F>,
            Global,
            MAX_LOADED,
        >>>,
    ) {
        // Acquire locks in correct order
        let meta_guard = meta_data.read().await;
        let graph_guard = inter_graph.read().await;
        let buffer_guard = partition_buffer.write().await;

        // Your training logic goes here
        {
            let mut partitions_to_train: HashSet<Uuid> = HashSet::new();

            for vector_entry in &self.dirty_vecs {
                let vec_id = vector_entry.id;

                // Find the partition containing this vector by checking meta_data centroid distance or membership
                // This example just finds the closest partition based on centroid distance (like your example)

                let mut closest_partition: Option<(Uuid, F)> = None;

                for (partition_id, meta_arc) in meta_guard.iter() {
                    let meta = meta_arc.read().await;
                    let dist = HV::dist(&vector_entry.vector, &meta.centroid);
                    match &closest_partition {
                        Some((_, best_dist)) if *best_dist <= dist => {}
                        _ => {
                            closest_partition = Some((*partition_id, dist));
                        }
                    }
                }

                if let Some((closest_id, _dist)) = closest_partition {
                    partitions_to_train.insert(closest_id);
                } else {
                    // No partition found; TODO: handle empty partitions or create new partition
                    tracing::event!(tracing::Level::WARN, "No partition found for vector id {:?}", vec_id);
                }
            }
        }

        todo!("Implement training logic using meta_data, inter_graph, and partition_buffer");

        // Locks auto-release at end of scope
        drop(meta_guard);
        drop(graph_guard);
        drop(buffer_guard);
    }
}

// pub struct IncrementalConfig {

// }

// todo
// umap model struct
// pipe that
