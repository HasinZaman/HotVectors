use std::{collections::HashMap, sync::mpsc};

use db::{component::graph::InterPartitionGraph, operations::LoadedPartitions};
use vector::{Extremes, Vector};

mod db;
pub mod ops;
pub mod vector;

fn main() {
    // println!("Hello, world!");

    // let a = Vector::<f32, 10>::max();
    // println!("{a:#?}");
    let loaded_partitions: LoadedPartitions<f32, Vector<f32, 10>, 500, 500, 500> =
        LoadedPartitions::new();
    let inter_partition_graph: InterPartitionGraph<f32> = InterPartitionGraph::new();
    // let partition_meta_data = HashMap::new();

    // let (tx, cmd_input) = mpsc::channel();
    // let (logger, rx) = mpsc::channel();

    // db_loop(
    //     loaded_partitions,
    //     inter_partition_graph,
    //     partition_meta_data,
    //     cmd_input,
    //     logger,
    // );
}
