use core::panic;
use std::{fs::File, thread};

use burn::backend::{wgpu::Wgpu, Autodiff};
use HotVectors::db::db_loop;
use HotVectors::interface::{conn_pool, rest_api};
use std::io;
use tokio::{runtime, sync::mpsc::channel};
use HotVectors::vector::Vector;

use HotVectors::db::component::{ids::PartitionId, partition::VectorEntry};

const VECTOR_CAP: usize = 128;

fn main() {
    // test umap trainer

    // 6 random vectors
    // const PARTITION_SIZE: usize = 150;
    // let mut rng = rand::rng();
    // let mut data: Vec<(PartitionId, Vector<f32, 3>)> = Vec::with_capacity(PARTITION_SIZE);

    // for _ in 0..50 {
    //     // Generate a random array of three f32 in [0.0, 1.0)
    //     let arr: [f32; 3] = [rng.random(), rng.random(), rng.random()];
    //     data.push((PartitionId(Uuid::new_v4()), Vector(arr)));
    // }
    // for _ in 0..50 {
    //     // Generate a random array of three f32 in [0.0, 1.0)
    //     let arr: [f32; 3] = [2.5 + rng.random::<f32>(), 2.5 + rng.random::<f32>(), 2.5 + rng.random::<f32>()];
    //     data.push((PartitionId(Uuid::new_v4()), Vector(arr)));
    // }
    // for _ in 0..25 {
    //     // Generate a random array of three f32 in [0.0, 1.0)
    //     let arr: [f32; 3] = [2.5 + rng.random::<f32>(), 7.5 + rng.random::<f32>(), 2.5 + rng.random::<f32>()];
    //     data.push((PartitionId(Uuid::new_v4()), Vector(arr)));
    // }
    // for _ in 0..25 {
    //     // Generate a random array of three f32 in [0.0, 1.0)
    //     let arr: [f32; 3] = [2.5 + rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()];
    //     data.push((PartitionId(Uuid::new_v4()), Vector(arr)));
    // }

    // let config: UMapTrainingConfig = UMapTrainingConfig {
    //     optimizer: AdamConfig::new(),
    //     attractive_size: 16,
    //     repulsive_size: 8,
    //     epoch: 3,
    //     seed: 42,
    //     learning_rate: 0.01,
    // };

    // type B = Wgpu<f32, i32>;
    // type AutoB = Autodiff<B>;

    // let model =
    //     train_umap::<AutoB, Vector<f32, 3>, Vector<f32, 2>, 3, 3, 16, 2>(data.clone(), config);

    // println!("Projected positions:");
    // let device = WgpuDevice::default();
    // for (partition_id, original_vec) in &data {
    //     let projected: Vector<f32, 2> = model.forward(*original_vec);
    //     println!("Partition {:?} ({:?}) -> Projected: {:?}", partition_id, original_vec, projected);
    // }

    // // Group 1 (low range)
    // for _ in 0..25 {
    //     let arr: [f32; 3] = [rng.random(), rng.random(), rng.random()];
    //     let test_vec = Vector(arr);
    //     let projected: Vector<f32, 2> = model.forward(test_vec);
    //     println!("Test (Group 1) {:?} -> Projected: {:?}", test_vec, projected);
    // }

    // // Group 2 (high range)
    // for _ in 0..30 {
    //     let arr: [f32; 3] = [
    //         2.5 + rng.random::<f32>(),
    //         2.5 + rng.random::<f32>(),
    //         2.5 + rng.random::<f32>(),
    //     ];
    //     let test_vec = Vector(arr);
    //     let projected: Vector<f32, 2> = model.forward(test_vec);
    //     println!("Test (Group 2) {:?} -> Projected: {:?}", test_vec, projected);
    // }

    let file = File::create("debug.log");
    let file = match file {
        Ok(file) => file,
        Err(error) => panic!("Error: {:?}", error),
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(io::stderr)
        .with_writer(file)
        .init();
    const DB_THREADS: usize = 10;
    const INTERFACE_THREADS: usize = 4;

    let (cmd_sender, cmd_receiver) = channel(64);

    let db_thread = thread::Builder::new()
        .name("db_thread".to_string())
        .spawn(move || {
            db_loop::<
                Autodiff<Wgpu<f32, i32>>,
                f32,
                Vector<f32, VECTOR_CAP>,
                20,
                VECTOR_CAP,
                128,
                DB_THREADS,
            >(cmd_receiver);
        });

    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(INTERFACE_THREADS)
        .enable_io()
        .build()
        .unwrap();

    rt.block_on(async {
        conn_pool::input_loop::<f32, Vector<f32, VECTOR_CAP>>(cmd_sender, None).await;
    });
}
