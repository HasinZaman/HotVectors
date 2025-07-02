#![recursion_limit = "512"]

use core::panic;
use std::sync::Arc;
use std::{fs::File, thread};

use burn::backend::{wgpu::Wgpu, Autodiff};
use burn::prelude::Backend;
use burn_cuda::Cuda;
use tokio::sync::RwLock;
// use burn_tch::LibTorch;
use std::{default, io};
use tokio::{runtime, sync::mpsc::channel};
use HotVectors::db::component::umap::model::Model;
use HotVectors::db::db_loop;
use HotVectors::interface::{conn_pool, rest_api};
use HotVectors::vector::Vector;

use HotVectors::db::component::{ids::PartitionId, partition::VectorEntry};

const VECTOR_CAP: usize = 384;

fn main() {
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
                Autodiff<Wgpu<f32>>,
                // f32,
                Vector<f32, VECTOR_CAP>,
                Vector<f32, 2>,
                Model<Autodiff<Wgpu<f32>>, { VECTOR_CAP }, 3, 16, 2>,
                64,
                VECTOR_CAP,
                64,
                DB_THREADS,
                1,
                { VECTOR_CAP * 3 },
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
