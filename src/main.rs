use std::{fs::File, thread};

use db::db_loop;
use interface::rest_api;
use std::io;
use tokio::{runtime, sync::mpsc::channel};
use vector::Vector;

mod db;
pub mod interface;
pub mod ops;
pub mod vector;

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
    const INTERFACE_THREADS: usize = 2;

    let (cmd_sender, cmd_receiver) = channel(64);

    let db_thread = thread::Builder::new()
        .name("db_thread".to_string())
        .spawn(move || {
            db_loop::<f32, Vector<f32, 2>, 20, 2, 10, DB_THREADS>(cmd_receiver);
        });

    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(INTERFACE_THREADS)
        .enable_io()
        .build()
        .unwrap();

    rt.block_on(async {
        rest_api::input_loop::<f32, Vector<f32, 2>>(cmd_sender).await;
    });
}
