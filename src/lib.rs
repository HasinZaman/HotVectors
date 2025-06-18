use core::panic;
use std::{fs::File, thread};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{wgpu::Wgpu, Autodiff};
use burn::optim::AdamConfig;
use burn::tensor::Tensor;
use db::db_loop;
use interface::{conn_pool, rest_api};
use rand::Rng;
use std::io;
use tokio::{runtime, sync::mpsc::channel};
use uuid::Uuid;
use vector::Vector;

use crate::db::component::{ids::PartitionId, partition::VectorEntry};

pub mod db;
pub mod interface;
pub mod ops;
pub mod vector;

const VECTOR_CAP: usize = 384;

// placeholder impl until better solution
impl From<Vector<f32, VECTOR_CAP>> for Vector<f32, 2> {
    fn from(value: Vector<f32, VECTOR_CAP>) -> Self {
        // trait needs to be implemented but the conversion shouldn't be used in practice
        panic!("This conversion is not implemented and should not be used in practice.");
    }
}
