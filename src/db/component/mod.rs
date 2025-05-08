pub mod cluster;
pub mod data_buffer;
pub mod graph;
pub mod ids;
pub mod meta;
pub mod partition;
pub mod serial;

#[cfg(feature = "benchmark")]
pub mod benchmark;

pub trait Initialize {
    fn initialize_dir() -> bool
    where
        Self: Sized;
}
