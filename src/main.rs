use vector::{Extremes, Vector};

mod db;
pub mod ops;
pub mod vector;

fn main() {
    println!("Hello, world!");

    let a = Vector::<f32, 10>::max();
    println!("{a:#?}");
}
