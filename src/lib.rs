use vector::Vector;

pub mod db;
pub mod interface;
pub mod ops;
pub mod vector;

const VECTOR_CAP: usize = 384;

// placeholder impl until better solution
impl From<Vector<f32, VECTOR_CAP>> for Vector<f32, 2> {
    fn from(_: Vector<f32, VECTOR_CAP>) -> Self {
        // trait needs to be implemented but the conversion shouldn't be used in practice
        panic!("This conversion is not implemented and should not be used in practice.");
    }
}
// #![recursion_limit = "512"]
