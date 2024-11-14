use vector::Vector;

mod vector;

fn main() {
    println!("Hello, world!");

    let a = Vector::<f32, 10>::max();
    println!("{a:#?}");
}
