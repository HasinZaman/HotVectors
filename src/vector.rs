use std::{
    array,
    ops::{Add, Mul, Sub},
};

use serde::{Deserialize, Serialize};

pub trait Extremes {
    fn min() -> Self;
    fn max() -> Self;
    fn additive_identity() -> Self;
    fn multiplicative_identity() -> Self;
}

pub trait VectorSpace<A> {
    fn add(lhs: &Self, rhs: &Self) -> Self;

    fn sub(lhs: &Self, rhs: &Self) -> Self;

    fn dot(lhs: &Self, rhs: &Self) -> Self;

    fn scalar_mult(lhs: &Self, scalar: &A) -> Self;
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Vector<A: PartialEq + Clone + Copy, const CAP: usize>([A; CAP]);

impl<
        A: PartialEq + Clone + Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
        const CAP: usize,
    > VectorSpace<A> for Vector<A, CAP>
{
    fn add(lhs: &Self, rhs: &Self) -> Self {
        let mut iter = (lhs.0).iter().zip(rhs.0.iter()).map(|(a, b)| *a + *b);

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }

    fn sub(lhs: &Self, rhs: &Self) -> Self {
        let mut iter = (lhs.0).iter().zip(rhs.0.iter()).map(|(a, b)| *a - *b);

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }

    fn dot(lhs: &Self, rhs: &Self) -> Self {
        let mut iter = (lhs.0).iter().zip(rhs.0.iter()).map(|(a, b)| *a * *b);

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }

    fn scalar_mult(vec: &Self, scalar: &A) -> Self {
        let mut iter = vec.0.iter().map(|a| *scalar * *a);
        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorSerial<A: Clone + Copy>(Vec<A>);

impl<A: PartialEq + Clone + Copy, const CAP: usize> From<VectorSerial<A>> for Vector<A, CAP> {
    fn from(value: VectorSerial<A>) -> Self {
        if value.0.len() != CAP {
            panic!("Invalid size");
        }

        let mut iter = value.0.iter();

        Vector(array::from_fn(|_| *(iter.next().unwrap())))
    }
}
impl<A: PartialEq + Clone + Copy, const CAP: usize> From<Vector<A, CAP>> for VectorSerial<A> {
    fn from(value: Vector<A, CAP>) -> Self {
        VectorSerial(value.0.to_vec())
    }
}

impl<A: PartialEq + Clone + Copy, const CAP: usize> Vector<A, CAP> {
    pub fn splat(val: A) -> Self {
        let mut iter = (0..CAP).map(|_| val);
        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
}

macro_rules! extremes_int {
    [$head: tt , $($tail: tt),+ ] => {
        extremes_int!($head);
        extremes_int![$($tail),+];
    };
    [$head: tt] => {
        impl<const CAP: usize> Extremes for Vector<$head, CAP> {
            fn min() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| $head::MIN);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            fn max() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| $head::MAX);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            fn additive_identity() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| 0);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            fn multiplicative_identity() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| 1);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
        }
    };
}
macro_rules! extremes_float {
    [$head: tt , $($tail: tt),+ ] => {
        extremes_float!($head);
        extremes_float![$($tail),+];
    };
    [$head: tt] => {
        impl<const CAP: usize> Extremes for Vector<$head, CAP> {
            fn min() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| $head::MIN);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            fn max() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| $head::MAX);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            fn additive_identity() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| 0.);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            fn multiplicative_identity() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| 1.);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
        }
    };
}

extremes_int![i8, i16, i32, i64, i128];
extremes_int![u8, u16, u32, u64, u128];
extremes_float![f32, f64];
