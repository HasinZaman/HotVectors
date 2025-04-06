use std::{array, fmt::Debug};

use rkyv::{Archive, Deserialize, Serialize};
use spade::{HasPosition, Point2, SpadeNum};

pub trait Extremes {
    fn min() -> Self;
    fn max() -> Self;
    fn additive_identity() -> Self;
    fn multiplicative_identity() -> Self;
}

pub trait Field<A> {
    fn add(lhs: &Self, rhs: &Self) -> Self;
    fn sub(lhs: &Self, rhs: &Self) -> Self;
    fn mult(lhs: &Self, rhs: &Self) -> Self;
    fn div(lhs: &Self, rhs: &Self) -> Self;
    fn additive_identity() -> Self;
    fn multiplicative_identity() -> Self;
    fn from_usize(x: usize) -> Self;
}

pub trait VectorSpace<A: Field<A>> {
    fn add(lhs: &Self, rhs: &Self) -> Self;

    fn sub(lhs: &Self, rhs: &Self) -> Self;

    fn dot(lhs: &Self, rhs: &Self) -> A;

    fn scalar_mult(lhs: &Self, scalar: &A) -> Self;

    fn dist(lhs: &Self, rhs: &Self) -> A
    where
        Self: Sized,
    {
        let dist: Self = Self::sub(lhs, rhs);

        Self::dot(&dist, &dist)
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Vector<A: PartialEq + Clone + Copy, const CAP: usize>(pub [A; CAP]);

impl<A: PartialEq + Clone + Copy + Field<A>, const CAP: usize> VectorSpace<A> for Vector<A, CAP> {
    fn add(lhs: &Self, rhs: &Self) -> Self {
        let mut iter = (lhs.0).iter().zip(rhs.0.iter()).map(|(a, b)| A::add(a, b));

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }

    fn sub(lhs: &Self, rhs: &Self) -> Self {
        let mut iter = (lhs.0).iter().zip(rhs.0.iter()).map(|(a, b)| A::sub(a, b));

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }

    fn dot(lhs: &Self, rhs: &Self) -> A {
        (lhs.0)
            .iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| A::mult(a, b))
            .fold(A::additive_identity(), |acc, next| A::add(&acc, &next))
    }

    fn scalar_mult(vec: &Self, scalar: &A) -> Self {
        let mut iter = vec.0.iter().map(|a| A::mult(scalar, a));
        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
}

impl<A: PartialEq + Clone + Copy + Field<A> + Into<f32>> HasPosition for Vector<A, 2> {
    type Scalar = f32;

    fn position(&self) -> Point2<Self::Scalar> {
        Point2 {
            x: self.0[0].into(),
            y: self.0[1].into(),
        }
    }
}
// impl HasPosition for Vector<f32, 2> {
//     type Scalar = f32;

//     fn position(&self) -> Point2<Self::Scalar> {
//         Point2 { x: self.0[0], y: self.0[1] }
//     }
// }
// impl HasPosition for Vector<f64, 2> {
//     type Scalar = f64;

//     fn position(&self) -> Point2<Self::Scalar> {
//         Point2 { x: self.0[0], y: self.0[1] }
//     }
// }

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
pub struct VectorSerial<A: Clone + Copy>(pub Vec<A>);

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
        impl Extremes for $head {
            fn min() -> Self {
                $head::MIN
            }
            fn max() -> Self {
                $head::MAX
            }
            fn additive_identity() -> Self {
                0
            }
            fn multiplicative_identity() -> Self {
                1
            }
        }
        impl Field<$head> for $head {
            fn add(lhs: &Self, rhs: &Self) -> Self {
                lhs + rhs
            }

            fn sub(lhs: &Self, rhs: &Self) -> Self {
                lhs - rhs
            }

            fn mult(lhs: &Self, rhs: &Self) -> Self {
                lhs * rhs
            }

            fn div(lhs: &Self, rhs: &Self) -> Self {
                lhs / rhs
            }

            fn additive_identity() -> Self {
                0
            }

            fn multiplicative_identity() -> Self {
                1
            }

            fn from_usize(x: usize) -> Self {
                x as $head
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
        impl Extremes for $head {
            fn min() -> Self {
                $head::MIN
            }
            fn max() -> Self {
                $head::MAX
            }
            fn additive_identity() -> Self {
                0.
            }
            fn multiplicative_identity() -> Self {
                1.
            }
        }
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
        impl Field<$head> for $head {
            fn add(lhs: &Self, rhs: &Self) -> Self {
                lhs + rhs
            }

            fn sub(lhs: &Self, rhs: &Self) -> Self {
                lhs - rhs
            }

            fn mult(lhs: &Self, rhs: &Self) -> Self {
                lhs * rhs
            }

            fn div(lhs: &Self, rhs: &Self) -> Self {
                lhs / rhs
            }

            fn additive_identity() -> Self {
                0.
            }

            fn multiplicative_identity() -> Self {
                1.
            }

            fn from_usize(x: usize) -> Self {
                x as $head
            }
        }
    };
}

extremes_int![i8, i16, i32, i64, i128];
extremes_int![u8, u16, u32, u64, u128];
extremes_float![f32, f64];
