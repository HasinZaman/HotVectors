use std::{
    array,
    ops::{Add, Mul, Sub},
};

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Vector<A: PartialEq +Clone + Copy, const CAP: usize>([A; CAP]);

impl<A: PartialEq + Clone + Copy, const CAP: usize> Vector<A, CAP> {
    pub fn splat(val: A) -> Self {
        let mut iter = (0..CAP)
            .map(|_| val);
        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
    // pub fn zero() -> Self {
    //     todo!()
    // }
    // pub fn one() -> Self {
    //     todo!()
    // }

    // pub fn max() -> Self {
    //     todo!()
    // } 
    // pub fn min() -> Self {
    //     todo!()
    // } 

    pub fn scalar_mul(scalar: A, vec: &Self) -> Self
    where A: Default + Mul<Output = A>
    {
        let mut iter = vec.0
            .iter()
            .map(|a| scalar * *a);
        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
}

macro_rules! extremes {
    [$head: tt , $($tail: tt),+ ] => {
        extremes!($head);
        extremes![$($tail),+];
    };
    [$head: tt] => {
        impl<const CAP: usize> Vector<$head, CAP> {
            pub fn min() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| $head::MIN);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            pub fn max() -> Self {
                let mut iter = (0..CAP)
                    .map(|_| $head::MAX);
                Vector(array::from_fn(|_| iter.next().unwrap()))
            }
            
        }
    };
}

extremes![i8, i16, i32, i64, i128];
extremes![u8, u16, u32, u64, u128];
extremes![f32, f64];

impl<A, const CAP: usize> Add for Vector<A, CAP>
where
    A: Add<Output = A> + PartialEq + Clone + Copy,
{
    type Output = Vector<A, CAP>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut iter = (&self.0).iter().zip(rhs.0.iter()).map(|(a, b)| *a + *b);

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
}

impl<A, const CAP: usize> Sub for Vector<A, CAP>
where
    A: Sub<Output = A> + PartialEq + Clone + Copy,
{
    type Output = Vector<A, CAP>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut iter = (&self.0).iter().zip(rhs.0.iter()).map(|(a, b)| *a - *b);

        Vector(array::from_fn(|_| iter.next().unwrap()))
    }
}

impl<A, const CAP: usize> Mul for Vector<A, CAP>
where
    A: Default + Mul<Output = A> + Add<Output = A> + PartialEq + Clone + Copy,
{
    type Output = A;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self.0)
            .iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| *a * *b)
            .fold(A::default(), |acc, next| acc + next)
    }
}

