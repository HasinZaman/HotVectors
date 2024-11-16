use std::cmp::min;

use heapify::{make_heap, pop_heap};

use crate::vector::{self, Field, VectorSpace};

struct KeyValuePair<A, B: PartialOrd + PartialEq>(pub(crate) A, pub(crate) B);

impl<A, B: PartialOrd + PartialEq> PartialOrd for KeyValuePair<A, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
}
impl<A, B: PartialOrd + PartialEq> PartialEq for KeyValuePair<A, B> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

pub fn knn<'a, A, B: PartialOrd + PartialEq>(
    data: &[&'a [A]],
    metric_func: fn(&A) -> B,
    neighbors: usize,
) -> Vec<&'a A> {
    let mut data = data
        .iter()
        .map(|x| x.iter())
        .flatten()
        .map(|x| x)
        .map(|x| KeyValuePair(x, metric_func(x)))
        .collect::<Vec<KeyValuePair<&A, B>>>();

    make_heap(&mut data);

    (0..min(neighbors, data.len()))
        .map(|_| {
            pop_heap(&mut data);
            data.pop().unwrap()
        })
        .map(|KeyValuePair(key, _val)| key)
        .collect::<Vec<&A>>()
}

pub fn euclidean_dist<A: PartialOrd + PartialEq + Field<A>, B: VectorSpace<A>>(
    vector_1: &B,
    vector_2: &B,
) -> A {
    let delta = B::sub(vector_1, vector_2);
    
    B::dot(&delta, &delta)
}
