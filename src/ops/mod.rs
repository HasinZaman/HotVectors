use std::cmp::{min, Reverse};

use heapify::{make_heap, pop_heap};

use crate::vector::{Field, VectorSpace};

struct KeyValuePair<A, B: PartialOrd + PartialEq>(pub(crate) A, pub(crate) Reverse<B>);

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
        .map(|x| KeyValuePair(x, Reverse(metric_func(x))))
        .collect::<Vec<KeyValuePair<&A, B>>>();

    if data.len() == 0 {
        return vec![];
    }

    make_heap(&mut data);

    (0..min(neighbors, data.len()))
        .map(|_| {
            pop_heap(&mut data);
            data.pop().unwrap()
        })
        .map(|KeyValuePair(key, _val)| key)
        .collect::<Vec<&'a A>>()
}

pub fn euclidean_dist<A: PartialOrd + PartialEq + Field<A>, B: VectorSpace<A>>(
    vector_1: &B,
    vector_2: &B,
) -> A {
    let delta = B::sub(vector_1, vector_2);

    B::dot(&delta, &delta)
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{euclidean_dist, knn},
        vector::Vector,
    };

    #[test]
    fn test_knn_single_neighbor() {
        let data = vec![
            vec![Vector([1.0, 2.0]), Vector([2.0, 3.0])],
            vec![Vector([3.0, 4.0]), Vector([5.0, 6.0])],
        ];

        let result = knn(
            vec![data[0].as_slice(), data[1].as_slice()].as_slice(),
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            1,
        );

        // Closest to origin is Vector([1.0, 2.0])
        assert_eq!(*result[0], Vector([1.0, 2.0]));
    }

    #[test]
    fn test_knn_multiple_neighbors() {
        let data = vec![
            Vector([1.0, 2.0]),
            Vector([2.0, 3.0]),
            Vector([3.0, 4.0]),
            Vector([5.0, 6.0]),
        ];

        let result = knn(
            &[data.as_slice()],
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            3,
        );

        // Closest vectors to origin
        let expected = vec![Vector([1.0, 2.0]), Vector([2.0, 3.0]), Vector([3.0, 4.0])];

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_eq!(**res, *exp);
        }
    }

    #[test]
    fn test_knn_with_ties() {
        let data = vec![Vector([1.0, 1.0]), Vector([1.0, 1.0])];

        let result = knn(
            &[data.as_slice()],
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            2,
        );

        // Both vectors are identical, so any order is acceptable
        assert_eq!(*result[0], Vector([1.0, 1.0]));
        assert_eq!(*result[1], Vector([1.0, 1.0]));
    }

    #[test]
    fn test_knn_no_data() {
        let data = vec![];

        let result = knn(
            &[data.as_slice()],
            |x| euclidean_dist(&Vector([1.0, 2.0]), x),
            3,
        );

        // No neighbors should be returned
        assert!(result.is_empty());
    }
}
