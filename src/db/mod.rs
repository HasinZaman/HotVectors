use std::{marker::PhantomData, sync::mpsc::{Receiver, Sender}};

use partition::{External, Internal, LoadedPartitions, Partition, PartitionGraph, VectorEntry};
use tokio::runtime;
use uuid::Uuid;

use crate::vector::{Extremes, Field, VectorSpace};

mod partition;
mod serialization;

pub enum AtomicCmd<A: Field<A>, B: VectorSpace<A> + Sized> {
    // transaction commands
    StartTransaction(Uuid),
    EndTransaction(Uuid),
    UndoTransaction(Uuid),

    // Write
    InsertVector{
        vector: B,
        transaction_id: Option<Uuid>
    },

    // projections
    GetIds{
        transaction_id: Option<Uuid>
    },
    GetVectors {
        transaction_id: Option<Uuid>
    },

    // Filters
    Knn {
        transaction_id: Option<Uuid>
    },
    SelectedUUID {
        transaction_id: Option<Uuid>
    },

    PhantomData(PhantomData<A>)
}

pub struct ChainedCmd<A: Field<A>, B: VectorSpace<A> + Sized>(Vec<AtomicCmd<A, B>>);

pub enum Cmd<A: Field<A>, B: VectorSpace<A> + Sized> {
    Atomic(AtomicCmd<A, B>),
    Chained(ChainedCmd<A, B>)
}

pub fn db_loop<
    A: PartialEq + Clone + Copy + Field<A>,
    B: VectorSpace<A> + Sized + Clone + Copy,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
    const MAX_LOADED: usize,
>(
    loaded_partitions: LoadedPartitions<A, B, PARTITION_CAP, VECTOR_CAP, MAX_LOADED>,
    external_graph: PartitionGraph<A, External>,
    cmd_input: Receiver<Sender<Cmd<A, B>>>,
) -> ! {
    // initialize internal graphs
    // initialize locks
    let main_rt = runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();

    // let read_rt = runtime::Builder::new_multi_thread()
    //     .worker_threads(1)
    //     .build()
    //     .unwrap();

    // let write_rt = runtime::Builder::new_multi_thread()
    //     .worker_threads(1)
    //     .build()
    //     .unwrap();
    // if
    main_rt.block_on(async {
        let tmp = cmd_input.try_recv();
        // check if any new cmds

        // create new action
        //  -> start transaction
        //  -> read
        //      -> knn
        //      -> clusters
        //      -> read vectors?
        //      -> filter
        //  -> write
        //      -> add vector
        //      -> batch add vector
        //      -> remove vector
        //      -> batch remove vector
    });
    panic!()
}

fn split_partition<
    'a,
    A: Clone + Copy + PartialEq + Field<A>,
    B: Clone + Copy + VectorSpace<A> + Extremes,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    partition: &'a mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    internal_graphs: &[&'a mut PartitionGraph<A, Internal>],
    external_graphs: &[&'a mut PartitionGraph<A, External>],
) -> (
    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    Vec<PartitionGraph<A, Internal>>,
) {
    // select random
    let mut new_partition: Partition<A, B, PARTITION_CAP, VECTOR_CAP> = Partition::new();

    // select to random points

    // divide

    todo!()
}

fn merge_partition<
    'a,
    A: Clone + Copy + PartialEq + Field<A>,
    B: Clone + Copy + VectorSpace<A>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    sink_partitions: &'a mut Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    sink_internal_graphs: &[&'a mut PartitionGraph<A, Internal>],

    source_partitions: &[&'a Partition<A, B, PARTITION_CAP, VECTOR_CAP>],
    source_internal_graphs: &[&[&'a mut PartitionGraph<A, Internal>]],

    external_graphs: &[&'a mut PartitionGraph<A, External>],
) -> (
    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    Vec<PartitionGraph<A, Internal>>,
) {
    // select random
    todo!()
}

async fn update_partition<
    A: Clone + Copy + PartialEq + Field<A>,
    B: Clone + Copy + VectorSpace<A>,
    const PARTITION_CAP: usize,
    const VECTOR_CAP: usize,
>(
    start_partitions: Uuid,
    external_graphs: Vec<PartitionGraph<A, External>>,

    partition_receiver: Receiver<(
        Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
        Vec<PartitionGraph<A, Internal>>,
    )>,
    partition_request: Sender<Uuid>,
) -> (
    Partition<A, B, PARTITION_CAP, VECTOR_CAP>,
    Vec<PartitionGraph<A, Internal>>,
) {
    todo!()
}
