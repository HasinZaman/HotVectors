use crate::{
    db::{
        component::{
            ids::PartitionId,
            partition::{ArchivedVectorEntrySerial, VectorEntrySerial},
        },
        AtomicCmd, Cmd, Response, Success,
    },
    interface::HotRequest,
    vector::{Field, VectorSerial, VectorSpace},
};
use std::{fmt::Debug, mem::MaybeUninit, str::FromStr, sync::Arc};

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    from_bytes,
    rend::u32_le,
    ser::{allocator::ArenaHandle, sharing::Share, Serializer},
    to_bytes,
    tuple::ArchivedTuple3,
    util::AlignedVec,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, DeserializeUnsized, Serialize,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    sync::mpsc::{channel, Sender},
};
use uuid::Uuid;

type PartitionIdUUid = String;
type ClusterIdUUid = String;

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum ReadCmd<A: Archive> {
    Meta { filter: Option<PartitionIdUUid> },
    PartitionVectors { partition_id: PartitionIdUUid },
    ClusterVectors { threshold: A },
}

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum RequestCmd<A: Clone + Copy + Archive> {
    StartTransaction,
    EndTransaction,
    Read(ReadCmd<A>),
    InsertVector(VectorSerial<A>),
    CreateCluster(A),
}

// #[derive(Archive, Debug, Serialize, Deserialize, Clone)]
// enum IdSerial{
//     ClusterId(String),
//     PartitionId(String),
//     VectorId(String)
// }

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum Data<A: Archive + Clone + Copy> {
    ClusterId(String),
    PartitionId(String),

    Vector(Option<String>, Option<VectorSerial<A>>),
    Meta {
        id: String,
        size: usize,
        centroid: VectorSerial<A>,
    },
}

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum TmpResponse<A: Archive + Clone + Copy> {
    // Streaming?
    Start,
    End,

    //Organizational
    Pair(Data<A>, Data<A>),
    // Group(Vec<Data<A>>),
    Data(Data<A>),
}

pub async fn input_loop<
    A: Archive
        + Field<A>
        + Clone
        + Copy
        + Sized
        + Send
        + Sync
        + Debug
        + From<f32>
        + Archive
        + 'static
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
    B: VectorSpace<A> + Sized + Send + Sync + From<VectorSerial<A>> + 'static,
    BS: TryFrom<Vec<f32>>,
>(
    sender: Sender<(Cmd<A, B>, Sender<Response<A>>)>,
    max_conn: Option<usize>,
) -> !
where
    <BS as TryFrom<Vec<f32>>>::Error: Debug,
    f32: From<A>,
    VectorSerial<A>: From<B>,
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    [ArchivedVectorEntrySerial<A>]:
        DeserializeUnsized<[VectorEntrySerial<A>], Strategy<Pool, rancor::Error>>,
    [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    [ArchivedTuple3<u32_le, u32_le, <A as Archive>::Archived>]:
        DeserializeUnsized<[(usize, usize, A)], Strategy<Pool, rancor::Error>>,
    f32: From<A>,
    <A as Archive>::Archived: Deserialize<A, Strategy<Pool, rancor::Error>>,
{
    let shared_state = Arc::new(HotRequest {
        sender: Arc::new(sender.clone()),
    });

    // Start the Axum server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    let mut workers = Vec::new();

    loop {
        let (mut tcp_stream, socket_addr) = match listener.accept().await {
            Ok((tcp_stream, socket_addr)) => (tcp_stream, socket_addr),
            Err(_) => todo!(),
        };

        let state: Arc<HotRequest<A, B>> = shared_state.clone();
        let worker = tokio::spawn(async move {
            loop {
                // Read 4-byte length prefix
                let mut len_buf = [0u8; 4];
                if tcp_stream.read_exact(&mut len_buf).await.is_err() {
                    eprintln!("Client {} disconnected or sent invalid length", socket_addr);
                    break;
                }
                let msg_len = u32::from_le_bytes(len_buf) as usize;

                // Read the full message
                let mut data_buf = vec![0u8; msg_len];
                if tcp_stream.read_exact(&mut data_buf).await.is_err() {
                    eprintln!("Client {} disconnected during payload read", socket_addr);
                    break;
                }

                // Deserialize using rkyv
                match from_bytes::<RequestCmd<A>, rancor::Error>(&data_buf) {
                    Ok(cmd) => {
                        println!("[{}] Received: {:?}", socket_addr, cmd);

                        match cmd {
                            RequestCmd::StartTransaction => todo!(),
                            RequestCmd::EndTransaction => todo!(),
                            RequestCmd::Read(cmd) => {
                                read_cmd(state.clone(), cmd, &mut tcp_stream).await
                            }
                            RequestCmd::InsertVector(vector_serial) => {
                                insert_vector(state.clone(), vector_serial, &mut tcp_stream).await
                            }
                            RequestCmd::CreateCluster(threshold) => create_cluster(state.clone(), threshold, &mut tcp_stream).await,
                        }
                    }
                    Err(e) => {
                        eprintln!("Deserialization failed from {}: {:?}", socket_addr, e);
                    }
                }
            }

            println!("Connection closed: {}", socket_addr);
        });

        workers.push(worker);
    }

    // panic!("Server stopped unexpectedly");
}

async fn read_cmd<
    A: Clone
        + Copy
        + Debug
        + Field<A>
        + Into<f32>
        + Archive
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
    B: VectorSpace<A> + Sized + Send + Sync + From<VectorSerial<A>>,
>(
    state: Arc<HotRequest<A, B>>,
    read_cmd: ReadCmd<A>,
    tcp_stream: &mut TcpStream,
) where
    f32: From<A>,
{
    match read_cmd {
        ReadCmd::Meta { filter } => {
            let (tx, mut rx) = channel(64);

            let _ = state
                .sender
                .send((
                    Cmd::Atomic(AtomicCmd::GetMetaData {
                        transaction_id: None,
                    }),
                    tx,
                ))
                .await;

            // let mut data = Vec::new();
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&TmpResponse::<A>::Start).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
            while let Some(Response::Success(meta_data)) = rx.recv().await {
                let Success::MetaData(id, size, vector_serial) = meta_data else {
                    panic!("")
                };

                let meta_data = Data::Meta {
                    id: (*id).to_string(),
                    size: size,
                    centroid: vector_serial,
                };

                {
                    let bytes: AlignedVec =
                        to_bytes::<rancor::Error>(&TmpResponse::Data(meta_data)).unwrap();

                    let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
                }
            }

            {
                let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::End).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
        }
        ReadCmd::PartitionVectors { partition_id } => {
            let (tx, mut rx) = channel(64);

            let _ = state
                .sender
                .send((
                    Cmd::Atomic(AtomicCmd::Partitions {
                        ids: vec![PartitionId(Uuid::from_str(&partition_id).unwrap())],
                    }),
                    tx,
                ))
                .await;

            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&TmpResponse::<A>::Start).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
            while let Some(Response::Success(data)) = rx.recv().await {
                match data {
                    Success::Partition(partition_id) => {
                        let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::Data(
                            Data::PartitionId(partition_id.0.to_string()),
                        ))
                        .unwrap();

                        let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
                    }
                    Success::Vector(vector_id, vector_serial) => {
                        let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::Data(
                            Data::Vector(Some(vector_id.0.to_string()), Some(vector_serial)),
                        ))
                        .unwrap();

                        let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
                    }
                    _ => panic!(""),
                };
            }

            {
                let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::End).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
        }
        ReadCmd::ClusterVectors { threshold } => {
            let (tx, mut rx) = channel(64);
            let _ = state
                .sender
                .send((
                    Cmd::Atomic(AtomicCmd::GetClusters {
                        threshold: threshold,
                    }),
                    tx,
                ))
                .await;

            {
                let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::End).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }

            while let Some(data) = rx.recv().await {
                match data {
                    Response::Success(Success::Cluster(cluster_id)) => {
                        let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::Data(
                            Data::ClusterId(cluster_id.0.to_string()),
                        ))
                        .unwrap();

                        let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
                    }
                    Response::Success(Success::Vector(vector_id, _)) => {
                        let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::Data(
                            Data::Vector(Some(vector_id.0.to_string()), None),
                        ))
                        .unwrap();

                        let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
                    }
                    Response::Success(_) => {
                        panic!()
                    }
                    Response::Fail => {
                        todo!()
                    }
                    Response::Done => {
                        break;
                    }
                }
            }
        }
    }

    // let _ = state
    //     .sender
    //     .send((
    //         Cmd::Atomic(AtomicCmd::InsertVector {
    //             vector: vector_serial.into(),
    //             transaction_id: None,
    //         }),
    //         tx,
    //     ))
    //     .await;
    // let inserted_id = match rx.recv().await {
    //     Some(Response::Success(Success::Vector(id, _))) => id,
    //     None => {
    //         todo!()
    //     }
    //     _ => {
    //         todo!()
    //     }
    // };
    // match rx.recv().await {
    //     Some(Response::Done) => {
    //         let bytes: AlignedVec = to_bytes::<rancor::Error>(
    //             &(*inserted_id).to_string()
    //         ).unwrap();

    //         tcp_stream.write_all(
    //             bytes.as_slice()
    //         ).await;
    //     },
    //     None => {
    //         todo!()
    //     }
    //     _ => {
    //         todo!()
    //     }
    // }
}

async fn create_cluster<
    A: Clone
        + Copy
        + Debug
        + Field<A>
        + Into<f32>
        + Archive
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
    B: VectorSpace<A> + Sized + Send + Sync + From<VectorSerial<A>>,
>(
    state: Arc<HotRequest<A, B>>,
    threshold: A,
    tcp_stream: &mut TcpStream,
) where
    f32: From<A>,
{
    let (tx, mut rx) = channel(64);
    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::CreateCluster {
                threshold: threshold,
            }),
            tx,
        ))
        .await;

    let Some(Response::Done) = rx.recv().await else {
        todo!();
    };

    {
        let bytes: AlignedVec =
            to_bytes::<rancor::Error>(&TmpResponse::<A>::Start).unwrap();

        let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
    }
    
    {
        let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::End).unwrap();

        let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
    }
}

async fn insert_vector<
    A: Clone
        + Copy
        + Field<A>
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
    B: VectorSpace<A> + Sized + Send + Sync + From<VectorSerial<A>>,
>(
    state: Arc<HotRequest<A, B>>,
    vector_serial: VectorSerial<A>,
    tcp_stream: &mut TcpStream,
) {
    let (tx, mut rx) = channel(2);
    let _ = state
        .sender
        .send((
            Cmd::Atomic(AtomicCmd::InsertVector {
                vector: vector_serial.into(),
                transaction_id: None,
            }),
            tx,
        ))
        .await;
    let inserted_id = match rx.recv().await {
        Some(Response::Success(Success::Vector(id, _))) => {
            TmpResponse::<A>::Data(Data::Vector(Some(id.to_string()), None))
        }
        None => {
            todo!()
        }
        _ => {
            todo!()
        }
    };
    match rx.recv().await {
        Some(Response::Done) => {
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&TmpResponse::<A>::Start).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
            {
                let bytes: AlignedVec = to_bytes::<rancor::Error>(&inserted_id).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
            {
                let bytes: AlignedVec = to_bytes::<rancor::Error>(&TmpResponse::<A>::End).unwrap();

                let _ = tcp_stream.write_all(bytes.as_slice()).await.unwrap();
            }
        }
        None => {
            todo!()
        }
        _ => {
            todo!()
        }
    }
}
