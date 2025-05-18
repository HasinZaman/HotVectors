use crate::{
    db::{
        component::{
            ids::{PartitionId, VectorId},
            partition::{ArchivedVectorEntrySerial, VectorEntrySerial},
        },
        AtomicCmd, Cmd, Response, Success,
    },
    interface::HotRequest,
    vector::{Field, VectorSerial, VectorSpace},
};
use std::{collections::HashSet, fmt::Debug, io::ErrorKind, str::FromStr, sync::Arc};

use futures::FutureExt;
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
    io::{AsyncReadExt, AsyncWriteExt},
    net::{tcp::OwnedWriteHalf, TcpStream},
    sync::mpsc::{channel, Sender},
    task::JoinHandle,
};
use uuid::Uuid;

type PartitionIdUUid = String;

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum EdgeType{
    Intra(PartitionIdUUid),
    Inter
}

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum ReadCmd<A: Archive> {
    Meta { filter: Option<PartitionIdUUid> },
    PartitionVectors { partition_id: PartitionIdUUid },
    ClusterVectors { threshold: A },
    GraphEdges(EdgeType)
}

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum RequestCmd<A: Clone + Copy + Archive> {
    StartTransaction,
    EndTransaction,
    Read(ReadCmd<A>),
    InsertVector(VectorSerial<A>),
    CreateCluster(A),
}

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

    InterEdge(A, (String, String), (String, String)),
    IntraEdge(A, String, String),
}

#[derive(Archive, Debug, Serialize, Deserialize, Clone)]
enum ProtocolMessage<A: Archive + Clone + Copy> {
    // Streaming?
    Start,
    End,

    //Organizational
    Pair(Data<A>, Data<A>),
    // Group(Vec<Data<A>>),
    Data(Data<A>),

    Open,
    TooManyConnections,
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
    // BS: TryFrom<Vec<f32>>,
>(
    sender: Sender<(Cmd<A, B>, Sender<Response<A>>)>,
    max_conn: Option<usize>,
) -> !
where
    // <BS as TryFrom<Vec<f32>>>::Error: Debug,
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
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    let _thread = tokio::spawn(async move {
        println!("Starting conn pool");
        let mut workers: Vec<JoinHandle<()>> = Vec::new();

        loop {
            let (mut tcp_stream, socket_addr) = match listener.accept().await {
                Ok((tcp_stream, socket_addr)) => (tcp_stream, socket_addr),
                Err(_) => todo!(),
            };

            println!("Got a connection from {:?}", socket_addr);

            workers.retain_mut(|handle: &mut JoinHandle<()>| {
                match handle.now_or_never() {
                    Some(Ok(_)) => false,
                    Some(Err(_err)) => false,
                    None => true, // Still running, keep it
                }
            });

            let valid = match max_conn {
                Some(max) => workers.len() < max,
                None => true,
            };

            if !valid {
                let _ = tokio::spawn(async move {
                    let bytes: AlignedVec =
                        to_bytes::<rancor::Error>(&ProtocolMessage::<A>::TooManyConnections)
                            .unwrap();

                    write_all(&mut tcp_stream, bytes).await;
                })
                .await;
                continue;
            }

            let state: Arc<HotRequest<A, B>> = shared_state.clone();
            let worker = tokio::spawn(async move {
                worker_loop(tcp_stream, socket_addr, state).await;
            });

            workers.push(worker);
        }
    })
    .await
    .unwrap();
    panic!("Server stopped unexpectedly");
}

async fn worker_loop<
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
>(
    mut tcp_stream: TcpStream,
    socket_addr: std::net::SocketAddr,
    state: Arc<HotRequest<A, B>>,
) where
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
    {
        let bytes: AlignedVec = to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Open).unwrap();
        write_all(&mut tcp_stream, bytes).await;
    }

    let (mut reader, mut writer) = tcp_stream.into_split();

    let (tx, mut rx) = channel(16);

    let _input_worker = tokio::spawn(async move {
        loop {
            let mut len_buf = [0u8; 4];
            if let Err(e) = reader.read_exact(&mut len_buf).await {
                eprintln!(
                    "Client {} disconnected or sent invalid length: {}",
                    socket_addr, e
                );
                return;
            }
            let msg_len = u32::from_le_bytes(len_buf) as usize;

            // Read the full message
            let mut data_buf = vec![0u8; msg_len];
            match reader.read_exact(&mut data_buf).await {
                Ok(_) => {
                    // Process the data
                }
                Err(ref e) if e.kind() == ErrorKind::ConnectionReset => {
                    eprintln!("Connection was reset by the remote host");
                    // Handle the connection reset
                }
                Err(e) => {
                    eprintln!("An error occurred: {}", e);
                    // Handle other errors
                }
            }

            match from_bytes::<RequestCmd<A>, rancor::Error>(&data_buf) {
                Ok(cmd) => {
                    println!("Decoded cmd {:?}", &cmd);
                    if let Err(e) = tx.send(cmd).await {
                        eprintln!("Failed to send command for client {}: {}", socket_addr, e);
                        return;
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Failed to deserialize command from client {}: {}",
                        socket_addr, e
                    );
                    return;
                }
            }
        }
    });

    while let Some(cmd) = rx.recv().await {
        match cmd {
            RequestCmd::StartTransaction => todo!(),
            RequestCmd::EndTransaction => todo!(),
            RequestCmd::Read(cmd) => read_cmd(state.clone(), cmd, &mut writer).await,
            RequestCmd::InsertVector(vector_serial) => {
                insert_vector(state.clone(), vector_serial, &mut writer).await
            }
            RequestCmd::CreateCluster(threshold) => {
                create_cluster(state.clone(), threshold, &mut writer).await
            }
        }
    }

    println!("Connection closed: {}", socket_addr);
}

async fn write_all<'w, W: AsyncWriteExt + Unpin>(tcp_stream: &'w mut W, bytes: AlignedVec) {
    let len = (bytes.len() as u32).to_le_bytes();

    tcp_stream.write_all(&len).await.unwrap();
    // Send length
    tcp_stream.write_all(&bytes).await.unwrap();
    // Send actual data
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
    tcp_stream: &mut OwnedWriteHalf,
) where
    f32: From<A>,
{
    match read_cmd {
        ReadCmd::Meta { .. } => {
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
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

                write_all(tcp_stream, bytes).await;
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
                        to_bytes::<rancor::Error>(&ProtocolMessage::Data(meta_data)).unwrap();

                    write_all(tcp_stream, bytes).await;
                }
            }

            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

                write_all(tcp_stream, bytes).await;
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
                        to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

                    write_all(tcp_stream, bytes).await;
                }
                while let Some(Response::Success(data)) = rx.recv().await {
                    match data {
                        Success::Partition(partition_id) => {
                            let bytes: AlignedVec =
                                to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Data(
                                    Data::PartitionId(partition_id.0.to_string()),
                                ))
                                .unwrap();

                            write_all(tcp_stream, bytes).await;
                        }
                        Success::Vector(vector_id, vector_serial) => {
                            let bytes: AlignedVec =
                                to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Data(Data::Vector(
                                    Some(vector_id.0.to_string()),
                                    Some(vector_serial),
                                )))
                                .unwrap();

                            write_all(tcp_stream, bytes).await;
                        }
                        _ => panic!(""),
                    };
                }

                {
                    let bytes: AlignedVec =
                        to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

                    write_all(tcp_stream, bytes).await;
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
                    let bytes: AlignedVec =
                        to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

                    write_all(tcp_stream, bytes).await;
                }

                while let Some(data) = rx.recv().await {
                    match data {
                        Response::Success(Success::Cluster(cluster_id)) => {
                            let bytes: AlignedVec = to_bytes::<rancor::Error>(
                                &ProtocolMessage::<A>::Data(Data::ClusterId(cluster_id.0.to_string())),
                            )
                            .unwrap();

                            write_all(tcp_stream, bytes).await;
                        }
                        Response::Success(Success::Vector(vector_id, _)) => {
                            let bytes: AlignedVec =
                                to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Data(Data::Vector(
                                    Some(vector_id.0.to_string()),
                                    None,
                                )))
                                .unwrap();

                            write_all(tcp_stream, bytes).await;
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

                {
                    let bytes: AlignedVec =
                        to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

                    write_all(tcp_stream, bytes).await;
                }
            }
        ReadCmd::GraphEdges(EdgeType::Inter) => {
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

                write_all(tcp_stream, bytes).await;
            }
            let partition_ids = {
                let mut partition_ids = Vec::new();
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

                while let Some(Response::Success(meta_data)) = rx.recv().await {
                    let Success::MetaData(id, size, vector_serial) = meta_data else {
                        panic!("")
                    };

                    partition_ids.push(id);
                }

                partition_ids
            };

            let mut visited_edges: HashSet<((PartitionId, VectorId), (PartitionId, VectorId))> = HashSet::new();
            for id in partition_ids {
                let (tx, mut rx) = channel(64);

                let _ = state
                    .sender
                    .send((
                        Cmd::Atomic(AtomicCmd::GetInterPartitionGraph {
                            transaction_id: None,
                            partition_id: id,
                        }),
                        tx,
                    ))
                    .await;

                while let Some(Response::Success(data)) = rx.recv().await {
                    match data {
                        Success::InterEdge(
                            (PartitionId(source_partition), VectorId(source_vector)),
                            (PartitionId(target_partition), VectorId(target_vector)),
                            dist
                        ) => {
                            let id_1 = (
                                (PartitionId(source_partition), VectorId(source_vector)),
                                (PartitionId(target_partition), VectorId(target_vector)),
                            );
                            let id_2 = (
                                (PartitionId(target_partition), VectorId(target_vector)),
                                (PartitionId(source_partition), VectorId(source_vector)),
                            );
                            
                            if visited_edges.contains(&id_1) || visited_edges.contains(&id_2) {
                                continue;
                            }
                            visited_edges.insert(id_1);
                            visited_edges.insert(id_2);

                            let bytes: AlignedVec =
                                to_bytes::<rancor::Error>(
                                    &ProtocolMessage::<A>::Data(
                                        Data::InterEdge(
                                            dist,
                                            (
                                                source_partition.to_string(),
                                                source_vector.to_string()
                                            ),
                                            (
                                                target_partition.to_string(),
                                                target_vector.to_string()
                                            )
                                        )
                                    )
                                ).unwrap();

                            write_all(tcp_stream, bytes).await;
                        }
                        _ => panic!(""),
                    };
                }
            }
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

                write_all(tcp_stream, bytes).await;
            }
        },
        ReadCmd::GraphEdges(EdgeType::Intra(partition_id)) => {
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

                write_all(tcp_stream, bytes).await;
            }

            let (tx, mut rx) = channel(64);

            let _ = state
                .sender
                .send((
                    Cmd::Atomic(AtomicCmd::GetPartitionGraph {
                        transaction_id: None,
                        partition_id: PartitionId(Uuid::from_str(&partition_id).unwrap()),
                    }),
                    tx,
                ))
                .await;
            while let Some(Response::Success(data)) = rx.recv().await {
                match data {
                    Success::Edge(VectorId(source), VectorId(target), dist) => {
                        let bytes: AlignedVec =
                            to_bytes::<rancor::Error>(
                                &ProtocolMessage::<A>::Data(
                                    Data::IntraEdge(
                                        dist,
                                        source.to_string(),
                                        target.to_string(),
                                    )
                                )
                            ).unwrap();

                        write_all(tcp_stream, bytes).await;
                    }
                    _ => panic!(""),
                };
            }
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

                write_all(tcp_stream, bytes).await;
            }
        },
    }
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
    tcp_stream: &mut OwnedWriteHalf,
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
        let bytes: AlignedVec = to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

        write_all(tcp_stream, bytes).await;
    }

    {
        let bytes: AlignedVec = to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

        write_all(tcp_stream, bytes).await;
    }
}

async fn insert_vector<
    A: Clone
        + Copy
        + Debug
        + Field<A>
        + for<'a> Serialize<Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>>,
    B: VectorSpace<A> + Sized + Send + Sync + From<VectorSerial<A>>,
>(
    state: Arc<HotRequest<A, B>>,
    vector_serial: VectorSerial<A>,
    tcp_stream: &mut OwnedWriteHalf,
) {
    let (tx, mut rx) = channel(2);
    println!("sending {:?}", &vector_serial);
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
            ProtocolMessage::<A>::Data(Data::Vector(Some(id.to_string()), None))
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
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::Start).unwrap();

                write_all(tcp_stream, bytes).await;
            }
            {
                let bytes: AlignedVec = to_bytes::<rancor::Error>(&inserted_id).unwrap();

                write_all(tcp_stream, bytes).await;
            }
            {
                let bytes: AlignedVec =
                    to_bytes::<rancor::Error>(&ProtocolMessage::<A>::End).unwrap();

                write_all(tcp_stream, bytes).await;
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
