use std::collections::{HashMap, HashSet};
use tokio::sync::{mpsc::Receiver, oneshot};
use uuid::Uuid;

use super::component::ids::PartitionId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode {
    Read,
    Write,
}

#[derive(Debug)]
pub enum AccessResponse {
    Granted,
    Denied,
}

pub type TransactionId = Uuid;

#[derive(Debug)]
pub struct AccessRequest {
    pub transaction_id: TransactionId,
    pub partitions: Vec<(PartitionId, AccessMode)>,
    pub respond_to: oneshot::Sender<AccessResponse>,
}
#[derive(Debug)]
pub enum BankerMessage {
    RequestAccess {
        transaction_id: TransactionId,
        partitions: Vec<(PartitionId, AccessMode)>,
        respond_to: oneshot::Sender<AccessResponse>,
    },
    ReleaseAccess {
        transaction_id: TransactionId,
        partitions: Vec<PartitionId>,
    },
}

/// Internal state for a single partition
#[derive(Default, Debug)]
struct PartitionState {
    readers: HashSet<TransactionId>,
    writer: Option<TransactionId>,
}

impl PartitionState {
    fn can_grant(&self, tx: TransactionId, mode: AccessMode) -> bool {
        match mode {
            AccessMode::Read => self.writer.is_none() || self.writer == Some(tx),
            AccessMode::Write => {
                (self.writer.is_none() || self.writer == Some(tx)) && self.readers.is_empty()
            }
        }
    }

    fn grant(&mut self, tx: TransactionId, mode: AccessMode) {
        match mode {
            AccessMode::Read => {
                self.readers.insert(tx);
            }
            AccessMode::Write => {
                self.writer = Some(tx);
            }
        }
    }

    fn release(&mut self, tx: TransactionId) {
        self.readers.remove(&tx);
        if self.writer == Some(tx) {
            self.writer = None;
        }
    }

    fn is_empty(&self) -> bool {
        self.readers.is_empty() && self.writer.is_none()
    }
}

/// Runs the banker event loop — handles access and release requests
pub async fn banker(mut rx: Receiver<BankerMessage>) {
    let mut partition_map: HashMap<PartitionId, PartitionState> = HashMap::new();

    while let Some(msg) = rx.recv().await {
        match msg {
            BankerMessage::RequestAccess {
                transaction_id,
                partitions,
                respond_to,
            } => {
                let can_grant = partitions.iter().all(|(pid, mode)| {
                    partition_map
                        .get(pid)
                        .map(|s| s.can_grant(transaction_id, *mode))
                        .unwrap_or(true)
                });

                if can_grant {
                    for (pid, mode) in &partitions {
                        partition_map
                            .entry(*pid)
                            .or_default()
                            .grant(transaction_id, *mode);
                    }
                    let _ = respond_to.send(AccessResponse::Granted);
                } else {
                    let _ = respond_to.send(AccessResponse::Denied);
                }
            }

            BankerMessage::ReleaseAccess {
                transaction_id,
                partitions,
            } => {
                for pid in partitions {
                    if let Some(state) = partition_map.get_mut(&pid) {
                        state.release(transaction_id);
                        if state.is_empty() {
                            partition_map.remove(&pid);
                        }
                    }
                }
            }
        }
    }
}
