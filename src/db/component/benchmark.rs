

use std::ops::Deref;
use std::sync::mpsc::{Receiver, Sender, RecvTimeoutError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::sync::Arc;

use async_sqlite::rusqlite::{self, Connection};
use tokio::time::sleep;
use uuid::Uuid;

#[derive(Clone, Copy)]
pub enum BenchmarkId{
    ID(Uuid),
    Child(Uuid , Uuid)
}

impl Deref for BenchmarkId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        match self {
            BenchmarkId::ID(uuid) => uuid,
            BenchmarkId::Child(_, uuid) => uuid,
        }
    }
}

#[derive(Clone)]
pub struct Benchmark{
    id: BenchmarkId,
    start: u64,
    pub message: String,

    sender: Arc<Sender<(BenchmarkId, u64, u64, String)>>
}

impl Benchmark {
    pub fn new(message: String, writer: Arc<Sender<(BenchmarkId, u64, u64, String)>>) -> Self {
        Self {
            id: BenchmarkId::ID(Uuid::new_v4()),
            start: get_timestamp(),
            message: message,
            sender: writer,
        }
    }

    pub fn spawn_child(message: String, parent: &Self,) -> Self {
        Self {
            id: BenchmarkId::Child(*parent.id, Uuid::new_v4()),
            start: get_timestamp(),
            message,
            sender: parent.sender.clone(),
        }
    }
}

impl Drop for Benchmark{
    fn drop(&mut self) {
        let _ = self.sender.send((
            self.id,
            self.start,
            get_timestamp(),
            self.message.clone()
        ));
    }
}

fn get_timestamp() -> u64 {
    let now = SystemTime::now();
    let since_epoch = now.duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    since_epoch.as_secs() as u64
}

pub async fn benchmark_logger(receiver: Receiver<(BenchmarkId, u64, u64, String)>) -> ! {
    let conn = match Connection::open("benchmark_logs.db") {
        Ok(c) => c,
        Err(e) => {
            panic!("Failed to open SQLite DB: {}", e);
        }
    };

    if let Err(e) = conn.execute(
        "CREATE TABLE IF NOT EXISTS benchmark (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            start INTEGER,
            end INTEGER,
            message TEXT
        )",
        [],
    ) {
        panic!("Failed to create table: {}", e);
    }

    let timeout = Duration::from_secs(1); // Timeout duration between polls

    loop {
        let first_msg = match receiver.recv_timeout(timeout) {
            Ok(msg) => msg,
            Err(RecvTimeoutError::Timeout) => {
                sleep(timeout).await;
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                println!("Receiver disconnected. Exiting benchmark logger.");
                panic!();
            }
        };

        if let Err(e) = conn.execute("BEGIN", []) {
            eprintln!("Failed to begin transaction: {}", e);
            continue;
        } else {
            println!("Begin Transaction");
        }

        // Process the first message
        process_benchmark_message(&conn, first_msg);

        let mut i1 = 1;
        while i1 < 100 {
            match receiver.recv_timeout(timeout) {
                Ok(msg) => {
                    process_benchmark_message(&conn, msg);
                    i1 += 1;
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    println!("Receiver disconnected. Exiting benchmark logger.");
                    panic!();
                }
            }
        }

        if let Err(e) = conn.execute("COMMIT", []) {
            eprintln!("Failed to commit transaction: {}", e);
        }

        sleep(timeout).await;
    }
}

fn process_benchmark_message(
    conn: &rusqlite::Connection,
    msg: (BenchmarkId, u64, u64, String),
) {
    match msg {
        (BenchmarkId::ID(id), start, end, message) => {
            if let Err(e) = conn.execute(
                "INSERT INTO benchmark (id, parent_id, start, end, message) VALUES (?, NULL, ?, ?, ?)",
                (id.to_string(), start, end, message),
            ) {
                eprintln!("Failed to insert benchmark: {}", e);
            } else {
                println!("Benchmark recorded: {}", id);
            }
        }
        (BenchmarkId::Child(parent_id, child_id), start, end, message) => {
            if let Err(e) = conn.execute(
                "INSERT INTO benchmark (id, parent_id, start, end, message) VALUES (?, ?, ?, ?, ?)",
                (child_id.to_string(), parent_id.to_string(), start, end, message),
            ) {
                eprintln!("Failed to insert benchmark: {}", e);
            } else {
                println!("Benchmark recorded: {}", child_id);
            }
        }
    }
}
