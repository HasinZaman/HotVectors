use std::{cmp::Ordering, collections::HashSet, fmt::Debug, str::FromStr, time::Duration};

use async_sqlite::{rusqlite::params_from_iter, Client, ClientBuilder, JournalMode};
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    from_bytes, to_bytes,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, Serialize,
};
use tokio::{fs::read_dir, sync::oneshot};
use uuid::Uuid;

use crate::vector::Field;

use super::{
    ids::{ClusterId, VectorId},
    serial::FileExtension,
};

#[derive()]
pub struct ClusterSet<A: Field<A> + Debug + Clone> {
    pub threshold: A,
    pub id: Uuid,

    conn: Client,
}

impl<A: Field<A> + Debug + Clone> ClusterSet<A> {
    pub async fn load_from_folder(dir: &str) -> Vec<Self>
    where
        A: Archive,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
        <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<rkyv::de::Pool, rancor::Error>>,
    {
        let mut results = Vec::new();
        let mut entries = read_dir(dir).await.unwrap();

        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();
            if let Some(extension) = path.extension() {
                if extension == ClusterSetSerial::<A>::extension() {
                    // let file_name = path.file_stem().unwrap().to_string_lossy();
                    let bytes = tokio::fs::read(&path).await.unwrap();
                    let meta_serial: ClusterSetSerial<A> =
                        from_bytes::<ClusterSetSerial<A>, rancor::Error>(&bytes).unwrap();
                    let meta: ClusterSet<A> = meta_serial.into();
                    results.push(meta);
                }
            }
        }

        results
    }

    pub async fn from_smaller_cluster_set(threshold: A, other: &Self) -> Self
    where
        A: Into<f32>,
    {
        let id = Uuid::new_v4();

        let _ = other.copy(id.to_string()).await.unwrap();

        let conn = ClientBuilder::new()
            .path(&format!("data/clusters//{}.db", id.to_string()))
            .journal_mode(JournalMode::Memory)
            .open()
            .await
            .unwrap();

        let threshold_f32: f32 = threshold.clone().into();
        let _ = conn
            .conn(move |conn| {
                conn.execute("BEGIN TRANSACTION", [])?;

                let mut stmt =
                    conn.prepare("SELECT source, target FROM ClusterEdge WHERE weight < ?")?;

                let rows: Vec<(String, String)> = stmt
                    .query_map([threshold_f32], |row| {
                        let source: String = row.get(0)?;
                        let target: String = row.get(1)?;
                        Ok((source, target))
                    })?
                    .filter_map(Result::ok)
                    .collect();

                for (source, target) in rows {
                    let cluster_id_1: String = conn
                        .query_row(
                            "SELECT cluster_id FROM Clusters WHERE vector_id = ?",
                            [&source],
                            |row| row.get(0),
                        )
                        .expect("source missing");
                    let cluster_id_2: String = conn
                        .query_row(
                            "SELECT cluster_id FROM Clusters WHERE vector_id = ?",
                            [&target],
                            |row| row.get(0),
                        )
                        .expect("target missing");

                    if cluster_id_1 == cluster_id_2 {
                        continue;
                    }
                    let size_1: i64 = conn
                        .query_row(
                            "SELECT size FROM ClusterMeta WHERE cluster_id = ?",
                            [&cluster_id_1],
                            |row| row.get(0),
                        )
                        .expect("source missing");

                    let size_2: i64 = conn
                        .query_row(
                            "SELECT size FROM ClusterMeta WHERE cluster_id = ?",
                            [&cluster_id_2],
                            |row| row.get(0),
                        )
                        .expect("target missing");

                    let merge_size = size_1 + size_2;

                    let (cluster_id_1, cluster_id_2) = match size_1 > size_2 {
                        true => (&cluster_id_1, &cluster_id_2),
                        false => (&cluster_id_2, &cluster_id_1),
                    };

                    let _ = conn.execute(
                        "UPDATE Clusters SET cluster_id=? WHERE cluster_id=?",
                        (cluster_id_1, cluster_id_2),
                    );
                    let _ = conn.execute(
                        "UPDATE ClusterMeta SET size=?, merged_into=? WHERE cluster_id=?",
                        (merge_size, cluster_id_1, cluster_id_2),
                    );
                    let _ = conn.execute(
                        "UPDATE ClusterMeta SET size=? WHERE cluster_id=?",
                        (merge_size, cluster_id_1),
                    );

                    conn.execute(
                        "DELETE FROM ClusterEdge WHERE source=? AND target=?",
                        [source, target],
                    )?;
                }

                conn.execute("COMMIT", [])?;

                Ok(())
            })
            .await
            .unwrap();

        Self {
            threshold,
            id: Uuid::new_v4(),

            conn,
        }
    }

    pub async fn new(threshold: A, dir: String) -> Self {
        let id = Uuid::new_v4();
        let conn = ClientBuilder::new()
            .path(&format!("{}/{}.db", dir, id.to_string()))
            .journal_mode(JournalMode::Memory)
            .open()
            .await
            .unwrap();
        conn.conn(|conn| {
            conn.execute(
                "CREATE TABLE Clusters(vector_id STRING PRIMARY KEY, cluster_id STRING)", // replace cluster_id from String to INT & Sort by cluster_id
                [],
            )?;
            conn.execute(
                "CREATE TABLE ClusterEdge(source STRING, target STRING, weight REAL, FOREIGN KEY (source) REFERENCES Clusters(vector_id), FOREIGN KEY (target) REFERENCES Clusters(vector_id))", // replace cluster_id from String to INT & Sort by cluster_id
                [],
            )?;
            conn.execute(
                "CREATE TABLE ClusterMeta(cluster_id STRING PRIMARY KEY, size INT, merged_into STRING)", // should map String -> Int
                [],
            )
        })
        .await
        .unwrap();

        Self {
            threshold,
            id: Uuid::new_v4(),

            conn,
        }
    }

    pub async fn create_local(&self, vectors: &[VectorId], transaction_id: Uuid) -> Self {
        let local = Self::new(
            self.threshold.clone(),
            format!("data/local/{}/clusters", transaction_id.to_string()),
        )
        .await;

        // get all cluster ids from vectorId
        let vector_ids: Vec<_> = vectors.into();
        let (cluster_meta, clusters, cluster_edges) = self
            .conn
            .conn(move |conn| {
                conn.execute("BEGIN TRANSACTION", [])?;

                // get cluster
                let cluster_ids = {
                    let placeholders = std::iter::repeat("?")
                        .take(vector_ids.len())
                        .collect::<Vec<_>>()
                        .join(",");

                    let query = format!(
                        "SELECT DISTINCT cluster_id FROM Clusters WHERE vector_id IN ({})",
                        placeholders
                    );

                    let mut stmt = conn.prepare(&query)?;
                    let rows = stmt.query_map(
                        params_from_iter(vector_ids.iter().map(|x| (**x).to_string())),
                        |row| row.get::<_, String>(0),
                    )?;

                    let mut cluster_ids = HashSet::new();
                    for row in rows {
                        cluster_ids.insert(row?);
                    }

                    cluster_ids
                };

                let cluster_meta = {
                    let mut cluster_meta = Vec::new();
                    // get cluster meta
                    for id in cluster_ids.iter() {
                        let mut stmt =
                            conn.prepare("SELECT * FROM ClusterMeta WHERE cluster_id = ?")?;

                        let rows = stmt.query_map([id], |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, u64>(1)?,
                                row.get::<_, Option<String>>(2)?,
                            ))
                        })?;

                        for row in rows {
                            cluster_meta.push(row?);
                        }
                    }

                    cluster_meta
                };

                let clusters = {
                    let mut clusters = Vec::new();
                    // get cluster meta
                    for id in cluster_ids.iter() {
                        let mut stmt =
                            conn.prepare("SELECT * FROM Clusters WHERE cluster_id = ?")?;

                        let rows = stmt.query_map([id], |row| {
                            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                        })?;

                        for row in rows {
                            clusters.push(row?);
                        }
                    }

                    clusters
                };

                let cluster_edges = {
                    let mut cluster_edges = Vec::new();
                    // get cluster meta
                    for (id, _) in clusters.iter() {
                        let mut stmt = conn
                            .prepare("SELECT * FROM ClusterEdge WHERE source = ? OR target = ?")?;

                        let rows = stmt.query_map([id, id], |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, i64>(2)?,
                            ))
                        })?;

                        for row in rows {
                            cluster_edges.push(row?);
                        }
                    }

                    cluster_edges
                };

                conn.execute("COMMIT", [])?;

                Ok((cluster_meta, clusters, cluster_edges))
            })
            .await
            .unwrap();

        let _ = local
            .conn
            .conn(move |conn| {
                conn.execute("BEGIN TRANSACTION", [])?;

                {
                    let mut stmt = conn.prepare(
                        "INSERT INTO ClusterMeta (cluster_id, size, merged_into) VALUES (?, ?, ?)",
                    )?;
                    for (cluster_id, size, merged_into) in cluster_meta.iter() {
                        stmt.execute((cluster_id, &size.to_string(), merged_into))?;
                    }
                }

                {
                    let mut stmt =
                        conn.prepare("INSERT INTO Clusters (cluster_id, vector_id) VALUES (?, ?)")?;
                    for (cluster_id, vector_id) in clusters.iter() {
                        stmt.execute([cluster_id, vector_id])?;
                    }
                }

                {
                    let mut stmt = conn.prepare(
                        "INSERT INTO ClusterEdge (source, target, weight) VALUES (?, ?, ?)",
                    )?;
                    for (source, target, weight) in cluster_edges.iter() {
                        stmt.execute((source, target, weight))?;
                    }
                }

                conn.execute("COMMIT", [])?;
                Ok(())
            })
            .await
            .unwrap();

        local
    }

    pub async fn save(&self, dir: &str, name: &str)
    where
        A: Archive
            + for<'a> rkyv::Serialize<
                rancor::Strategy<
                    rkyv::ser::Serializer<
                        rkyv::util::AlignedVec,
                        rkyv::ser::allocator::ArenaHandle<'a>,
                        rkyv::ser::sharing::Share,
                    >,
                    rancor::Error,
                >,
            >,
    {
        let serial_value: ClusterSetSerial<A> = self.into();

        let bytes = to_bytes::<rancor::Error>(&serial_value).unwrap();

        tokio::fs::write(
            &format!("{dir}//{name}.{}", ClusterSetSerial::<A>::extension()),
            bytes.as_slice(),
        )
        .await
        .unwrap();
    }

    pub async fn new_cluster(&self) -> Result<ClusterId, ()> {
        let cluster_id = ClusterId(Uuid::new_v4());

        self.conn
            .conn({
                let cluster_id = cluster_id.clone();

                move |conn| {
                    conn.execute(
                        "INSERT iNTO ClusterMeta(cluster_id, size) VALUES (?, 0)",
                        [cluster_id.0.to_string()],
                    )
                }
            })
            .await
            .expect("Failed to insert into ClusterMeta");

        Ok(cluster_id)
    }

    pub async fn new_cluster_from_vector(
        &mut self,
        vector_id: VectorId,
        cluster_id: ClusterId,
    ) -> Result<(), ()> {
        self.conn
            .conn(move |conn| {
                let _ = conn.execute(
                    "UPDATE ClusterMeta SET size=size+1 WHERE cluster_id=?",
                    [cluster_id.0.to_string()],
                );
                conn.execute(
                    "INSERT iNTO Clusters(vector_id, cluster_id) VALUES (?, ?)",
                    [vector_id.0.to_string(), cluster_id.0.to_string()],
                )
            })
            .await
            .expect("Failed to insert into ClusterMeta");

        Ok(())
    }

    pub async fn add_edge(
        &mut self,
        vector_id_1: VectorId,
        vector_id_2: VectorId,
        weight: A,
    ) -> Result<(), ()>
    where
        A: Into<f32>,
    {
        let weight = weight.into();

        let _ = self
            .conn
            .conn(move |conn| {
                let affected = conn.execute(
                    "INSERT INTO ClusterEdge(source, target, weight)
                     SELECT ?, ?, ?
                     WHERE (
                         (SELECT cluster_id FROM Clusters WHERE vector_id = ?) !=
                         (SELECT cluster_id FROM Clusters WHERE vector_id = ?)
                     )",
                    (
                        vector_id_1.to_string(),
                        vector_id_2.to_string(),
                        weight,
                        vector_id_1.to_string(),
                        vector_id_2.to_string(),
                    ),
                )?;
                Ok(affected > 0)
            })
            .await
            .unwrap();
        Ok(())
    }

    pub async fn delete_edge(
        &mut self,
        vector_id_1: VectorId,
        vector_id_2: VectorId,
    ) -> Result<(), ()> {
        let _ = self.conn
            .conn(move |conn| {
                conn.execute(
                    "DELETE FROM ClusterEdge WHERE (source=? AND target=?) OR (source=? AND target=?) ",
                    (
                        vector_id_1.0.to_string(), vector_id_2.0.to_string(),
                        vector_id_2.0.to_string(), vector_id_1.0.to_string()
                    ),
                )
            })
            .await
            .unwrap();
        Ok(())
    }

    pub async fn get_edges(&self) -> Result<Vec<(VectorId, VectorId, A)>, ()> {
        todo!()
    }

    pub async fn merge_clusters<const MAX_ATTEMPTS: usize>(
        &mut self,
        cluster_id_1: ClusterId,
        cluster_id_2: ClusterId,
    ) -> Result<(), ()> {
        for _ in 0..MAX_ATTEMPTS {
            let result = self
                .conn
                .conn(move |conn| {
                    conn.busy_timeout(Duration::from_secs(5)).ok();

                    if let Err(e) = conn.execute("BEGIN IMMEDIATE", []) {
                        if e.to_string().contains("SQLITE_BUSY") {
                            return Err(e); // Retry outside
                        } else {
                            return Err(e);
                        }
                    }

                    let size_1: i64 = conn
                        .query_row(
                            "SELECT size FROM ClusterMeta WHERE cluster_id = ?",
                            [cluster_id_1.0.to_string()],
                            |row| row.get(0),
                        )
                        .expect("cluster_id_1 missing");

                    let size_2: i64 = conn
                        .query_row(
                            "SELECT size FROM ClusterMeta WHERE cluster_id = ?",
                            [cluster_id_2.0.to_string()],
                            |row| row.get(0),
                        )
                        .expect("cluster_id_2 missing");

                    let merge_size = size_1 + size_2;

                    let (cluster_id_1, cluster_id_2) = match size_1 > size_2 {
                        true => (cluster_id_1, cluster_id_2),
                        false => (cluster_id_2, cluster_id_1),
                    };

                    let _ = conn.execute(
                        "UPDATE Clusters SET cluster_id=? WHERE cluster_id=?",
                        (cluster_id_1.0.to_string(), cluster_id_2.0.to_string()),
                    );
                    let _ = conn.execute(
                        "UPDATE ClusterMeta SET size=?, merged_into=? WHERE cluster_id=?",
                        (
                            merge_size,
                            cluster_id_1.0.to_string(),
                            cluster_id_2.0.to_string(),
                        ),
                    );
                    let _ = conn.execute(
                        "UPDATE ClusterMeta SET size=? WHERE cluster_id=?",
                        (merge_size, cluster_id_1.0.to_string()),
                    );

                    // let _ = conn.execute(
                    //     "DELETE FROM ClusterMeta WHERE cluster_id=?",
                    //     (cluster_id_2.0.to_string(),),
                    // );

                    let _ = conn
                        .execute("COMMIT", [])
                        .expect("Failed to commit transaction");

                    Ok(())
                })
                .await;
            if result.is_ok() {
                return Ok(());
            }
        }

        Err(())
    }

    pub async fn insert_cluster_edge(
        &mut self,
        vector_id_1: VectorId,
        vector_id_2: VectorId,
    ) -> Result<(), ()> {
        self.conn
            .conn(move |conn| {
                conn.execute(
                    "INSERT iNTO ClusterEdge(source, target) VALUES (?, ?)",
                    [vector_id_1.0.to_string(), vector_id_2.0.to_string()],
                )
            })
            .await
            .expect("Failed to insert into ClusterMeta");

        Ok(())
    }

    pub async fn get_clusters(&self) -> Vec<ClusterId> {
        let cluster_ids = self
            .conn
            .conn(|conn| {
                let mut stmt = conn
                    .prepare(
                        "
                SELECT
                    cluster_id
                FROM
                    ClusterMeta
                WHERE
                    merged_into IS NULL AND size > 0
                ",
                    )
                    .unwrap();

                let rows: Vec<ClusterId> = stmt
                    .query_map([], |row| {
                        let uuid_str: String = row.get(0).unwrap();
                        let uuid = Uuid::from_str(&uuid_str).unwrap();

                        Ok(ClusterId(uuid))
                    })
                    .unwrap()
                    .map(|x| x.unwrap())
                    .collect();

                Ok(rows)
            })
            .await;

        cluster_ids.unwrap()
    }

    pub async fn get_cluster(&self, vector_id: VectorId) -> Result<ClusterId, ()> {
        let cluster_id = self
            .conn
            .conn(move |conn| {
                let mut stmt = conn
                    .prepare("SELECT cluster_id FROM Clusters WHERE vector_id = ?")
                    .unwrap();

                let cluster_id: ClusterId = stmt
                    .query_map([vector_id.0.to_string()], |row| {
                        let uuid_str: String = row.get(0).unwrap();
                        let uuid = Uuid::from_str(&uuid_str).unwrap();

                        Ok(ClusterId(uuid))
                    })
                    .unwrap()
                    .map(|x| x.unwrap())
                    .next()
                    .unwrap();

                Ok(cluster_id)
            })
            .await;

        Ok(cluster_id.unwrap())
    }

    pub async fn get_cluster_members<const MAX_ATTEMPTS: usize>(
        &self,
        id: ClusterId,
    ) -> Result<Vec<VectorId>, ()> {
        for _ in 0..MAX_ATTEMPTS {
            let vectors = self
                .conn
                .conn(move |conn| {
                    conn.busy_timeout(Duration::from_secs(5)).ok();

                    if let Err(e) = conn.execute("BEGIN DEFERRED", []) {
                        return Err(e);
                    }

                    let mut stmt = conn
                        .prepare("SELECT merged_into FROM ClusterMeta WHERE cluster_id = ? AND merged_into != NULL")
                        .expect("Failed to prepare statement");
                    
                    let result: Option<String> = stmt
                        .query_row([id.0.to_string()], |row| {
                        
                            println!("{:?}", row.get::<_, Option<String>>(0));
                            row.get::<_, Option<String>>(0)
                        })
                        .ok()
                        .flatten();

                    let id = result.unwrap_or(id.0.to_string());

                    // println!("{:?}", id);
                    // let merged_cluster_id: Option<String> = conn
                    //     .query_row(
                    //         "SELECT merged_into FROM ClusterMeta WHERE cluster_id = ? AND merged_into IS NOT NULL",
                    //         [id.0.to_string()],
                    //         |row| row.get(0),
                    //     )
                    //     .ok();

                    // let id = merged_cluster_id.unwrap_or_else(|| id.0.to_string());

                    let mut stmt = conn
                        .prepare("SELECT vector_id FROM Clusters WHERE cluster_id = ?")
                        .unwrap();

                    let rows: Vec<VectorId> = stmt
                        .query_map([id], |row| {
                            let uuid_str: String = row.get(0).unwrap();
                            let uuid = Uuid::from_str(&uuid_str).unwrap();

                            Ok(VectorId(uuid))
                        })
                        .unwrap()
                        .map(|x| x.unwrap())
                        .collect();
                    let _ = conn
                        .execute("COMMIT", [])
                        .expect("Failed to commit transaction");

                    Ok(rows)
                })
                .await;
            if vectors.is_ok() {
                return Ok(vectors.unwrap());
            }
        }

        Err(())
    }

    pub async fn clean_up(&mut self) -> Result<(), ()> {
        let _ = self
            .conn
            .conn(move |conn| {
                conn.execute("DELETE FROM ClusterMeta WHERE merged_into=NULL", [])
                    .unwrap();

                conn.execute("VACUUM", []).unwrap();
                Ok(())
            })
            .await;

        Ok(())
    }

    pub async fn copy(&self, file_name: String) -> Result<(), ()> {
        let path = format!("./data/clusters/{}.db", file_name);

        let _ = self
            .conn
            .conn(move |conn| {
                let sql = format!("VACUUM INTO '{}'", path);
                conn.execute(&sql, []).unwrap();
                Ok(())
            })
            .await;

        Ok(())
    }
}

impl<A: Field<A> + PartialEq + Debug + Clone> PartialEq for ClusterSet<A> {
    fn eq(&self, other: &Self) -> bool {
        self.threshold == other.threshold
    }
}

// impl PartialEq for ClusterSet<f32> {
//     fn eq(&self, other: &Self) -> bool {
//         (self.threshold - other.threshold).abs() < f32::EPSILON
//     }
// }

impl<A: Field<A> + PartialOrd + Debug + Clone> PartialOrd for ClusterSet<A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.threshold.partial_cmp(&other.threshold) {
            Some(core::cmp::Ordering::Equal) => None,
            ord => ord,
        }
    }
}
impl<A: Field<A> + PartialEq + Debug + Clone> Eq for ClusterSet<A> {}
impl<A: Field<A> + PartialOrd + Debug + Clone> Ord for ClusterSet<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// pub fn update<A: Field<A> + Debug>(cluster_set: &mut ClusterSet<A>, affected_vectors: Vec<VectorId>, graph: &IntraPartitionGraph<A>, inter_graph: &InterPartitionGraph<A>) -> Result<(), ()> {
//     todo!()
// }

#[derive(Archive, Serialize, Deserialize)]
pub struct ClusterSetSerial<A: Field<A> + Debug> {
    threshold: A,
    id: String,
}

impl<A: Field<A> + Debug> FileExtension for ClusterSetSerial<A> {
    fn extension() -> &'static str {
        "cluster"
    }
}

impl<A: Field<A> + Clone + Debug> From<&ClusterSet<A>> for ClusterSetSerial<A> {
    fn from(value: &ClusterSet<A>) -> Self {
        ClusterSetSerial {
            threshold: value.threshold.clone(),
            id: value.id.to_string(),
        }
    }
}

impl<A: Field<A> + Debug + Clone> From<ClusterSetSerial<A>> for ClusterSet<A> {
    fn from(value: ClusterSetSerial<A>) -> Self {
        ClusterSet {
            threshold: value.threshold,
            id: Uuid::from_str(&value.id).unwrap(),
            conn: ClientBuilder::new()
                .path(&format!("data/clusters//{}.db", value.id))
                .journal_mode(JournalMode::Memory)
                .open_blocking()
                .unwrap(),
        }
    }
}
