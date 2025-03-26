use std::{cmp::Ordering, fmt::Debug, str::FromStr};

use async_sqlite::{Client, ClientBuilder, JournalMode};
use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    from_bytes, to_bytes,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, Serialize,
};
use tokio::fs::read_dir;
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

    pub async fn new(threshold: A) -> Self {
        let id = Uuid::new_v4();
        let conn = ClientBuilder::new()
            .path(&format!("data/clusters//{}.db", id.to_string()))
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
                "CREATE TABLE ClusterEdge(source STRING, target STRING)", // replace cluster_id from String to INT & Sort by cluster_id
                [],
            )?;
            conn.execute(
                "CREATE TABLE ClusterMeta(cluster_id STRING PRIMARY KEY)", // should map String -> Int
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
                        "INSERT iNTO ClusterMeta(cluster_id) VALUES (?)",
                        [cluster_id.0.to_string()],
                    )
                }
            })
            .await
            .expect("Failed to insert into ClusterMeta");

        Ok(cluster_id)
    }

    pub async fn insert(&mut self, vector_id: VectorId, cluster_id: ClusterId) -> Result<(), ()> {
        self.conn
            .conn(move |conn| {
                conn.execute(
                    "INSERT iNTO Clusters(vector_id, cluster_id) VALUES (?, ?)",
                    [vector_id.0.to_string(), cluster_id.0.to_string()],
                )
            })
            .await
            .expect("Failed to insert into ClusterMeta");

        Ok(())
    }

    pub async fn merge_clusters(
        &mut self,
        cluster_id_1: ClusterId,
        cluster_id_2: ClusterId,
    ) -> Result<(), ()> {
        self.conn
            .conn(move |conn| {
                let _ = conn.execute(
                    "UPDATE Clusters SET cluster_id=? WHERE cluster_id=?",
                    (cluster_id_1.0.to_string(), cluster_id_2.0.to_string()),
                );

                let _ = conn.execute(
                    "DELETE FROM ClusterMeta WHERE cluster_id=?",
                    (cluster_id_2.0.to_string(),),
                );

                Ok(())
            })
            .await
            .expect("Failed to merge clusters");

        Ok(())
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
                let mut stmt = conn.prepare("SELECT cluster_id FROM ClusterMeta").unwrap();

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

    pub async fn get_cluster_members(&self, id: ClusterId) -> Vec<VectorId> {
        let vectors = self
            .conn
            .conn(move |conn| {
                let mut stmt = conn
                    .prepare("SELECT vector_id FROM Clusters WHERE cluster_id = ?")
                    .unwrap();

                let rows: Vec<VectorId> = stmt
                    .query_map([id.0.to_string()], |row| {
                        let uuid_str: String = row.get(0).unwrap();
                        let uuid = Uuid::from_str(&uuid_str).unwrap();

                        Ok(VectorId(uuid))
                    })
                    .unwrap()
                    .map(|x| x.unwrap())
                    .collect();

                Ok(rows)
            })
            .await;

        vectors.unwrap()
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
