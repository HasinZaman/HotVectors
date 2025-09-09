use std::{cmp::Ordering, collections::HashSet, fmt::Debug, str::FromStr};

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes, from_bytes, from_bytes_unchecked, ser::{allocator::ArenaHandle, sharing::Share, Serializer}, to_bytes, util::AlignedVec, validation::{archive::ArchiveValidator, shared::SharedValidator, Validator}, Archive, Deserialize, Serialize
};
use sled::Db;
use tokio::fs::read_dir;
use uuid::Uuid;

use crate::vector::Field;

use super::{
    ids::{ClusterId, VectorId},
    serial::FileExtension,
};

pub enum MergeClusterError{
    SameCluster
}

#[derive(Debug)]
struct ClusterMembership {
    cluster_to_vector: Db,
    vector_to_cluster: Db,
}

impl ClusterMembership {
    pub fn get_cluster_id(&self, vector_id: &VectorId) -> Result<ClusterId, sled::Error> {
        let bytes = self.vector_to_cluster.get(vector_id.as_ref())?.unwrap();

        let cluster_id = from_bytes::<ClusterId, rancor::Error>(&bytes).unwrap();

        Ok(cluster_id)
    }

    pub fn merge_clusters(
        &mut self,
        absorbing_cluster_id: &ClusterId,
        merging_cluster_id: &ClusterId,
    ) -> Result<(), sled::Error> {
        let absorbing_cluster: HashSet<VectorId> = {
            let bytes = self
                .cluster_to_vector
                .get(absorbing_cluster_id.as_ref())?
                .expect(&format!("Cannot find absorbing cluster: {absorbing_cluster_id:?}"));

            from_bytes::<Vec<VectorId>, rancor::Error>(&bytes)
                .unwrap()
                .into_iter()
                .collect()
        };

        let merging_cluster: HashSet<VectorId> = {
            let bytes = self
                .cluster_to_vector
                .get(merging_cluster_id.as_ref())?
                .expect(&format!("Cannot find merging cluster: {merging_cluster_id:?}"));

            from_bytes::<Vec<VectorId>, rancor::Error>(&bytes)
                .unwrap()
                .into_iter()
                .collect()
        };

        // update vtc
        {
            for vector_id in &merging_cluster {
                let _ = self
                    .vector_to_cluster
                    .insert(
                        vector_id.as_ref(),
                        to_bytes::<rancor::Error>(absorbing_cluster_id)
                            .unwrap()
                            .as_ref(),
                    )
                    .unwrap();
            }
        }

        // add to merge
        {
            let merged_vectors: Vec<VectorId> =
                absorbing_cluster.union(&merging_cluster).cloned().collect();
            let _ = self
                .cluster_to_vector
                .insert(
                    absorbing_cluster_id.as_ref(),
                    to_bytes::<rancor::Error>(&merged_vectors).unwrap().as_ref(),
                )
                .unwrap();
        }
        // delete
        {
            let _ = self
                .cluster_to_vector
                .remove(merging_cluster_id.as_ref())
                .unwrap();
        }
        Ok(())
    }
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
struct ClusterMeta {
    size: usize,
    merged_into: Option<ClusterId>,
}

#[derive(Debug)]
pub struct ClusterSet<A: Field<A> + Debug + Clone> {
    pub threshold: A,
    pub id: Uuid,

    edges: Db,
    meta: Db,
    membership: ClusterMembership,
}

impl<
        A: Field<A>
            + Debug
            + Clone
            + Copy
            + 'static
            + Archive
            + PartialOrd
            + for<'a> Serialize<
                Strategy<Serializer<AlignedVec, ArenaHandle<'a>, Share>, rancor::Error>,
            >,
    > ClusterSet<A>
where
    for<'a> <A as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
    <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<rkyv::de::Pool, rancor::Error>>,
{
    pub async fn load_from_folder(dir: &str) -> Vec<Self> {
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

    pub fn from_smaller_cluster_set<'a>(
        threshold: A,
        other: &'a Self,
    ) -> Result<Self, sled::Error> {
        let Self {
            ref edges,
            ref meta,
            membership:
                ClusterMembership {
                    ref vector_to_cluster,
                    ref cluster_to_vector,
                },
            ..
        } = other;

        // let _ = other.copy(id.to_string()).await.unwrap();

        let new_id = Uuid::new_v4();
        let new_edges: Db = sled::open(&format!("data/clusters/{}/edges", new_id.to_string()))?;
        let new_meta: Db = sled::open(&format!("data/clusters/{}/meta", new_id.to_string()))?;
        let new_membership = ClusterMembership {
            cluster_to_vector: sled::open(&format!("data/clusters/{}/ctv", new_id.to_string()))?,
            vector_to_cluster: sled::open(&format!("data/clusters/{}/vtc", new_id.to_string()))?,
        };

        let mut new_cluster_set = Self {
            threshold,
            id: Uuid::new_v4(),

            edges: new_edges,
            meta: new_meta,
            membership: new_membership,
        };

        // copy edges
        {
            let _copy_edges = edges
                .iter()
                .map(|x| x.unwrap())
                .filter(|(_, dist_bytes)| {
                    println!("{dist_bytes:?}");
                    // let dist = from_bytes::<A, rancor::Error>(&dist_bytes).unwrap();
                    let dist = (unsafe { from_bytes_unchecked::<A, rancor::Error>(&dist_bytes) }).unwrap();

                    println!("{dist:?}");

                    dist >= threshold
                })
                .map(|(key, val)| new_cluster_set.edges.insert(key, val)
                    .or_else(|x| {
                        println!("{x:?}");
                        Err(x)
                    })
                )
                .all(|val| val.is_ok());

            if !_copy_edges {
                return Err(todo!())
            }
        }
        // copy meta
        {
            let _copy_meta = meta.iter()
                .map(|x| x.unwrap())
                .map(|(key, val)| new_cluster_set.meta.insert(key, val)
                    .or_else(|x| {
                        println!("{x:?}");
                        Err(x)
                    })
                )
                .all(|val| val.is_ok());

            if !_copy_meta {
                return Err(todo!())
            }
        }

        // Copy vector membership
        {
            let _copy_vtc = vector_to_cluster
                .iter()
                .map(|x| x.unwrap())
                .map(|(key, val)| {
                    new_cluster_set.membership.vector_to_cluster.insert(&key, &val)
                    .or_else(|x| {
                        println!("{x:?}");
                        Err(x)
                    })
                })
                .all(|val| val.is_ok());

            if !_copy_vtc {
                return Err(todo!())
            }
            let _copy_ctv = cluster_to_vector
                .iter()
                .map(|x| x.unwrap())
                .map(|(key, val)| {
                    new_cluster_set.membership.cluster_to_vector.insert(key, val)
                    .or_else(|x| {
                        println!("{x:?}");
                        Err(x)
                    })
                })
                .all(|val| val.is_ok());
            
            if !_copy_ctv {
                return Err(todo!())
            }
        }


        // merge clusters
        {
            let iter = edges
                .iter()
                .map(|x| x.unwrap())
                .filter(|(_, dist_bytes)| {
                    let dist = (unsafe { from_bytes_unchecked::<A, rancor::Error>(&dist_bytes) }).unwrap();
                    // let dist = from_bytes::<A, rancor::Error>(&dist).unwrap();

                    println!("{dist:?} <= {threshold:?}");

                    dist <= threshold
                })
                .map(|(bytes, _)| bytes_to_uuid_pair(&bytes).unwrap())
                .map(|(id_1, id_2)| (VectorId(id_1), VectorId(id_2)))
                .map(|(id_1, id_2)| {
                    (
                        (other.membership.get_cluster_id(&id_1).unwrap(), id_1),
                        (other.membership.get_cluster_id(&id_2).unwrap(), id_2),
                    )
                });


            for ((cluster_id_1, vector_id_1), (cluster_id_2, vector_id_2)) in iter {
                match new_cluster_set.merge_clusters::<5>(cluster_id_1, cluster_id_2) {
                    Ok(_) | Err(MergeClusterError::SameCluster) => {},
                    Err(_) => {
                        return Err(todo!())
                    },
                }
                if new_cluster_set.delete_edge(vector_id_1, vector_id_2).is_err() {
                    return Err(todo!())
                }
            }
        }

        Ok(new_cluster_set)
    }

    pub fn new(threshold: A, dir: String) -> Self {
        let id = Uuid::new_v4();
        let edges: Db = sled::open(&format!("data/clusters/{}/edges", id.to_string())).unwrap();
        let meta: Db = sled::open(&format!("data/clusters/{}/meta", id.to_string())).unwrap();
        let membership = ClusterMembership {
            cluster_to_vector: sled::open(&format!("data/clusters/{}/ctv", id.to_string()))
                .unwrap(),
            vector_to_cluster: sled::open(&format!("data/clusters/{}/vtc", id.to_string()))
                .unwrap(),
        };

        Self {
            threshold,
            id,
            edges,
            meta,
            membership,
        }
    }

    pub async fn create_local(&self, vectors: &[VectorId], transaction_id: Uuid) -> Self {
        todo!()
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

    pub fn new_cluster(&mut self) -> Result<ClusterId, ()> {
        let cluster_id = ClusterId(Uuid::new_v4());

        let meta = ClusterMeta {
            size: 0,
            merged_into: None,
        };

        self.meta
            .insert(
                cluster_id.as_ref(),
                to_bytes::<rancor::Error>(&meta).unwrap().as_ref(),
            )
            .unwrap();

        let vec: Vec<VectorId> = Vec::new();

        // self.membership
        //     .cluster_to_vector
        //     .insert(
        //         cluster_id.as_ref(),
        //         to_bytes::<rancor::Error>(&vec).unwrap().as_ref()
        //     ).unwrap();

        Ok(cluster_id)
    }

    pub fn new_cluster_from_vector(
        &mut self,
        vector_id: VectorId,
        cluster_id: ClusterId,
    ) -> Result<(), ()> {
        // increase size
        {
            let bytes = self.meta.get(cluster_id.as_ref()).unwrap().unwrap();

            let mut meta = from_bytes::<ClusterMeta, rancor::Error>(bytes.as_ref()).unwrap();
            meta.size += 1;

            let _ = self
                .meta
                .insert(
                    cluster_id.as_ref(),
                    to_bytes::<rancor::Error>(&meta).unwrap().as_ref(),
                )
                .unwrap();
        }
        // update membership
        {
            let _ = self
                .membership
                .vector_to_cluster
                .insert(
                    vector_id.as_ref(),
                    to_bytes::<rancor::Error>(&cluster_id).unwrap().as_ref(),
                )
                .unwrap();

            let bytes = self
                .membership
                .cluster_to_vector
                .get(cluster_id.as_ref())
                .unwrap();

            let vectors: Vec<VectorId> = match bytes {
                Some(bytes) => {
                    let mut vectors: Vec<VectorId> =
                        from_bytes::<Vec<VectorId>, rancor::Error>(bytes.as_ref()).unwrap();
                    vectors.push(vector_id);

                    vectors
                }
                None => {
                    vec![vector_id]
                }
            };

            let _ = self
                .membership
                .cluster_to_vector
                .insert(
                    cluster_id.as_ref(),
                    to_bytes::<rancor::Error>(&vectors).unwrap().as_ref(),
                )
                .unwrap();
        }

        Ok(())
    }

    pub fn add_edge(
        &mut self,
        vector_id_1: VectorId,
        vector_id_2: VectorId,
        weight: A,
    ) -> Result<(), ()> {
        let cluster_id_1: ClusterId = {
            let bytes = self
                .membership
                .vector_to_cluster
                .get(vector_id_1.as_ref())
                .unwrap()
                .unwrap();

            from_bytes::<ClusterId, rancor::Error>(&bytes.as_ref()).unwrap()
        };
        let cluster_id_2: ClusterId = {
            let bytes = self
                .membership
                .vector_to_cluster
                .get(vector_id_2.as_ref())
                .unwrap()
                .unwrap();

            from_bytes::<ClusterId, rancor::Error>(&bytes.as_ref()).unwrap()
        };

        if cluster_id_1 != cluster_id_2 {
            let weight = to_bytes::<rancor::Error>(&weight).unwrap();
            // println!("add weight: {:?}",weight.as_ref());
            if let Err(err) = self.edges.insert(
                uuid_pair_to_bytes(vector_id_1.0, vector_id_2.0),
                weight.as_ref(),
            ) {
                println!("{:?}",err);
                return Err(())
            }
            if let Err(err) = self.edges.insert(
                uuid_pair_to_bytes(vector_id_2.0, vector_id_1.0),
                weight.as_ref(),
            ) {
                println!("{:?}",err);
                return Err(())
            }
        }

        Ok(())
    }

    pub fn delete_edge(&mut self, vector_id_1: VectorId, vector_id_2: VectorId) -> Result<(), ()> {
        let uuids = uuid_pair_to_bytes(vector_id_1.0, vector_id_2.0);
        if let Err(err) = self.edges.remove(uuids) {
            println!("{:?}",err);
            return Err(())
        }
        let uuids = uuid_pair_to_bytes(vector_id_2.0, vector_id_1.0);
        if let Err(err) = self.edges.remove(uuids) {
            println!("{:?}",err);
            return Err(())
        }
        Ok(())
    }

    pub fn get_edges(&self) -> Result<Vec<(VectorId, VectorId, A)>, ()> {
        todo!()
    }

    pub fn merge_clusters<const MAX_ATTEMPTS: usize>(
        &mut self,
        cluster_id_1: ClusterId,
        cluster_id_2: ClusterId,
    ) -> Result<(), MergeClusterError> {
        
        let cluster_id_1 = {
            let mut cluster_id_1 = cluster_id_1;
            // println!("cluster_id_1: {cluster_id_1:?}");
            let mut cluster_meta_1: ClusterMeta = from_bytes::<ClusterMeta, rancor::Error>(
                &(self.meta.get(&cluster_id_1).unwrap().unwrap()),
            )
            .unwrap();

            while let Some(new_cluster_id) = cluster_meta_1.merged_into {
                cluster_id_1 = new_cluster_id;
                // println!("updated cluster_id_1: {cluster_id_1:?}");

                cluster_meta_1 = from_bytes::<ClusterMeta, rancor::Error>(
                    &(self.meta.get(&cluster_id_1).unwrap().unwrap()),
                )
                .unwrap();
            }
            // println!("Final cluster_id_1: {cluster_id_1:?}");

            cluster_id_1
        };
        

        let cluster_id_2 = {
            let mut cluster_id_2= cluster_id_2;
            // println!("cluster_id_2: {cluster_id_2:?}");
            let mut cluster_meta_2: ClusterMeta = from_bytes::<ClusterMeta, rancor::Error>(
                &(self.meta.get(&cluster_id_2).unwrap().unwrap()),
            )
            .unwrap();

            while let Some(new_cluster_id) = cluster_meta_2.merged_into {
                cluster_id_2 = new_cluster_id;
                // println!("updated cluster_id_2: {cluster_id_2:?}");

                cluster_meta_2 = from_bytes::<ClusterMeta, rancor::Error>(
                    &(self.meta.get(&cluster_id_2).unwrap().unwrap()),
                )
                .unwrap();
            }
            // println!("Final cluster_id_2: {cluster_id_2:?}");

            cluster_id_2
        };

        if cluster_id_1 == cluster_id_2 {
            return Err(MergeClusterError::SameCluster)
        }


        let mut cluster_meta_1: ClusterMeta = from_bytes::<ClusterMeta, rancor::Error>(
            &self.meta.get(&cluster_id_1).unwrap().unwrap(),
        )
        .unwrap();
        let mut cluster_meta_2: ClusterMeta = from_bytes::<ClusterMeta, rancor::Error>(
            &self.meta.get(&cluster_id_2).unwrap().unwrap(),
        )
        .unwrap();

        let ((absorbing_cluster_id, absorbing_cluster_meta), (merging_cluster_id, merging_cluster_meta)) =
            match cluster_meta_1.size < cluster_meta_2.size {
                true => (
                    (cluster_id_2, &mut cluster_meta_2),
                    (cluster_id_1, &mut cluster_meta_1),
                ),
                false => (
                    (cluster_id_1, &mut cluster_meta_1),
                    (cluster_id_2, &mut cluster_meta_2),
                ),
            };
        
        // update meta
        {
            absorbing_cluster_meta.size += merging_cluster_meta.size;

            let bytes: AlignedVec = to_bytes::<rancor::Error>(absorbing_cluster_meta).unwrap();

            self.meta.insert(absorbing_cluster_id, bytes.as_ref()).unwrap();
        }
        {
            merging_cluster_meta.merged_into = Some(absorbing_cluster_id);

            let bytes: AlignedVec = to_bytes::<rancor::Error>(merging_cluster_meta).unwrap();

            self.meta.insert(merging_cluster_id, bytes.as_ref())
                .unwrap();
        }


        // update membership
        // println!("merge {merging_cluster_id:?} |-> {absorbing_cluster_id:?}");
        let _ = self
            .membership
            .merge_clusters(&absorbing_cluster_id, &merging_cluster_id)
            .unwrap();

        Ok(())
    }

    pub fn get_clusters(&self) -> Vec<ClusterId> {
        // could decode key after filters
        self.meta
            .iter()
            .map(|val| val.unwrap())
            .map(|(key, value)| {
                (
                    from_bytes::<ClusterId, rancor::Error>(key.as_ref()).unwrap(),
                    from_bytes::<ClusterMeta, rancor::Error>(value.as_ref()).unwrap(),
                )
            })
            .filter(|(_, meta)| meta.size > 0)
            .filter(|(_, meta)| meta.merged_into.is_none())
            .map(|(cluster_id, _)| cluster_id)
            .collect()
    }

    pub fn get_cluster(&self, vector_id: VectorId) -> Result<ClusterId, ()> {
        let bytes = match self.membership.vector_to_cluster.get(&vector_id.as_ref()) {
            Ok(Some(val)) => val,
            _ => return Err(()),
        };

        Ok(from_bytes::<ClusterId, rancor::Error>(bytes.as_ref()).unwrap())
    }

    pub fn get_cluster_members(&self, cluster_id: ClusterId) -> Result<Vec<VectorId>, ()> {
        let bytes = self
            .membership
            .cluster_to_vector
            .get(cluster_id.as_ref())
            .unwrap()
            .unwrap();

        let vectors = from_bytes::<Vec<VectorId>, rancor::Error>(bytes.as_ref()).unwrap();

        Ok(vectors)
    }

    pub fn clean_up(&mut self) -> Result<(), ()> {
        todo!()
        // Ok(())
    }

    // pub fn copy(&self, file_name: String) -> Result<(), ()> {
    //     let path = format!("./data/clusters/{}.db", file_name);

    //     let _ = self
    //         .conn
    //         .conn(move |conn| {
    //             let sql = format!("VACUUM INTO '{}'", path);
    //             conn.execute(&sql, []).unwrap();
    //             Ok(())
    //         })
    //         .await;

    //     Ok(())
    // }
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
        let id = Uuid::from_str(&value.id).unwrap();

        let edges: Db = sled::open(&format!("data/clusters/{}/edges", id.to_string())).unwrap();
        let meta: Db = sled::open(&format!("data/clusters/{}/meta", id.to_string())).unwrap();
        let membership = ClusterMembership {
            cluster_to_vector: sled::open(&format!("data/clusters/{}/ctv", id.to_string()))
                .unwrap(),
            vector_to_cluster: sled::open(&format!("data/clusters/{}/vtc", id.to_string()))
                .unwrap(),
        };

        ClusterSet {
            threshold: value.threshold,
            id: id,
            edges: edges,
            meta: meta,
            membership: membership,
        }
    }
}

fn bytes_to_uuid_pair(bytes: &[u8]) -> Result<(Uuid, Uuid), String> {
    if bytes.len() < 32 {
        return Err("Input must be at least 32 bytes".to_string());
    }

    let uuid_1 = Uuid::from_slice(&bytes[0..16])
        .map_err(|e| format!("Failed to parse first UUID: {}", e))?;
    let uuid_2 = Uuid::from_slice(&bytes[16..32])
        .map_err(|e| format!("Failed to parse second UUID: {}", e))?;
    println!("bytes len: {bytes:?}\nuuid_1: {uuid_1:?}\nuuid_2: {uuid_2:?}");

    Ok((uuid_1, uuid_2))
}
pub fn uuid_pair_to_bytes(uuid_1: Uuid, uuid_2: Uuid) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(uuid_1.as_bytes());
    bytes.extend_from_slice(uuid_2.as_bytes());

    println!("first:{uuid_1:?}\nsecond:{uuid_2:?}\nbytes: {bytes:?}");

    bytes
}
