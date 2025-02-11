use std::{marker::PhantomData, str::FromStr};

use rancor::Strategy;
use rkyv::{
    bytecheck::CheckBytes,
    de::Pool,
    from_bytes, to_bytes,
    validation::{archive::ArchiveValidator, shared::SharedValidator, Validator},
    Archive, Deserialize, DeserializeUnsized, Serialize,
};
use tokio::fs::read_dir;
use uuid::Uuid;

use crate::vector::{Field, VectorSerial, VectorSpace};

use super::{ids::PartitionId, serial::FileExtension};

#[derive(Clone, Debug)]
pub struct Meta<A: Field<A>, B: VectorSpace<A> + Clone> {
    pub id: PartitionId,
    pub size: usize,
    pub centroid: B,

    pub edge_length: (A, A),
}

impl<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Clone + From<VectorSerial<A>>> Meta<A, B> {
    pub fn new(id: PartitionId, size: usize, centroid: B, edge_length: (A, A)) -> Self {
        Self {
            id,
            size,
            centroid,
            edge_length,
        }
    }

    pub async fn load(dir: &str, name: &str) -> Self
    where
        A: Clone + Copy + Archive,
        B: Into<VectorSerial<A>>,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
        [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
        <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<Pool, rancor::Error>>,
    {
        let bytes = tokio::fs::read(&format!("{dir}//{name}.{}", MetaSerial::<A>::extension()))
            .await
            .unwrap();

        from_bytes::<MetaSerial<A>, rancor::Error>(&bytes)
            .unwrap()
            .into()
    }

    pub async fn save(&self, dir: &str, name: &str)
    where
        A: Clone
            + Copy
            + Archive
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
        B: Into<VectorSerial<A>>,
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
        [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
    {
        let serial_value: MetaSerial<A> = self.into();

        let bytes = to_bytes::<rancor::Error>(&serial_value).unwrap();

        tokio::fs::write(
            &format!("{dir}//{name}.{}", MetaSerial::<A>::extension()),
            bytes.as_slice(),
        )
        .await
        .unwrap();
    }

    pub async fn load_from_folder(dir: &str) -> Vec<Self>
    where
        A: Clone
            + Copy
            + Archive
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
        for<'a> <A as Archive>::Archived:
            CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rancor::Error>>,
        [<A as Archive>::Archived]: DeserializeUnsized<[A], Strategy<Pool, rancor::Error>>,
        <A as Archive>::Archived: rkyv::Deserialize<A, Strategy<Pool, rancor::Error>>,
    {
        let mut results = Vec::new();
        let mut entries = read_dir(dir).await.unwrap();

        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();
            if let Some(extension) = path.extension() {
                if extension == MetaSerial::<A>::extension() {
                    let file_name = path.file_stem().unwrap().to_string_lossy();
                    let bytes = tokio::fs::read(&path).await.unwrap();
                    let meta_serial: MetaSerial<A> =
                        from_bytes::<MetaSerial<A>, rancor::Error>(&bytes).unwrap();
                    let meta: Meta<A, B> = meta_serial.into();
                    results.push(meta);
                }
            }
        }

        results
    }
}

#[derive(Archive, Deserialize, Serialize)]
pub struct MetaSerial<A: Clone + Copy> {
    pub id: String,
    pub size: usize,
    pub centroid: VectorSerial<A>,

    pub edge_length: (A, A),
}

impl<A: Clone + Copy> FileExtension for MetaSerial<A> {
    fn extension() -> &'static str {
        "meta"
    }
}

impl<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Clone + From<VectorSerial<A>>>
    From<MetaSerial<A>> for Meta<A, B>
{
    fn from(value: MetaSerial<A>) -> Self {
        Self {
            id: PartitionId(Uuid::from_str(&value.id).unwrap()),
            size: value.size,
            centroid: value.centroid.into(),
            edge_length: value.edge_length,
        }
    }
}

impl<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Clone + Into<VectorSerial<A>>> From<Meta<A, B>>
    for MetaSerial<A>
{
    fn from(value: Meta<A, B>) -> Self {
        MetaSerial {
            id: value.id.0.to_string(),
            size: value.size,
            centroid: value.centroid.into(),

            edge_length: value.edge_length,
        }
    }
}

impl<A: Field<A> + Clone + Copy, B: VectorSpace<A> + Clone + Into<VectorSerial<A>>>
    From<&Meta<A, B>> for MetaSerial<A>
{
    fn from(value: &Meta<A, B>) -> Self {
        MetaSerial {
            id: value.id.0.to_string(),
            size: value.size,
            centroid: value.centroid.clone().into(),

            edge_length: value.edge_length,
        }
    }
}
