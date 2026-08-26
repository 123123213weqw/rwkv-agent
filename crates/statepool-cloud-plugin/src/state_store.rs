use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredObject {
    pub uri: String,
    pub checksum: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct StateStoreError(pub String);

#[async_trait]
pub trait StateStore: Send + Sync {
    async fn put_immutable(
        &self,
        key: &str,
        payload: &[u8],
    ) -> Result<StoredObject, StateStoreError>;
    async fn get(&self, uri: &str) -> Result<Vec<u8>, StateStoreError>;
    async fn delete(&self, uri: &str) -> Result<(), StateStoreError>;
}

#[derive(Clone, Debug)]
pub struct LocalFsStateStore {
    root: PathBuf,
}

impl LocalFsStateStore {
    pub fn new(root: PathBuf) -> Result<Self, StateStoreError> {
        if root.as_os_str().is_empty() {
            return Err(StateStoreError("State store root must not be empty".into()));
        }
        Ok(Self { root })
    }

    fn object_path(&self, key: &str) -> Result<PathBuf, StateStoreError> {
        let key = Path::new(key);
        if key.is_absolute()
            || key
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(StateStoreError("Unsafe State object key".into()));
        }
        Ok(self.root.join(key))
    }

    fn uri_path(&self, uri: &str) -> Result<PathBuf, StateStoreError> {
        let raw = uri
            .strip_prefix("file://")
            .ok_or_else(|| StateStoreError("LocalFS store only accepts file:// URIs".into()))?;
        let path = PathBuf::from(raw);
        if !path.starts_with(&self.root) {
            return Err(StateStoreError("State URI escapes configured root".into()));
        }
        Ok(path)
    }
}

#[async_trait]
impl StateStore for LocalFsStateStore {
    async fn put_immutable(
        &self,
        key: &str,
        payload: &[u8],
    ) -> Result<StoredObject, StateStoreError> {
        let path = self.object_path(key)?;
        let parent = path
            .parent()
            .ok_or_else(|| StateStoreError("State object has no parent directory".into()))?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| StateStoreError(format!("Create State directory: {error}")))?;

        let checksum = sha256_checksum(payload);
        if tokio::fs::try_exists(&path)
            .await
            .map_err(|error| StateStoreError(format!("Inspect State object: {error}")))?
        {
            let existing = tokio::fs::read(&path)
                .await
                .map_err(|error| StateStoreError(format!("Read existing State object: {error}")))?;
            if sha256_checksum(&existing) != checksum {
                return Err(StateStoreError(
                    "Immutable State key already contains different bytes".into(),
                ));
            }
            return Ok(stored_object(path, checksum, payload.len()));
        }

        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp = path.with_extension(format!("state.tmp.{}.{}", std::process::id(), sequence));
        let mut options = tokio::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = options
            .open(&temp)
            .await
            .map_err(|error| StateStoreError(format!("Create temporary State object: {error}")))?;
        if let Err(error) = file.write_all(payload).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(StateStoreError(format!("Write State object: {error}")));
        }
        if let Err(error) = file.sync_all().await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(StateStoreError(format!("Sync State object: {error}")));
        }
        drop(file);
        if let Err(error) = tokio::fs::rename(&temp, &path).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(StateStoreError(format!(
                "Publish State object atomically: {error}"
            )));
        }
        Ok(stored_object(path, checksum, payload.len()))
    }

    async fn get(&self, uri: &str) -> Result<Vec<u8>, StateStoreError> {
        let path = self.uri_path(uri)?;
        tokio::fs::read(path)
            .await
            .map_err(|error| StateStoreError(format!("Read State object: {error}")))
    }

    async fn delete(&self, uri: &str) -> Result<(), StateStoreError> {
        let path = self.uri_path(uri)?;
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(StateStoreError(format!("Delete State object: {error}"))),
        }
    }
}

fn stored_object(path: PathBuf, checksum: String, len: usize) -> StoredObject {
    StoredObject {
        uri: format!("file://{}", path.to_string_lossy()),
        checksum,
        size_bytes: len.try_into().unwrap_or(u64::MAX),
    }
}

pub fn sha256_checksum(payload: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(payload))
}
