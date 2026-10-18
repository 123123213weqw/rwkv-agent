use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore as ObjectStoreClient, ObjectStoreExt, PutMode, PutOptions};
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
pub struct S3StateStoreConfig {
    pub bucket: String,
    pub region: String,
    pub endpoint: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub prefix: String,
    pub allow_http: bool,
}

#[derive(Clone)]
pub struct S3StateStore {
    store: std::sync::Arc<dyn ObjectStoreClient>,
    bucket: String,
    prefix: String,
}

impl std::fmt::Debug for S3StateStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("S3StateStore")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl S3StateStore {
    pub fn new(config: S3StateStoreConfig) -> Result<Self, StateStoreError> {
        if config.bucket.trim().is_empty() || config.region.trim().is_empty() {
            return Err(StateStoreError(
                "S3 bucket and region must not be empty".into(),
            ));
        }
        if config.access_key_id.is_some() != config.secret_access_key.is_some() {
            return Err(StateStoreError(
                "S3 access key and secret key must be configured together".into(),
            ));
        }
        let mut builder = AmazonS3Builder::from_env()
            .with_bucket_name(&config.bucket)
            .with_region(&config.region)
            .with_allow_http(config.allow_http)
            .with_virtual_hosted_style_request(false);
        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if let (Some(access), Some(secret)) = (&config.access_key_id, &config.secret_access_key) {
            builder = builder
                .with_access_key_id(access)
                .with_secret_access_key(secret);
        }
        let store = builder
            .build()
            .map_err(|error| StateStoreError(format!("Build S3 State store: {error}")))?;
        Ok(Self::from_store(
            std::sync::Arc::new(store),
            config.bucket,
            config.prefix,
        ))
    }

    fn from_store(
        store: std::sync::Arc<dyn ObjectStoreClient>,
        bucket: String,
        prefix: String,
    ) -> Self {
        Self {
            store,
            bucket,
            prefix: prefix.trim_matches('/').to_string(),
        }
    }

    fn object_key(&self, key: &str) -> Result<String, StateStoreError> {
        validate_object_key(key)?;
        Ok(if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{key}", self.prefix)
        })
    }

    fn uri_key(&self, uri: &str) -> Result<String, StateStoreError> {
        let prefix = format!("s3://{}/", self.bucket);
        let key = uri.strip_prefix(&prefix).ok_or_else(|| {
            StateStoreError("State URI is outside the configured S3 bucket".into())
        })?;
        validate_object_key(key)?;
        if !self.prefix.is_empty()
            && key != self.prefix
            && !key.starts_with(&format!("{}/", self.prefix))
        {
            return Err(StateStoreError(
                "State URI is outside the configured S3 prefix".into(),
            ));
        }
        Ok(key.to_string())
    }

    async fn read_key(&self, key: &str) -> Result<Vec<u8>, StateStoreError> {
        self.store
            .get(&ObjectPath::from(key))
            .await
            .map_err(|error| StateStoreError(format!("Get S3 State object: {error}")))?
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .map_err(|error| StateStoreError(format!("Read S3 State object: {error}")))
    }
}

#[async_trait]
impl StateStore for S3StateStore {
    async fn put_immutable(
        &self,
        key: &str,
        payload: &[u8],
    ) -> Result<StoredObject, StateStoreError> {
        let key = self.object_key(key)?;
        let path = ObjectPath::from(key.as_str());
        let checksum = sha256_checksum(payload);
        let result = self
            .store
            .put_opts(
                &path,
                payload.to_vec().into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..PutOptions::default()
                },
            )
            .await;
        if let Err(put_error) = result {
            match self.read_key(&key).await {
                Ok(existing) if sha256_checksum(&existing) == checksum => {}
                Ok(_) => {
                    return Err(StateStoreError(
                        "Immutable S3 State key already contains different bytes".into(),
                    ));
                }
                Err(_) => {
                    return Err(StateStoreError(format!(
                        "Create immutable S3 State object: {put_error}"
                    )));
                }
            }
        }
        Ok(StoredObject {
            uri: format!("s3://{}/{}", self.bucket, key),
            checksum,
            size_bytes: payload.len().try_into().unwrap_or(u64::MAX),
        })
    }

    async fn get(&self, uri: &str) -> Result<Vec<u8>, StateStoreError> {
        let key = self.uri_key(uri)?;
        self.read_key(&key).await
    }

    async fn delete(&self, uri: &str) -> Result<(), StateStoreError> {
        let key = self.uri_key(uri)?;
        self.store
            .delete(&ObjectPath::from(key))
            .await
            .map_err(|error| StateStoreError(format!("Delete S3 State object: {error}")))
    }
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

fn validate_object_key(key: &str) -> Result<(), StateStoreError> {
    let key = Path::new(key);
    if key.as_os_str().is_empty()
        || key.is_absolute()
        || key
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(StateStoreError("Unsafe State object key".into()));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[tokio::test]
    async fn s3_adapter_preserves_immutable_keys_and_scoped_uris() {
        let store = S3StateStore::from_store(
            std::sync::Arc::new(InMemory::new()),
            "state-bucket".into(),
            "tenant-prefix".into(),
        );
        let first = store
            .put_immutable("session/v1.state", b"state-one")
            .await
            .expect("first write");
        assert_eq!(
            first.uri,
            "s3://state-bucket/tenant-prefix/session/v1.state"
        );
        assert_eq!(store.get(&first.uri).await.expect("read"), b"state-one");
        let repeated = store
            .put_immutable("session/v1.state", b"state-one")
            .await
            .expect("idempotent repeat");
        assert_eq!(repeated, first);
        let conflict = store
            .put_immutable("session/v1.state", b"state-two")
            .await
            .expect_err("immutable conflict");
        assert!(conflict.0.contains("different bytes"));
        assert!(
            store
                .get("s3://other-bucket/tenant-prefix/session/v1.state")
                .await
                .is_err()
        );
        assert!(store.put_immutable("../escape", b"bad").await.is_err());
        store.delete(&first.uri).await.expect("delete");
        assert!(store.get(&first.uri).await.is_err());
    }
}
