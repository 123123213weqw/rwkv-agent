use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, OwnedMutexGuard};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exchange {
    pub user: String,
    pub assistant: String,
}

#[derive(Clone)]
pub struct SessionStore {
    root: Arc<PathBuf>,
    locks: Arc<Vec<Arc<Mutex<()>>>>,
}

impl SessionStore {
    pub async fn new(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(|e| format!("create session directory: {e}"))?;
        Ok(Self {
            root: Arc::new(root),
            locks: Arc::new((0..64).map(|_| Arc::new(Mutex::new(()))).collect()),
        })
    }

    pub fn normalize_session_id(value: &str) -> Result<String, String> {
        let value = value.trim();
        if value.is_empty() {
            return Err("session_id must not be empty".into());
        }
        if value.chars().count() > 128 || value.chars().any(char::is_control) {
            return Err("session_id must be at most 128 non-control characters".into());
        }
        Ok(value.to_string())
    }

    pub async fn lock(&self, session_id: &str) -> Result<OwnedMutexGuard<()>, String> {
        let session_id = Self::normalize_session_id(session_id)?;
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        session_id.hash(&mut hash);
        let lock = self.locks[hash.finish() as usize % self.locks.len()].clone();
        Ok(lock.lock_owned().await)
    }

    fn path(&self, session_id: &str) -> Result<PathBuf, String> {
        let normalized = Self::normalize_session_id(session_id)?;
        let name = normalized
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(self.root.join(format!("{name}.jsonl")))
    }

    pub async fn history(&self, session_id: &str, limit: usize) -> Result<Vec<Exchange>, String> {
        let path = self.path(session_id)?;
        let raw = match tokio::fs::read_to_string(path).await {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(format!("read session transcript: {error}")),
        };
        let mut rows = Vec::new();
        let lines = raw.lines().collect::<Vec<_>>();
        for (index, line) in lines.iter().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Exchange>(line) {
                Ok(exchange) => rows.push(exchange),
                Err(_) if index + 1 == lines.len() && !raw.ends_with('\n') => break,
                Err(error) => return Err(format!("invalid session transcript: {error}")),
            }
        }
        if rows.len() > limit {
            rows.drain(..rows.len() - limit);
        }
        Ok(rows)
    }

    pub async fn append(
        &self,
        session_id: &str,
        user: &str,
        assistant: &str,
    ) -> Result<(), String> {
        let path = self.path(session_id)?;
        let exchange = Exchange {
            user: user.to_string(),
            assistant: assistant.to_string(),
        };
        let mut encoded = serde_json::to_vec(&exchange).map_err(|e| e.to_string())?;
        encoded.push(b'\n');
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(|e| format!("open session transcript: {e}"))?;
        file.write_all(&encoded)
            .await
            .map_err(|e| format!("append session transcript: {e}"))?;
        file.flush()
            .await
            .map_err(|e| format!("flush session transcript: {e}"))
    }

    pub fn root(&self) -> &Path {
        self.root.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn transcript_is_session_scoped_and_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::new(directory.path()).await.unwrap();
        store.append("a", "u1", "a1").await.unwrap();
        store.append("a", "u2", "a2").await.unwrap();
        store.append("b", "ub", "ab").await.unwrap();
        assert_eq!(store.history("a", 1).await.unwrap()[0].user, "u2");
        assert_eq!(store.history("b", 12).await.unwrap()[0].user, "ub");
        assert!(store.history("c", 12).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn ignores_only_a_torn_final_jsonl_record() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::new(directory.path()).await.unwrap();
        store.append("a", "u1", "a1").await.unwrap();
        let path = store.path("a").unwrap();
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .await
            .unwrap();
        file.write_all(b"{\"user\":\"torn").await.unwrap();
        file.flush().await.unwrap();
        assert_eq!(store.history("a", 12).await.unwrap().len(), 1);
    }

    #[test]
    fn session_id_never_becomes_a_path() {
        assert!(SessionStore::normalize_session_id("../x").is_ok());
        assert!(SessionStore::normalize_session_id("").is_err());
        assert!(SessionStore::normalize_session_id(&"x".repeat(129)).is_err());
    }
}
