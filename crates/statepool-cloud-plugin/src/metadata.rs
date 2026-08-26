use std::collections::HashMap;

use async_trait::async_trait;
use rwkv_statepool_plugin_api::{
    AcquireLeaseRequest, LEASE_CONTRACT_VERSION, Lease, RenewLeaseRequest, StateReference,
};
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls, Transaction};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataErrorKind {
    OwnerMismatch,
    VersionConflict,
    LeaseHeld,
    LeaseExpired,
    StaleFence,
    StateConflict,
    NotFound,
    BackendFailure,
}

#[derive(Clone, Debug)]
pub struct MetadataError {
    pub kind: MetadataErrorKind,
    pub message: String,
}

impl MetadataError {
    fn new(kind: MetadataErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        match self.kind {
            MetadataErrorKind::OwnerMismatch => "owner_mismatch",
            MetadataErrorKind::VersionConflict => "state_version_conflict",
            MetadataErrorKind::LeaseHeld => "lease_held",
            MetadataErrorKind::LeaseExpired => "lease_expired",
            MetadataErrorKind::StaleFence => "stale_fencing_token",
            MetadataErrorKind::StateConflict => "state_conflict",
            MetadataErrorKind::NotFound => "state_not_found",
            MetadataErrorKind::BackendFailure => "metadata_failure",
        }
    }
}

#[async_trait]
pub trait MetadataStore: Send + Sync {
    async fn acquire(
        &self,
        request: &AcquireLeaseRequest,
        now_ms: u64,
    ) -> Result<Lease, MetadataError>;

    async fn renew(&self, request: &RenewLeaseRequest, now_ms: u64)
    -> Result<Lease, MetadataError>;

    async fn release(&self, lease: &Lease, now_ms: u64) -> Result<(), MetadataError>;

    async fn assert_lease(
        &self,
        lease: &Lease,
        require_current_version: bool,
        now_ms: u64,
    ) -> Result<(), MetadataError>;

    async fn commit_state(
        &self,
        lease: &Lease,
        expected_version: u64,
        state_ref: StateReference,
        now_ms: u64,
    ) -> Result<(), MetadataError>;

    async fn current_state(
        &self,
        session_id: &str,
        owner_id: &str,
    ) -> Result<StateReference, MetadataError>;
}

#[derive(Default)]
pub struct InMemoryMetadataStore {
    sessions: Mutex<HashMap<String, SessionRecord>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SessionRecord {
    owner_id: String,
    current_state: Option<StateReference>,
    last_fencing_token: u64,
    active_lease: Option<Lease>,
}

pub struct PostgresMetadataStore {
    client: Mutex<Client>,
}

impl PostgresMetadataStore {
    pub async fn connect(database_url: &str) -> Result<Self, MetadataError> {
        if database_url.trim().is_empty() {
            return Err(MetadataError::new(
                MetadataErrorKind::BackendFailure,
                "PostgreSQL URL must not be empty",
            ));
        }
        let (client, connection) = tokio_postgres::connect(database_url, NoTls)
            .await
            .map_err(database_error)?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                eprintln!("StatePool PostgreSQL connection ended: {error}");
            }
        });
        client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS statepool_sessions (\
                    session_id TEXT PRIMARY KEY,\
                    record JSONB NOT NULL,\
                    updated_at_ms BIGINT NOT NULL\
                );",
            )
            .await
            .map_err(database_error)?;
        Ok(Self {
            client: Mutex::new(client),
        })
    }

    async fn locked_record(
        transaction: &Transaction<'_>,
        session_id: &str,
    ) -> Result<Option<SessionRecord>, MetadataError> {
        let row = transaction
            .query_opt(
                "SELECT record FROM statepool_sessions WHERE session_id = $1 FOR UPDATE",
                &[&session_id],
            )
            .await
            .map_err(database_error)?;
        row.map(|row| decode_record(row.get(0))).transpose()
    }

    async fn lock_session(
        transaction: &Transaction<'_>,
        session_id: &str,
    ) -> Result<(), MetadataError> {
        // Row locks do not protect a Session before its first row exists.
        // A transaction-scoped advisory lock serializes creation and every
        // subsequent mutation for the same stable Session identity.
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                &[&session_id],
            )
            .await
            .map_err(database_error)?;
        Ok(())
    }

    async fn save_record(
        transaction: &Transaction<'_>,
        session_id: &str,
        record: &SessionRecord,
        now_ms: u64,
    ) -> Result<(), MetadataError> {
        let value = serde_json::to_value(record).map_err(|error| {
            MetadataError::new(
                MetadataErrorKind::BackendFailure,
                format!("Serialize PostgreSQL Session record: {error}"),
            )
        })?;
        let now = i64::try_from(now_ms).unwrap_or(i64::MAX);
        transaction
            .execute(
                "INSERT INTO statepool_sessions(session_id, record, updated_at_ms) \
                 VALUES($1, $2, $3) ON CONFLICT(session_id) DO UPDATE SET \
                 record = EXCLUDED.record, updated_at_ms = EXCLUDED.updated_at_ms",
                &[&session_id, &value, &now],
            )
            .await
            .map_err(database_error)?;
        Ok(())
    }
}

fn database_error(error: tokio_postgres::Error) -> MetadataError {
    MetadataError::new(
        MetadataErrorKind::BackendFailure,
        format!("PostgreSQL metadata operation failed: {error}"),
    )
}

fn decode_record(value: serde_json::Value) -> Result<SessionRecord, MetadataError> {
    serde_json::from_value(value).map_err(|error| {
        MetadataError::new(
            MetadataErrorKind::BackendFailure,
            format!("Decode PostgreSQL Session record: {error}"),
        )
    })
}

impl SessionRecord {
    fn current_version(&self) -> u64 {
        self.current_state
            .as_ref()
            .map(|state| state.version)
            .unwrap_or(0)
    }
}

#[async_trait]
impl MetadataStore for InMemoryMetadataStore {
    async fn acquire(
        &self,
        request: &AcquireLeaseRequest,
        now_ms: u64,
    ) -> Result<Lease, MetadataError> {
        let mut sessions = self.sessions.lock().await;
        let record = match sessions.get_mut(&request.session_id) {
            Some(record) => record,
            None if request.expected_state_version == 0 => sessions
                .entry(request.session_id.clone())
                .or_insert(SessionRecord {
                    owner_id: request.owner_id.clone(),
                    current_state: None,
                    last_fencing_token: 0,
                    active_lease: None,
                }),
            None => {
                return Err(MetadataError::new(
                    MetadataErrorKind::VersionConflict,
                    "Unknown Session can only start at State version 0",
                ));
            }
        };
        ensure_owner(record, &request.owner_id)?;
        if record.current_version() != request.expected_state_version {
            return Err(MetadataError::new(
                MetadataErrorKind::VersionConflict,
                format!(
                    "Expected State version {}, current version is {}",
                    request.expected_state_version,
                    record.current_version()
                ),
            ));
        }
        if record
            .active_lease
            .as_ref()
            .is_some_and(|lease| lease.expires_at_ms > now_ms)
        {
            return Err(MetadataError::new(
                MetadataErrorKind::LeaseHeld,
                "Session already has an unexpired writer Lease",
            ));
        }
        record.active_lease = None;
        record.last_fencing_token = record.last_fencing_token.checked_add(1).ok_or_else(|| {
            MetadataError::new(
                MetadataErrorKind::StaleFence,
                "Fencing token space exhausted",
            )
        })?;
        let token = record.last_fencing_token;
        let lease = Lease {
            contract_version: LEASE_CONTRACT_VERSION.into(),
            lease_id: format!("lease-{now_ms:016x}-{token:016x}"),
            session_id: request.session_id.clone(),
            owner_id: request.owner_id.clone(),
            holder_id: request.holder_id.clone(),
            fencing_token: token,
            expected_state_version: request.expected_state_version,
            expires_at_ms: now_ms.saturating_add(request.ttl_ms),
        };
        record.active_lease = Some(lease.clone());
        Ok(lease)
    }

    async fn renew(
        &self,
        request: &RenewLeaseRequest,
        now_ms: u64,
    ) -> Result<Lease, MetadataError> {
        let mut sessions = self.sessions.lock().await;
        let record = session_mut(&mut sessions, &request.lease)?;
        validate_active(record, &request.lease, now_ms)?;
        let mut renewed = request.lease.clone();
        renewed.expires_at_ms = now_ms.saturating_add(request.ttl_ms);
        record.active_lease = Some(renewed.clone());
        Ok(renewed)
    }

    async fn release(&self, lease: &Lease, now_ms: u64) -> Result<(), MetadataError> {
        let mut sessions = self.sessions.lock().await;
        let record = session_mut(&mut sessions, lease)?;
        validate_active(record, lease, now_ms)?;
        record.active_lease = None;
        Ok(())
    }

    async fn assert_lease(
        &self,
        lease: &Lease,
        require_current_version: bool,
        now_ms: u64,
    ) -> Result<(), MetadataError> {
        let sessions = self.sessions.lock().await;
        let record = session(&sessions, lease)?;
        validate_active(record, lease, now_ms)?;
        if require_current_version && record.current_version() != lease.expected_state_version {
            return Err(MetadataError::new(
                MetadataErrorKind::VersionConflict,
                "Lease was acquired for a different State version",
            ));
        }
        Ok(())
    }

    async fn commit_state(
        &self,
        lease: &Lease,
        expected_version: u64,
        state_ref: StateReference,
        now_ms: u64,
    ) -> Result<(), MetadataError> {
        let mut sessions = self.sessions.lock().await;
        let record = session_mut(&mut sessions, lease)?;
        validate_active(record, lease, now_ms)?;
        if record.current_version() != expected_version
            || lease.expected_state_version != expected_version
        {
            return Err(MetadataError::new(
                MetadataErrorKind::VersionConflict,
                "State compare-and-swap version no longer matches",
            ));
        }
        if state_ref.session_id != lease.session_id
            || state_ref.owner_id != lease.owner_id
            || state_ref.version != expected_version.saturating_add(1)
            || state_ref.fencing_token != Some(lease.fencing_token)
            || !state_ref.atomic
        {
            return Err(MetadataError::new(
                MetadataErrorKind::StateConflict,
                "Committed State identity, version, fencing token or atomic marker is invalid",
            ));
        }
        record.current_state = Some(state_ref);
        Ok(())
    }

    async fn current_state(
        &self,
        session_id: &str,
        owner_id: &str,
    ) -> Result<StateReference, MetadataError> {
        let sessions = self.sessions.lock().await;
        let record = sessions
            .get(session_id)
            .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
        ensure_owner(record, owner_id)?;
        record.current_state.clone().ok_or_else(|| {
            MetadataError::new(
                MetadataErrorKind::NotFound,
                "Session has no committed State",
            )
        })
    }
}

#[async_trait]
impl MetadataStore for PostgresMetadataStore {
    async fn acquire(
        &self,
        request: &AcquireLeaseRequest,
        now_ms: u64,
    ) -> Result<Lease, MetadataError> {
        let mut client = self.client.lock().await;
        let transaction = client.transaction().await.map_err(database_error)?;
        Self::lock_session(&transaction, &request.session_id).await?;
        let mut record = match Self::locked_record(&transaction, &request.session_id).await? {
            Some(record) => record,
            None if request.expected_state_version == 0 => SessionRecord {
                owner_id: request.owner_id.clone(),
                current_state: None,
                last_fencing_token: 0,
                active_lease: None,
            },
            None => {
                return Err(MetadataError::new(
                    MetadataErrorKind::VersionConflict,
                    "Unknown Session can only start at State version 0",
                ));
            }
        };
        ensure_owner(&record, &request.owner_id)?;
        if record.current_version() != request.expected_state_version {
            return Err(MetadataError::new(
                MetadataErrorKind::VersionConflict,
                format!(
                    "Expected State version {}, current version is {}",
                    request.expected_state_version,
                    record.current_version()
                ),
            ));
        }
        if record
            .active_lease
            .as_ref()
            .is_some_and(|lease| lease.expires_at_ms > now_ms)
        {
            return Err(MetadataError::new(
                MetadataErrorKind::LeaseHeld,
                "Session already has an unexpired writer Lease",
            ));
        }
        record.last_fencing_token = record.last_fencing_token.checked_add(1).ok_or_else(|| {
            MetadataError::new(
                MetadataErrorKind::StaleFence,
                "Fencing token space exhausted",
            )
        })?;
        let lease = Lease {
            contract_version: LEASE_CONTRACT_VERSION.into(),
            lease_id: format!("lease-{now_ms:016x}-{:016x}", record.last_fencing_token),
            session_id: request.session_id.clone(),
            owner_id: request.owner_id.clone(),
            holder_id: request.holder_id.clone(),
            fencing_token: record.last_fencing_token,
            expected_state_version: request.expected_state_version,
            expires_at_ms: now_ms.saturating_add(request.ttl_ms),
        };
        record.active_lease = Some(lease.clone());
        Self::save_record(&transaction, &request.session_id, &record, now_ms).await?;
        transaction.commit().await.map_err(database_error)?;
        Ok(lease)
    }

    async fn renew(
        &self,
        request: &RenewLeaseRequest,
        now_ms: u64,
    ) -> Result<Lease, MetadataError> {
        let mut client = self.client.lock().await;
        let transaction = client.transaction().await.map_err(database_error)?;
        Self::lock_session(&transaction, &request.lease.session_id).await?;
        let mut record = Self::locked_record(&transaction, &request.lease.session_id)
            .await?
            .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
        ensure_owner(&record, &request.lease.owner_id)?;
        validate_active(&record, &request.lease, now_ms)?;
        let mut renewed = request.lease.clone();
        renewed.expires_at_ms = now_ms.saturating_add(request.ttl_ms);
        record.active_lease = Some(renewed.clone());
        Self::save_record(&transaction, &request.lease.session_id, &record, now_ms).await?;
        transaction.commit().await.map_err(database_error)?;
        Ok(renewed)
    }

    async fn release(&self, lease: &Lease, now_ms: u64) -> Result<(), MetadataError> {
        let mut client = self.client.lock().await;
        let transaction = client.transaction().await.map_err(database_error)?;
        Self::lock_session(&transaction, &lease.session_id).await?;
        let mut record = Self::locked_record(&transaction, &lease.session_id)
            .await?
            .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
        ensure_owner(&record, &lease.owner_id)?;
        validate_active(&record, lease, now_ms)?;
        record.active_lease = None;
        Self::save_record(&transaction, &lease.session_id, &record, now_ms).await?;
        transaction.commit().await.map_err(database_error)
    }

    async fn assert_lease(
        &self,
        lease: &Lease,
        require_current_version: bool,
        now_ms: u64,
    ) -> Result<(), MetadataError> {
        let client = self.client.lock().await;
        let row = client
            .query_opt(
                "SELECT record FROM statepool_sessions WHERE session_id = $1",
                &[&lease.session_id],
            )
            .await
            .map_err(database_error)?
            .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
        let record = decode_record(row.get(0))?;
        ensure_owner(&record, &lease.owner_id)?;
        validate_active(&record, lease, now_ms)?;
        if require_current_version && record.current_version() != lease.expected_state_version {
            return Err(MetadataError::new(
                MetadataErrorKind::VersionConflict,
                "Lease was acquired for a different State version",
            ));
        }
        Ok(())
    }

    async fn commit_state(
        &self,
        lease: &Lease,
        expected_version: u64,
        state_ref: StateReference,
        now_ms: u64,
    ) -> Result<(), MetadataError> {
        let mut client = self.client.lock().await;
        let transaction = client.transaction().await.map_err(database_error)?;
        Self::lock_session(&transaction, &lease.session_id).await?;
        let mut record = Self::locked_record(&transaction, &lease.session_id)
            .await?
            .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
        ensure_owner(&record, &lease.owner_id)?;
        validate_active(&record, lease, now_ms)?;
        if record.current_version() != expected_version
            || lease.expected_state_version != expected_version
        {
            return Err(MetadataError::new(
                MetadataErrorKind::VersionConflict,
                "State compare-and-swap version no longer matches",
            ));
        }
        if state_ref.session_id != lease.session_id
            || state_ref.owner_id != lease.owner_id
            || state_ref.version != expected_version.saturating_add(1)
            || state_ref.fencing_token != Some(lease.fencing_token)
            || !state_ref.atomic
        {
            return Err(MetadataError::new(
                MetadataErrorKind::StateConflict,
                "Committed State identity, version, fencing token or atomic marker is invalid",
            ));
        }
        record.current_state = Some(state_ref);
        Self::save_record(&transaction, &lease.session_id, &record, now_ms).await?;
        transaction.commit().await.map_err(database_error)
    }

    async fn current_state(
        &self,
        session_id: &str,
        owner_id: &str,
    ) -> Result<StateReference, MetadataError> {
        let client = self.client.lock().await;
        let row = client
            .query_opt(
                "SELECT record FROM statepool_sessions WHERE session_id = $1",
                &[&session_id],
            )
            .await
            .map_err(database_error)?
            .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
        let record = decode_record(row.get(0))?;
        ensure_owner(&record, owner_id)?;
        record.current_state.ok_or_else(|| {
            MetadataError::new(
                MetadataErrorKind::NotFound,
                "Session has no committed State",
            )
        })
    }
}

fn ensure_owner(record: &SessionRecord, owner_id: &str) -> Result<(), MetadataError> {
    if record.owner_id != owner_id {
        return Err(MetadataError::new(
            MetadataErrorKind::OwnerMismatch,
            "Session is owned by a different principal",
        ));
    }
    Ok(())
}

fn session<'a>(
    sessions: &'a HashMap<String, SessionRecord>,
    lease: &Lease,
) -> Result<&'a SessionRecord, MetadataError> {
    let record = sessions
        .get(&lease.session_id)
        .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
    ensure_owner(record, &lease.owner_id)?;
    Ok(record)
}

fn session_mut<'a>(
    sessions: &'a mut HashMap<String, SessionRecord>,
    lease: &Lease,
) -> Result<&'a mut SessionRecord, MetadataError> {
    let record = sessions
        .get_mut(&lease.session_id)
        .ok_or_else(|| MetadataError::new(MetadataErrorKind::NotFound, "Unknown Session"))?;
    ensure_owner(record, &lease.owner_id)?;
    Ok(record)
}

fn validate_active(
    record: &SessionRecord,
    lease: &Lease,
    now_ms: u64,
) -> Result<(), MetadataError> {
    if lease.fencing_token < record.last_fencing_token {
        return Err(MetadataError::new(
            MetadataErrorKind::StaleFence,
            "Lease carries an obsolete fencing token",
        ));
    }
    let active = record.active_lease.as_ref().ok_or_else(|| {
        MetadataError::new(MetadataErrorKind::StaleFence, "No active writer Lease")
    })?;
    if active.lease_id != lease.lease_id
        || active.holder_id != lease.holder_id
        || active.fencing_token != lease.fencing_token
    {
        return Err(MetadataError::new(
            MetadataErrorKind::StaleFence,
            "Lease holder or fencing token does not match the active writer",
        ));
    }
    if active.expires_at_ms <= now_ms {
        return Err(MetadataError::new(
            MetadataErrorKind::LeaseExpired,
            "Writer Lease expired",
        ));
    }
    Ok(())
}
