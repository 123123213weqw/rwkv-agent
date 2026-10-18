use std::future::Future;
use std::pin::Pin;

use crate::{
    CheckpointRef, ContinueRequest, ContinueResult, CreateRequest, ReleaseOutcome, RestoreRequest,
    SessionDescription, SessionHandle, SnapshotRequest, StateContractError,
};

pub type ProviderFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, StateContractError>> + Send + 'a>>;

pub trait StatefulInferenceProvider: Send + Sync + 'static {
    fn create(&self, request: CreateRequest) -> ProviderFuture<'_, SessionHandle>;
    fn continue_session(&self, request: ContinueRequest) -> ProviderFuture<'_, ContinueResult>;
    fn snapshot(&self, request: SnapshotRequest) -> ProviderFuture<'_, CheckpointRef>;
    fn restore(&self, request: RestoreRequest) -> ProviderFuture<'_, SessionHandle>;
    fn describe(
        &self,
        owner_id: String,
        session_handle: SessionHandle,
    ) -> ProviderFuture<'_, SessionDescription>;
    fn release(
        &self,
        owner_id: String,
        session_handle: SessionHandle,
    ) -> ProviderFuture<'_, ReleaseOutcome>;
}
