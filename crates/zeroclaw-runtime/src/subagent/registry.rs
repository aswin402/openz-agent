//! Process-wide registry of running SubAgents.
//!
//! The registry is the canonical place where the primary model
//! discovers, monitors, stops, and changes the model of subagents it
//! has spawned. It is a *runtime cache* — the persistent identity of
//! each agent still lives in `Config::agents[<alias>]`, and the
//! registry only tracks in-flight runs that exist right now.
//!
//! ## Design notes
//!
//! - **State is per-parent, keyed by `SubAgentId`.** Each handle is a
//!   cheap `Arc<SubAgentHandle>` so the agent loop, the cancellation
//!   path, and the monitor tool can all reach the same status cell
//!   without locks-on-locks. The internal `status`/`current_step` are
//!   `parking_lot::Mutex` because the updates are short and frequent.
//! - **The registry does NOT duplicate config.** `Config::agents` is
//!   the source of truth for which aliases exist and what their
//!   defaults are. The handle holds only runtime-only fields
//!   (`started_at`, `current_step`, `last_update`, `prompt`,
//!   `provider/model`) — and even those last two can be re-resolved
//!   from `Config` if the handle outlives a config reload.
//! - **Cancellation is a `tokio_util::sync::CancellationToken`.** The
//!   agent loop is expected to `tokio::select!` on it the same way it
//!   already does for the parent tool's cancellation token. When the
//!   primary model calls `subagent_manage` with `action: "stop"`, the
//!   token is cancelled and the run aborts cooperatively.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Globally-unique identifier for a running subagent. Stable across
/// the run's lifetime — the primary model refers to it in every
/// follow-up `subagent_manage` call (status / stop / set_model).
pub type SubAgentId = String;

/// Lifecycle state of a running subagent. Transitions are:
///
/// `Pending` → `Running` → (`Completed` | `Failed` | `Cancelled`)
///
/// `set_model` while `Running` is allowed and the agent loop is
/// expected to read `pending_model_override` between turns. (See
/// `SubAgentManageTool::set_model_action` for the simplified
/// "stop + respawn" implementation that ships today.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubAgentStatus {
    /// Reserved a slot, not yet started.
    Pending,
    /// Actively executing the agent loop.
    Running,
    /// Finished successfully; `output` is populated.
    Completed,
    /// Finished with an error; `error` is populated.
    Failed,
    /// Primary model cancelled the run.
    Cancelled,
}

impl SubAgentStatus {
    /// Returns `true` for terminal states (the run will not transition
    /// further on its own; the registry entry can be GC'd).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            SubAgentStatus::Completed | SubAgentStatus::Failed | SubAgentStatus::Cancelled
        )
    }
}

/// Read-only snapshot of a handle's state. Returned by
/// `SubAgentRegistry::list_active` and `status` so the caller can
/// serialize the snapshot without holding a lock.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActiveSubAgentInfo {
    pub id: SubAgentId,
    pub parent_alias: String,
    pub target_alias: String,
    pub status: SubAgentStatus,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub prompt: String,
    pub started_at_ms: u64,
    pub last_update_ms: u64,
    pub current_step: String,
    pub error: Option<String>,
    pub output: Option<String>,
}

/// Live handle for a single subagent run. Cloning yields a new
/// reference to the same cell, so the registry, the running loop, and
/// the monitor tool all see the same status updates.
#[derive(Debug)]
pub struct SubAgentHandle {
    /// Globally-unique run id.
    pub id: SubAgentId,
    /// Alias of the agent that spawned this run (the primary model).
    pub parent_alias: String,
    /// Alias of the agent this run is *executing* as. Inherits from
    /// the parent by default but the primary model can spawn a child
    /// under a different alias (e.g. spawn a `coder` subagent from the
    /// `assistant` primary).
    pub target_alias: String,
    /// The prompt the subagent is working on.
    pub prompt: String,
    /// Cancellation token; cancels the in-flight run.
    pub cancel: CancellationToken,
    /// Resolved provider at spawn time (None = inherit parent).
    provider: Mutex<Option<String>>,
    /// Resolved model at spawn time (None = inherit parent).
    model: Mutex<Option<String>>,
    /// Live status; updated by the agent loop on every transition.
    status: Mutex<SubAgentStatus>,
    /// Free-text step description; updated as the run progresses
    /// (e.g. "calling tool: shell", "step 3/10", "waiting for tool").
    current_step: Mutex<String>,
    /// Wall-clock instant the run started.
    started_at: Instant,
    /// Wall-clock instant of the most recent status update.
    last_update: Mutex<Instant>,
    /// Final output (set on `Completed`).
    output: Mutex<Option<String>>,
    /// Error message (set on `Failed` or `Cancelled`).
    error: Mutex<Option<String>>,
}

impl SubAgentHandle {
    /// Construct a new handle. Called by `SubAgentSpawn` /
    /// `SpawnSubagentTool` once the run is committed (after the
    /// `SubAgentSpawn::build` validator returns `Ok`).
    pub fn new(
        parent_alias: impl Into<String>,
        target_alias: impl Into<String>,
        prompt: impl Into<String>,
        provider: Option<String>,
        model: Option<String>,
    ) -> Arc<Self> {
        let id = Uuid::new_v4().to_string();
        let now = Instant::now();
        Arc::new(Self {
            id,
            parent_alias: parent_alias.into(),
            target_alias: target_alias.into(),
            prompt: prompt.into(),
            provider: Mutex::new(provider),
            model: Mutex::new(model),
            cancel: CancellationToken::new(),
            status: Mutex::new(SubAgentStatus::Pending),
            current_step: Mutex::new("starting".to_string()),
            started_at: now,
            last_update: Mutex::new(now),
            output: Mutex::new(None),
            error: Mutex::new(None),
        })
    }

    /// Current status. Cheap; no allocation.
    pub fn status(&self) -> SubAgentStatus {
        *self.status.lock()
    }

    /// Update the status. No-op if the new status is terminal and the
    /// existing one is already terminal — protects against accidental
    /// double-finalization when both the loop and the cancellation
    /// path try to mark the run as `Completed`/`Failed`/`Cancelled`.
    pub fn set_status(&self, new: SubAgentStatus) {
        let mut s = self.status.lock();
        let current = *s;
        if current.is_terminal() && new.is_terminal() && current != new {
            // Don't overwrite a real terminal status with a different
            // one. Cancellation racing completion keeps whichever won.
            return;
        }
        *s = new;
        *self.last_update.lock() = Instant::now();
    }

    /// Free-text current step (e.g. tool name being called).
    pub fn set_step(&self, step: impl Into<String>) {
        *self.current_step.lock() = step.into();
        *self.last_update.lock() = Instant::now();
    }

    /// Set the final output. Idempotent — only the first call wins,
    /// so a later error path doesn't clobber a successful output.
    pub fn set_output(&self, output: impl Into<String>) {
        let mut slot = self.output.lock();
        if slot.is_none() {
            *slot = Some(output.into());
        }
        *self.last_update.lock() = Instant::now();
    }

    /// Set the final error. Idempotent on the same way as `set_output`.
    pub fn set_error(&self, error: impl Into<String>) {
        let mut slot = self.error.lock();
        if slot.is_none() {
            *slot = Some(error.into());
        }
        *self.last_update.lock() = Instant::now();
    }

    /// Resolved provider at spawn time (None = inherit parent).
    pub fn provider(&self) -> Option<String> {
        self.provider.lock().clone()
    }

    /// Resolved model at spawn time (None = inherit parent).
    pub fn model(&self) -> Option<String> {
        self.model.lock().clone()
    }

    /// Update the provider and model on a running handle. Called by
    /// the orchestrator after a successful provider resolution so the
    /// primary model can read the actual provider/model used via
    /// `subagent_manage` status.
    pub fn set_model(&self, provider: Option<String>, model: Option<String>) {
        *self.provider.lock() = provider;
        *self.model.lock() = model;
        *self.last_update.lock() = Instant::now();
    }

    /// Cancellation token — clone to share with the run loop.
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Snapshot for serialization. Reads all fields under their
    /// respective locks and clones. Use from monitor/list endpoints.
    pub fn snapshot(&self) -> ActiveSubAgentInfo {
        ActiveSubAgentInfo {
            id: self.id.clone(),
            parent_alias: self.parent_alias.clone(),
            target_alias: self.target_alias.clone(),
            status: self.status(),
            provider: self.provider.lock().clone(),
            model: self.model.lock().clone(),
            prompt: self.prompt.clone(),
            started_at_ms: millis_since(self.started_at),
            last_update_ms: millis_since(*self.last_update.lock()),
            current_step: self.current_step.lock().clone(),
            error: self.error.lock().clone(),
            output: self.output.lock().clone(),
        }
    }
}

fn millis_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

/// Process-wide registry. Backed by a `OnceLock` so the singleton is
/// initialized lazily on first use. Cheap to clone (it's an `Arc`).
#[derive(Debug, Clone, Default)]
pub struct SubAgentRegistry {
    inner: Arc<Mutex<HashMap<SubAgentId, Arc<SubAgentHandle>>>>,
}

impl SubAgentRegistry {
    /// The process-wide singleton. Use this from agent-loop / tool
    /// sites that don't have a more specific handle.
    pub fn global() -> Self {
        static GLOBAL: OnceLock<SubAgentRegistry> = OnceLock::new();
        GLOBAL
            .get_or_init(|| Self {
                inner: Arc::new(Mutex::new(HashMap::new())),
            })
            .clone()
    }

    /// Register a freshly-constructed handle. Returns the registered
    /// `SubAgentId` so the caller can refer to it. Idempotent on
    /// existing id (re-registration is a no-op).
    pub fn register(&self, handle: Arc<SubAgentHandle>) -> SubAgentId {
        let id = handle.id.clone();
        self.inner.lock().insert(id.clone(), handle);
        id
    }

    /// Remove a handle from the registry. Idempotent.
    pub fn deregister(&self, id: &str) -> Option<Arc<SubAgentHandle>> {
        self.inner.lock().remove(id)
    }

    /// Look up a handle by id.
    pub fn get(&self, id: &str) -> Option<Arc<SubAgentHandle>> {
        self.inner.lock().get(id).cloned()
    }

    /// Snapshot every *active* (non-terminal) subagent that is owned
    /// by `parent_alias`. The primary model's `subagent_manage.list`
    /// call passes its own alias here so it only sees its own children.
    pub fn list_active_for(&self, parent_alias: &str) -> Vec<ActiveSubAgentInfo> {
        self.inner
            .lock()
            .values()
            .filter(|h| h.parent_alias == parent_alias && !h.status().is_terminal())
            .map(|h| h.snapshot())
            .collect()
    }

    /// Total number of non-terminal runs owned by `parent_alias`.
    /// Used by the spawn gate so the primary model can't overwhelm
    /// itself with concurrent children.
    pub fn count_active_for(&self, parent_alias: &str) -> usize {
        self.inner
            .lock()
            .values()
            .filter(|h| h.parent_alias == parent_alias && !h.status().is_terminal())
            .count()
    }

    /// Snapshot every handle (any parent, any status) — useful for
    /// `/debug subagents` style introspection. Not gated.
    pub fn list_all(&self) -> Vec<ActiveSubAgentInfo> {
        self.inner.lock().values().map(|h| h.snapshot()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(parent: &str, target: &str) -> Arc<SubAgentHandle> {
        SubAgentHandle::new(parent, target, "do a thing", None, None)
    }

    #[test]
    fn register_and_get_round_trip() {
        let reg = SubAgentRegistry::default();
        let h = handle("primary", "coder");
        let id = reg.register(h.clone());
        assert_eq!(reg.get(&id).map(|h| h.id.clone()), Some(id.clone()));
    }

    #[test]
    fn list_active_for_filters_by_parent_and_terminal() {
        let reg = SubAgentRegistry::default();
        let primary_a = handle("primary_a", "coder");
        let primary_b = handle("primary_b", "coder");
        primary_b.set_status(SubAgentStatus::Completed);
        reg.register(primary_a);
        reg.register(primary_b);
        assert_eq!(reg.count_active_for("primary_a"), 1);
        assert_eq!(reg.count_active_for("primary_b"), 0);
        assert_eq!(reg.list_active_for("primary_a").len(), 1);
    }

    #[test]
    fn terminal_status_does_not_get_overwritten_by_different_terminal() {
        let h = handle("primary", "coder");
        h.set_status(SubAgentStatus::Completed);
        h.set_status(SubAgentStatus::Failed);
        assert_eq!(h.status(), SubAgentStatus::Completed);
    }

    #[test]
    fn output_is_first_wins() {
        let h = handle("primary", "coder");
        h.set_output("first");
        h.set_output("second");
        assert_eq!(h.output.lock().as_deref(), Some("first"));
    }

    #[test]
    fn cancel_token_is_unique_per_handle() {
        let h1 = handle("primary", "coder");
        let h2 = handle("primary", "coder");
        assert!(!h1.cancel_token().is_cancelled());
        h1.cancel_token().cancel();
        assert!(h1.cancel_token().is_cancelled());
        assert!(!h2.cancel_token().is_cancelled());
    }
}
