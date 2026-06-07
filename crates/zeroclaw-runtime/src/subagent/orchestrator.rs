//! Runtime-side SubAgent control plane.
//!
//! This module is the OpenZ-authority surface for subagents. The
//! model-facing `subagent_manage` tool exposes
//! `spawn`/`list`/`status`/`stop` to the primary LLM, but the *runtime*
//! — not the model — also needs to spawn subagents (image
//! pre-describe, cron-launched jobs, future deterministic routing).
//! [`SubAgentDispatcher`] is that runtime hook.
//!
//! It consolidates three responsibilities that were previously split
//! across `agentz.rs` and `tools/subagent_manage.rs`:
//!
//! 1. **Auto-spawn** with a deterministic 1 + 3 fallback chain — the
//!    target alias's `model_provider` is the primary; up to three
//!    entries from `model_fallbacks` are tried in order on failure.
//!    The first provider that returns a non-error wins.
//!
//! 2. **Stop** — cancel a running subagent by handle id via the
//!    [`SubAgentRegistry`]'s `CancellationToken`. The agent loop
//!    aborts cooperatively at its next checkpoint.
//!
//! 3. **Monitor** — a background task that auto-stops any subagent
//!    whose total runtime exceeds its per-agent timeout. This is the
//!    same `tokio::time::timeout` per-attempt budget surfaced as a
//!    process-wide watchdog so an agent loop that ignores its
//!    cancellation token (e.g. a hung tool call) still gets reaped.
//!
//! ## Single source of truth
//!
//! The handle / registry is the same one the model-facing
//! `subagent_manage` tool populates. The dispatcher does not duplicate
//! the registry — it just calls `SubAgentRegistry::global()`,
//! registers its own handle, and removes it on terminal transition.
//! The model can `list`/`status`/`stop` a runtime-spawned subagent
//! the same way it can one it spawned itself.

use crate::agent::loop_::AgentRunOverrides;
use crate::subagent::{ActiveSubAgentInfo, SubAgentHandle, SubAgentRegistry, SubAgentStatus};
use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use zeroclaw_config::schema::{AliasedAgentConfig, Config};

/// Per-target default timeout table. Matches the table in `agentz.rs`;
/// the monitor reads the same numbers so the two paths stay in sync.
fn default_timeout_for(target_alias: &str) -> Duration {
    match target_alias {
        "vision-agent" => Duration::from_secs(180),
        "research-agent" => Duration::from_secs(300),
        "openz-planagent" => Duration::from_secs(300),
        "coder" | "worker" => Duration::from_secs(900),
        "reviewer" => Duration::from_secs(600),
        "docs-agent" => Duration::from_secs(300),
        _ => Duration::from_secs(600),
    }
}

/// Result of a successful dispatch.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Registry handle id, returned so the caller (or the
    /// `subagent_manage` tool) can `status` / `stop` the run.
    pub handle_id: String,
    /// The subagent's final text output.
    pub output: String,
    /// Provider that succeeded (e.g. `"openrouter.default"`).
    pub provider_used: String,
    /// Model id that succeeded (resolved from the provider's config).
    pub model_used: String,
    /// Providers that were attempted and failed before the winner, in
    /// order. Empty when the primary succeeded on the first try.
    pub fallbacks_attempted: Vec<String>,
    /// Wall-clock duration of the entire dispatch, including all
    /// fallback attempts.
    pub duration_ms: u64,
}

/// The runtime-side subagent control plane. One per parent agent.
#[derive(Clone)]
pub struct SubAgentDispatcher {
    config: Arc<Config>,
    parent_alias: String,
    registry: SubAgentRegistry,
    /// Override the per-agent default timeout. `None` = use the table.
    timeout_override: Option<Duration>,
    /// Hard cap on the number of attempt providers (primary +
    /// fallbacks). The user explicitly asked for "3 fallbacks" so we
    /// cap the chain at 1 + 3 = 4. Pinned as a constant so the
    /// dispatcher and its tests agree on the shape.
    max_attempts: usize,
}

// --- Public accessors (used by planning, reflection, orchestrate) ---

impl SubAgentDispatcher {
    /// The application config this dispatcher was created from.
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    /// The agent alias that owns this dispatcher (the "parent").
    pub fn parent_alias(&self) -> &str {
        &self.parent_alias
    }
}

impl SubAgentDispatcher {
    /// Construct a dispatcher scoped to a parent agent. The parent
    /// alias is the `parent_alias` field on every spawned handle, so
    /// `list` / `status` / `stop` only see the parent's own children
    /// (matching the model-facing `subagent_manage` authority model).
    pub fn new(config: Arc<Config>, parent_alias: impl Into<String>) -> Self {
        Self {
            config,
            parent_alias: parent_alias.into(),
            registry: SubAgentRegistry::global(),
            timeout_override: None,
            max_attempts: 4, // 1 primary + 3 fallbacks
        }
    }

    /// Override the per-agent default timeout table. Useful for
    /// tests and for callers that already have a budget in mind.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_override = Some(timeout);
        self
    }

    /// Override the max attempts cap. Defaults to 4 (1 primary + 3
    /// fallbacks). Pinned for testability.
    #[must_use]
    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    /// Dispatch a prompt to `target_alias`. Resolves the alias's
    /// configured primary + up to 3 fallbacks, runs each in turn with
    /// the per-agent timeout, returns the first successful output.
    /// Registers a handle in the registry before the first attempt
    /// so a concurrent `stop` or monitor tick can find and cancel an
    /// in-flight run.
    pub async fn dispatch(
        &self,
        target_alias: &str,
        prompt: String,
        allowed_tools: Option<Vec<String>>,
    ) -> Result<DispatchResult> {
        let started = Instant::now();

        // Pre-flight: target alias must exist in the parent's config.
        // Unknown alias is a structured failure the caller can surface
        // — not a panic, not a recursion.
        let agent_cfg = self
            .config
            .agents
            .get(target_alias)
            .cloned()
            .with_context(|| format!("subagent_dispatch: unknown target alias {target_alias:?}"))?;

        let attempt_providers = self.attempt_providers(&agent_cfg);
        if attempt_providers.is_empty() {
            anyhow::bail!("subagent_dispatch: no model providers configured for {target_alias:?}");
        }

        // Register the handle BEFORE any provider attempt so a
        // concurrent `stop` or monitor tick can see and cancel an
        // in-flight run.
        let handle = SubAgentHandle::new(&self.parent_alias, target_alias, &prompt, None, None);
        let handle_id = self.registry.register(handle.clone());
        handle.set_status(SubAgentStatus::Running);
        handle.set_step(&format!("dispatching ({}/{})", 1, attempt_providers.len()));

        let timeout = self.timeout_for(target_alias);
        let mut fallbacks_attempted: Vec<String> = Vec::new();
        let mut last_error: Option<anyhow::Error> = None;

        for (idx, provider) in attempt_providers.iter().enumerate() {
            handle.set_step(&format!(
                "attempt {}/{}: {provider}",
                idx + 1,
                attempt_providers.len()
            ));

            match self
                .try_provider(
                    target_alias,
                    provider,
                    &prompt,
                    allowed_tools.clone(),
                    timeout,
                )
                .await
            {
                Ok((output, model)) => {
                    let provider_used = provider.clone();
                    let model_used = model;
                    handle.set_model(Some(provider.clone()), Some(model_used.clone()));
                    handle.set_output(&output);
                    handle.set_step("completed");
                    handle.set_status(SubAgentStatus::Completed);
                    self.registry.deregister(&handle_id);

                    return Ok(DispatchResult {
                        handle_id,
                        output,
                        provider_used,
                        model_used,
                        fallbacks_attempted,
                        duration_ms: started.elapsed().as_millis() as u64,
                    });
                }
                Err(e) => {
                    fallbacks_attempted.push(provider.clone());
                    last_error = Some(e);
                }
            }
        }

        let err = last_error.unwrap_or_else(|| {
            anyhow::anyhow!("subagent_dispatch: all providers failed for {target_alias:?}")
        });
        handle.set_error(&err.to_string());
        handle.set_status(SubAgentStatus::Failed);
        self.registry.deregister(&handle_id);
        Err(err)
    }

    /// Build the attempt chain for a target agent: the primary
    /// `model_provider` followed by up to `max_attempts - 1` entries
    /// from `model_fallbacks`. Deduplicated and emptied of blanks so
    /// a misconfigured agent doesn't retry the same provider twice.
    pub(crate) fn attempt_providers(&self, agent_cfg: &AliasedAgentConfig) -> Vec<String> {
        let mut chain: Vec<String> = Vec::with_capacity(self.max_attempts);
        let primary = agent_cfg.model_provider.as_str().trim();
        if !primary.is_empty() {
            chain.push(primary.to_string());
        }
        for fb in agent_cfg.model_fallbacks.iter() {
            let trimmed = fb.trim();
            if trimmed.is_empty() {
                continue;
            }
            if chain.iter().any(|p| p == trimmed) {
                continue;
            }
            chain.push(trimmed.to_string());
            if chain.len() >= self.max_attempts {
                break;
            }
        }
        chain
    }

    async fn try_provider(
        &self,
        target_alias: &str,
        provider: &str,
        prompt: &str,
        allowed_tools: Option<Vec<String>>,
        timeout: Duration,
    ) -> Result<(String, String)> {
        // Build a child config where the target agent's
        // `model_provider` is pinned to this attempt's provider. The
        // agent loop resolves the model id from there.
        let mut try_config = (*self.config).clone();
        if let Some(cfg) = try_config.agents.get_mut(target_alias) {
            cfg.model_provider = provider.into();
        }

        let overrides = AgentRunOverrides {
            is_subagent: true,
            ..Default::default()
        };

        let run_fut = crate::agent::run_boxed(
            try_config,
            target_alias,
            Some(prompt.to_string()),
            None,
            None,
            None,
            vec![],
            false, // non-interactive — subagents never prompt
            None,
            allowed_tools,
            overrides,
        );

        let res = tokio::time::timeout(timeout, run_fut)
            .await
            .map_err(|_| anyhow::anyhow!("timeout after {timeout:?}"))??;

        // Resolve the actual model id used so the caller knows what
        // served the response. Read off the *original* config (not
        // the pinned try_config) so the model id matches the
        // canonical entry, not the cloned one.
        let model = self
            .config
            .model_provider_for_agent(target_alias)
            .and_then(|cfg| cfg.model.clone())
            .unwrap_or_default();

        Ok((res, model))
    }

    fn timeout_for(&self, target_alias: &str) -> Duration {
        self.timeout_override
            .unwrap_or_else(|| default_timeout_for(target_alias))
    }

    /// Stop a running subagent by handle id. Idempotent — calling on
    /// an already-stopped or already-finished handle is a no-op.
    pub fn stop(&self, handle_id: &str, reason: &str) -> Result<()> {
        match self.registry.get(handle_id) {
            Some(h) if h.parent_alias == self.parent_alias => {
                if !h.cancel_token().is_cancelled() {
                    h.cancel_token().cancel();
                }
                h.set_step(&format!("cancelled by {}: {reason}", self.parent_alias));
                Ok(())
            }
            Some(_) => Err(anyhow::anyhow!(
                "subagent_dispatch.stop: {handle_id} not owned by {}",
                self.parent_alias
            )),
            None => Err(anyhow::anyhow!(
                "subagent_dispatch.stop: {handle_id} not found (already finished?)"
            )),
        }
    }

    /// Get a snapshot of a subagent's state. Returns `None` when the
    /// handle is unknown OR not owned by this dispatcher.
    pub fn status(&self, handle_id: &str) -> Option<ActiveSubAgentInfo> {
        self.registry
            .get(handle_id)
            .filter(|h| h.parent_alias == self.parent_alias)
            .map(|h| h.snapshot())
    }

    /// List all active (non-terminal) subagents owned by the parent.
    pub fn list(&self) -> Vec<ActiveSubAgentInfo> {
        self.registry.list_active_for(&self.parent_alias)
    }

    /// Spawn a background monitor task. The task watches the registry
    /// every `tick_interval` and auto-stops any subagent whose total
    /// runtime exceeds its per-agent timeout. Exits when `shutdown`
    /// flips to `true`.
    pub fn spawn_monitor(
        self: Arc<Self>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let tick_interval = Duration::from_secs(15);
            let mut interval = tokio::time::interval(tick_interval);
            // Skip the first immediate tick — handles created at
            // startup need at least one interval to elapse before
            // we'd consider them stuck.
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        self.tick_monitor();
                    }
                    _ = shutdown.changed() => {
                        if *shutdown.borrow() {
                            break;
                        }
                    }
                }
            }
        })
    }

    fn tick_monitor(&self) {
        for info in self.list() {
            let timeout = self.timeout_for(&info.target_alias);
            let max_ms = timeout.as_millis() as u64;
            if info.started_at_ms <= max_ms {
                continue;
            }
            let Some(h) = self.registry.get(&info.id) else {
                continue;
            };
            if h.cancel_token().is_cancelled() {
                continue;
            }
            h.cancel_token().cancel();
            h.set_step(&format!(
                "auto-stopped by monitor: exceeded {:?} budget",
                timeout
            ));
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Timeout,)
                    .with_outcome(::zeroclaw_log::EventOutcome::Success)
                    .with_attrs(::serde_json::json!({
                        "subagent_id": &info.id,
                        "target": &info.target_alias,
                        "elapsed_ms": info.started_at_ms,
                        "timeout_ms": max_ms,
                        "parent": &self.parent_alias,
                    })),
                "subagent monitor: auto-stopped run exceeding max duration"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};

    fn base_config() -> Config {
        let mut config = Config::default();
        config
            .risk_profiles
            .insert("default".to_string(), RiskProfileConfig::default());
        config
    }

    fn agent_with_fallbacks(primary: &str, fallbacks: Vec<&str>) -> AliasedAgentConfig {
        AliasedAgentConfig {
            model_provider: primary.into(),
            model_fallbacks: fallbacks.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn attempt_providers_includes_primary_then_fallbacks() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let cfg = agent_with_fallbacks(
            "openrouter.default",
            vec!["google.default", "openai.default"],
        );
        assert_eq!(
            dispatcher.attempt_providers(&cfg),
            vec![
                "openrouter.default".to_string(),
                "google.default".to_string(),
                "openai.default".to_string(),
            ]
        );
    }

    #[test]
    fn attempt_providers_caps_at_max_attempts() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha").with_max_attempts(3);
        let cfg = agent_with_fallbacks(
            "openrouter.default",
            vec![
                "google.default",
                "openai.default",
                "anthropic.default",
                "groq.default",
            ],
        );
        assert_eq!(dispatcher.attempt_providers(&cfg).len(), 3);
        assert_eq!(
            dispatcher.attempt_providers(&cfg),
            vec![
                "openrouter.default".to_string(),
                "google.default".to_string(),
                "openai.default".to_string(),
            ]
        );
    }

    #[test]
    fn attempt_providers_dedupes_primary_listed_again_in_fallbacks() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let cfg = agent_with_fallbacks(
            "openrouter.default",
            vec!["openrouter.default", "google.default"],
        );
        assert_eq!(
            dispatcher.attempt_providers(&cfg),
            vec![
                "openrouter.default".to_string(),
                "google.default".to_string()
            ]
        );
    }

    #[test]
    fn attempt_providers_skips_blank_entries() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let cfg = agent_with_fallbacks("openrouter.default", vec!["", "   ", "google.default"]);
        assert_eq!(
            dispatcher.attempt_providers(&cfg),
            vec![
                "openrouter.default".to_string(),
                "google.default".to_string()
            ]
        );
    }

    #[test]
    fn attempt_providers_handles_missing_primary() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let cfg = AliasedAgentConfig {
            model_provider: "".into(),
            model_fallbacks: vec!["google.default".to_string()],
            ..Default::default()
        };
        assert_eq!(
            dispatcher.attempt_providers(&cfg),
            vec!["google.default".to_string()]
        );
    }

    #[test]
    fn attempt_providers_with_no_providers_returns_empty() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let cfg = AliasedAgentConfig::default();
        assert!(dispatcher.attempt_providers(&cfg).is_empty());
    }

    #[test]
    fn timeout_for_falls_back_to_default_table() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        assert_eq!(
            dispatcher.timeout_for("vision-agent"),
            Duration::from_secs(180)
        );
        assert_eq!(dispatcher.timeout_for("coder"), Duration::from_secs(900));
        assert_eq!(
            dispatcher.timeout_for("unknown-agent"),
            Duration::from_secs(600)
        );
    }

    #[test]
    fn timeout_for_respects_override() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha")
            .with_timeout(Duration::from_secs(42));
        assert_eq!(
            dispatcher.timeout_for("vision-agent"),
            Duration::from_secs(42)
        );
        assert_eq!(dispatcher.timeout_for("unknown"), Duration::from_secs(42));
    }

    #[test]
    fn stop_rejects_handle_owned_by_other_parent() {
        let config = base_config();
        // Seed another parent's handle in the global registry.
        let other = SubAgentHandle::new("bravo", "coder", "other work", None, None);
        let other_id = SubAgentRegistry::global().register(other);
        // Alpha can't stop it.
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let err = dispatcher
            .stop(&other_id, "test")
            .expect_err("cross-parent stop must be rejected");
        assert!(
            err.to_string().contains("not owned by"),
            "expected ownership error, got: {err}"
        );
        // Cleanup
        SubAgentRegistry::global().deregister(&other_id);
    }

    #[test]
    fn stop_rejects_unknown_handle() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let err = dispatcher
            .stop("00000000-0000-0000-0000-000000000000", "test")
            .expect_err("unknown id must error");
        assert!(
            err.to_string().contains("not found"),
            "expected not-found error, got: {err}"
        );
    }

    #[test]
    fn status_returns_none_for_unknown_or_foreign_handle() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        assert!(
            dispatcher
                .status("00000000-0000-0000-0000-000000000000")
                .is_none()
        );

        let foreign = SubAgentHandle::new("bravo", "coder", "x", None, None);
        let foreign_id = SubAgentRegistry::global().register(foreign);
        assert!(dispatcher.status(&foreign_id).is_none());
        SubAgentRegistry::global().deregister(&foreign_id);
    }

    #[test]
    fn list_filters_by_parent_alias() {
        let config = base_config();
        let alpha = SubAgentHandle::new("alpha", "coder", "a1", None, None);
        let bravo = SubAgentHandle::new("bravo", "coder", "b1", None, None);
        let alpha_id = SubAgentRegistry::global().register(alpha);
        let _bravo_id = SubAgentRegistry::global().register(bravo);

        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let active = dispatcher.list();
        let ids: HashSet<String> = active.iter().map(|h| h.id.clone()).collect();
        assert!(ids.contains(&alpha_id));
        assert!(!ids.iter().any(|i| {
            SubAgentRegistry::global()
                .get(i)
                .map_or(false, |h| h.parent_alias == "bravo")
        }));

        // Cleanup
        SubAgentRegistry::global().deregister(&alpha_id);
        SubAgentRegistry::global().deregister(&_bravo_id);
    }

    #[tokio::test]
    async fn dispatch_unknown_alias_returns_error() {
        let config = base_config();
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let err = dispatcher
            .dispatch("does-not-exist", "hello".into(), None)
            .await
            .expect_err("unknown alias must error");
        assert!(
            err.to_string().contains("unknown target alias"),
            "expected unknown-alias error, got: {err}"
        );
    }

    #[tokio::test]
    async fn dispatch_no_providers_returns_error() {
        let mut config = base_config();
        config.agents.insert(
            "empty-agent".to_string(),
            AliasedAgentConfig {
                risk_profile: "default".to_string(),
                ..Default::default()
            },
        );
        let dispatcher = SubAgentDispatcher::new(Arc::new(config), "alpha");
        let err = dispatcher
            .dispatch("empty-agent", "hello".into(), None)
            .await
            .expect_err("no providers must error");
        assert!(
            err.to_string().contains("no model providers"),
            "expected no-providers error, got: {err}"
        );
    }

    #[test]
    fn monitor_cancels_oversized_runs() {
        // Manually register a handle and age it past the timeout by
        // backdating `last_update`. The monitor reads elapsed time
        // from `last_update_ms` / `started_at_ms`, both of which are
        // computed off the handle's `Instant`s, so we can't
        // time-travel — but we can use a tiny timeout_override so
        // any in-flight handle exceeds it on the first tick.
        let config = base_config();
        let dispatcher = Arc::new(
            SubAgentDispatcher::new(Arc::new(config), "alpha")
                .with_timeout(Duration::from_millis(1)),
        );
        let handle = SubAgentHandle::new("alpha", "coder", "stuck", None, None);
        let id = SubAgentRegistry::global().register(handle.clone());
        // Sleep just over the 1ms budget so `started_at_ms > 1`.
        std::thread::sleep(Duration::from_millis(10));
        dispatcher.tick_monitor();
        // Monitor should have cancelled the token.
        assert!(handle.cancel_token().is_cancelled());
        SubAgentRegistry::global().deregister(&id);
    }

    #[test]
    fn monitor_skips_handles_below_threshold() {
        let config = base_config();
        let dispatcher = Arc::new(
            SubAgentDispatcher::new(Arc::new(config), "alpha")
                .with_timeout(Duration::from_secs(60)),
        );
        let handle = SubAgentHandle::new("alpha", "coder", "fresh", None, None);
        let id = SubAgentRegistry::global().register(handle.clone());
        dispatcher.tick_monitor();
        // Fresh handle — token must NOT be cancelled.
        assert!(!handle.cancel_token().is_cancelled());
        SubAgentRegistry::global().deregister(&id);
    }

    #[test]
    fn monitor_skips_foreign_parents() {
        // Tick monitor for "alpha" — handles owned by "bravo" should
        // not be touched even if they're over budget (they aren't
        // visible to alpha's `list()` anyway, so this is mostly a
        // sanity test of the parent filter).
        let config = base_config();
        let dispatcher = Arc::new(
            SubAgentDispatcher::new(Arc::new(config), "alpha")
                .with_timeout(Duration::from_millis(1)),
        );
        let bravo = SubAgentHandle::new("bravo", "coder", "stuck", None, None);
        let id = SubAgentRegistry::global().register(bravo.clone());
        std::thread::sleep(Duration::from_millis(10));
        dispatcher.tick_monitor();
        assert!(!bravo.cancel_token().is_cancelled());
        SubAgentRegistry::global().deregister(&id);
    }
}
