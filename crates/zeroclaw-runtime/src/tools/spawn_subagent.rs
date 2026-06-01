//! Agent-loop tool that spawns an ephemeral SubAgent inheriting the
//! parent's identity, security policy, and memory allowlist, runs a
//! focused prompt, and returns the response. Cron's `JobType::Agent`
//! dispatch is the other SubAgent spawn site; both funnel through
//! [`crate::subagent::SubAgentSpawn`] so permission inheritance,
//! tracing-span shape, and audit attribution stay uniform.

use crate::agent::loop_::AgentRunOverrides;
use crate::subagent::{SubAgentOverrides, SubAgentSpawn};
use crate::tools::McpRegistry;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::schema::Config;
use zeroclaw_log::scope;

/// Spawn an ephemeral SubAgent that inherits the parent agent's
/// identity and runs a focused prompt under the same alias.
pub struct SpawnSubagentTool {
    config: Arc<Config>,
    parent_alias: String,
    /// Shared MCP registry from the parent agent — passed so the subagent
    /// doesn't re-spawn MCP server processes.
    mcp_registry: Option<Arc<McpRegistry>>,
    /// `true` when this tool is registered inside a run that is itself
    /// a SubAgent. Triggers a depth-1 cap refusal in `execute` before
    /// any spawn work happens. Set by the agent loop from
    /// `AgentRunOverrides.is_subagent` at registry construction time.
    is_subagent_caller: bool,
}

impl SpawnSubagentTool {
    pub fn new(
        config: Arc<Config>,
        parent_alias: impl Into<String>,
        mcp_registry: Option<Arc<McpRegistry>>,
    ) -> Self {
        Self {
            config,
            parent_alias: parent_alias.into(),
            mcp_registry,
            is_subagent_caller: false,
        }
    }

    /// Mark this tool instance as belonging to a SubAgent's tool
    /// registry. Triggers the depth-1 refusal on `execute`. The agent
    /// loop sets this from `AgentRunOverrides.is_subagent`.
    #[must_use]
    pub fn with_subagent_caller(mut self, is_subagent_caller: bool) -> Self {
        self.is_subagent_caller = is_subagent_caller;
        self
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};

static ACTIVE_SUBAGENTS: AtomicUsize = AtomicUsize::new(0);

#[async_trait]
impl Tool for SpawnSubagentTool {
    fn name(&self) -> &str {
        "spawn_subagent"
    }

    fn description(&self) -> &str {
        "Spawn an ephemeral SubAgent that inherits this agent's identity, \
         security policy, and memory allowlist. The SubAgent runs the supplied \
         prompt to completion under the parent's permissions envelope and \
         returns its response. Use for focused subtasks (research lookup, \
         multi-step reasoning, etc.) that should not pollute this agent's main \
         conversation history. Cost-aware: each SubAgent run is a full agent \
         loop and consumes provider tokens."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question for the SubAgent. Be specific and self-contained — the SubAgent does not see this conversation's history."
                },
                "allowed_mcp_servers": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Optional list of MCP server names that this subagent is allowed to access (e.g. ['wikipedia', 'git']). If omitted, the subagent gets access to all parent MCP tools."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        // Enforce the concurrency limit: max 2 active subagents spawned simultaneously
        let active = ACTIVE_SUBAGENTS.load(Ordering::SeqCst);
        if active >= 2 {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "spawn_subagent: refused — maximum concurrent subagent limit reached (active: {active}/2)"
                )),
            });
        }

        // Increment active subagents count and use a guard to automatically decrement on exit
        ACTIVE_SUBAGENTS.fetch_add(1, Ordering::SeqCst);
        struct ActiveGuard;
        impl Drop for ActiveGuard {
            fn drop(&mut self) {
                ACTIVE_SUBAGENTS.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let _guard = ActiveGuard;

        // Depth-1 cap: a SubAgent may not spawn its own subagents.
        // The caller-side flag is set at registry construction time
        // from `AgentRunOverrides.is_subagent`, so the refusal fires
        // before any spawn work and before the risk_profile gate.
        if self.is_subagent_caller {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "spawn_subagent: a subagent may not spawn its own subagents (depth-1 cap)"
                        .into(),
                ),
            });
        }

        // risk_profile gate: a parent's risk_profile.allowed_tools that
        // omits `spawn_subagent` must refuse pre-spawn. The agent-loop
        // dispatch filter (apply_policy_tool_filter) already drops the
        // tool from the registry when the policy excludes it, but this
        // tool also runs from cron and other registry construction
        // sites that don't currently apply the filter; refuse here so
        // the gate is honored everywhere the tool is reachable.
        let risk_profile = self.config.risk_profile_for_agent(&self.parent_alias);
        if let Some(rp) = risk_profile {
            let excluded = rp.excluded_tools.iter().any(|t| t == "spawn_subagent");
            let allowed_when_listed = rp.allowed_tools.is_empty()
                || rp.allowed_tools.iter().any(|t| t == "spawn_subagent");
            if excluded || !allowed_when_listed {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "spawn_subagent: refused — agent '{}' risk_profile does not list spawn_subagent in allowed_tools",
                        self.parent_alias
                    )),
                });
            }
        }

        // Argument validation surfaces as a structured `ToolResult`
        // failure (matching the unknown-parent and run-failure shapes
        // below) so the agent loop receives a uniform "tool reported
        // failure" signal regardless of which step rejected the call.
        let prompt = match args
            .get("prompt")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(p) => p.to_string(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing or empty 'prompt' parameter".into()),
                });
            }
        };

        let allowed_mcp_servers: Option<std::collections::HashSet<String>> = args
            .get("allowed_mcp_servers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|val| val.as_str().map(|s| s.to_string()))
                    .collect()
            });

        let subagent_ctx = match SubAgentSpawn::for_agent(&self.config, &self.parent_alias)
            .and_then(|spawn| spawn.build(SubAgentOverrides::default()))
        {
            Ok(ctx) => ctx,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("subagent spawn failed: {e:#}")),
                });
            }
        };

        let run_id = uuid::Uuid::new_v4().to_string();

        let temperature: Option<f64> = self
            .config
            .model_provider_for_agent(&self.parent_alias)
            .and_then(|e| e.temperature);
        let session_path = std::path::PathBuf::from(format!("subagent-{run_id}"));

        // Pass the validated SubAgent context as run-time overrides so
        // the subset-confirmed policy reaches the agent loop instead
        // of being silently re-derived from config. `is_subagent: true`
        // marks the child run so its own SpawnSubagentTool is
        // registered with the depth-cap refusal armed.
        let run_overrides = AgentRunOverrides {
            security: Some(subagent_ctx.policy.clone()),
            memory: None,
            is_subagent: true,
            tui_sender: None,
            mcp_registry: self.mcp_registry.clone(),
            allowed_mcp_servers,
        };
        let parent_alias = subagent_ctx.parent_alias.clone();
        let run_result = Box::pin(scope!(
            agent_alias: parent_alias,
            session_key: run_id,
            =>
            crate::agent::run(
                (*self.config).clone(),
                &self.parent_alias,
                Some(prompt),
                None,
                None,
                temperature,
                vec![],
                false,
                Some(session_path),
                None,
                run_overrides,
            )
        ))
        .await;

        match run_result {
            Ok(response) => Ok(ToolResult {
                success: true,
                output: if response.trim().is_empty() {
                    "subagent completed without output".to_string()
                } else {
                    response
                },
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("subagent run failed: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};

    fn config_with_agent(alias: &str) -> Config {
        let mut config = Config::default();
        config
            .risk_profiles
            .insert("default".to_string(), RiskProfileConfig::default());
        config.agents.insert(
            alias.to_string(),
            AliasedAgentConfig {
                risk_profile: "default".to_string(),
                ..AliasedAgentConfig::default()
            },
        );
        config
    }

    #[tokio::test]
    async fn empty_or_missing_prompt_is_rejected() {
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        for args in [json!({}), json!({ "prompt": "   " })] {
            let result = tool
                .execute(args)
                .await
                .expect("execute returns Ok with structured failure");
            assert!(!result.success);
            assert!(
                result
                    .error
                    .as_deref()
                    .unwrap_or_default()
                    .contains("prompt"),
                "expected prompt-validation error, got: {:?}",
                result.error
            );
        }
    }

    #[tokio::test]
    async fn unknown_parent_alias_surfaces_spawn_failure() {
        // Parent alias that is not configured: SubAgentSpawn::for_agent
        // returns Err, the tool reports a structured spawn failure
        // (no panic, no recursion attempt).
        let tool = SpawnSubagentTool::new(Arc::new(Config::default()), "missing-alpha", None);
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok with structured failure");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("subagent spawn failed"),
            "expected spawn-failure error, got: {:?}",
            result.error
        );
    }

    // ── Depth-1 cap: subagent may not spawn its own subagent ──

    #[tokio::test]
    async fn refuses_recursive_spawn_when_caller_is_subagent() {
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha", None)
            .with_subagent_caller(true);
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok with structured failure");
        assert!(!result.success);
        let err = result.error.as_deref().unwrap_or_default();
        assert!(
            err.contains("subagent") && err.contains("depth"),
            "expected depth-cap refusal mentioning subagent + depth, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn allows_top_level_spawn_when_caller_is_not_subagent() {
        // The top-level path may still fail later for unrelated reasons
        // (e.g. no model provider configured in this minimal harness),
        // but it MUST NOT trip the depth-cap refusal. Pin that the
        // depth-cap error is absent.
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha", None)
            .with_subagent_caller(false);
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok");
        let err = result.error.as_deref().unwrap_or_default();
        assert!(
            !(err.contains("subagent") && err.contains("depth")),
            "top-level caller must not see the depth-cap refusal, got: {err:?}"
        );
    }

    // ── risk_profile.allowed_tools gates spawn_subagent ──

    fn config_with_allowed_tools(alias: &str, allowed_tools: Vec<String>) -> Config {
        let mut config = Config::default();
        config.risk_profiles.insert(
            "default".to_string(),
            RiskProfileConfig {
                allowed_tools,
                ..RiskProfileConfig::default()
            },
        );
        config.agents.insert(
            alias.to_string(),
            AliasedAgentConfig {
                risk_profile: "default".to_string(),
                ..AliasedAgentConfig::default()
            },
        );
        config
    }

    #[tokio::test]
    async fn refuses_when_risk_profile_excludes_spawn_subagent() {
        // Parent's risk_profile.allowed_tools omits "spawn_subagent" —
        // the tool itself refuses pre-spawn so the dispatch-site filter
        // doesn't have to be the only line of defense.
        let config = config_with_allowed_tools("alpha", vec!["shell".into()]);
        let tool = SpawnSubagentTool::new(Arc::new(config), "alpha", None);
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok with structured failure");
        assert!(!result.success);
        let err = result.error.as_deref().unwrap_or_default();
        assert!(
            err.contains("risk_profile") && err.contains("spawn_subagent"),
            "expected risk_profile-gate refusal naming spawn_subagent, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn admits_when_risk_profile_lists_spawn_subagent() {
        // When the parent's risk_profile.allowed_tools explicitly lists
        // spawn_subagent, the tool does NOT short-circuit on the gate.
        // It may still fail later for unrelated reasons; pin only that
        // the gate refusal is absent.
        let config =
            config_with_allowed_tools("alpha", vec!["spawn_subagent".into(), "shell".into()]);
        let tool = SpawnSubagentTool::new(Arc::new(config), "alpha", None);
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok");
        let err = result.error.as_deref().unwrap_or_default();
        assert!(
            !(err.contains("risk_profile") && err.contains("spawn_subagent")),
            "spawn_subagent in allowed_tools must not trigger the gate refusal, got: {err:?}"
        );
    }

    // ── Cron path stays depth-0: AgentRunOverrides::default() ──
    //
    // The cron `JobType::Agent` site constructs `AgentRunOverrides`
    // without explicit `is_subagent`, so a `false` Default is the
    // load-bearing invariant. A future refactor flipping the default
    // would silently turn every cron-launched agent into a depth-1
    // subagent and break recursive-spawn guarantees from the other
    // direction. Pin the default explicitly.

    #[test]
    fn agent_run_overrides_default_is_top_level() {
        use crate::agent::loop_::AgentRunOverrides;
        let overrides = AgentRunOverrides::default();
        assert!(
            !overrides.is_subagent,
            "AgentRunOverrides::default().is_subagent must be false so cron paths inherit a top-level shape"
        );
    }

    // ── Tool : Attributable contract ──────────────────────────
    //
    // Every Tool impl carries a structured role + alias the same way
    // channels do, so log emissions, audit traces, and ops banners can
    // tag tool activity with the same `<kind>.<alias>` composite shape
    // they use for the rest of the runtime. The trait supertrait is
    // the load-bearing piece: a `&dyn Tool` must coerce to a
    // `&dyn Attributable` automatically. Without `Tool: Attributable`
    // the line below does not compile.

    #[test]
    fn spawn_subagent_dyn_tool_implements_attributable() {
        use zeroclaw_api::attribution::{Attributable, Role, ToolKind};

        let tool: Box<dyn Tool> = Box::new(SpawnSubagentTool::new(
            Arc::new(config_with_agent("alpha")),
            "alpha",
            None,
        ));
        assert_eq!(
            Attributable::role(tool.as_ref()),
            Role::Tool(ToolKind::SpawnSubagent),
            "SpawnSubagentTool must surface its kind through the Tool trait object"
        );
        assert!(
            !Attributable::alias(tool.as_ref()).is_empty(),
            "Attributable::alias on a Tool must be non-empty so composite keys never produce `.<bare>`"
        );
    }

    #[tokio::test]
    async fn enforces_concurrency_limit() {
        ACTIVE_SUBAGENTS.store(2, Ordering::SeqCst);
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("maximum concurrent subagent limit reached"),
            "expected concurrency refusal, got: {:?}",
            result.error
        );
        ACTIVE_SUBAGENTS.store(0, Ordering::SeqCst);
    }
}
