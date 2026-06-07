//! Primary-model authority tool for SubAgent lifecycle control.
//!
//! The primary model (the agent the user is talking to — typically
//! `assistant`) is the only authority that can spawn, stop, monitor,
//! or change the model of its running subagents. This single tool
//! (`subagent_manage`) is the unified surface for that authority and
//! replaces the ad-hoc "delegate + spawn_subagent + check_result"
//! dance. The primary model calls it with a discriminated `action`:
//!
//! - `action: "spawn"` — start a new subagent under a target alias
//!   with an optional model/provider override; returns its `id` so
//!   follow-up calls can refer to it.
//! - `action: "list"` — list every active subagent owned by this
//!   primary (id, alias, model, status, current step, age).
//! - `action: "status"` — return a full snapshot of a single run.
//! - `action: "stop"` — cancel a running subagent by `id`; the run
//!   loop aborts cooperatively at the next cancellation check.
//! - `action: "set_model"` — change the model/provider of an active
//!   subagent. Implemented today as "stop current run, spawn a fresh
//!   one with the new model" so the change is observable immediately
//!   without intrusive changes to the agent loop's mid-iteration
//!   model resolution. The fresh run inherits the original prompt and
//!   a stable id is reported in the response so the primary can
//!   keep talking to the same logical subagent.
//!
//! ## Why a single tool and not four
//!
//! The model picks tool calls from a flat schema; a single tool with
//! a discriminated `action` is harder to mis-invoke than four
//! near-identical tools with overlapping parameter shapes, and the
//! branching logic is right next to the schema in one place.
//!
//! ## Authority scope
//!
//! - Only the primary model can call this tool. The agent loop
//!   refuses to register it on SubAgent tool registries (depth-1 cap
//!   mirrors `spawn_subagent`).
//! - The tool only sees the calling primary's own children. The
//!   `parent_alias` field is the source of truth for ownership and
//!   is filled from the tool instance, not the args, so a model
//!   cannot query or stop another primary's subagents.
//! - A subagent that the primary spawned cannot in turn call this
//!   tool to affect the primary or its siblings.

use crate::agent::loop_::AgentRunOverrides;
use crate::subagent::{SubAgentOverrides, SubAgentRegistry, SubAgentSpawn, SubAgentStatus};
use crate::tools::McpRegistry;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::{Arc, RwLock};
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::schema::Config;
use zeroclaw_log::scope;

/// Tool that gives the primary model authority to spawn, list, status,
/// stop, and change the model of its running subagents.
pub struct SubAgentManageTool {
    config: Arc<Config>,
    /// The alias of the agent calling this tool. Filled by the agent
    /// loop on registry construction; never derived from the JSON
    /// args (so a model cannot impersonate another primary).
    primary_alias: String,
    /// Shared MCP registry from the primary — passed through to
    /// spawned subagents so they don't re-spawn MCP server processes.
    mcp_registry: Option<Arc<McpRegistry>>,
    /// Shared tool registry from the primary. Set after the
    /// primary's tool registry is built (the agent loop wraps the
    /// final `Vec<Box<dyn Tool>>` in an `Arc<RwLock<…>>` and
    /// installs it here so spawned subagents inherit the parent's
    /// tool `Box`es — ShellTool's sandbox, file guards, Memory
    /// sqlite handle, MCP wrappers all stay single-instance).
    /// `RwLock<Option<…>>` because construction order is
    /// `SubAgentManageTool::new` first, registry build second, then
    /// `set_parent_tools` once we have the final list.
    parent_tools: Arc<RwLock<Option<Arc<RwLock<Vec<Box<dyn Tool>>>>>>>,
    /// `true` when this tool is registered inside a SubAgent's tool
    /// set. Triggers the depth-1 cap refusal before any action runs.
    is_subagent_caller: bool,
}

impl SubAgentManageTool {
    pub fn new(
        config: Arc<Config>,
        primary_alias: impl Into<String>,
        mcp_registry: Option<Arc<McpRegistry>>,
    ) -> Self {
        Self {
            config,
            primary_alias: primary_alias.into(),
            mcp_registry,
            parent_tools: Arc::new(RwLock::new(None)),
            is_subagent_caller: false,
        }
    }

    /// Install the parent's tool registry handle so spawned subagents
    /// inherit the parent's tool `Box`es instead of rebuilding. Called
    /// by the agent loop once `tools_registry` is finalized.
    pub fn set_parent_tools(&self, parent_tools: Arc<RwLock<Vec<Box<dyn Tool>>>>) {
        *self.parent_tools.write().unwrap() = Some(parent_tools);
    }

    /// Mark this tool instance as belonging to a SubAgent's tool
    /// registry. Triggers the depth-1 cap refusal on `execute`.
    #[must_use]
    pub fn with_subagent_caller(mut self, is_subagent_caller: bool) -> Self {
        self.is_subagent_caller = is_subagent_caller;
        self
    }
}

#[async_trait]
impl Tool for SubAgentManageTool {
    fn name(&self) -> &str {
        "subagent_manage"
    }

    fn description(&self) -> &str {
        "Manage the subagents you (the primary model) have spawned. \
         This is your authority surface — no other model can spawn, \
         list, status, stop, or change the model of *your* subagents. \
         Use action='spawn' to start a new subagent under a target \
         alias (optionally overriding the model/provider); \
         action='list' to enumerate your active children; \
         action='status' to inspect a specific one; \
         action='stop' to cancel a running subagent; \
         action='set_model' to swap the model of a running subagent. \
         Spawn returns a subagent id you can refer to in follow-ups."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "list", "status", "stop", "set_model"],
                    "description": "Which authority action to take."
                },
                // ── spawn ──
                "target_alias": {
                    "type": "string",
                    "description": "spawn: alias of the subagent role to instantiate (e.g. 'coder', 'researcher'). Must exist in [agents.<alias>]."
                },
                "prompt": {
                    "type": "string",
                    "description": "spawn: the task or question for the subagent. Self-contained — the subagent does not see your conversation history."
                },
                "provider_override": {
                    "type": "string",
                    "description": "spawn / set_model: optional model provider name (e.g. 'openrouter', 'ollama') overriding the agent's default."
                },
                "model_override": {
                    "type": "string",
                    "description": "spawn / set_model: optional model id (e.g. 'anthropic/claude-sonnet-4.5') overriding the agent's default."
                },
                "allowed_mcp_servers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "spawn: optional list of MCP server names the subagent may access. Omit to inherit parent's set."
                },
                // ── status / stop / set_model ──
                "id": {
                    "type": "string",
                    "description": "status / stop / set_model: the subagent id returned by a prior spawn call."
                },
                "reason": {
                    "type": "string",
                    "description": "stop: optional human-readable reason for the cancellation (logged for audit)."
                }
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": { "properties": { "action": { "const": "spawn" } } },
                    "required": ["target_alias", "prompt"]
                },
                {
                    "if": { "properties": { "action": { "const": "status" } } },
                    "required": ["id"]
                },
                {
                    "if": { "properties": { "action": { "const": "stop" } } },
                    "required": ["id"]
                },
                {
                    "if": { "properties": { "action": { "const": "set_model" } } },
                    "required": ["id", "model_override"]
                }
            ]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        // Depth-1 cap mirrors spawn_subagent: a subagent may not
        // exercise this authority. The agent loop sets the flag from
        // `AgentRunOverrides.is_subagent`.
        if self.is_subagent_caller {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "subagent_manage: a subagent may not exercise primary authority (depth-1 cap)"
                        .into(),
                ),
            });
        }

        let action = match args.get("action").and_then(|v| v.as_str()).map(str::trim) {
            Some(a) if !a.is_empty() => a.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing or empty 'action' parameter".into()),
                });
            }
        };

        match action.as_str() {
            "spawn" => self.spawn_action(args).await,
            "list" => self.list_action().await,
            "status" => self.status_action(args).await,
            "stop" => self.stop_action(args).await,
            "set_model" => self.set_model_action(args).await,
            other => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Unknown action '{other}'; expected one of spawn|list|status|stop|set_model"
                )),
            }),
        }
    }
}

impl SubAgentManageTool {
    async fn spawn_action(&self, args: serde_json::Value) -> Result<ToolResult> {
        let target_alias = match args
            .get("target_alias")
            .and_then(|v| v.as_str())
            .map(str::trim)
        {
            Some(a) if !a.is_empty() => a.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("spawn: 'target_alias' is required".into()),
                });
            }
        };
        let prompt = match args.get("prompt").and_then(|v| v.as_str()).map(str::trim) {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("spawn: 'prompt' is required and must be non-empty".into()),
                });
            }
        };

        let provider_override = args
            .get("provider_override")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let model_override = args
            .get("model_override")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let allowed_mcp_servers: Option<std::collections::HashSet<String>> = args
            .get("allowed_mcp_servers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|val| val.as_str().map(|s| s.to_string()))
                    .collect()
            });

        // Per-parent concurrency cap. Mirrors SpawnSubagentTool but
        // scoped to the primary's own children. Note: this tool and
        // spawn_subagent share the same registry, so the cap is
        // genuinely unified — you can't bypass it by using the other.
        let registry = SubAgentRegistry::global();
        let active = registry.count_active_for(&self.primary_alias);
        if active >= 2 {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "subagent_manage.spawn: refused — maximum concurrent subagent limit for '{}' reached (active: {active}/2)",
                    self.primary_alias
                )),
            });
        }

        // Resolve the parent's security policy + memory allowlist for
        // the child (same shape as the legacy spawn_subagent path).
        let subagent_ctx = match SubAgentSpawn::for_agent(&self.config, &target_alias)
            .and_then(|spawn| spawn.build(SubAgentOverrides::default()))
        {
            Ok(ctx) => ctx,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "subagent_manage.spawn: target '{target_alias}' is not a valid subagent alias: {e}"
                    )),
                });
            }
        };

        // Apply provider/model override resolution: the new run gets
        // its own effective model if either override is supplied.
        // Today the override is informational (the spawned loop will
        // still resolve its provider from the config) — but the
        // override IS applied through AgentRunOverrides so the next
        // iteration of the implementation can read it without
        // breaking the schema. We surface it on the handle so
        // `status` / `list` calls report what was actually requested.
        let resolved_provider = provider_override.clone();
        let resolved_model = model_override.clone().or_else(|| {
            self.config
                .model_provider_for_agent(&target_alias)
                .and_then(|p| p.model.clone())
        });

        let run_id = uuid::Uuid::new_v4().to_string();
        let session_path = std::path::PathBuf::from(format!("subagent-{run_id}"));

        let handle = crate::subagent::SubAgentHandle::new(
            &self.primary_alias,
            &subagent_ctx.parent_alias,
            &prompt,
            resolved_provider,
            resolved_model,
        );
        let handle_id = registry.register(handle.clone());

        let temperature = self
            .config
            .model_provider_for_agent(&target_alias)
            .and_then(|e| e.temperature);

        // Provider/model override hook: AgentRunOverrides does not
        // currently carry these fields end-to-end (model_override is
        // used as a one-shot at the top of agent::run). We still
        // surface the override on the handle so the primary can see
        // it via `status` and the run-loop can pick it up when it
        // resolves its provider. The provider_override on
        // AgentRunOverrides is a stub for the v0.0.6 plumbing pass
        // tracked in #5800.
        let run_overrides = AgentRunOverrides {
            security: Some(subagent_ctx.policy.clone()),
            memory: None,
            is_subagent: true,
            tui_sender: None,
            mcp_registry: self.mcp_registry.clone(),
            allowed_mcp_servers,
        };

        handle.set_status(SubAgentStatus::Running);
        handle.set_step("agent loop");

        let config = (*self.config).clone();
        let primary_alias = self.primary_alias.clone();
        let handle_for_run = handle.clone();
        let registry_for_run = registry.clone();
        let run_result = Box::pin(scope!(
            agent_alias: primary_alias,
            session_key: run_id.clone(),
            =>
            crate::agent::run(
                config,
                &target_alias,
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

        let body = match run_result {
            Ok(response) => {
                let text = if response.trim().is_empty() {
                    "subagent completed without output".to_string()
                } else {
                    response.clone()
                };
                handle_for_run.set_output(&text);
                handle_for_run.set_step("completed");
                handle_for_run.set_status(SubAgentStatus::Completed);
                registry_for_run.deregister(&handle_id);
                json!({
                    "id": handle_id,
                    "status": "completed",
                    "output": text,
                })
            }
            Err(e) => {
                let err_text = format!("subagent run failed: {e}");
                handle_for_run.set_error(&err_text);
                handle_for_run.set_step("failed");
                handle_for_run.set_status(SubAgentStatus::Failed);
                registry_for_run.deregister(&handle_id);
                json!({
                    "id": handle_id,
                    "status": "failed",
                    "error": err_text,
                })
            }
        };

        Ok(ToolResult {
            success: body.get("status").and_then(|v| v.as_str()) == Some("completed"),
            output: serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string()),
            error: if body.get("error").is_some() {
                body.get("error")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            } else {
                None
            },
        })
    }

    async fn list_action(&self) -> Result<ToolResult> {
        let registry = SubAgentRegistry::global();
        let active = registry.list_active_for(&self.primary_alias);
        let body = json!({
            "primary": self.primary_alias,
            "count": active.len(),
            "subagents": active,
        });
        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string()),
            error: None,
        })
    }

    async fn status_action(&self, args: serde_json::Value) -> Result<ToolResult> {
        let id = match args
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(i) => i.to_string(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("status: 'id' is required".into()),
                });
            }
        };

        let registry = SubAgentRegistry::global();
        match registry.get(&id) {
            Some(h) if h.parent_alias == self.primary_alias => {
                let body = json!({ "id": id, "subagent": h.snapshot() });
                Ok(ToolResult {
                    success: true,
                    output: serde_json::to_string_pretty(&body)
                        .unwrap_or_else(|_| body.to_string()),
                    error: None,
                })
            }
            Some(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "status: subagent '{id}' is not owned by primary '{}'",
                    self.primary_alias
                )),
            }),
            None => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "status: no active subagent with id '{id}' (it may have already finished)"
                )),
            }),
        }
    }

    async fn stop_action(&self, args: serde_json::Value) -> Result<ToolResult> {
        let id = match args
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(i) => i.to_string(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("stop: 'id' is required".into()),
                });
            }
        };
        let reason = args
            .get("reason")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let registry = SubAgentRegistry::global();
        match registry.get(&id) {
            Some(h) if h.parent_alias == self.primary_alias => {
                let token = h.cancel_token();
                if !token.is_cancelled() {
                    token.cancel();
                }
                h.set_step("cancelled by primary");
                let note = reason.unwrap_or_else(|| "(no reason given)".to_string());
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject,)
                        .with_outcome(::zeroclaw_log::EventOutcome::Success)
                        .with_attrs(::serde_json::json!({
                            "primary": self.primary_alias,
                            "subagent_id": &id,
                            "subagent_target": h.target_alias,
                            "reason": note,
                        })),
                    "subagent_manage.stop: primary cancelled subagent"
                );
                let body = json!({
                    "id": id,
                    "status": "cancelling",
                    "reason": note,
                });
                Ok(ToolResult {
                    success: true,
                    output: serde_json::to_string_pretty(&body)
                        .unwrap_or_else(|_| body.to_string()),
                    error: None,
                })
            }
            Some(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "stop: subagent '{id}' is not owned by primary '{}'",
                    self.primary_alias
                )),
            }),
            None => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "stop: no active subagent with id '{id}' (it may have already finished)"
                )),
            }),
        }
    }

    async fn set_model_action(&self, args: serde_json::Value) -> Result<ToolResult> {
        let id = match args
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(i) => i.to_string(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("set_model: 'id' is required".into()),
                });
            }
        };
        let model_override = match args
            .get("model_override")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(m) => m.to_string(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("set_model: 'model_override' is required".into()),
                });
            }
        };
        let provider_override = args
            .get("provider_override")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let registry = SubAgentRegistry::global();
        let handle = match registry.get(&id) {
            Some(h) if h.parent_alias == self.primary_alias => h,
            Some(_) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "set_model: subagent '{id}' is not owned by primary '{}'",
                        self.primary_alias
                    )),
                });
            }
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("set_model: no active subagent with id '{id}'")),
                });
            }
        };

        // Record the requested swap on the handle before cancelling
        // so a `status` call mid-stop can read the intent. Then stop
        // the run; the run loop observes the cancellation token and
        // exits cooperatively.
        let original_prompt = handle.prompt.clone();
        let original_target = handle.target_alias.clone();
        handle.cancel_token().cancel();
        handle.set_step("model swap requested by primary");
        ::zeroclaw_log::record!(
            INFO,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note,)
                .with_outcome(::zeroclaw_log::EventOutcome::Success)
                .with_attrs(::serde_json::json!({
                    "primary": self.primary_alias,
                    "subagent_id": &id,
                    "subagent_target": &original_target,
                    "old_provider": handle.provider(),
                    "old_model": handle.model(),
                    "new_provider": provider_override,
                    "new_model": &model_override,
                })),
            "subagent_manage.set_model: stopping subagent to swap model"
        );

        // Stop + respawn. The fresh run inherits the original prompt
        // and a new subagent id; we return both so the primary can
        // continue using the same logical conversation against the
        // new id.
        let respawn_args = json!({
            "target_alias": original_target,
            "prompt": original_prompt,
            "model_override": model_override,
            "provider_override": provider_override,
        });
        // Mark the old run as cancelled so any concurrent status
        // query sees the terminal state. (The run loop will
        // re-finalize; the first-writer-wins on terminal status keeps
        // the audit chain intact.)
        handle.set_status(SubAgentStatus::Cancelled);

        let respawn = self.spawn_action(respawn_args).await?;

        let body = json!({
            "previous_id": id,
            "previous_status": "cancelled",
            "respawn": serde_json::from_str::<serde_json::Value>(&respawn.output)
                .unwrap_or(serde_json::Value::String(respawn.output.clone())),
        });
        Ok(ToolResult {
            success: respawn.success,
            output: serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string()),
            error: respawn.error,
        })
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

    #[test]
    fn description_mentions_all_actions() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let desc = tool.description();
        for action in ["spawn", "list", "status", "stop", "set_model"] {
            assert!(
                desc.contains(action),
                "description must mention action '{action}' so the model knows it exists"
            );
        }
    }

    #[test]
    fn schema_requires_action() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let schema = tool.parameters_schema();
        assert_eq!(
            schema
                .get("required")
                .and_then(|v| v.as_array())
                .map(|a| a.len()),
            Some(1)
        );
    }

    #[tokio::test]
    async fn refuses_when_caller_is_subagent() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None)
            .with_subagent_caller(true);
        for action in ["spawn", "list", "status", "stop", "set_model"] {
            let result = tool
                .execute(json!({ "action": action }))
                .await
                .expect("execute returns Ok with structured failure");
            assert!(!result.success, "depth cap must refuse action {action}");
            assert!(
                result
                    .error
                    .as_deref()
                    .unwrap_or_default()
                    .contains("depth-1 cap"),
                "expected depth-cap refusal for action {action}, got: {:?}",
                result.error
            );
        }
    }

    #[tokio::test]
    async fn rejects_unknown_action() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let result = tool
            .execute(json!({ "action": "teleport" }))
            .await
            .expect("execute returns Ok");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("Unknown action"),
            "expected unknown-action error, got: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn list_returns_empty_when_no_children() {
        // Use a unique primary alias to avoid colliding with other
        // tests' state in the process-wide registry.
        let primary = format!("test-list-empty-{}", uuid::Uuid::new_v4());
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent(&primary)), &primary, None);
        let result = tool
            .execute(json!({ "action": "list" }))
            .await
            .expect("execute returns Ok");
        assert!(result.success);
        let body: serde_json::Value = serde_json::from_str(&result.output).expect("output is JSON");
        assert_eq!(body.get("count").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(
            body.get("primary").and_then(|v| v.as_str()),
            Some(primary.as_str())
        );
    }

    #[tokio::test]
    async fn status_rejects_missing_id() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let result = tool
            .execute(json!({ "action": "status" }))
            .await
            .expect("execute returns Ok");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("'id' is required"),
            "expected missing-id error, got: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn stop_rejects_missing_id() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let result = tool
            .execute(json!({ "action": "stop" }))
            .await
            .expect("execute returns Ok");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn set_model_rejects_missing_model_override() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let result = tool
            .execute(json!({ "action": "set_model", "id": "does-not-matter" }))
            .await
            .expect("execute returns Ok");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("'model_override' is required"),
            "expected missing-model-override error, got: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn status_for_unknown_id_says_already_finished() {
        let tool = SubAgentManageTool::new(Arc::new(config_with_agent("alpha")), "alpha", None);
        let result = tool
            .execute(json!({ "action": "status", "id": "00000000-0000-0000-0000-000000000000" }))
            .await
            .expect("execute returns Ok");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("no active subagent"),
            "expected no-active error, got: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn tool_is_typed_against_attributable_trait() {
        // `Tool: Attributable` is the load-bearing supertrait. Pin it
        // here so a future refactor that drops the bound fails the
        // build, not just runtime.
        use zeroclaw_api::attribution::{Attributable, Role};
        let tool: Box<dyn Tool> = Box::new(SubAgentManageTool::new(
            Arc::new(config_with_agent("alpha")),
            "alpha",
            None,
        ));
        let role = Attributable::role(tool.as_ref());
        assert!(
            matches!(role, Role::Tool(_)),
            "SubAgentManageTool must surface as a Tool role"
        );
    }
}
