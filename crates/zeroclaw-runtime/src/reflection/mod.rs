//! Self-repair reflection loop for agent task outputs.
//!
//! Adapted from OpenMAD's reflection system. Instead of calling model
//! APIs directly (like OpenMAD's `ModelRouter`), this module delegates
//! review and repair to ZeroClaw's [`SubAgentDispatcher`] so that:
//!
//! - The reviewer is a configured agent alias (e.g. `"reviewer"`) with
//!   its own fallback chain and security policy.
//! - Repair re-executes through the same dispatcher, gaining retries
//!   and timeout protection.
//! - The process-wide [`SubAgentRegistry`] tracks each review+repair
//!   cycle so the operator can monitor progress.
//!
//! ## Self-repair loop
//!
//! 1. Agent produces an output for a task.
//! 2. Reviewer agent audits the output against task requirements.
//! 3. If approved → return output.
//! 4. If rejected → provide feedback, agent retries (up to N times).
//! 5. After N retries → return best-effort output.

use crate::planning::core_memory::{AgentMemoryStore, CoreMemory, parse_and_apply_memory_updates};
use crate::subagent::orchestrator::{DispatchResult, SubAgentDispatcher};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use zeroclaw_log::{Action, Event, EventOutcome, record};

/// Result of a single review cycle.
#[derive(Debug, Clone, Serialize)]
pub struct ReviewResult {
    pub approved: bool,
    pub feedback: String,
    pub reviewer_provider: String,
    pub reviewer_model: String,
}

/// Result of a self-repair loop.
#[derive(Debug, Clone, Serialize)]
pub struct RepairOutcome {
    /// The final output (best-effort after N retries).
    pub output: String,
    /// Number of repair iterations performed.
    pub retries: usize,
    /// Whether the output was approved on the final iteration.
    pub approved: bool,
    /// Feedback from the final review (empty if approved).
    pub final_feedback: String,
    /// Duration of the entire repair loop.
    pub duration_ms: u64,
}

/// Self-repair reflection system.
///
/// Wraps a [`SubAgentDispatcher`] for both review and re-execution,
/// so all the ZeroClaw machinery (fallback chains, timeouts,
/// cancellation, registry) is available to the reflection loop.
///
/// Optionally integrates with [`CoreMemory`] — injects memory context
/// into task prompts and parses Letta-style memory update tags from
/// agent output.
#[derive(Clone)]
pub struct ReflectionSystem {
    dispatcher: Arc<SubAgentDispatcher>,
    max_retries: usize,
    /// Agent alias to use for review (e.g. `"reviewer"`).
    reviewer_alias: String,
    /// Optional core memory injected into task prompts.
    core_memory: Option<CoreMemory>,
    /// Optional memory store for persisting autonomous memory updates.
    memory_store: Option<AgentMemoryStore>,
    /// Agent name used for memory store lookups.
    agent_name: Option<String>,
}

impl ReflectionSystem {
    /// Create a new reflection system.
    ///
    /// * `dispatcher` — the parent's [`SubAgentDispatcher`] (scoped to
    ///   the parent agent so registry entries are visible).
    /// * `max_retries` — how many times a failed task should retry
    ///   before accepting best-effort output.
    /// * `reviewer_alias` — which agent alias to use as the reviewer
    ///   (default: `"reviewer"`).
    pub fn new(
        dispatcher: Arc<SubAgentDispatcher>,
        max_retries: usize,
        reviewer_alias: impl Into<String>,
    ) -> Self {
        Self {
            dispatcher,
            max_retries,
            reviewer_alias: reviewer_alias.into(),
            core_memory: None,
            memory_store: None,
            agent_name: None,
        }
    }

    /// Attach core memory context to this reflection system.
    ///
    /// When set:
    /// - The core memory XML is injected into every task prompt.
    /// - After each execution, memory update tags in the output
    ///   are parsed and applied.
    #[must_use]
    pub fn with_core_memory(
        mut self,
        core_memory: CoreMemory,
        memory_store: AgentMemoryStore,
        agent_name: impl Into<String>,
    ) -> Self {
        self.core_memory = Some(core_memory);
        self.memory_store = Some(memory_store);
        self.agent_name = Some(agent_name.into());
        self
    }

    /// Review a task output using the configured reviewer agent.
    ///
    /// The reviewer gets:
    /// - The task title and description (requirements).
    /// - The agent's output to evaluate.
    ///
    /// Returns `(approved, feedback, provider_used, model_used)`.
    pub async fn review_output(
        &self,
        task_title: &str,
        task_desc: &str,
        output: &str,
    ) -> anyhow::Result<ReviewResult> {
        let prompt = format!(
            r#"You are an expert reviewer. Your role is to examine task results and determine if they satisfy the requirements.

Task Title: {task_title}
Task Description: {task_desc}

Agent Output to Review:
{output}

Does this output meet ALL requirements?

If YES, begin your response with exactly: "STATUS: APPROVED"
If NO, begin your response with exactly: "STATUS: REJECTED" followed by specific, actionable feedback on what needs to be fixed and how."#,
        );

        let DispatchResult {
            output: review_text,
            provider_used,
            model_used,
            ..
        } = self
            .dispatcher
            .dispatch(&self.reviewer_alias, prompt, None)
            .await?;

        let approved =
            review_text.contains("STATUS: APPROVED") || !review_text.contains("STATUS: REJECTED");

        record!(
            INFO,
            Event::new(module_path!(), Action::Validate)
                .with_outcome(if approved {
                    EventOutcome::Success
                } else {
                    EventOutcome::Failure
                })
                .with_attrs(serde_json::json!({
                    "task_title": task_title,
                    "approved": approved,
                    "reviewer": &self.reviewer_alias,
                })),
            if approved {
                "review approved"
            } else {
                "review rejected"
            }
        );

        Ok(ReviewResult {
            approved,
            feedback: review_text,
            reviewer_provider: provider_used,
            reviewer_model: model_used,
        })
    }

    /// Run a task through the self-repair loop.
    ///
    /// 1. Execute the task via the dispatcher (if `initial_output` is
    ///    empty — else skip to review).
    /// 2. Review the output.
    /// 3. If rejected and retries remain, re-execute with feedback.
    /// 4. Return the best-effort output (approved or last attempt).
    ///
    /// When [`CoreMemory`] is configured (via [`Self::with_core_memory`]):
    /// - Memory XML is injected into every execution prompt.
    /// - Letta-style `<update_core_memory ...>` tags are parsed from
    ///   output and persisted automatically.
    ///
    /// * `task_alias` — the agent alias to run (e.g. `"coder"`).
    /// * `task_title` — human-readable task name (for the reviewer).
    /// * `task_desc` — full task description (prompt body).
    /// * `initial_output` — pre-existing output to review (pass empty
    ///   string to trigger execution).
    /// * `allowed_tools` — optional tool allowlist for the task agent.
    pub async fn run_self_repair(
        &self,
        task_alias: &str,
        task_title: &str,
        task_desc: &str,
        initial_output: String,
        allowed_tools: Option<Vec<String>>,
    ) -> anyhow::Result<RepairOutcome> {
        let started = std::time::Instant::now();

        // Inject core memory context into every execution prompt.
        let memory_context = self
            .core_memory
            .as_ref()
            .map(|m| {
                format!(
                    "\n\n=== LETTA-STYLE HIERARCHICAL CORE MEMORY ===\n{}\n\n\
                     === MEMORY UPDATE PROTOCOL ===\n\
                     You can autonomously update your core memory blocks. \
                     If you learn anything new about the user or your goals, \
                     or wish to refine your persona instructions, output your \
                     changes using this exact tag format:\n\
                     <update_core_memory block=\"human\">new info about the user</update_core_memory>\n\
                     <update_core_memory block=\"persona\">new self-instructions or skill notes</update_core_memory>\n\
                     DO NOT output placeholders. Write the full updated value.",
                    m.to_xml(),
                )
            })
            .unwrap_or_default();

        let enhanced_desc = format!("{task_desc}{memory_context}");

        // Step 1: Initial execution (if no output provided).
        let mut current_output = if initial_output.is_empty() {
            let DispatchResult {
                output,
                provider_used: _,
                model_used: _,
                ..
            } = self
                .dispatcher
                .dispatch(task_alias, enhanced_desc.clone(), allowed_tools.clone())
                .await?;
            output
        } else {
            initial_output
        };

        // Step 2 & 3: Review → repair loop.
        let mut retry_count = 0;
        let mut last_feedback;

        loop {
            let review = self
                .review_output(task_title, task_desc, &current_output)
                .await?;

            if review.approved {
                // Parse and apply any memory updates from the approved output.
                self.apply_memory_updates(&current_output);

                return Ok(RepairOutcome {
                    output: current_output,
                    retries: retry_count,
                    approved: true,
                    final_feedback: String::new(),
                    duration_ms: started.elapsed().as_millis() as u64,
                });
            }

            last_feedback = review.feedback;
            retry_count += 1;

            if retry_count > self.max_retries {
                record!(
                    WARN,
                    Event::new(module_path!(), Action::Fail)
                        .with_outcome(EventOutcome::Success)
                        .with_attrs(serde_json::json!({
                            "task_title": task_title,
                            "retries": retry_count,
                            "max_retries": self.max_retries,
                        })),
                    "reflection: max retries reached, accepting current output"
                );

                // Parse and apply any memory updates even from unapproved output.
                self.apply_memory_updates(&current_output);

                return Ok(RepairOutcome {
                    output: current_output,
                    retries: retry_count - 1,
                    approved: false,
                    final_feedback: last_feedback,
                    duration_ms: started.elapsed().as_millis() as u64,
                });
            }

            record!(
                INFO,
                Event::new(module_path!(), Action::Retry).with_attrs(serde_json::json!({
                    "task_title": task_title,
                    "retry": retry_count,
                    "max_retries": self.max_retries,
                })),
                "reflection: repairing output"
            );

            let repair_prompt = format!(
                "{enhanced_desc}\n\nPrevious attempt failed review.\nFeedback: {feedback}\nPlease correct the output according to the feedback.",
                feedback = last_feedback,
            );

            let DispatchResult {
                output,
                provider_used: _,
                model_used: _,
                ..
            } = self
                .dispatcher
                .dispatch(task_alias, repair_prompt, allowed_tools.clone())
                .await?;

            current_output = output;
        }
    }

    /// Parse and apply Letta-style memory updates from the agent's output.
    fn apply_memory_updates(&self, output: &str) {
        let (Some(ref memory_store), Some(ref agent_name)) =
            (self.memory_store.as_ref(), self.agent_name.as_ref())
        else {
            return;
        };

        if self.core_memory.is_none() {
            return;
        }

        // Clone the CoreMemory for mutation, then write back to store.
        // The shared Arc'd self.core_memory is intentionally NOT updated
        // because the ReflectionSystem is shared across tasks via Arc.
        // The memory store on disk is the source of truth.
        // Next load from disk will pick up the changes.
        let mut mem = self.core_memory.clone().unwrap();
        parse_and_apply_memory_updates(output, agent_name, &mut mem, memory_store);
    }

    /// Set a custom timeout for the dispatcher.
    /// Returns a new [`ReflectionSystem`] with the override applied.
    #[must_use]
    pub fn with_dispatcher_timeout(mut self, timeout: Duration) -> Self {
        let new_dispatcher = SubAgentDispatcher::new(
            self.dispatcher.config().clone(),
            self.dispatcher.parent_alias(),
        )
        .with_timeout(timeout);
        self.dispatcher = Arc::new(new_dispatcher);
        self
    }

    /// Access the reviewer alias for display/logging.
    pub fn reviewer_alias(&self) -> &str {
        &self.reviewer_alias
    }

    /// Access the configured max retries.
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    /// Check if this system has core memory configured.
    pub fn has_core_memory(&self) -> bool {
        self.core_memory.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::core_memory::CoreMemory;

    #[test]
    fn reflection_system_construction() {
        let config = Arc::new(zeroclaw_config::schema::Config::default());
        let dispatcher = Arc::new(SubAgentDispatcher::new(config, "test-parent"));
        let sys = ReflectionSystem::new(dispatcher, 2, "reviewer");
        assert_eq!(sys.max_retries, 2);
        assert_eq!(sys.reviewer_alias(), "reviewer");
        assert!(!sys.has_core_memory());
    }

    #[test]
    fn reflection_system_with_core_memory() {
        let config = Arc::new(zeroclaw_config::schema::Config::default());
        let dispatcher = Arc::new(SubAgentDispatcher::new(config, "test-parent"));
        let mem = CoreMemory::new("Expert coder", "User likes Rust");
        let store = AgentMemoryStore::new("/tmp/test_mem.json");
        let sys = ReflectionSystem::new(dispatcher, 2, "reviewer").with_core_memory(
            mem,
            store,
            "test-agent",
        );
        assert!(sys.has_core_memory());
    }
}
