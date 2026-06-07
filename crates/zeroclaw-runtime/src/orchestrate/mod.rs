//! Multi-agent orchestration engine.
//!
//! Integrates OpenMAD-inspired DAG planning and self-repair reflection
//! with ZeroClaw's native [`SubAgentDispatcher`] to decompose a
//! high-level goal into a dependency-resolved task graph, execute tasks
//! in parallel (respecting dependencies), and optionally repair failed
//! outputs through a reviewer-led self-repair loop.
//!
//! ## Extended features (ported from OpenMAD)
//!
//! - **Core Memory** — Letta-style hierarchical agent memory with
//!   autonomous `<update_core_memory>` XML tags in agent output.
//! - **Semantic Memory** — [`MemoryEngine`] injects relevant context
//!   from past tasks into each execution prompt.
//! - **Shared Agent Memory** — agents publish artifacts (`SharedAgentMemory`)
//!   that later tasks can read.
//! - **Model Performance Tracking** — [`ModelTracker`] records per-model
//!   success/failure and provides dynamic routing.
//! - **Dynamic Team Composition** — [`AgentSpawner`] selects the right
//!   agent team based on goal keywords.
//!
//! ## Flow
//!
//! 1. **Plan** — a planner agent (e.g. `"planner"`) decomposes the
//!    goal into a structured JSON task list. The orchestrator parses
//!    this into a [`TaskDag`].
//! 2. **Execute** — tasks whose dependencies are met run concurrently
//!    via [`SubAgentDispatcher`]. Each task is dispatched to the
//!    appropriate agent alias (determined by its [`TaskType`]).
//! 3. **Reflect** — optionally, each completed task output is reviewed
//!    by a reviewer agent. Rejected outputs trigger a self-repair loop
//!    (re-execution with reviewer feedback, up to N retries).
//! 4. **Memorize** — task results are stored in semantic and shared
//!    memory for use by downstream tasks. Core memory updates are
//!    parsed and persisted automatically.
//! 5. **Aggregate** — once all tasks are complete (or failed), the
//!    orchestrator returns the full results map.
//!
//! ## Single source of truth
//!
//! - Agent aliases and their model provider chains live in
//!   [`zeroclaw_config::schema::Config`] — the orchestrator reads from
//!   config, never duplicates it.
//! - Task DAG is ephemeral runtime state (not persisted).
//! - Core memory is persisted to a JSON file by [`AgentMemoryStore`].
//! - Semantic memory and model stats are ephemeral (in-memory).
//! - Running tasks register handles in the process-wide
//!   [`SubAgentRegistry`] so the operator can monitor/cancel them.

use crate::planning::core_memory::{AgentMemoryStore, CoreMemory};
use crate::planning::model_tracker::ModelTracker;
use crate::planning::semantic_memory::{MemoryEngine, SharedAgentMemory};
use crate::planning::spawner::AgentSpawner;
use crate::planning::{TaskDag, TaskStatus, TaskType};
use crate::reflection::{ReflectionSystem, RepairOutcome};
use crate::subagent::orchestrator::{DispatchResult, SubAgentDispatcher};
use regex::Regex;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use zeroclaw_config::schema::Config;
use zeroclaw_log::{Action, Event, EventOutcome, record};

/// Regex matching local image file paths (same pattern as agent/history.rs).
static LOCAL_IMAGE_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"/[^\s<>'"`\]\)]+?\.(?i:png|jpe?g|webp|gif|bmp)"#).expect("valid image path regex")
});

/// How the orchestrator handles task failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FailureMode {
    /// Stop at the first failed task; return partial results.
    FailFast,
    /// Continue executing remaining tasks; report failures at the end.
    ContinueOnFailure,
}

/// Configuration for a multi-agent orchestration run.
#[derive(Debug, Clone, Serialize)]
pub struct OrchestrationConfig {
    /// Agent alias for the planner (goal decomposition).
    pub planner_alias: String,
    /// Agent alias for the reviewer (output audit).
    pub reviewer_alias: String,
    /// Maximum retries per task in the self-repair loop.
    /// Set to 0 to skip reflection entirely.
    pub max_repair_retries: usize,
    /// How to handle task failures.
    pub failure_mode: FailureMode,
    /// Per-task timeout for the SubAgentDispatcher.
    pub task_timeout: Option<Duration>,
    /// Tool allowlist passed to every task (None = all tools allowed).
    pub allowed_tools: Option<Vec<String>>,
    /// Max semantic memory entries to inject per task (0 = disable).
    pub semantic_memory_limit: usize,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            planner_alias: "planner".to_string(),
            reviewer_alias: "reviewer".to_string(),
            max_repair_retries: 2,
            failure_mode: FailureMode::FailFast,
            task_timeout: None,
            allowed_tools: None,
            semantic_memory_limit: 3,
        }
    }
}

/// A single task execution result.
#[derive(Debug, Clone, Serialize)]
pub struct TaskResult {
    pub task_id: String,
    pub title: String,
    pub task_type: TaskType,
    pub status: TaskStatus,
    pub output: Option<String>,
    pub repair: Option<RepairOutcome>,
    pub provider_used: Option<String>,
    pub model_used: Option<String>,
    pub duration_ms: u64,
}

/// Result of a full orchestration run.
#[derive(Debug, Clone, Serialize)]
pub struct OrchestrationResult {
    pub goal: String,
    pub success: bool,
    pub task_results: HashMap<String, TaskResult>,
    pub total_duration_ms: u64,
    pub plan_visualization: String,
}

/// Multi-agent orchestration engine with OpenMAD-inspired extensions.
///
/// Wraps a [`SubAgentDispatcher`] and provides:
/// - Goal decomposition via a planner agent.
/// - DAG-based parallel task execution.
/// - Optional self-repair reflection per task.
/// - Optional semantic memory context injection.
/// - Optional core memory (Letta-style autonomous updates).
/// - Optional shared artifact memory between tasks.
/// - Optional model performance tracking.
#[derive(Clone)]
pub struct MultiAgentOrchestrator {
    dispatcher: Arc<SubAgentDispatcher>,
    config: OrchestrationConfig,
    /// Semantic memory engine for context injection (optional).
    memory_engine: Option<Arc<MemoryEngine>>,
    /// Shared artifact memory for cross-task data exchange (optional).
    shared_memory: Option<Arc<SharedAgentMemory>>,
    /// Agent spawner for dynamic team composition (optional).
    spawner: Option<Arc<AgentSpawner>>,
    /// Model performance tracker (optional).
    model_tracker: Option<Arc<ModelTracker>>,
    /// Core memory for the planner agent (optional).
    planner_core_memory: Option<CoreMemory>,
    /// Core memory store for persistence (optional).
    planner_memory_store: Option<AgentMemoryStore>,
}

impl MultiAgentOrchestrator {
    /// Create a new orchestrator bound to a parent agent.
    ///
    /// * `parent_config` — application config (agent aliases, providers).
    /// * `parent_alias` — the agent alias that spawns the orchestration.
    /// * `orchestration_config` — how to run (or default).
    pub fn new(
        parent_config: Arc<Config>,
        parent_alias: impl Into<String>,
        orchestration_config: OrchestrationConfig,
    ) -> Self {
        Self {
            dispatcher: Arc::new(SubAgentDispatcher::new(parent_config, parent_alias)),
            config: orchestration_config,
            memory_engine: None,
            shared_memory: None,
            spawner: None,
            model_tracker: None,
            planner_core_memory: None,
            planner_memory_store: None,
        }
    }

    // --- Builder methods for optional features ---

    /// Attach a semantic memory engine for context injection.
    #[must_use]
    pub fn with_memory_engine(mut self, engine: Arc<MemoryEngine>) -> Self {
        self.memory_engine = Some(engine);
        self
    }

    /// Attach a shared artifact memory.
    #[must_use]
    pub fn with_shared_memory(mut self, shared: Arc<SharedAgentMemory>) -> Self {
        self.shared_memory = Some(shared);
        self
    }

    /// Attach a dynamic team spawner.
    #[must_use]
    pub fn with_spawner(mut self, spawner: Arc<AgentSpawner>) -> Self {
        self.spawner = Some(spawner);
        self
    }

    /// Attach a model performance tracker.
    #[must_use]
    pub fn with_model_tracker(mut self, tracker: Arc<ModelTracker>) -> Self {
        self.model_tracker = Some(tracker);
        self
    }

    /// Attach planner core memory for Letta-style autonomous updates.
    #[must_use]
    pub fn with_planner_core_memory(
        mut self,
        core_memory: CoreMemory,
        memory_store: AgentMemoryStore,
    ) -> Self {
        self.planner_core_memory = Some(core_memory);
        self.planner_memory_store = Some(memory_store);
        self
    }

    // --- Core orchestration ---

    /// Decompose a goal into a [`TaskDag`] using the planner agent.
    ///
    /// Sends the goal to the configured planner alias (e.g. `"planner"`)
    /// and expects a JSON array of tasks back.
    ///
    /// If a [`MemoryEngine`] is attached, relevant semantic memories
    /// are injected into the planner prompt.
    /// If a [`ModelTracker`] is attached, planner model performance
    /// is recorded.
    pub async fn plan_goal(&self, goal: &str) -> anyhow::Result<TaskDag> {
        // Inject semantic memory context if available.
        let memory_context = self
            .memory_engine
            .as_ref()
            .map(|m| {
                let results = m.query_semantic(goal, self.config.semantic_memory_limit);
                if results.is_empty() {
                    String::new()
                } else {
                    let mut ctx = "\n\n--- Relevant Memory Context ---\n".to_string();
                    for (text, score) in results {
                        ctx.push_str(&format!("* (score: {score:.2}) {text}\n"));
                    }
                    ctx
                }
            })
            .unwrap_or_default();

        let system_prompt = format!(
            r#"You are a goal decomposition planner. Break down the user's objective into 3-7 sequential, structured tasks with clear dependencies.

You MUST return ONLY a valid JSON array. No markdown. No explanation.

Each entry has:
- "id": unique task ID (e.g. "T1", "T2")
- "title": short task name
- "description": what this task does
- "type": one of Planning, Research, Coding, Review, Testing, Documentation, Vision
- "dependencies": array of task IDs this depends on (empty for root tasks)

Example:
[
  {{
    "id": "T1",
    "title": "Analyze requirements",
    "description": "Gather and analyze project requirements",
    "type": "Research",
    "dependencies": []
  }},
  {{
    "id": "T2",
    "title": "Implement core logic",
    "description": "Write the main implementation",
    "type": "Coding",
    "dependencies": ["T1"]
  }}
]{memory_context}"#,
        );

        let user_prompt = format!("Decompose this goal into structured tasks:\n\n{goal}");

        let prompt = format!("{system_prompt}\n\n{user_prompt}");

        let DispatchResult {
            output: plan_output,
            provider_used: _,
            model_used,
            ..
        } = self
            .dispatcher
            .dispatch(&self.config.planner_alias, prompt, None)
            .await?;

        // Track model performance if tracker is available.
        if let Some(ref tracker) = self.model_tracker {
            if !model_used.is_empty() {
                tracker.record_success(&model_used);
            }
        }

        // Parse the JSON response into a TaskDag.
        let mut dag = TaskDag::new();

        // Try to extract JSON from markdown code fences first, then
        // try parsing raw text.
        let json_str = extract_json(&plan_output).unwrap_or(&plan_output);

        if let Ok(tasks_json) = serde_json::from_str::<serde_json::Value>(json_str) {
            let arr: Vec<&serde_json::Value> = match tasks_json {
                serde_json::Value::Array(ref a) => a.iter().collect(),
                _ => {
                    // Maybe wrapped in an object with a "tasks" key.
                    tasks_json
                        .get("tasks")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().collect())
                        .unwrap_or_default()
                }
            };

            for t_val in arr {
                let id = t_val["id"].as_str().unwrap_or_default().to_string();
                if id.is_empty() {
                    continue;
                }
                let title = t_val["title"].as_str().unwrap_or_default().to_string();
                let desc = t_val["description"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let t_type = TaskType::from_str(t_val["type"].as_str().unwrap_or_default());
                let deps: Vec<String> = t_val["dependencies"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v: &serde_json::Value| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                dag.add_task(id, title, desc, t_type, deps);
            }
        }

        // Fallback for empty or failed parsing.
        if dag.tasks.is_empty() {
            record!(
                WARN,
                Event::new(module_path!(), Action::Read)
                    .with_outcome(EventOutcome::Success)
                    .with_attrs(serde_json::json!({"reason": "parse_failure"})),
                "planner returned unparseable output, using fallback DAG"
            );
            dag.add_task(
                "T1".into(),
                "Analyze and design".into(),
                format!("Assess requirements for: {goal}"),
                TaskType::Planning,
                vec![],
            );
            dag.add_task(
                "T2".into(),
                "Implement".into(),
                "Write core implementation.".into(),
                TaskType::Coding,
                vec!["T1".into()],
            );
            dag.add_task(
                "T3".into(),
                "Review and test".into(),
                "Review implementation and run tests.".into(),
                TaskType::Testing,
                vec!["T2".into()],
            );
        }

        Ok(dag)
    }

    /// Execute a planned [`TaskDag`] and return results.
    ///
    /// Runs tasks in parallel where dependencies allow. Optionally
    /// applies self-repair reflection after each task completes.
    /// When semantic memory is attached, task results are stored
    /// and relevant context is injected into downstream tasks.
    /// When shared memory is attached, task outputs are published
    /// for later tasks to consume.
    pub async fn execute_dag(&self, goal: &str, dag: &mut TaskDag) -> OrchestrationResult {
        let started = std::time::Instant::now();
        let mut task_results: HashMap<String, TaskResult> = HashMap::new();

        let dag_arc = Arc::new(std::sync::Mutex::new(dag));
        let shared_memory = self.shared_memory.clone();
        let memory_engine = self.memory_engine.clone();

        loop {
            let ready_tasks: Vec<String> = {
                let guard = dag_arc.lock().unwrap();
                guard.get_ready_tasks()
            };

            if ready_tasks.is_empty() {
                let guard = dag_arc.lock().unwrap();
                if guard.is_complete() || guard.has_failures() {
                    break;
                }
                record!(
                    WARN,
                    Event::new(module_path!(), Action::Fail).with_outcome(EventOutcome::Failure),
                    "orchestrator: stall detected — no ready tasks but DAG incomplete"
                );
                break;
            }

            record!(
                INFO,
                Event::new(module_path!(), Action::Tick)
                    .with_attrs(serde_json::json!({ "ready_count": ready_tasks.len() })),
                "orchestrator: executing ready tasks"
            );

            let failure_mode = self.config.failure_mode;
            let mut futures = Vec::new();

            for task_id in ready_tasks {
                let snapshot = {
                    let guard = dag_arc.lock().unwrap();
                    guard.tasks.get(&task_id).cloned()
                };
                let Some(task) = snapshot else {
                    continue;
                };

                // Mark Running.
                {
                    let mut guard = dag_arc.lock().unwrap();
                    guard.update_status(&task_id, TaskStatus::Running, None);
                }

                let dispatcher = self.dispatcher.clone();
                let reflector = if self.config.max_repair_retries > 0 {
                    let mut rs = ReflectionSystem::new(
                        self.dispatcher.clone(),
                        self.config.max_repair_retries,
                        &self.config.reviewer_alias,
                    );

                    // Wire core memory into reflection for this task.
                    // Load fresh memory from the store for the task's agent.
                    if let (Some(_), Some(store)) =
                        (&self.planner_core_memory, &self.planner_memory_store)
                    {
                        let agent_name = task.task_type.default_agent_alias();
                        let default_persona = format!("You are a {agent_name} agent.");
                        let core_mem = store.load_memory(agent_name, &default_persona);
                        rs = rs.with_core_memory(core_mem, store.clone(), agent_name);
                    }

                    Some(rs)
                } else {
                    None
                };
                let allowed_tools = self.config.allowed_tools.clone();
                let dag_arc = dag_arc.clone();
                let task_title = task.title.clone();
                let task_type = task.task_type;
                let task_alias = task.task_type.default_agent_alias().to_string();
                let description = task.description.clone();
                let task_id_clone = task_id.clone();
                let shared_memory = shared_memory.clone();
                let memory_engine = memory_engine.clone();
                let goal = goal.to_string();

                let fut = async move {
                    let task_started = std::time::Instant::now();

                    // Inject semantic memory context into the task description.
                    let enhanced_desc = memory_engine.as_ref().map_or_else(
                        || description.clone(),
                        |mem| {
                            let results = mem.query_semantic(&task_title, 3);
                            if results.is_empty() {
                                description.clone()
                            } else {
                                let mut ctx =
                                    format!("{goal}\n\nTask: {task_title}\n\n{description}");
                                ctx.push_str("\n\n--- Relevant Memory Context ---\n");
                                for (text, score) in results {
                                    ctx.push_str(&format!("* (score: {score:.2}) {text}\n"));
                                }
                                ctx
                            }
                        },
                    );

                    let shared_keys = shared_memory
                        .as_ref()
                        .map(|s| s.list_keys())
                        .unwrap_or_default();
                    let final_desc = if shared_keys.is_empty() {
                        enhanced_desc
                    } else {
                        format!(
                            "{enhanced_desc}\n\nAvailable Shared Artifacts:\n{:?}",
                            shared_keys
                        )
                    };
                    let dispatcher_for_vision = dispatcher.clone();

                    // ── Vision preprocessing ──────────────────────────────
                    // If the task description contains image references, spawn
                    // a vision-agent to analyze them and inject the analysis
                    // into the task context before dispatching.
                    let final_desc =
                        preprocess_task_vision(&dispatcher_for_vision, &final_desc).await;

                    let (output, repair) = if let Some(ref sys) = reflector {
                        match sys
                            .run_self_repair(
                                &task_alias,
                                &task_title,
                                &final_desc,
                                String::new(),
                                allowed_tools,
                            )
                            .await
                        {
                            Ok(outcome) => (Some(outcome.output.clone()), Some(outcome)),
                            Err(e) => {
                                record!(
                                    ERROR,
                                    Event::new(module_path!(), Action::Fail)
                                        .with_outcome(EventOutcome::Failure)
                                        .with_attrs(serde_json::json!({
                                            "task_id": &task_id_clone,
                                            "error": e.to_string(),
                                        })),
                                    "task with reflection failed"
                                );
                                (None, None)
                            }
                        }
                    } else {
                        match dispatcher
                            .dispatch(&task_alias, final_desc, allowed_tools)
                            .await
                        {
                            Ok(DispatchResult { output, .. }) => (Some(output), None),
                            Err(e) => {
                                record!(
                                    ERROR,
                                    Event::new(module_path!(), Action::Fail)
                                        .with_outcome(EventOutcome::Failure)
                                        .with_attrs(serde_json::json!({
                                            "task_id": &task_id_clone,
                                            "error": e.to_string(),
                                        })),
                                    "task execution failed"
                                );
                                (None, None)
                            }
                        }
                    };

                    let duration_ms = task_started.elapsed().as_millis() as u64;
                    let succeeded = output.is_some();

                    // Store output in semantic and shared memory.
                    if let Some(ref out) = output {
                        if let Some(ref mem) = memory_engine {
                            mem.store_memory(out, &format!("task-{}", &task_id_clone));
                        }
                        if let Some(ref shared) = shared_memory {
                            shared.publish(&task_id_clone, out);
                        }
                    }

                    // Update DAG.
                    {
                        let mut guard = dag_arc.lock().unwrap();
                        if succeeded {
                            guard.update_status(
                                &task_id_clone,
                                TaskStatus::Completed,
                                output.clone(),
                            );
                        } else {
                            guard.update_status(&task_id_clone, TaskStatus::Failed, None);
                        }
                    }

                    TaskResult {
                        task_id: task_id_clone,
                        title: task_title,
                        task_type,
                        status: if succeeded {
                            TaskStatus::Completed
                        } else {
                            TaskStatus::Failed
                        },
                        output,
                        repair,
                        provider_used: None,
                        model_used: None,
                        duration_ms,
                    }
                };

                futures.push(fut);
            }

            // ⚡ Run all ready tasks concurrently.
            let results: Vec<TaskResult> = futures_util::future::join_all(futures).await;

            for result in results {
                let failed = result.status == TaskStatus::Failed;
                task_results.insert(result.task_id.clone(), result);

                if failed && failure_mode == FailureMode::FailFast {
                    record!(
                        WARN,
                        Event::new(module_path!(), Action::Cancel)
                            .with_outcome(EventOutcome::Failure),
                        "orchestrator: fail-fast triggered by task failure"
                    );
                    let plan_visualization = dag_arc.lock().unwrap().visualize();
                    return OrchestrationResult {
                        goal: goal.to_string(),
                        success: false,
                        task_results,
                        total_duration_ms: started.elapsed().as_millis() as u64,
                        plan_visualization,
                    };
                }
            }
        }

        let dag_final = dag_arc.lock().unwrap();
        let success = dag_final.is_complete() && !dag_final.has_failures();
        let plan_visualization = dag_final.visualize();

        record!(
            INFO,
            Event::new(module_path!(), Action::Complete)
                .with_outcome(if success {
                    EventOutcome::Success
                } else {
                    EventOutcome::Failure
                })
                .with_attrs(serde_json::json!({
                    "success": success,
                    "total_duration_ms": started.elapsed().as_millis() as u64,
                })),
            "orchestration complete"
        );

        OrchestrationResult {
            goal: goal.to_string(),
            success,
            task_results,
            total_duration_ms: started.elapsed().as_millis() as u64,
            plan_visualization,
        }
    }
}

/// If the task description contains image references (`[IMAGE:` markers
/// or local image file paths), preprocess by dispatching a vision-agent
/// to analyze the image and inject the description into the task context.
async fn preprocess_task_vision(dispatcher: &SubAgentDispatcher, task_desc: &str) -> String {
    if !task_desc.contains("[IMAGE:") && !LOCAL_IMAGE_PATH_RE.is_match(task_desc) {
        return task_desc.to_string();
    }

    record!(
        INFO,
        Event::new(module_path!(), Action::Invoke).with_attrs(serde_json::json!({
            "phase": "vision-preprocess",
            "desc_preview": &task_desc[..task_desc.len().min(100)],
        })),
        "orchestrator: image detected, spawning vision-agent"
    );

    match dispatcher
        .dispatch("vision-agent", task_desc.to_string(), None)
        .await
    {
        Ok(DispatchResult { output, .. }) => {
            let analysis = output.trim();
            if !analysis.is_empty() {
                let enhanced =
                    format!("{task_desc}\n\n### [Image Description (vision-agent)]\n{analysis}\n");
                record!(
                    INFO,
                    Event::new(module_path!(), Action::Complete)
                        .with_outcome(EventOutcome::Success),
                    "orchestrator: vision-agent analysis injected into task context"
                );
                enhanced
            } else {
                task_desc.to_string()
            }
        }
        Err(e) => {
            record!(
                WARN,
                Event::new(module_path!(), Action::Fail)
                    .with_outcome(EventOutcome::Failure)
                    .with_attrs(serde_json::json!({"error": e.to_string()})),
                "orchestrator: vision-agent dispatch failed, continuing without analysis"
            );
            task_desc.to_string()
        }
    }
}

impl MultiAgentOrchestrator {
    /// Full orchestration pipeline: plan → execute.
    ///
    /// 1. Decompose the goal into a DAG (plan).
    /// 2. Execute the DAG (with optional reflection, memory, etc.).
    /// 3. Return the aggregated result.
    pub async fn run_goal(&self, goal: &str) -> anyhow::Result<OrchestrationResult> {
        println!("\n=== Multi-Agent Orchestration ===");
        println!("Goal: {goal}\n");

        let mut dag = self.plan_goal(goal).await?;

        println!("{}", dag.visualize());

        let result = self.execute_dag(goal, &mut dag).await;

        println!(
            "\n=== Orchestration {} ===",
            if result.success {
                "SUCCESS ✅"
            } else {
                "FAILED ❌"
            }
        );
        println!("Duration: {}ms", result.total_duration_ms);
        println!("\nTask Results:");
        for (id, tr) in &result.task_results {
            let status = match tr.status {
                TaskStatus::Completed => "✅",
                TaskStatus::Failed => "❌",
                _ => "⏳",
            };
            println!("  {status} [{id}] {} — {:?}", tr.title, tr.task_type);
            if let Some(ref repair) = tr.repair {
                if repair.approved {
                    println!("       Repaired ({} retries)", repair.retries);
                } else if repair.retries > 0 {
                    println!("       Max retries ({}) — best effort", repair.retries);
                }
            }
        }

        Ok(result)
    }

    /// Access the underlying dispatcher for monitor/stop operations.
    pub fn dispatcher(&self) -> &SubAgentDispatcher {
        &self.dispatcher
    }

    /// Access the model tracker, if attached.
    pub fn model_tracker(&self) -> Option<&Arc<ModelTracker>> {
        self.model_tracker.as_ref()
    }

    /// Access the shared memory, if attached.
    pub fn shared_memory(&self) -> Option<&Arc<SharedAgentMemory>> {
        self.shared_memory.as_ref()
    }

    /// Access the memory engine, if attached.
    pub fn memory_engine(&self) -> Option<&Arc<MemoryEngine>> {
        self.memory_engine.as_ref()
    }
}

/// Extract a JSON array from a response that might contain markdown
/// code fences or other wrapping text.
fn extract_json(text: &str) -> Option<&str> {
    // Try to extract from ```json ... ``` fences.
    if let Some(start) = text.find("```json") {
        let rest = &text[start + 7..];
        if let Some(end) = rest.find("```") {
            return Some(rest[..end].trim());
        }
    }

    // Try ``` (without language tag).
    if let Some(start) = text.find("```") {
        let rest = &text[start + 3..];
        // Skip the language tag line if present.
        let content_start = rest.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &rest[content_start..];
        if let Some(end) = content.find("```") {
            return Some(content[..end].trim());
        }
    }

    // Fallback: check if the text itself starts with '['.
    let trimmed = text.trim();
    if trimmed.starts_with('[') {
        return Some(trimmed);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_from_fenced_block() {
        let text = r#"Here's the plan:
```json
[{"id":"T1","title":"Test","description":"","type":"Coding","dependencies":[]}]
```
Done."#;
        assert_eq!(
            extract_json(text),
            Some(
                r#"[{"id":"T1","title":"Test","description":"","type":"Coding","dependencies":[]}]"#
            )
        );
    }

    #[test]
    fn extract_json_from_bare_array() {
        let text =
            r#"[{"id":"T1","title":"Test","description":"","type":"Coding","dependencies":[]}]"#;
        assert_eq!(extract_json(text), Some(text));
    }

    #[test]
    fn extract_json_from_unlabeled_fence() {
        let text = r#"
```
[{"id":"T1","title":"Test","description":"","type":"Coding","dependencies":[]}]
```
"#;
        assert!(extract_json(text).unwrap_or("").contains("T1"));
    }

    #[test]
    fn extract_json_returns_none_for_non_json() {
        assert!(extract_json("Hello world").is_none());
    }

    #[test]
    fn orchestration_config_defaults() {
        let cfg = OrchestrationConfig::default();
        assert_eq!(cfg.planner_alias, "planner");
        assert_eq!(cfg.reviewer_alias, "reviewer");
        assert_eq!(cfg.max_repair_retries, 2);
        assert_eq!(cfg.failure_mode, FailureMode::FailFast);
        assert_eq!(cfg.semantic_memory_limit, 3);
    }

    #[test]
    fn failure_mode_serialization() {
        let ff = serde_json::to_value(&FailureMode::FailFast).unwrap();
        assert_eq!(ff, serde_json::json!("FailFast"));

        let cont = serde_json::to_value(&FailureMode::ContinueOnFailure).unwrap();
        assert_eq!(cont, serde_json::json!("ContinueOnFailure"));
    }

    #[test]
    fn orchestrator_builder_methods() {
        let config = Arc::new(Config::default());
        let orch = MultiAgentOrchestrator::new(config, "parent", OrchestrationConfig::default())
            .with_memory_engine(Arc::new(MemoryEngine::new()))
            .with_shared_memory(Arc::new(SharedAgentMemory::new()))
            .with_model_tracker(Arc::new(ModelTracker::new()));

        assert!(orch.memory_engine().is_some());
        assert!(orch.shared_memory().is_some());
        assert!(orch.model_tracker().is_some());
    }

    #[test]
    fn orchestrator_semantic_memory_integration() {
        let engine = Arc::new(MemoryEngine::new());
        let shared = Arc::new(SharedAgentMemory::new());
        let tracker = Arc::new(ModelTracker::new());

        // Store some test memories.
        engine.store_memory("Rust is great for systems programming", "lesson");
        engine.store_memory("Use async/await for concurrency", "lesson");

        // Verify recent history.
        let recent = engine.get_recent_history(2);
        assert_eq!(recent.len(), 2);

        // Verify semantic query returns results.
        let results = engine.query_semantic("programming", 5);
        assert!(!results.is_empty());

        // Test shared memory.
        shared.publish("task-1", "output-1");
        assert_eq!(shared.get("task-1").unwrap(), "output-1");
        assert_eq!(shared.len(), 1);

        // Test model tracker.
        tracker.record_success("gemini-pro");
        tracker.record_success("gemini-pro");
        tracker.record_failure("claude-sonnet");
        assert!((tracker.success_rate("gemini-pro") - 1.0).abs() < 1e-6);
        assert!((tracker.success_rate("claude-sonnet") - 0.0).abs() < 1e-6);
        assert!(
            tracker
                .best_model(&["gemini-pro", "claude-sonnet"])
                .unwrap()
                == "gemini-pro"
        );
    }

    #[test]
    fn orchestrator_core_memory_integration() {
        use crate::planning::core_memory::{AgentMemoryStore, CoreMemory};

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = AgentMemoryStore::new(tmp.path().to_str().unwrap());
        let mut mem = CoreMemory::new("I am a coder.", "User likes Rust.");

        // Test XML serialization.
        let xml = mem.to_xml();
        assert!(xml.contains("<persona>"));
        assert!(xml.contains("<human>"));

        // Test roundtrip persistence.
        store.save_memory("coder", &mem).unwrap();
        let loaded = store.load_memory("coder", "default");
        assert_eq!(loaded.get_block("persona").unwrap(), "I am a coder.");

        // Test memory update parsing.
        let output = r#"Some work done.
<update_core_memory block="persona">I now know async Rust.</update_core_memory>
Done."#;
        crate::planning::core_memory::parse_and_apply_memory_updates(
            output, "coder", &mut mem, &store,
        );
        let reloaded = store.load_memory("coder", "default");
        assert_eq!(
            reloaded.get_block("persona").unwrap(),
            "I now know async Rust."
        );
    }

    #[test]
    fn orchestrator_spawner_integration() {
        use crate::planning::spawner::AgentSpawner;

        let spawner = AgentSpawner::new();

        // Test team composition.
        let full_team = spawner.spawn_team_for_goal("Build a website with React");
        assert!(full_team.len() >= 5);

        let debug_team = spawner.spawn_team_for_goal("Fix the login bug");
        assert_eq!(debug_team.len(), 2);

        // Test message hub.
        spawner
            .send_message("sender", "Amelia", "receiver", "hello")
            .unwrap();
        let msg = spawner.try_recv_message().unwrap();
        assert_eq!(msg.content, "hello");

        // Test agent spawning with core memory.
        let agent = spawner.spawn_agent_for_task(TaskType::Coding);
        assert_eq!(agent.persona.name, "Amelia");
        assert!(agent.core_memory.get_block("persona").is_some());
    }

    #[test]
    fn orchestrator_persona_integration() {
        use crate::planning::personas::AgentRegistry;

        let reg = AgentRegistry::new();
        assert_eq!(reg.len(), 7);

        // Verify each persona has a system prompt and 3 principles.
        for key in reg.keys() {
            let persona = reg.get(&key).unwrap();
            assert!(!persona.system_prompt.is_empty());
            assert_eq!(persona.principles.len(), 3);
        }

        // Verify persona-to-task mapping.
        assert_eq!(
            crate::planning::personas::persona_key_for_task(TaskType::Coding),
            "amelia"
        );
        assert_eq!(
            crate::planning::personas::persona_key_for_task(TaskType::Vision),
            "vivian"
        );
    }

    #[test]
    fn orchestrator_model_tracker_json_roundtrip() {
        let tracker = ModelTracker::new();
        tracker.record_success("model-a");
        tracker.record_success("model-a");
        tracker.record_failure("model-b");

        let json = tracker.to_json();
        let tracker2 = ModelTracker::new();
        tracker2.from_json(&json);

        assert_eq!(tracker2.get_performance("model-a").successes, 2);
        assert_eq!(tracker2.get_performance("model-b").failures, 1);
        assert_eq!(tracker2.len(), 2);
    }

    #[test]
    fn orchestrator_memory_engine_zero_confidence_pattern() {
        let engine = MemoryEngine::new();
        // Even without embeddings, the engine should store and return recent history.
        engine.store_memory("test entry 1", "test");
        engine.store_memory("test entry 2", "test");
        assert_eq!(engine.short_term_count(), 2);
        assert_eq!(engine.long_term_count(), 2);

        let history = engine.get_recent_history(1);
        assert_eq!(history.len(), 1);
        assert!(history[0].contains("test entry 2"));
    }

    #[test]
    fn vision_regex_matches_image_paths() {
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/path/to/image.png"));
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/tmp/screenshot.jpg"));
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/home/user/photo.jpeg"));
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/var/data/pic.webp"));
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/tmp/animated.gif"));
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/dir/img.bmp"));
        assert!(LOCAL_IMAGE_PATH_RE.is_match("/path/to/image.PNG"));
    }

    #[test]
    fn vision_regex_does_not_match_non_images() {
        assert!(!LOCAL_IMAGE_PATH_RE.is_match("/path/to/file.txt"));
        assert!(!LOCAL_IMAGE_PATH_RE.is_match("/path/to/file.pdf"));
        assert!(!LOCAL_IMAGE_PATH_RE.is_match("/path/to/file.html"));
        assert!(!LOCAL_IMAGE_PATH_RE.is_match("no/path/here"));
        assert!(!LOCAL_IMAGE_PATH_RE.is_match("Just text without any path"));
        assert!(!LOCAL_IMAGE_PATH_RE.is_match("relative/path.txt"));
    }

    #[test]
    fn vision_detection_via_image_marker() {
        // [IMAGE: markers are detected by string contains, not regex
        let desc_with_marker = "Analyze this [IMAGE:/tmp/photo.jpg]";
        assert!(desc_with_marker.contains("[IMAGE:"));

        let desc_without = "Just a regular task description";
        assert!(!desc_without.contains("[IMAGE:"));
    }
}
