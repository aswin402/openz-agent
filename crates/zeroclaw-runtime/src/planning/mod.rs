//! DAG-based task planning for multi-agent orchestration.
//!
//! Provides a Directed Acyclic Graph (DAG) planner that decomposes
//! high-level user goals into structured, dependency-resolved tasks.
//! Adapted from OpenMAD's DAG planner — ported to use ZeroClaw's
//! SubAgentDispatcher for LLM calls and ZeroClaw's Config for
//! agent resolution.
//!
//! ## Single source of truth
//!
//! The task DAG is ephemeral runtime state (not persisted in config).
//! Task type → agent alias mappings are resolved from
//! [`zeroclaw_config::schema::Config::agents`] at plan-build time.

pub mod core_memory;
pub mod model_tracker;
pub mod personas;
pub mod semantic_memory;
pub mod spawner;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Semantic type of a planned task. Each type maps to a configured
/// agent alias that can execute it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    Planning,
    Research,
    Coding,
    Review,
    Testing,
    Documentation,
    Vision,
}

impl TaskType {
    /// Returns the default ZeroClaw agent alias for this task type.
    /// These match the canonical agent names in `zeroclaw.json`.
    pub fn default_agent_alias(&self) -> &'static str {
        match self {
            TaskType::Planning => "planner",
            TaskType::Research => "researcher",
            TaskType::Coding => "coder",
            TaskType::Review => "reviewer",
            TaskType::Testing => "tester",
            TaskType::Documentation => "docs-agent",
            TaskType::Vision => "vision-agent",
        }
    }

    /// Parse from a JSON string returned by the LLM planner.
    pub fn from_str(s: &str) -> Self {
        match s {
            "Planning" => TaskType::Planning,
            "Research" => TaskType::Research,
            "Coding" => TaskType::Coding,
            "Review" => TaskType::Review,
            "Testing" => TaskType::Testing,
            "Documentation" => TaskType::Documentation,
            "Vision" => TaskType::Vision,
            _ => TaskType::Documentation,
        }
    }
}

/// Execution status of a single task node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
}

/// A single node in the task DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub description: String,
    pub task_type: TaskType,
    pub status: TaskStatus,
    pub dependencies: Vec<String>,
    pub result: Option<String>,
    pub assigned_agent: Option<String>,
}

/// A Directed Acyclic Graph of tasks with dependency resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskDag {
    pub tasks: HashMap<String, Task>,
}

impl TaskDag {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    /// Add a task node. Tasks with no dependencies start as `Ready`.
    pub fn add_task(
        &mut self,
        id: String,
        title: String,
        description: String,
        task_type: TaskType,
        dependencies: Vec<String>,
    ) {
        let status = if dependencies.is_empty() {
            TaskStatus::Ready
        } else {
            TaskStatus::Pending
        };

        self.tasks.insert(
            id.clone(),
            Task {
                id,
                title,
                description,
                task_type,
                status,
                dependencies,
                result: None,
                assigned_agent: None,
            },
        );
    }

    /// Returns all tasks that are ready to execute (dependencies met).
    pub fn get_ready_tasks(&self) -> Vec<String> {
        let mut ready = Vec::new();
        for (id, task) in &self.tasks {
            if task.status == TaskStatus::Ready {
                ready.push(id.clone());
            } else if task.status == TaskStatus::Pending {
                let all_deps_done = task.dependencies.iter().all(|dep_id| {
                    self.tasks
                        .get(dep_id)
                        .map_or(true, |dep| dep.status == TaskStatus::Completed)
                });
                if all_deps_done {
                    ready.push(id.clone());
                }
            }
        }
        ready
    }

    /// True when every task is `Completed`.
    pub fn is_complete(&self) -> bool {
        self.tasks
            .values()
            .all(|t| t.status == TaskStatus::Completed)
    }

    /// True when any task is `Failed`.
    pub fn has_failures(&self) -> bool {
        self.tasks.values().any(|t| t.status == TaskStatus::Failed)
    }

    /// Update a task's status and re-evaluate downstream dependencies.
    pub fn update_status(&mut self, id: &str, status: TaskStatus, result: Option<String>) {
        if let Some(task) = self.tasks.get_mut(id) {
            task.status = status;
            if result.is_some() {
                task.result = result;
            }
        }

        // Cascade: completed tasks unlock downstream deps.
        if status == TaskStatus::Completed {
            let to_ready: Vec<String> = self
                .tasks
                .iter()
                .filter(|(_, t)| t.status == TaskStatus::Pending)
                .filter(|(_, t)| {
                    t.dependencies.iter().all(|dep_id| {
                        self.tasks
                            .get(dep_id)
                            .map_or(true, |dep| dep.status == TaskStatus::Completed)
                    })
                })
                .map(|(id, _)| id.clone())
                .collect();

            for tid in to_ready {
                if let Some(t) = self.tasks.get_mut(&tid) {
                    t.status = TaskStatus::Ready;
                }
            }
        }
    }

    /// Human-readable DAG visualization with status emoji.
    pub fn visualize(&self) -> String {
        let mut visual = String::new();
        visual.push_str("Task DAG:\n");
        for task in self.tasks.values() {
            let emoji = match task.status {
                TaskStatus::Pending => "⏳",
                TaskStatus::Ready => "⚡",
                TaskStatus::Running => "🌀",
                TaskStatus::Completed => "✅",
                TaskStatus::Failed => "❌",
            };
            visual.push_str(&format!(
                "  {} [{}] {} — {:?}\n",
                emoji, task.id, task.title, task.task_type
            ));
            if !task.dependencies.is_empty() {
                visual.push_str(&format!(
                    "     Depends on: {}\n",
                    task.dependencies.join(", ")
                ));
            }
        }
        visual
    }

    /// Serialize the DAG to a JSON string (for LLM consumption or
    /// persistence).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.tasks).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_with_no_deps_starts_ready() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Do thing".into(),
            "".into(),
            TaskType::Coding,
            vec![],
        );
        assert_eq!(dag.tasks["T1"].status, TaskStatus::Ready);
    }

    #[test]
    fn task_with_deps_starts_pending() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Plan".into(),
            "".into(),
            TaskType::Planning,
            vec![],
        );
        dag.add_task(
            "T2".into(),
            "Code".into(),
            "".into(),
            TaskType::Coding,
            vec!["T1".into()],
        );
        assert_eq!(dag.tasks["T2"].status, TaskStatus::Pending);
    }

    #[test]
    fn get_ready_tasks_returns_only_ready() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Plan".into(),
            "".into(),
            TaskType::Planning,
            vec![],
        );
        dag.add_task(
            "T2".into(),
            "Code".into(),
            "".into(),
            TaskType::Coding,
            vec!["T1".into()],
        );
        let ready = dag.get_ready_tasks();
        assert_eq!(ready, vec!["T1".to_string()]);
    }

    #[test]
    fn completing_task_unlocks_dependents() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Plan".into(),
            "".into(),
            TaskType::Planning,
            vec![],
        );
        dag.add_task(
            "T2".into(),
            "Code".into(),
            "".into(),
            TaskType::Coding,
            vec!["T1".into()],
        );
        dag.update_status("T1", TaskStatus::Completed, None);
        assert_eq!(dag.tasks["T2"].status, TaskStatus::Ready);
    }

    #[test]
    fn is_complete_when_all_done() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Plan".into(),
            "".into(),
            TaskType::Planning,
            vec![],
        );
        dag.update_status("T1", TaskStatus::Completed, None);
        assert!(dag.is_complete());
    }

    #[test]
    fn has_failures_detects_failed() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Plan".into(),
            "".into(),
            TaskType::Planning,
            vec![],
        );
        dag.update_status("T1", TaskStatus::Failed, None);
        assert!(dag.has_failures());
    }

    #[test]
    fn default_agent_alias_maps_correctly() {
        assert_eq!(TaskType::Planning.default_agent_alias(), "planner");
        assert_eq!(TaskType::Research.default_agent_alias(), "researcher");
        assert_eq!(TaskType::Coding.default_agent_alias(), "coder");
        assert_eq!(TaskType::Review.default_agent_alias(), "reviewer");
        assert_eq!(TaskType::Testing.default_agent_alias(), "tester");
        assert_eq!(TaskType::Documentation.default_agent_alias(), "docs-agent");
        assert_eq!(TaskType::Vision.default_agent_alias(), "vision-agent");
    }

    #[test]
    fn visualize_does_not_panic() {
        let mut dag = TaskDag::new();
        dag.add_task(
            "T1".into(),
            "Test".into(),
            "".into(),
            TaskType::Coding,
            vec![],
        );
        let vis = dag.visualize();
        assert!(vis.contains("T1"));
    }
}
