//! Dynamic Team Composition and Agent Message Hub.
//!
//! Adapted from OpenMAD's spawner.rs.
//!
//! - **AgentMessage** hub: typed inter-agent communication via flume
//!   channels.
//! - **Dynamic team building**: keyword-based goal scanning selects
//!   the appropriate agent team (full stack, vision, debug, or minimal).
//!
//! ## Single source of truth
//!
//! Agent personas live in the [`AgentRegistry`] (code defaults or
//! `planning_agents.json`). Agent model/provider configuration lives
//! in `zeroclaw.json`. The spawner combines both into ready-to-run
//! [`AgentInstance`]s.

use uuid::Uuid;

use crate::planning::TaskType;
use crate::planning::core_memory::AgentMemoryStore;
use crate::planning::personas::{AgentInstance, AgentPersona, AgentRegistry, persona_key_for_task};

/// A typed message between agents.
#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub sender_id: String,
    pub sender_name: String,
    pub recipient_id: String,
    pub content: String,
}

/// Spawns agent instances and manages inter-agent communication.
pub struct AgentSpawner {
    registry: AgentRegistry,
    message_hub_tx: flume::Sender<AgentMessage>,
    message_hub_rx: flume::Receiver<AgentMessage>,
    pub memory_store: AgentMemoryStore,
}

impl AgentSpawner {
    /// Create a new spawner with the default registry and memory store.
    pub fn new() -> Self {
        let (tx, rx) = flume::unbounded();
        Self {
            registry: AgentRegistry::new(),
            message_hub_tx: tx,
            message_hub_rx: rx,
            memory_store: AgentMemoryStore::new("letta_memory_store.json"),
        }
    }

    /// Create a spawner with a custom memory store path.
    pub fn with_memory_store(memory_path: &str) -> Self {
        let (tx, rx) = flume::unbounded();
        Self {
            registry: AgentRegistry::new(),
            message_hub_tx: tx,
            message_hub_rx: rx,
            memory_store: AgentMemoryStore::new(memory_path),
        }
    }

    /// Dynamically compose a team based on goal keywords.
    ///
    /// Returns agent instances matched to the complexity domain:
    /// - Full team for website/saas/app goals.
    /// - Vision team for UI/design/frontend goals.
    /// - Debug team for fix/bug/refactor goals.
    /// - Minimal (coder only) for simple goals.
    pub fn spawn_team_for_goal(&self, goal: &str) -> Vec<AgentInstance> {
        let goal_lower = goal.to_lowercase();
        let mut team = Vec::new();

        let full_team_keywords = ["website", "saas", "app", "application", "full-stack"];
        let vision_keywords = [
            "ui", "design", "css", "frontend", "visual", "vision", "layout",
        ];
        let debug_keywords = ["fix", "bug", "refactor", "debug", "issue", "error"];

        if full_team_keywords.iter().any(|k| goal_lower.contains(k)) {
            // Full development team
            self.push_agent(&mut team, "john", TaskType::Planning);
            self.push_agent(&mut team, "mary", TaskType::Research);
            self.push_agent(&mut team, "winston", TaskType::Planning);
            self.push_agent(&mut team, "amelia", TaskType::Coding);
            self.push_agent(&mut team, "tester", TaskType::Testing);
            self.push_agent(&mut team, "paige", TaskType::Documentation);
            self.push_agent(&mut team, "vivian", TaskType::Vision);
        } else if vision_keywords.iter().any(|k| goal_lower.contains(k)) {
            // Vision/design team
            self.push_agent(&mut team, "john", TaskType::Planning);
            self.push_agent(&mut team, "amelia", TaskType::Coding);
            self.push_agent(&mut team, "vivian", TaskType::Vision);
        } else if debug_keywords.iter().any(|k| goal_lower.contains(k)) {
            // Debug/fix team
            self.push_agent(&mut team, "amelia", TaskType::Coding);
            self.push_agent(&mut team, "winston", TaskType::Review);
        } else {
            // Minimal team — just a coder
            self.push_agent(&mut team, "amelia", TaskType::Coding);
        }

        team
    }

    /// Spawn a single agent for a specific task type.
    pub fn spawn_agent_for_task(&self, task_type: TaskType) -> AgentInstance {
        let id = format!(
            "{}-{}",
            task_type.default_agent_alias(),
            &Uuid::new_v4().to_string()[..8]
        );

        let key = persona_key_for_task(task_type);
        let persona = self
            .registry
            .get(key)
            .cloned()
            .unwrap_or_else(|| AgentPersona {
                name: "AgentBot".to_string(),
                title: "Autonomous Agent".to_string(),
                system_prompt:
                    "You are an autonomous assistant. Work efficiently to solve the task."
                        .to_string(),
                principles: vec!["Execute tasks with precision.".to_string()],
            });

        self.create_instance(&id, &persona, task_type)
    }

    /// Get a clone of the message hub channels.
    pub fn get_message_hub(&self) -> (flume::Sender<AgentMessage>, flume::Receiver<AgentMessage>) {
        (self.message_hub_tx.clone(), self.message_hub_rx.clone())
    }

    /// Send a message to another agent via the hub.
    pub fn send_message(
        &self,
        sender_id: &str,
        sender_name: &str,
        recipient_id: &str,
        content: &str,
    ) -> anyhow::Result<()> {
        self.message_hub_tx.send(AgentMessage {
            sender_id: sender_id.to_string(),
            sender_name: sender_name.to_string(),
            recipient_id: recipient_id.to_string(),
            content: content.to_string(),
        })?;
        Ok(())
    }

    /// Try to receive a pending message (non-blocking).
    pub fn try_recv_message(&self) -> Option<AgentMessage> {
        self.message_hub_rx.try_recv().ok()
    }

    fn push_agent(&self, team: &mut Vec<AgentInstance>, key: &str, task_type: TaskType) {
        if let Some(persona) = self.registry.get(key) {
            let id = format!("{}-{}", key, &Uuid::new_v4().to_string()[..8]);
            team.push(self.create_instance(&id, persona, task_type));
        }
    }

    fn create_instance(
        &self,
        id: &str,
        persona: &AgentPersona,
        task_type: TaskType,
    ) -> AgentInstance {
        let core_memory = self
            .memory_store
            .load_memory(&persona.name, &persona.system_prompt);
        AgentInstance {
            id: id.to_string(),
            persona: persona.clone(),
            task_type,
            core_memory,
        }
    }
}

impl Default for AgentSpawner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_team_for_goal_full_team() {
        let spawner = AgentSpawner::new();
        let team = spawner.spawn_team_for_goal("Build a website with React");
        assert!(
            team.len() >= 5,
            "full team should have ≥5 agents, got {}",
            team.len()
        );
    }

    #[test]
    fn spawn_team_for_goal_vision_team() {
        let spawner = AgentSpawner::new();
        let team = spawner.spawn_team_for_goal("Design a beautiful UI layout");
        assert_eq!(team.len(), 3, "vision team should have 3 agents");
    }

    #[test]
    fn spawn_team_for_goal_debug_team() {
        let spawner = AgentSpawner::new();
        let team = spawner.spawn_team_for_goal("Fix the authentication bug");
        assert_eq!(team.len(), 2, "debug team should have 2 agents");
    }

    #[test]
    fn spawn_team_for_goal_minimal_team() {
        let spawner = AgentSpawner::new();
        let team = spawner.spawn_team_for_goal("Write a script to sort files");
        assert_eq!(team.len(), 1, "minimal team should have 1 agent");
    }

    #[test]
    fn spawn_agent_for_task_returns_correct_type() {
        let spawner = AgentSpawner::new();
        let agent = spawner.spawn_agent_for_task(TaskType::Testing);
        assert_eq!(agent.task_type, TaskType::Testing);
        assert_eq!(agent.persona.name, "TestBot");
    }

    #[test]
    fn message_hub_send_recv() {
        let spawner = AgentSpawner::new();
        spawner
            .send_message("sender-1", "Amelia", "recipient-1", "Hello")
            .unwrap();

        let msg = spawner.try_recv_message().unwrap();
        assert_eq!(msg.sender_id, "sender-1");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn message_hub_empty_recv() {
        let spawner = AgentSpawner::new();
        assert!(spawner.try_recv_message().is_none());
    }

    #[test]
    fn spawner_with_custom_memory_store() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spawner = AgentSpawner::with_memory_store(tmp.path().to_str().unwrap());
        let agent = spawner.spawn_agent_for_task(TaskType::Coding);
        assert_eq!(agent.persona.name, "Amelia");
    }
}
