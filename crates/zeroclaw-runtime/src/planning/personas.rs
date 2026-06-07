//! Agent Persona Templates with system prompts and guiding principles.
//!
//! Provides 7 canonical agent personas (matching the OpenMAD project
//! convention) plus a registry for custom persona overrides.
//!
//! Adapted from OpenMAD's agent.rs (AgentPersona, AgentRegistry).
//!
//! ## Single source of truth
//!
//! Persona definitions are code-constant defaults. Custom overrides
//! can be loaded from a `planning_agents.json` file next to the config,
//! but the canonical 7 are always available as fallbacks. Agent aliases
//! and model provider chains live in `zeroclaw.json` (ZeroClaw Config)
//! — this module only provides the *personality* (system prompt +
//! principles) that gets injected into the subagent's context.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::planning::TaskType;
use crate::planning::core_memory::CoreMemory;

/// An agent's identity: name, title, system prompt, and principles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPersona {
    pub name: String,
    pub title: String,
    pub system_prompt: String,
    pub principles: Vec<String>,
}

/// A fully-instantiated agent with persona and core memory.
#[derive(Debug, Clone)]
pub struct AgentInstance {
    pub id: String,
    pub persona: AgentPersona,
    pub task_type: TaskType,
    pub core_memory: CoreMemory,
}

/// Registry of available agent personas.
///
/// Pre-loaded with 7 canonical personas. Custom personas can be
/// registered at runtime or loaded from a JSON file.
pub struct AgentRegistry {
    personas: HashMap<String, AgentPersona>,
}

impl AgentRegistry {
    /// Create a new registry with all 7 default personas.
    pub fn new() -> Self {
        let mut registry = Self {
            personas: HashMap::new(),
        };
        registry.register_defaults();
        registry.try_load_custom();
        registry
    }

    /// Try to load custom persona overrides from `planning_agents.json`.
    fn try_load_custom(&mut self) {
        let path = std::path::Path::new("planning_agents.json");
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(custom) = serde_json::from_str::<HashMap<String, AgentPersona>>(&content)
                {
                    let count = custom.len();
                    for (key, persona) in custom {
                        self.personas.insert(key, persona);
                    }
                    if count > 0 {
                        zeroclaw_log::record!(
                            INFO,
                            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Read)
                                .with_outcome(zeroclaw_log::EventOutcome::Success)
                                .with_attrs(serde_json::json!({"count": count})),
                            "loaded {count} custom agent personas from planning_agents.json"
                        );
                    }
                }
            }
        }
    }

    /// Register or override a persona by key.
    pub fn register(&mut self, key: &str, persona: AgentPersona) {
        self.personas.insert(key.to_string(), persona);
    }

    /// Get a persona by key.
    pub fn get(&self, key: &str) -> Option<&AgentPersona> {
        self.personas.get(key)
    }

    /// Get all registered persona keys.
    pub fn keys(&self) -> Vec<String> {
        self.personas.keys().cloned().collect()
    }

    /// Number of registered personas.
    pub fn len(&self) -> usize {
        self.personas.len()
    }

    /// True if no personas are registered.
    pub fn is_empty(&self) -> bool {
        self.personas.is_empty()
    }

    fn register_defaults(&mut self) {
        // Business Analyst (BA) — Mary
        self.register(
            "mary",
            AgentPersona {
                name: "Mary".to_string(),
                title: "Business Analyst".to_string(),
                system_prompt: "You are Mary, the Expert Business Analyst. Your role is to analyze user requests, elicit hidden requirements, and detail product expectations. Focus on the value proposition, business edge cases, and client alignment.".to_string(),
                principles: vec![
                    "Identify hidden assumptions in user requests.".to_string(),
                    "Detail edge cases for business flows.".to_string(),
                    "Maintain extreme clarity in all specs.".to_string(),
                ],
            },
        );

        // Product Manager (PM) — John
        self.register(
            "john",
            AgentPersona {
                name: "John".to_string(),
                title: "Product Manager".to_string(),
                system_prompt: "You are John, the Product Manager. Your role is to formulate high-quality PRDs, epics, and modular user stories with clear acceptance criteria (UAT). You prioritize work and organize milestones.".to_string(),
                principles: vec![
                    "Decompose complex goals into clear, actionable stories.".to_string(),
                    "Define strict, testable acceptance criteria (UAT).".to_string(),
                    "Ensure user experience and functionality align with objectives.".to_string(),
                ],
            },
        );

        // System Architect — Winston
        self.register(
            "winston",
            AgentPersona {
                name: "Winston".to_string(),
                title: "System Architect".to_string(),
                system_prompt: "You are Winston, the System Architect. Your role is to evaluate technical requirements, design file hierarchies, specify database models, choose external libraries, and design the communication flows. Ensure safety, modularity, and high-performance in designs.".to_string(),
                principles: vec![
                    "Design systems for high scalability and decoupling.".to_string(),
                    "Audit libraries for security and licensing compatibility.".to_string(),
                    "Document interface signatures and data models cleanly.".to_string(),
                ],
            },
        );

        // Senior Engineer — Amelia
        self.register(
            "amelia",
            AgentPersona {
                name: "Amelia".to_string(),
                title: "Senior Software Engineer".to_string(),
                system_prompt: "You are Amelia, the Senior Engineer. Your role is to implement features, fix bugs, and refactor code according to architectural specifications. Write idiomatic, memory-safe, clean code with detailed comments.".to_string(),
                principles: vec![
                    "Follow language-specific idiomatic patterns (especially Rust safety rules).".to_string(),
                    "Write extensive unit tests and document public interfaces.".to_string(),
                    "Perform robust error handling without ignoring failures.".to_string(),
                ],
            },
        );

        // QA Tester — TestBot
        self.register(
            "tester",
            AgentPersona {
                name: "TestBot".to_string(),
                title: "QA Test Automator".to_string(),
                system_prompt: "You are the QA Test Automator. Your role is to write automated test suites (unit, integration, regression), run coverage checks, and verify correctness of engineer code against the PM's acceptance criteria.".to_string(),
                principles: vec![
                    "Test for failure conditions and empty values, not just happy paths.".to_string(),
                    "Aim for high test coverage and independent test execution.".to_string(),
                    "Generate clear reports of test failures with repro steps.".to_string(),
                ],
            },
        );

        // Technical Writer — Paige
        self.register(
            "paige",
            AgentPersona {
                name: "Paige".to_string(),
                title: "Technical Writer".to_string(),
                system_prompt: "You are Paige, the Technical Writer. Your role is to write comprehensive user guides, project READMEs, architecture summaries, and API documentation. Ensure professional, technical, and readable documentation.".to_string(),
                principles: vec![
                    "Use precise markdown styling and clear formatting.".to_string(),
                    "Generate diagrams to represent data flows and structures.".to_string(),
                    "Verify file links and references are valid.".to_string(),
                ],
            },
        );

        // Vision Auditor — Vivian
        self.register(
            "vivian",
            AgentPersona {
                name: "Vivian".to_string(),
                title: "Vision Auditor".to_string(),
                system_prompt: "You are Vivian, the Vision Auditor. Your role is to analyze user interface designs, layouts, wireframes, diagrams, and evaluate visual compliance of frontend outputs. Ensure high aesthetic standards, consistent UI guidelines, and responsive designs.".to_string(),
                principles: vec![
                    "Verify alignment, color consistency, and spacing in user interfaces.".to_string(),
                    "Interpret visual diagrams and check design fidelity.".to_string(),
                    "Recommend layout improvements for high aesthetic quality.".to_string(),
                ],
            },
        );
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Lookup table: TaskType → registry key (persona key).
pub const TASK_TYPE_TO_PERSONA_KEY: &[(TaskType, &str)] = &[
    (TaskType::Planning, "john"),
    (TaskType::Research, "mary"),
    (TaskType::Coding, "amelia"),
    (TaskType::Review, "winston"),
    (TaskType::Testing, "tester"),
    (TaskType::Documentation, "paige"),
    (TaskType::Vision, "vivian"),
];

/// Get the default persona key for a task type.
pub fn persona_key_for_task(task_type: TaskType) -> &'static str {
    TASK_TYPE_TO_PERSONA_KEY
        .iter()
        .find(|(t, _)| *t == task_type)
        .map(|(_, k)| *k)
        .unwrap_or("amelia")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_all_7_personas() {
        let reg = AgentRegistry::new();
        assert_eq!(reg.len(), 7);
    }

    #[test]
    fn registry_contains_all_expected_keys() {
        let reg = AgentRegistry::new();
        for (_, key) in TASK_TYPE_TO_PERSONA_KEY {
            assert!(reg.get(key).is_some(), "missing persona key: {key}");
        }
    }

    #[test]
    fn each_persona_has_3_principles() {
        let reg = AgentRegistry::new();
        for key in reg.keys() {
            let p = reg.get(&key).unwrap();
            assert_eq!(
                p.principles.len(),
                3,
                "persona '{key}' should have 3 principles, got {}",
                p.principles.len()
            );
        }
    }

    #[test]
    fn persona_key_for_task_mapping() {
        assert_eq!(persona_key_for_task(TaskType::Planning), "john");
        assert_eq!(persona_key_for_task(TaskType::Coding), "amelia");
        assert_eq!(persona_key_for_task(TaskType::Vision), "vivian");
    }

    #[test]
    fn custom_registration_overrides() {
        let mut reg = AgentRegistry::new();
        reg.register(
            "amelia",
            AgentPersona {
                name: "Amelia-V2".to_string(),
                title: "Senior Engineer V2".to_string(),
                system_prompt: "Custom prompt.".to_string(),
                principles: vec!["Principle 1.".to_string()],
            },
        );
        let amelia = reg.get("amelia").unwrap();
        assert_eq!(amelia.name, "Amelia-V2");
    }

    #[test]
    fn agent_instance_creation() {
        let amelia = AgentPersona {
            name: "Amelia".to_string(),
            title: "Engineer".to_string(),
            system_prompt: "You are an engineer.".to_string(),
            principles: vec![],
        };
        let instance = AgentInstance {
            id: "amelia-1".to_string(),
            persona: amelia,
            task_type: TaskType::Coding,
            core_memory: CoreMemory::new("Engineer", "User"),
        };
        assert_eq!(instance.task_type, TaskType::Coding);
        assert_eq!(
            instance.core_memory.get_block("persona").unwrap(),
            "Engineer"
        );
    }
}
