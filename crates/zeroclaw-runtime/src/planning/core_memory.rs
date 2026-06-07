//! Letta-Style Core Memory System
//!
//! Agents can autonomously self-modify their memory via XML tags in
//! their output. The orchestrator parses these and persists changes.
//!
//! Adapted from OpenMAD's memory.rs (CoreMemory, AgentMemoryStore)
//! and orchestrator.rs (parse_and_apply_memory_updates).
//!
//! ## Single source of truth
//!
//! Core memory is persisted per-agent in a JSON file on disk. It is
//! *not* stored in ZeroClaw Config — the agent's core memory is
//! ephemeral-to-persistent runtime state that the LLM can mutate
//! autonomously.
//!
//! ## Memory update protocol
//!
//! An agent can update its core memory by emitting XML in its output:
//!
//! ```xml
//! <update_core_memory block="persona">new self-instructions</update_core_memory>
//! <update_core_memory block="human">new info about the user</update_core_memory>
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core memory blocks (persona, human, and any custom labels).
///
/// This is the agent's persistent self-knowledge that it can read
/// and write through a structured XML protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMemory {
    pub blocks: HashMap<String, String>,
}

impl CoreMemory {
    /// Create a new core memory with persona and human blocks.
    pub fn new(persona_desc: &str, human_desc: &str) -> Self {
        let mut blocks = HashMap::new();
        blocks.insert("persona".to_string(), persona_desc.to_string());
        blocks.insert("human".to_string(), human_desc.to_string());
        Self { blocks }
    }

    /// Get the value of a named block.
    pub fn get_block(&self, label: &str) -> Option<&String> {
        self.blocks.get(label)
    }

    /// Set/update a named block.
    pub fn set_block(&mut self, label: &str, value: &str) {
        self.blocks.insert(label.to_string(), value.to_string());
    }

    /// Render the memory as XML that can be injected into LLM prompts.
    pub fn to_xml(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<core_memory>\n");
        for (label, val) in &self.blocks {
            xml.push_str(&format!("  <{label}>\n    {val}\n  </{label}>\n"));
        }
        xml.push_str("</core_memory>");
        xml
    }

    /// Render a compact single-line summary for logging.
    pub fn summary(&self) -> String {
        let count = self.blocks.len();
        let total_chars: usize = self.blocks.values().map(|v| v.len()).sum();
        format!("CoreMemory({count} blocks, {total_chars} chars)")
    }
}

/// Parse and apply Letta-style memory updates from an agent's output.
///
/// Scans `output` for `<update_core_memory block="X">value</update_core_memory>`
/// tags and applies them to `core_memory`, persisting through `memory_store`.
pub fn parse_and_apply_memory_updates(
    output: &str,
    agent_name: &str,
    core_memory: &mut CoreMemory,
    memory_store: &AgentMemoryStore,
) {
    let start_tag_prefix = "<update_core_memory block=\"";
    let end_tag = "</update_core_memory>";

    let mut current_pos = 0;
    let mut updated_blocks = Vec::new();

    while let Some(start_idx) = output[current_pos..].find(start_tag_prefix) {
        let absolute_start = current_pos + start_idx;
        let block_name_start = absolute_start + start_tag_prefix.len();

        if let Some(quote_idx) = output[block_name_start..].find('"') {
            let block_name_end = block_name_start + quote_idx;
            let block_name = &output[block_name_start..block_name_end];

            let val_start = block_name_end + 2; // skip `">`
            if val_start < output.len() {
                if let Some(end_idx) = output[val_start..].find(end_tag) {
                    let absolute_end = val_start + end_idx;
                    let val = &output[val_start..absolute_end];

                    core_memory.set_block(block_name, val.trim());
                    updated_blocks.push(block_name.to_string());

                    zeroclaw_log::record!(
                        INFO,
                        zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Write)
                            .with_outcome(zeroclaw_log::EventOutcome::Success)
                            .with_attrs(serde_json::json!({
                                "agent": agent_name,
                                "block": block_name,
                                "value_length": val.trim().len(),
                            })),
                        "core memory update: agent '{agent_name}' updated block '{block_name}'"
                    );

                    current_pos = absolute_end + end_tag.len();
                    continue;
                }
            }
        }
        current_pos += start_tag_prefix.len();
    }

    // Persist if any blocks were updated.
    if !updated_blocks.is_empty() {
        if let Err(e) = memory_store.save_memory(agent_name, core_memory) {
            zeroclaw_log::record!(
                ERROR,
                zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Fail)
                    .with_outcome(zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(serde_json::json!({
                        "agent": agent_name,
                        "error": e.to_string(),
                    })),
                "failed to save core memory for agent '{agent_name}': {e}"
            );
        }
    }
}

/// A file-backed store for agent core memories.
///
/// Each agent has a named entry in a single JSON file. The file is
/// created on first save if it doesn't exist.
#[derive(Debug, Clone)]
pub struct AgentMemoryStore {
    file_path: String,
}

impl AgentMemoryStore {
    /// Create a new store backed by the given JSON file path.
    pub fn new(path: &str) -> Self {
        Self {
            file_path: path.to_string(),
        }
    }

    /// Load core memory for an agent, falling back to defaults.
    pub fn load_memory(&self, agent_name: &str, default_persona: &str) -> CoreMemory {
        if std::path::Path::new(&self.file_path).exists() {
            if let Ok(content) = std::fs::read_to_string(&self.file_path) {
                if let Ok(mut store) = serde_json::from_str::<HashMap<String, CoreMemory>>(&content)
                {
                    if let Some(mem) = store.remove(agent_name) {
                        zeroclaw_log::record!(
                            INFO,
                            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Read)
                                .with_outcome(zeroclaw_log::EventOutcome::Success)
                                .with_attrs(serde_json::json!({
                                    "agent": agent_name,
                                })),
                            "loaded persistent core memory for '{agent_name}'"
                        );
                        return mem;
                    }
                }
            }
        }

        CoreMemory::new(
            default_persona,
            "The user wants to complete the orchestrator goals. Prefers clean code and clear logs.",
        )
    }

    /// Save core memory for an agent to the JSON file.
    pub fn save_memory(&self, agent_name: &str, memory: &CoreMemory) -> anyhow::Result<()> {
        let mut store: HashMap<String, CoreMemory> =
            if std::path::Path::new(&self.file_path).exists() {
                std::fs::read_to_string(&self.file_path)
                    .ok()
                    .and_then(|c| serde_json::from_str(&c).ok())
                    .unwrap_or_default()
            } else {
                HashMap::new()
            };

        store.insert(agent_name.to_string(), memory.clone());
        let serialized = serde_json::to_string_pretty(&store)?;

        // Ensure parent directory exists.
        if let Some(parent) = std::path::Path::new(&self.file_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&self.file_path, serialized)?;

        zeroclaw_log::record!(
            INFO,
            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Write)
                .with_outcome(zeroclaw_log::EventOutcome::Success)
                .with_attrs(serde_json::json!({
                    "agent": agent_name,
                })),
            "saved persistent core memory for '{agent_name}'"
        );

        Ok(())
    }

    /// Delete an agent's memory from the store.
    pub fn delete_memory(&self, agent_name: &str) -> anyhow::Result<()> {
        let mut store: HashMap<String, CoreMemory> =
            if std::path::Path::new(&self.file_path).exists() {
                std::fs::read_to_string(&self.file_path)
                    .ok()
                    .and_then(|c| serde_json::from_str(&c).ok())
                    .unwrap_or_default()
            } else {
                return Ok(());
            };

        store.remove(agent_name);
        let serialized = serde_json::to_string_pretty(&store)?;
        std::fs::write(&self.file_path, serialized)?;
        Ok(())
    }

    /// List all agent names with stored memories.
    pub fn list_agents(&self) -> Vec<String> {
        if std::path::Path::new(&self.file_path).exists() {
            std::fs::read_to_string(&self.file_path)
                .ok()
                .and_then(|c| serde_json::from_str::<HashMap<String, CoreMemory>>(&c).ok())
                .map(|map| map.into_keys().collect())
                .unwrap_or_default()
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_memory_new_creates_blocks() {
        let mem = CoreMemory::new("You are a coder.", "User likes Rust.");
        assert_eq!(mem.get_block("persona").unwrap(), "You are a coder.");
        assert_eq!(mem.get_block("human").unwrap(), "User likes Rust.");
    }

    #[test]
    fn core_memory_set_block() {
        let mut mem = CoreMemory::new("A", "B");
        mem.set_block("persona", "Updated persona");
        assert_eq!(mem.get_block("persona").unwrap(), "Updated persona");
    }

    #[test]
    fn core_memory_to_xml_contains_blocks() {
        let mem = CoreMemory::new("P", "H");
        let xml = mem.to_xml();
        assert!(xml.contains("<persona>"));
        assert!(xml.contains("P"));
        assert!(xml.contains("<human>"));
        assert!(xml.contains("H"));
        assert!(xml.contains("</core_memory>"));
    }

    #[test]
    fn parse_and_apply_memory_updates_parses_tags() {
        let output = r#"Some work done.
<update_core_memory block="persona">I am now a Rust expert.</update_core_memory>
<update_core_memory block="human">User prefers async patterns.</update_core_memory>
Done."#;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = AgentMemoryStore::new(tmp.path().to_str().unwrap());
        let mut mem = CoreMemory::new("Old persona", "Old human");

        parse_and_apply_memory_updates(output, "test-agent", &mut mem, &store);

        assert_eq!(mem.get_block("persona").unwrap(), "I am now a Rust expert.");
        assert_eq!(
            mem.get_block("human").unwrap(),
            "User prefers async patterns."
        );
    }

    #[test]
    fn parse_and_apply_ignores_output_without_tags() {
        let output = "Just a normal response. No updates here.";
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = AgentMemoryStore::new(tmp.path().to_str().unwrap());
        let mut mem = CoreMemory::new("P", "H");

        parse_and_apply_memory_updates(output, "agent", &mut mem, &store);

        assert_eq!(mem.get_block("persona").unwrap(), "P");
        assert_eq!(mem.get_block("human").unwrap(), "H");
    }

    #[test]
    fn agent_memory_store_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = AgentMemoryStore::new(tmp.path().to_str().unwrap());

        let mem = CoreMemory::new("Expert", "Loves Rust");
        store.save_memory("amelia", &mem).unwrap();

        let loaded = store.load_memory("amelia", "Default");
        assert_eq!(loaded.get_block("persona").unwrap(), "Expert");
        assert_eq!(loaded.get_block("human").unwrap(), "Loves Rust");
    }

    #[test]
    fn agent_memory_store_fallback_to_default() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = AgentMemoryStore::new(tmp.path().to_str().unwrap());

        let loaded = store.load_memory("unknown", "Default Persona");
        assert_eq!(loaded.get_block("persona").unwrap(), "Default Persona");
    }

    #[test]
    fn agent_memory_store_delete() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = AgentMemoryStore::new(tmp.path().to_str().unwrap());

        store
            .save_memory("agent1", &CoreMemory::new("A", "B"))
            .unwrap();
        store.delete_memory("agent1").unwrap();

        assert!(store.list_agents().is_empty());
    }

    #[test]
    fn core_memory_summary_format() {
        let mem = CoreMemory::new("ABC", "DE");
        let s = mem.summary();
        assert!(s.contains("2 blocks"));
    }
}
