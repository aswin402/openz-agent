//! Semantic Memory / Embedding Search for agent context injection.
//!
//! Provides a multi-tier memory system:
//!
//! - **Short-term history**: rolling window of recent events/utterances.
//! - **Long-term semantic store**: text entries with embeddings for
//!   cosine-similarity retrieval (when `memory-embeddings` feature is
//!   enabled).
//!
//! Adapted from OpenMAD's memory.rs (MemoryEngine, SharedAgentMemory).
//!
//! ## Feature flags
//!
//! * `memory-embeddings` — enables [`fastembed`]-based local embedding
//!   generation and [`dashmap`]-concurrent shared memory. Without this
//!   feature, semantic queries return empty results and the engine runs
//!   in offline/keyword-only mode.
//!
//! ## Single source of truth
//!
//! Memory entries are ephemeral runtime state — they are not persisted
//! in Config. Persisting to a file store is the caller's responsibility
//! (see [`crate::planning::core_memory::AgentMemoryStore`]).

use std::sync::{Arc, RwLock};

/// A single memory entry with its embedding vector.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: String,
}

/// Multi-tier memory engine.
///
/// Stores short-term conversational history and (optionally) long-term
/// semantic vectors for similarity search.
pub struct MemoryEngine {
    #[cfg(feature = "memory-embeddings")]
    model: Option<fastembed::TextEmbedding>,
    long_term_memories: Arc<RwLock<Vec<MemoryEntry>>>,
    short_term_history: Arc<RwLock<Vec<String>>>,
}

impl MemoryEngine {
    /// Create a new memory engine.
    ///
    /// When the `memory-embeddings` feature is enabled, this attempts
    /// to initialize the fastembed model. If that fails (or the feature
    /// is disabled), the engine degrades gracefully to text-only storage.
    pub fn new() -> Self {
        #[cfg(feature = "memory-embeddings")]
        let model = match fastembed::TextEmbedding::try_new(fastembed::InitOptions::default()) {
            Ok(m) => {
                zeroclaw_log::record!(
                    INFO,
                    zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Read)
                        .with_outcome(zeroclaw_log::EventOutcome::Success),
                    "semantic memory: fastembed model initialized"
                );
                Some(m)
            }
            Err(e) => {
                zeroclaw_log::record!(
                    WARN,
                    zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Fail)
                        .with_outcome(zeroclaw_log::EventOutcome::Success)
                        .with_attrs(serde_json::json!({"error": e.to_string()})),
                    "semantic memory: fastembed init failed, running in offline mode: {e}"
                );
                None
            }
        };

        Self {
            #[cfg(feature = "memory-embeddings")]
            model,
            long_term_memories: Arc::new(RwLock::new(Vec::new())),
            short_term_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Store a text entry in both short-term history and long-term
    /// semantic memory (with embedding if available).
    pub fn store_memory(&self, text: &str, metadata: &str) {
        // 1. Short-term history
        {
            let mut history = self.short_term_history.write().unwrap();
            history.push(format!("[{metadata}] {text}"));
        }

        // 2. Long-term semantic store (with optional embedding)
        #[cfg(feature = "memory-embeddings")]
        if let Some(ref model) = self.model {
            match model.embed(vec![text], None) {
                Ok(embeddings) => {
                    if !embeddings.is_empty() {
                        let mut long_term = self.long_term_memories.write().unwrap();
                        long_term.push(MemoryEntry {
                            text: text.to_string(),
                            embedding: embeddings[0].clone(),
                            metadata: metadata.to_string(),
                        });
                    }
                }
                Err(e) => {
                    zeroclaw_log::record!(
                        ERROR,
                        zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Fail)
                            .with_outcome(zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(serde_json::json!({"error": e.to_string()})),
                        "embedding generation failed: {e}"
                    );
                }
            }
            return;
        }

        // Offline fallback (no-embedding path also always runs):
        #[allow(unused)]
        {
            let mut long_term = self.long_term_memories.write().unwrap();
            long_term.push(MemoryEntry {
                text: text.to_string(),
                embedding: vec![],
                metadata: metadata.to_string(),
            });
        }
    }

    /// Query semantic memory by similarity to `query`.
    ///
    /// Returns up to `limit` entries sorted by relevance (descending).
    /// Without `memory-embeddings`, returns the most recent entries
    /// with zero scores.
    pub fn query_semantic(&self, _query: &str, limit: usize) -> Vec<(String, f32)> {
        #[cfg(feature = "memory-embeddings")]
        {
            let query_embedding = if let Some(ref model) = self.model {
                match model.embed(vec![query], None) {
                    Ok(embeddings) => {
                        if embeddings.is_empty() {
                            return vec![];
                        }
                        embeddings[0].clone()
                    }
                    Err(e) => {
                        zeroclaw_log::record!(
                            ERROR,
                            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Fail)
                                .with_outcome(zeroclaw_log::EventOutcome::Failure)
                                .with_attrs(serde_json::json!({"error": e.to_string()})),
                            "query embedding failed: {e}"
                        );
                        return vec![];
                    }
                }
            } else {
                return vec![];
            };

            let memories = self.long_term_memories.read().unwrap();
            let mut results = Vec::new();

            for entry in memories.iter() {
                if entry.embedding.is_empty() || query_embedding.is_empty() {
                    continue;
                }
                let score = cosine_similarity(&query_embedding, &entry.embedding);
                results.push((format!("[{}] {}", entry.metadata, entry.text), score));
            }

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);
            results
        }

        #[cfg(not(feature = "memory-embeddings"))]
        {
            // Without embeddings, return most recent entries with zero score.
            let history = self.short_term_history.read().unwrap();
            let start = history.len().saturating_sub(limit);
            history[start..].iter().map(|h| (h.clone(), 0.0)).collect()
        }
    }

    /// Get the most recent short-term history entries.
    pub fn get_recent_history(&self, limit: usize) -> Vec<String> {
        let history = self.short_term_history.read().unwrap();
        let start = history.len().saturating_sub(limit);
        history[start..].to_vec()
    }

    /// Total number of stored long-term entries.
    pub fn long_term_count(&self) -> usize {
        self.long_term_memories.read().unwrap().len()
    }

    /// Total number of short-term history entries.
    pub fn short_term_count(&self) -> usize {
        self.short_term_history.read().unwrap().len()
    }
}

impl Default for MemoryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe shared artifact memory.
///
/// Agents can publish and retrieve key-value artifacts that survive
/// across task boundaries within an orchestration run.
#[derive(Debug, Clone)]
pub struct SharedAgentMemory {
    #[cfg(feature = "memory-embeddings")]
    artifacts: Arc<dashmap::DashMap<String, String>>,
    #[cfg(not(feature = "memory-embeddings"))]
    artifacts: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

impl SharedAgentMemory {
    /// Create a new empty shared memory.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "memory-embeddings")]
            artifacts: Arc::new(dashmap::DashMap::new()),
            #[cfg(not(feature = "memory-embeddings"))]
            artifacts: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Publish a key-value artifact.
    pub fn publish(&self, key: &str, content: &str) {
        #[cfg(feature = "memory-embeddings")]
        self.artifacts.insert(key.to_string(), content.to_string());
        #[cfg(not(feature = "memory-embeddings"))]
        {
            let mut map = self.artifacts.write().unwrap();
            map.insert(key.to_string(), content.to_string());
        }
    }

    /// Retrieve an artifact by key.
    pub fn get(&self, key: &str) -> Option<String> {
        #[cfg(feature = "memory-embeddings")]
        {
            self.artifacts.get(key).map(|v| v.clone())
        }
        #[cfg(not(feature = "memory-embeddings"))]
        {
            self.artifacts.read().unwrap().get(key).cloned()
        }
    }

    /// List all artifact keys.
    pub fn list_keys(&self) -> Vec<String> {
        #[cfg(feature = "memory-embeddings")]
        {
            self.artifacts.iter().map(|kv| kv.key().clone()).collect()
        }
        #[cfg(not(feature = "memory-embeddings"))]
        {
            self.artifacts.read().unwrap().keys().cloned().collect()
        }
    }

    /// Number of stored artifacts.
    pub fn len(&self) -> usize {
        #[cfg(feature = "memory-embeddings")]
        {
            self.artifacts.len()
        }
        #[cfg(not(feature = "memory-embeddings"))]
        {
            self.artifacts.read().unwrap().len()
        }
    }

    /// True if no artifacts are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SharedAgentMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two equal-length float vectors.
///
/// Returns a value in [-1.0, 1.0] (typically [0.0, 1.0] for embeddings).
/// Returns 0.0 if vectors are empty or mismatched.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_similarity_mismatched_length_returns_zero() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn memory_engine_store_and_recent_history() {
        let engine = MemoryEngine::new();
        engine.store_memory("First message", "step-1");
        engine.store_memory("Second message", "step-2");
        let recent = engine.get_recent_history(1);
        assert_eq!(recent.len(), 1);
        assert!(recent[0].contains("Second message"));
    }

    #[test]
    fn memory_engine_query_semantic_returns_results() {
        let engine = MemoryEngine::new();
        engine.store_memory("Rust is a systems language", "doc");
        engine.store_memory("Python is interpreted", "doc");
        let results = engine.query_semantic("programming language", 5);
        // When feature is disabled, returns recent entries with 0.0 score.
        // When enabled, returns scored results.
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn shared_agent_memory_publish_get() {
        let sm = SharedAgentMemory::new();
        sm.publish("task-1", "result-1");
        assert_eq!(sm.get("task-1").unwrap(), "result-1");
    }

    #[test]
    fn shared_agent_memory_list_keys() {
        let sm = SharedAgentMemory::new();
        sm.publish("k1", "v1");
        sm.publish("k2", "v2");
        let keys = sm.list_keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"k1".to_string()));
        assert!(keys.contains(&"k2".to_string()));
    }

    #[test]
    fn shared_agent_memory_is_empty() {
        let sm = SharedAgentMemory::new();
        assert!(sm.is_empty());
        sm.publish("a", "b");
        assert!(!sm.is_empty());
    }

    #[test]
    fn cosine_similarity_partial_match() {
        let a = vec![1.0, 0.0, 0.5];
        let b = vec![0.8, 0.1, 0.4];
        let score = cosine_similarity(&a, &b);
        assert!(score > 0.0 && score < 1.0);
    }
}
