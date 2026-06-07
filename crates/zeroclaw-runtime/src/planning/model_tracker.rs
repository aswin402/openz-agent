//! Model Performance Tracking and Dynamic Routing.
//!
//! Tracks success/failure counts per model name and computes
//! historical success rates for dynamic routing decisions.
//!
//! Adapted from OpenMAD's model_router.rs (ModelPerformance).
//!
//! ## Usage
//!
//! The tracker wraps a `HashMap<String, ModelPerformance>` and
//! provides `record_success()`, `record_failure()`, and
//! `best_model()` for dynamic selection.
//!
//! ## Single source of truth
//!
//! Performance data is ephemeral runtime state (in-memory). It is
//! *not* persisted across restarts. If persistence is desired, the
//! caller can serialize via `to_json()` / `from_json()`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Performance counters for a single model.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub successes: u32,
    pub failures: u32,
}

impl ModelPerformance {
    /// Historical success rate in [0.0, 1.0].
    ///
    /// Returns 1.0 when no calls have been recorded (encourages trial
    /// of untested models).
    pub fn success_rate(&self) -> f64 {
        let total = self.successes + self.failures;
        if total == 0 {
            1.0
        } else {
            self.successes as f64 / total as f64
        }
    }

    /// Total number of recorded calls.
    pub fn total_calls(&self) -> u32 {
        self.successes + self.failures
    }
}

/// Thread-safe model performance tracker.
///
/// Records per-model success/failure and provides dynamic model
/// selection based on historical performance.
#[derive(Debug, Clone)]
pub struct ModelTracker {
    stats: Arc<RwLock<HashMap<String, ModelPerformance>>>,
}

impl ModelTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a successful call for a model.
    pub fn record_success(&self, model_name: &str) {
        let mut stats = self.stats.write().unwrap();
        let perf = stats.entry(model_name.to_string()).or_default();
        perf.successes += 1;
        let rate = perf.success_rate();

        zeroclaw_log::record!(
            INFO,
            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Complete)
                .with_outcome(zeroclaw_log::EventOutcome::Success)
                .with_attrs(serde_json::json!({
                    "model": model_name,
                    "successes": perf.successes,
                    "failures": perf.failures,
                    "rate": rate,
                })),
            "model '{model_name}' succeeded"
        );
    }

    /// Record a failed call for a model.
    pub fn record_failure(&self, model_name: &str) {
        let mut stats = self.stats.write().unwrap();
        let perf = stats.entry(model_name.to_string()).or_default();
        perf.failures += 1;
        let rate = perf.success_rate();

        zeroclaw_log::record!(
            WARN,
            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Fail)
                .with_outcome(zeroclaw_log::EventOutcome::Failure)
                .with_attrs(serde_json::json!({
                    "model": model_name,
                    "successes": perf.successes,
                    "failures": perf.failures,
                    "rate": rate,
                })),
            "model '{model_name}' failed"
        );
    }

    /// Get the performance stats for a model.
    pub fn get_performance(&self, model_name: &str) -> ModelPerformance {
        self.stats
            .read()
            .unwrap()
            .get(model_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Get the success rate for a model (1.0 if no data).
    pub fn success_rate(&self, model_name: &str) -> f64 {
        self.get_performance(model_name).success_rate()
    }

    /// Select the best model from a list based on historical rates.
    ///
    /// Returns the model with the highest success rate. Ties are
    /// broken in favor of the first model in the list.
    pub fn best_model(&self, candidates: &[&str]) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }

        candidates
            .iter()
            .map(|name| (name, self.success_rate(name)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.to_string())
    }

    /// Serialize all stats to JSON.
    pub fn to_json(&self) -> String {
        let stats = self.stats.read().unwrap();
        serde_json::to_string_pretty(&*stats).unwrap_or_default()
    }

    /// Load stats from a JSON string.
    pub fn from_json(&self, json: &str) {
        if let Ok(stats) = serde_json::from_str::<HashMap<String, ModelPerformance>>(json) {
            let mut current = self.stats.write().unwrap();
            for (key, perf) in stats {
                current.insert(key, perf);
            }
        }
    }

    /// Number of tracked models.
    pub fn len(&self) -> usize {
        self.stats.read().unwrap().len()
    }

    /// True if no models are tracked.
    pub fn is_empty(&self) -> bool {
        self.stats.read().unwrap().is_empty()
    }
}

impl Default for ModelTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_performance_default_rate_is_1() {
        let perf = ModelPerformance::default();
        assert!((perf.success_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn model_performance_rate_reflects_counts() {
        let mut perf = ModelPerformance::default();
        perf.successes = 3;
        perf.failures = 1;
        assert!((perf.success_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn tracker_record_success() {
        let tracker = ModelTracker::new();
        tracker.record_success("gemini-pro");
        assert_eq!(tracker.get_performance("gemini-pro").successes, 1);
    }

    #[test]
    fn tracker_record_failure() {
        let tracker = ModelTracker::new();
        tracker.record_failure("gemini-pro");
        assert_eq!(tracker.get_performance("gemini-pro").failures, 1);
    }

    #[test]
    fn tracker_best_model_selects_highest_rate() {
        let tracker = ModelTracker::new();
        tracker.record_success("model-a"); // 1.0 (1/0)
        tracker.record_failure("model-b"); // 0.0 (0/1)
        tracker.record_success("model-b"); // 0.5 (1/1)

        let best = tracker.best_model(&["model-a", "model-b"]);
        assert_eq!(best.unwrap(), "model-a");
    }

    #[test]
    fn tracker_best_model_empty_list() {
        let tracker = ModelTracker::new();
        assert!(tracker.best_model(&[]).is_none());
    }

    #[test]
    fn tracker_json_roundtrip() {
        let tracker = ModelTracker::new();
        tracker.record_success("model-a");
        tracker.record_success("model-a");
        tracker.record_failure("model-b");

        let json = tracker.to_json();
        let tracker2 = ModelTracker::new();
        tracker2.from_json(&json);

        assert_eq!(tracker2.get_performance("model-a").successes, 2);
        assert_eq!(tracker2.get_performance("model-b").failures, 1);
    }

    #[test]
    fn tracker_len_reflects_tracked_models() {
        let tracker = ModelTracker::new();
        assert!(tracker.is_empty());
        tracker.record_success("m1");
        tracker.record_failure("m2");
        assert_eq!(tracker.len(), 2);
    }
}
