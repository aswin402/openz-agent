//! Local Skill Auto-Evolution implementation.
//!
//! Analyzes shell command execution history from runtime logs and synthesizes
//! recurring command patterns into parameterized ZeroClaw skills.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use zeroclaw_config::schema::Config;
use zeroclaw_providers::ModelProvider;

/// Sequence of shell commands identified in a trace.
#[derive(Debug, Clone)]
pub struct DiscoveredSequence {
    pub trace_id: String,
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesizedSkillTool {
    pub name: String,
    pub description: String,
    pub kind: String,
    pub command: String,
    #[serde(default)]
    pub args: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesizedSkill {
    pub name: String,
    pub description: String,
    pub prompts: Vec<String>,
    pub tools: Vec<SynthesizedSkillTool>,
}

pub struct LocalEvolution {
    /// Source of truth: config.agent_workspace_dir(agent_alias)
    workspace_dir: PathBuf,
}

impl LocalEvolution {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }

    /// Parse the log trace file to find consecutive successful shell tool calls.
    pub fn parse_history(&self, log_path: &Path) -> Result<Vec<DiscoveredSequence>> {
        if !log_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(log_path)
            .with_context(|| format!("Failed to open log file: {}", log_path.display()))?;
        let reader = BufReader::new(file);

        // Group commands by trace_id and span_id.
        let mut trace_invocations: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut completed_spans: HashMap<String, bool> = HashMap::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let val: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let trace_id = match val.get("trace_id").and_then(|v| v.as_str()) {
                Some(tid) => tid.to_string(),
                None => continue,
            };

            let event = match val.get("event") {
                Some(e) => e,
                None => continue,
            };

            let category = event.get("category").and_then(|v| v.as_str()).unwrap_or("");
            let action = event.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let outcome = event.get("outcome").and_then(|v| v.as_str()).unwrap_or("");
            let span_id = val
                .get("span_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if category == "tool" {
                if action == "invoke" {
                    let tool = val
                        .get("zeroclaw")
                        .and_then(|z| z.get("fields"))
                        .and_then(|f| f.get("tool"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    if tool == "shell" {
                        if let Some(cmd) = val
                            .get("attributes")
                            .and_then(|a| a.get("input"))
                            .and_then(|i| i.get("command"))
                            .and_then(|c| c.as_str())
                        {
                            trace_invocations
                                .entry(trace_id)
                                .or_default()
                                .push((span_id, cmd.to_string()));
                        }
                    }
                } else if (action == "complete" || action == "fail") && !span_id.is_empty() {
                    let success = outcome == "success";
                    completed_spans.insert(span_id, success);
                }
            }
        }

        let mut sequences = Vec::new();
        for (trace_id, calls) in trace_invocations {
            let mut commands = Vec::new();
            for (span_id, cmd) in calls {
                if completed_spans.get(&span_id).copied().unwrap_or(false) {
                    commands.push(cmd);
                }
            }
            if commands.len() >= 2 {
                sequences.push(DiscoveredSequence { trace_id, commands });
            }
        }

        Ok(sequences)
    }

    /// Run the local evolution process: read history, call model provider, save skills.
    pub async fn evolve(
        &self,
        _agent_alias: &str,
        config: &Config,
        provider: &dyn ModelProvider,
        model_name: &str,
    ) -> Result<Vec<String>> {
        // Find log trace path
        let log_path = zeroclaw_log::current_log_path()
            .unwrap_or_else(|| self.workspace_dir.join("state").join("runtime-trace.jsonl"));

        let sequences = self.parse_history(&log_path)?;
        if sequences.is_empty() {
            ::zeroclaw_log::record!(
                INFO,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note),
                "No execution history sequences found for local skill evolution"
            );
            return Ok(Vec::new());
        }

        // Format sequences for LLM prompt
        let mut history_str = String::new();
        for (i, seq) in sequences.iter().enumerate() {
            history_str.push_str(&format!("Sequence {}:\n", i + 1));
            for cmd in &seq.commands {
                history_str.push_str(&format!("  - {}\n", cmd));
            }
        }

        let system_prompt = "You are a ZeroClaw system architect. Analyze the execution history sequences of shell commands, find patterns of 2 or more commands that should be packaged together, and synthesize them into reusable parameterized skills.

Each skill must have:
- A unique, URL-safe, lowercase, hyphenated name (e.g. `rust-test-build` or `git-commit-all`).
- A description of what it does.
- Triggering prompts (phrases or user descriptions that indicate when the skill is useful).
- An array of tools. For local skills, these are `kind = \"shell\"` and have `command` template strings featuring parameterized placeholders like `{{parameter_name}}` (e.g. `cargo test {{test_name}}`).
- The args map defining descriptions for each parameter.

Your response MUST be a valid JSON array of skill objects. Respond with ONLY the raw JSON block. No explanation outside the JSON.
JSON Schema:
[
  {
    \"name\": \"string\",
    \"description\": \"string\",
    \"prompts\": [\"string\"],
    \"tools\": [
      {
        \"name\": \"string\",
        \"description\": \"string\",
        \"kind\": \"shell\",
        \"command\": \"string\",
        \"args\": {
          \"parameter_name\": \"description\"
        }
      }
    ]
  }
]";

        let response = provider
            .chat_with_system(Some(system_prompt), &history_str, model_name, None)
            .await?;

        let json_text = extract_json_block(&response);
        let skills: Vec<SynthesizedSkill> = serde_json::from_str(json_text).with_context(|| {
            format!(
                "Failed to parse LLM response as SynthesizedSkill array. Raw response:\n{}",
                response
            )
        })?;

        let shared_skills_dir = config
            .shared_workspace_dir()
            .join("skills")
            .join("synthesized");
        fs::create_dir_all(&shared_skills_dir)?;

        let mut evolved_skills = Vec::new();
        for skill in skills {
            let skill_dir = shared_skills_dir.join(&skill.name);
            fs::create_dir_all(&skill_dir)?;

            // Generate SKILL.toml
            let mut toml_str = String::new();
            toml_str.push_str("[skill]\n");
            toml_str.push_str(&format!("name = \"{}\"\n", escape_toml(&skill.name)));
            toml_str.push_str(&format!(
                "description = \"{}\"\n",
                escape_toml(&skill.description)
            ));
            toml_str.push_str("version = \"0.1.0\"\n");
            toml_str.push_str("author = \"zeroclaw-auto\"\n");
            toml_str.push_str("tags = [\"auto-generated\", \"evolved\"]\n");

            if !skill.prompts.is_empty() {
                toml_str.push_str("prompts = [\n");
                for p in &skill.prompts {
                    toml_str.push_str(&format!("  \"{}\",\n", escape_toml(p)));
                }
                toml_str.push_str("]\n");
            }

            for tool in &skill.tools {
                toml_str.push_str("\n[[tools]]\n");
                toml_str.push_str(&format!("name = \"{}\"\n", escape_toml(&tool.name)));
                toml_str.push_str(&format!(
                    "description = \"{}\"\n",
                    escape_toml(&tool.description)
                ));
                toml_str.push_str("kind = \"shell\"\n");
                toml_str.push_str(&format!("command = \"{}\"\n", escape_toml(&tool.command)));
                if !tool.args.is_empty() {
                    toml_str.push_str("args = { ");
                    let args_formatted: Vec<String> = tool
                        .args
                        .iter()
                        .map(|(k, v)| format!("{} = \"{}\"", k, escape_toml(v)))
                        .collect();
                    toml_str.push_str(&args_formatted.join(", "));
                    toml_str.push_str(" }\n");
                }
            }

            fs::write(skill_dir.join("SKILL.toml"), toml_str)?;

            // Generate SKILL.md
            let mut md_str = String::new();
            md_str.push_str(&format!("# {}\n\n", skill.name));
            md_str.push_str(&format!("> {}\n\n", skill.description));
            md_str.push_str("## Evolved Tools\n\n");
            for tool in &skill.tools {
                md_str.push_str(&format!("### {}\n", tool.name));
                md_str.push_str(&format!("- **Description**: {}\n", tool.description));
                md_str.push_str(&format!("- **Template**: `{}`\n\n", tool.command));
            }
            fs::write(skill_dir.join("SKILL.md"), md_str)?;

            evolved_skills.push(skill.name.clone());
        }

        Ok(evolved_skills)
    }
}

fn extract_json_block(text: &str) -> &str {
    let text = text.trim();
    if let Some(start) = text.find("```json") {
        let rest = &text[start + 7..];
        if let Some(end) = rest.find("```") {
            return rest[..end].trim();
        }
    }
    if let Some(start) = text.find("```") {
        let rest = &text[start + 3..];
        if let Some(end) = rest.find("```") {
            return rest[..end].trim();
        }
    }
    text
}

fn escape_toml(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_history_extracts_sequences() {
        let tmp = tempdir().unwrap();
        let log_path = tmp.path().join("trace.jsonl");

        let log_content = r#"{"trace_id":"t1","span_id":"s1","event":{"category":"tool","action":"invoke"},"zeroclaw":{"fields":{"tool":"shell"}},"attributes":{"input":{"command":"cargo check"}}}
{"trace_id":"t1","span_id":"s1","event":{"category":"tool","action":"complete","outcome":"success"}}
{"trace_id":"t1","span_id":"s2","event":{"category":"tool","action":"invoke"},"zeroclaw":{"fields":{"tool":"shell"}},"attributes":{"input":{"command":"cargo test"}}}
{"trace_id":"t1","span_id":"s2","event":{"category":"tool","action":"complete","outcome":"success"}}
{"trace_id":"t2","span_id":"s3","event":{"category":"tool","action":"invoke"},"zeroclaw":{"fields":{"tool":"shell"}},"attributes":{"input":{"command":"git status"}}}
{"trace_id":"t2","span_id":"s3","event":{"category":"tool","action":"complete","outcome":"success"}}
"#;

        fs::write(&log_path, log_content).unwrap();

        let evolver = LocalEvolution::new(tmp.path().to_path_buf());
        let seqs = evolver.parse_history(&log_path).unwrap();

        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0].trace_id, "t1");
        assert_eq!(seqs[0].commands, vec!["cargo check", "cargo test"]);
    }
}
