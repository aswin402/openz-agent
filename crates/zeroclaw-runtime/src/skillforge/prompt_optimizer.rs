//! Genetic Prompt Optimizer implementation.
//!
//! Generates mutation variants of `IDENTITY.md`, evaluates them in a simulated
//! mock tool loop, and promotes the best performing prompt.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use zeroclaw_config::schema::Config;
use zeroclaw_providers::{ChatMessage, ModelProvider};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestCase {
    pub user_message: String,
    pub expected_tools: Vec<String>,
    pub mock_tool_results: HashMap<String, String>,
}

pub struct PromptOptimizer {
    /// Source of truth: config.agent_workspace_dir(agent_alias)
    workspace_dir: PathBuf,
}

impl PromptOptimizer {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }

    /// Load the current IDENTITY.md system prompt.
    pub fn load_identity(&self) -> Result<String> {
        let path = self.workspace_dir.join("IDENTITY.md");
        if !path.exists() {
            // Return a default baseline identity if none exists
            return Ok("# IDENTITY.md\nYou are ZeroClaw, a coding assistant.".to_string());
        }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read IDENTITY.md at {}", path.display()))?;
        Ok(content)
    }

    /// Save the winning prompt back to IDENTITY.md.
    pub fn save_identity(&self, content: &str) -> Result<()> {
        let path = self.workspace_dir.join("IDENTITY.md");
        fs::write(&path, content)
            .with_context(|| format!("Failed to write IDENTITY.md at {}", path.display()))?;
        Ok(())
    }

    /// Run the genetic optimization loop.
    pub async fn optimize(
        &self,
        _agent_alias: &str,
        _config: &Config,
        provider: &dyn ModelProvider,
        model_name: &str,
        generations: usize,
        eval_suite_path: &str,
    ) -> Result<String> {
        // Load evaluation suite
        let eval_suite = if eval_suite_path.is_empty() {
            default_eval_suite()
        } else {
            let path = Path::new(eval_suite_path);
            if path.exists() {
                let content = fs::read_to_string(path)?;
                serde_json::from_str(&content).context("Failed to parse evaluation suite JSON")?
            } else {
                println!(
                    "Warning: Evaluation suite path '{}' not found. Using default suite.",
                    eval_suite_path
                );
                default_eval_suite()
            }
        };

        // Load starting prompt
        let original_prompt = self.load_identity()?;
        println!("Loaded IDENTITY.md ({} bytes)", original_prompt.len());

        let pop_size = 4; // population size per generation
        let mut population = vec![original_prompt.clone()];

        // Generate initial variants
        println!("Generating initial prompt mutations...");
        for _i in 1..pop_size {
            match self
                .mutate_prompt(&original_prompt, provider, model_name)
                .await
            {
                Ok(mutated) => population.push(mutated),
                Err(e) => {
                    println!("Warning: prompt mutation failed: {}. Using original.", e);
                    population.push(original_prompt.clone());
                }
            }
        }

        let mut best_overall_prompt = original_prompt.clone();
        let mut best_overall_score = 0.0;

        for generation in 0..generations {
            println!("\n--- Generation {} ---", generation);
            let mut scores = Vec::new();

            for (idx, variant) in population.iter().enumerate() {
                let mut total_score = 0.0;
                for test in &eval_suite {
                    let score = self
                        .simulate_test_case(variant, test, provider, model_name)
                        .await;
                    total_score += score;
                }
                let avg_score = total_score / eval_suite.len() as f64;
                println!("  Variant {} average fitness: {:.4}", idx, avg_score);
                scores.push((idx, avg_score));
            }

            // Sort descending by score
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let best_idx = scores[0].0;
            let best_score = scores[0].1;
            let best_prompt = population[best_idx].clone();

            if best_score > best_overall_score {
                best_overall_score = best_score;
                best_overall_prompt = best_prompt.clone();
            }

            println!(
                "Generation {} Winner: Variant {} with score {:.4}",
                generation, best_idx, best_score
            );

            if generation + 1 < generations {
                // Keep the winner (elitism) and generate a mutated generation
                let mut next_population = vec![best_prompt.clone()];

                // Recombination of top 2
                if scores.len() > 1 {
                    let second_best_prompt = population[scores[1].0].clone();
                    match self
                        .crossover_prompts(&best_prompt, &second_best_prompt, provider, model_name)
                        .await
                    {
                        Ok(combined) => next_population.push(combined),
                        Err(_) => next_population.push(best_prompt.clone()),
                    }
                } else {
                    next_population.push(best_prompt.clone());
                }

                // Fill the rest with mutations of the winner
                while next_population.len() < pop_size {
                    match self.mutate_prompt(&best_prompt, provider, model_name).await {
                        Ok(mutated) => next_population.push(mutated),
                        Err(_) => next_population.push(best_prompt.clone()),
                    }
                }

                population = next_population;
            }
        }

        println!(
            "\nPromotion: Writing optimized prompt back to IDENTITY.md (Fitness: {:.4})",
            best_overall_score
        );
        self.save_identity(&best_overall_prompt)?;

        Ok(best_overall_prompt)
    }

    /// Simulate execution of a test case using the candidate prompt.
    async fn simulate_test_case(
        &self,
        prompt: &str,
        test_case: &TestCase,
        provider: &dyn ModelProvider,
        model_name: &str,
    ) -> f64 {
        let mut messages = vec![
            ChatMessage::system(prompt),
            ChatMessage::user(&test_case.user_message),
        ];

        let mut tools_called = Vec::new();
        let mut turn_count = 0;
        let mut total_tokens = 0;

        for _ in 0..5 {
            turn_count += 1;

            for msg in &messages {
                total_tokens += msg.content.len() / 4;
            }

            let response = match provider
                .chat_with_history(&messages, model_name, Some(0.0))
                .await
            {
                Ok(res) => res,
                Err(_) => break,
            };

            total_tokens += response.len() / 4;
            messages.push(ChatMessage::assistant(&response));

            let mut called_any = false;
            for expected_tool in &test_case.expected_tools {
                if response
                    .to_lowercase()
                    .contains(&expected_tool.to_lowercase())
                {
                    let mut mock_res = "Success".to_string();
                    for (key, val) in &test_case.mock_tool_results {
                        if response.contains(key) {
                            mock_res = val.clone();
                            break;
                        }
                    }
                    if mock_res == "Success" {
                        if let Some(val) = test_case.mock_tool_results.get(expected_tool) {
                            mock_res = val.clone();
                        }
                    }

                    tools_called.push(expected_tool.clone());
                    messages.push(ChatMessage::tool(mock_res));
                    called_any = true;
                    break;
                }
            }

            if !called_any {
                break;
            }
        }

        let matches = test_case
            .expected_tools
            .iter()
            .filter(|t| tools_called.contains(t))
            .count();

        let success_score = if test_case.expected_tools.is_empty() {
            1.0
        } else {
            matches as f64 / test_case.expected_tools.len() as f64
        };

        let turn_score = 1.0 / (turn_count as f64);
        let token_score = 1.0 / (total_tokens as f64 / 100.0).max(1.0);

        success_score * 0.7 + turn_score * 0.15 + token_score * 0.15
    }

    /// Ask the LLM to mutate a prompt.
    async fn mutate_prompt(
        &self,
        prompt: &str,
        provider: &dyn ModelProvider,
        model_name: &str,
    ) -> Result<String> {
        let system = "You are a expert prompt engineer. Mutate the user's agent system prompt to make it more concise, clear, and effective at tool usage. Output ONLY the new mutated prompt. No code blocks, no explanations.";
        let response = provider
            .chat_with_system(Some(system), prompt, model_name, Some(0.7))
            .await?;
        Ok(clean_response(&response))
    }

    /// Ask the LLM to combine two prompts.
    async fn crossover_prompts(
        &self,
        a: &str,
        b: &str,
        provider: &dyn ModelProvider,
        model_name: &str,
    ) -> Result<String> {
        let system = "You are a prompt engineer. Combine the best qualities, guidelines, and formatting of Prompt A and Prompt B into a single optimized prompt. Output ONLY the combined prompt. No explanations.";
        let msg = format!("Prompt A:\n```\n{}\n```\n\nPrompt B:\n```\n{}\n```", a, b);
        let response = provider
            .chat_with_system(Some(system), &msg, model_name, Some(0.7))
            .await?;
        Ok(clean_response(&response))
    }
}

fn clean_response(text: &str) -> String {
    let text = text.trim();
    if let Some(start) = text.find("```") {
        let rest = &text[start + 3..];
        let rest = if rest.starts_with("markdown") {
            &rest[8..]
        } else if rest.starts_with("text") {
            &rest[4..]
        } else {
            rest
        };
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }
    text.to_string()
}

fn default_eval_suite() -> Vec<TestCase> {
    vec![
        TestCase {
            user_message: "Format the codebase and check linting errors".to_string(),
            expected_tools: vec!["shell".to_string()],
            mock_tool_results: {
                let mut m = HashMap::new();
                m.insert("cargo fmt".to_string(), "Success".to_string());
                m
            },
        },
        TestCase {
            user_message: "Read AGENTS.md and summarize rules".to_string(),
            expected_tools: vec!["file_read".to_string()],
            mock_tool_results: {
                let mut m = HashMap::new();
                m.insert(
                    "AGENTS.md".to_string(),
                    "# AGENTS.md\nNo duplicate state rule.".to_string(),
                );
                m
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_load_save_identity() {
        let tmp = tempdir().unwrap();
        let optimizer = PromptOptimizer::new(tmp.path().to_path_buf());

        // Test loading default
        let default_prompt = optimizer.load_identity().unwrap();
        assert!(default_prompt.contains("You are ZeroClaw"));

        // Test saving and loading
        let new_prompt = "# IDENTITY.md\nYou are an optimized coding bot.";
        optimizer.save_identity(new_prompt).unwrap();
        let loaded = optimizer.load_identity().unwrap();
        assert_eq!(loaded, new_prompt);
    }

    #[test]
    fn test_clean_response() {
        let text1 = "```markdown\nHello World\n```";
        assert_eq!(clean_response(text1), "Hello World");

        let text2 = "```text\nTest\n```";
        assert_eq!(clean_response(text2), "Test");

        let text3 = "Hello World";
        assert_eq!(clean_response(text3), "Hello World");
    }
}
