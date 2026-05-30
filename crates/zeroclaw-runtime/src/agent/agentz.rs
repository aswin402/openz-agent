use crate::agent::loop_::AgentRunOverrides;
use anyhow::Result;
use std::path::Path;
use tokio::fs;
use zeroclaw_config::providers::ModelProviderRef;
use zeroclaw_config::schema::{AliasedAgentConfig, Config};

pub struct AgentzWorkflow;

impl AgentzWorkflow {
    pub async fn run(
        config: Config,
        agent_alias: &str,
        user_request: String,
        interactive: bool,
    ) -> Result<String> {
        let agent_alias = agent_alias.to_string();
        Box::pin(async move {
            let openz_dir = Path::new(".openz");
            let agents_dir = openz_dir.join("agents");
            let tasks_dir = openz_dir.join("tasks");

            fs::create_dir_all(&agents_dir).await?;
            fs::create_dir_all(&tasks_dir).await?;

            // Seed default subagent templates if they do not exist
            seed_default_agents(&agents_dir).await?;

            // Generate task name/directory using a slug of the prompt
            let date_str = chrono::Local::now().format("%Y-%m-%d").to_string();
            let slug = slugify(&user_request);
            let task_name = format!("{}-{}", date_str, slug);
            let task_path = tasks_dir.join(&task_name);
            fs::create_dir_all(&task_path).await?;

            // Write raw user request
            fs::write(task_path.join("request.md"), &user_request).await?;
            println!(
                "{}",
                console::style(format!(
                    "Initializing OpenZ Multi-Agent Task: {}",
                    task_name
                ))
                .cyan()
                .bold()
            );
            println!("Task directory: {}", task_path.display());

            // Preprocess user request for images (Vision subagent interception)
            let mut final_request = user_request.clone();
            if final_request.contains("[IMAGE:") {
                println!(
                    "{}",
                    console::style(
                        "Attached image detected! Preprocessing visual contents via vision-agent..."
                    )
                    .yellow()
                );
                let vision_description = run_vision_agent(&config, &agents_dir, &final_request).await?;
                final_request = replace_image_markers(&final_request, &vision_description);
                fs::write(task_path.join("request-with-vision.md"), &final_request).await?;
            }

            // 1. Research Phase
            println!(
                "{}",
                console::style("Executing Research Phase (research-agent)...").cyan()
            );
            let research_prompt = format!(
                "Analyze the following user task and perform local research on the codebase (using files, glob searches, content searches) or external search if needed to understand requirements, files to change, and target APIs.\n\n=== TASK ===\n{}",
                final_request
            );
            let research_report = run_research_agent(&config, &agents_dir, &research_prompt).await?;
            fs::write(task_path.join("research-report.md"), &research_report).await?;

            // 2. Planning & Lead Evaluation Loop
            let mut plan_approved = false;
            let mut plan_attempts = 0;
            let max_plan_attempts = 3;
            let mut spec = String::new();
            let mut plan = String::new();
            let mut files_to_change = String::new();
            let mut checklist = String::new();
            let mut lead_feedback = String::new();
            let mut current_research = research_report.clone();

            while !plan_approved && plan_attempts < max_plan_attempts {
                plan_attempts += 1;
                println!(
                    "{}",
                    console::style(format!(
                        "Executing Planning Phase (openz-planagent) - Attempt {}/{}...",
                        plan_attempts, max_plan_attempts
                    ))
                    .cyan()
                );

                let planner_prompt = if lead_feedback.is_empty() {
                    format!(
                        "Analyze the user request and research findings to generate the specification and implementation plan.\n\n=== USER REQUEST ===\n{}\n\n=== RESEARCH FINDINGS ===\n{}",
                        final_request, current_research
                    )
                } else {
                    format!(
                        "Your previous plan was REJECTED by the Lead Primary Agent. Analyze the feedback, perform any additional research if needed, and regenerate the spec and plan.\n\n=== USER REQUEST ===\n{}\n\n=== PREVIOUS SPEC ===\n{}\n\n=== PREVIOUS PLAN ===\n{}\n\n=== LEAD AGENT FEEDBACK ===\n{}",
                        final_request, spec, plan, lead_feedback
                    )
                };

                let (new_spec, new_plan, new_files, new_checklist) =
                    run_planner_agent(&config, &agents_dir, &planner_prompt).await?;

                spec = new_spec;
                plan = new_plan;
                files_to_change = new_files;
                checklist = new_checklist;

                fs::write(task_path.join("spec.md"), &spec).await?;
                fs::write(task_path.join("plan.md"), &plan).await?;
                fs::write(task_path.join("files-to-change.md"), &files_to_change).await?;
                fs::write(task_path.join("acceptance-checklist.md"), &checklist).await?;

                // Lead primary evaluation gate
                println!(
                    "{}",
                    console::style("Evaluating plan by Primary Lead Agent...").cyan()
                );
                let (approved, feedback) = run_primary_evaluation(
                    &config,
                    &agents_dir,
                    &agent_alias,
                    &final_request,
                    &current_research,
                    &spec,
                    &plan,
                    &files_to_change,
                    &checklist,
                )
                .await?;

                if approved {
                    println!(
                        "{}",
                        console::style("Plan APPROVED by Primary Lead Agent!").green().bold()
                    );
                    plan_approved = true;
                } else {
                    println!(
                        "{}",
                        console::style(format!(
                            "Plan REJECTED by Primary Lead Agent! Feedback:\n{}",
                            feedback
                        ))
                        .yellow()
                    );
                    lead_feedback = feedback.clone();
                    // Optionally let research-agent run again with lead feedback to get more context
                    let research_retry_prompt = format!(
                        "The Lead Primary Agent rejected the plan with feedback: {}.\nPerform further research to resolve this feedback.\n=== TASK ===\n{}",
                        feedback, final_request
                    );
                    if let Ok(new_research) = run_research_agent(&config, &agents_dir, &research_retry_prompt).await {
                        current_research.push_str(&format!("\n\n=== ADDITIONAL RESEARCH ATTEMPT {} ===\n{}", plan_attempts, new_research));
                        fs::write(task_path.join("research-report.md"), &current_research).await?;
                    }
                }
            }

            if !plan_approved {
                println!(
                    "{}",
                    console::style("Plan evaluation failed to converge after maximum attempts.").red()
                );
                return Err(anyhow::anyhow!("Plan evaluation failed to converge after {} attempts", max_plan_attempts));
            }

            // 3. User Approval Gate
            if interactive {
                println!(
                    "\nProposed Implementation Plan written to: {}",
                    task_path.join("plan.md").display()
                );
                println!(
                    "Files target for modifications:\n{}",
                    console::style(&files_to_change).dim()
                );
                use dialoguer::{theme::ColorfulTheme, Select};
                let mut theme = ColorfulTheme::default();
                let purple = console::Style::new().color256(99).bold();
                theme.active_item_style = purple;
                theme.prompt_style = console::Style::new().bold();

                let choices = vec!["Yes, approve plan", "No, abort workflow"];
                let selection = Select::with_theme(&theme)
                    .with_prompt("Do you approve this plan?")
                    .items(&choices)
                    .default(0)
                    .interact()?;

                if selection != 0 {
                    println!(
                        "{}",
                        console::style("Workflow execution aborted by user.")
                            .red()
                            .bold()
                    );
                    return Ok("Workflow aborted by user.".to_string());
                }
            }

            // 4. Task Execution Routing (Coder vs Worker)
            println!(
                "{}",
                console::style("Routing task to specialized execution agent...").cyan()
            );
            let execution_agent = determine_execution_agent(
                &config,
                &agents_dir,
                &agent_alias,
                &final_request,
                &spec,
                &plan,
            )
            .await?;

            println!(
                "Task routed to specialized agent: {}",
                console::style(&execution_agent).green().bold()
            );

            let execution_prompt = format!(
                "Implement the plan described in plan.md.\n\n=== SPEC ===\n{}\n\n=== PLAN ===\n{}\n\n=== TARGET FILES ===\n{}\n\nVerify that you only write/modify files listed in the TARGET FILES section. Limit edits to these files only.",
                spec, plan, files_to_change
            );

            let coder_logs = if execution_agent == "worker" {
                run_worker_agent(&config, &agents_dir, &execution_prompt).await?
            } else {
                run_coder_agent(&config, &agents_dir, &execution_prompt).await?
            };
            fs::write(task_path.join("implementation-log.md"), &coder_logs).await?;

            // 5. Verification & Testing Phase (reviewer)
            println!(
                "{}",
                console::style("Executing Verification Phase (reviewer agent)...").cyan()
            );
            let mut verification_report;
            let mut debug_attempts = 0;
            let max_debug_attempts = 3;

            loop {
                let tester_prompt = format!(
                    "Verify the implementation based on the spec, plan, and checklist. Run test/validation commands if needed.\n\n=== SPEC ===\n{}\n\n=== CHECKLIST ===\n{}",
                    spec, checklist
                );
                let test_results = run_reviewer_agent(&config, &agents_dir, &tester_prompt).await?;
                verification_report = test_results.clone();
                fs::write(
                    task_path.join("verification-report.md"),
                    &verification_report,
                )
                .await?;

                let has_failures = test_results.to_uppercase().contains("FAILED")
                    || test_results.to_uppercase().contains("ERROR")
                    || test_results.to_uppercase().contains("FAILURE");

                if !has_failures {
                    println!(
                        "{}",
                        console::style("Verification checks passed successfully!")
                            .green()
                            .bold()
                    );
                    break;
                }

                if debug_attempts >= max_debug_attempts {
                    println!(
                        "{}",
                        console::style(format!(
                            "Verification failed after {} attempts. Continuing to documentation.",
                            max_debug_attempts
                        ))
                        .red()
                    );
                    break;
                }

                debug_attempts += 1;
                println!(
                    "{}",
                    console::style(format!(
                        "Verification failed! Invoking Debugger loop (Attempt {}/{})...",
                        debug_attempts, max_debug_attempts
                    ))
                    .yellow()
                );

                // Run execution agent again to apply fix
                let coder_fix_prompt = format!(
                    "The verification failed with the following report:\n{}\n\nApply code fixes to resolve the issues.",
                    verification_report
                );

                let coder_fix_logs = if execution_agent == "worker" {
                    run_worker_agent(&config, &agents_dir, &coder_fix_prompt).await?
                } else {
                    run_coder_agent(&config, &agents_dir, &coder_fix_prompt).await?
                };

                let mut current_logs = fs::read_to_string(task_path.join("implementation-log.md"))
                    .await
                    .unwrap_or_default();
                current_logs.push_str(&format!(
                    "\n--- Debug Attempt {} ---\n{}",
                    debug_attempts, coder_fix_logs
                ));
                fs::write(task_path.join("implementation-log.md"), &current_logs).await?;
            }

            // 6. Documentation Phase (docs-agent)
            println!(
                "{}",
                console::style("Executing Documentation Phase (docs-agent)...").cyan()
            );
            let docs_prompt = format!(
                "Update or create relevant documentation (README.md, walkthrough.md, spec, API docs) based on implementation logs.\n\n=== SPEC ===\n{}\n\n=== IMPLEMENTATION LOG ===\n{}",
                spec,
                fs::read_to_string(task_path.join("implementation-log.md"))
                    .await
                    .unwrap_or_default()
            );
            let docs_feedback = run_docs_agent(&config, &agents_dir, &docs_prompt).await?;
            fs::write(task_path.join("documentation-log.md"), &docs_feedback).await?;

            // 7. Generate Final Task Summary
            println!("{}", console::style("Compiling final summary...").cyan());
            let summary_prompt = format!(
                "Create a clean technical summary of the completed task.\n\n=== USER REQUEST ===\n{}\n\n=== SPEC & PLAN ===\n{}\n\n=== VERIFICATION ===\n{}\n\n=== DOCUMENTATION ===\n{}",
                user_request,
                plan,
                verification_report,
                docs_feedback
            );
            let final_summary =
                run_planner_agent_summary(&config, &agents_dir, &summary_prompt).await?;
            fs::write(task_path.join("final-summary.md"), &final_summary).await?;

            println!(
                "{}",
                console::style("=== OpenZ Multi-Agent Task Execution Complete ===")
                    .green()
                    .bold()
            );
            println!("All output artifacts saved in: {}", task_path.display());

            Ok(final_summary)
        })
        .await
    }
}

fn run_specialized_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    agent_name: &'a str,
    prompt: &'a str,
    allowed_tools: Option<Vec<String>>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let prompt_file = agents_dir.join(format!("{}.md", agent_name));
        let system_prompt = if prompt_file.exists() {
            fs::read_to_string(&prompt_file).await.unwrap_or_default()
        } else {
            String::new()
        };

        let mut custom_config = config.clone();

        // Resolve base fallback model provider if none configured
        let mut default_model_provider = "openai.default".to_string();
        for p in [
            "google.default",
            "groq.default",
            "openai.default",
            "anthropic.default",
        ] {
            let provider_name = p.split('.').next().unwrap();
            if config
                .providers
                .models
                .iter_entries()
                .any(|(p_type, _, _)| p_type == provider_name)
            {
                default_model_provider = p.to_string();
                break;
            }
        }

        // Find primary agent's model provider and fallbacks if configured
        let primary_cfg = config
            .agents
            .get("agentz")
            .or_else(|| config.agents.get("oh-my-openagent"))
            .cloned();

        let (resolved_default_provider, primary_fallbacks) = if let Some(p_cfg) = &primary_cfg {
            (
                p_cfg.model_provider.as_str().to_string(),
                p_cfg.model_fallbacks.clone(),
            )
        } else {
            (default_model_provider, vec![])
        };

        let agent_cfg = custom_config
            .agents
            .get(agent_name)
            .cloned()
            .unwrap_or_else(|| {
                let mut default_agent = AliasedAgentConfig::default();
                default_agent.model_provider = ModelProviderRef::new(resolved_default_provider);
                default_agent.model_fallbacks = primary_fallbacks;
                default_agent.risk_profile = "default".to_string();
                default_agent.runtime_profile = "default".to_string();
                default_agent
            });

        let agent_workspace = custom_config.agent_workspace_dir(agent_name);
        if !system_prompt.is_empty() {
            fs::create_dir_all(&agent_workspace).await.ok();
            fs::write(agent_workspace.join("IDENTITY.md"), &system_prompt)
                .await
                .ok();
        }

        let primary_provider = agent_cfg.model_provider.clone();
        let mut fallbacks = agent_cfg.model_fallbacks.clone();

        if fallbacks.is_empty() {
            for p in [
                "google.default",
                "groq.default",
                "openai.default",
                "anthropic.default",
            ] {
                let provider_name = p.split('.').next().unwrap();
                if config
                    .providers
                    .models
                    .iter_entries()
                    .any(|(p_type, _, _)| p_type == provider_name)
                {
                    fallbacks.push(p.to_string());
                }
            }
        }

        let mut attempt_providers = vec![primary_provider];
        attempt_providers.extend(fallbacks.into_iter().map(ModelProviderRef::new));
        attempt_providers.retain(|p| !p.as_str().trim().is_empty());

        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        attempt_providers.retain(|p| seen.insert(p.clone()));

        if attempt_providers.is_empty() {
            attempt_providers.push(ModelProviderRef::new("openai.default"));
        }

        custom_config
            .agents
            .insert(agent_name.to_string(), agent_cfg);

        // Resource Scoping (Tool/MCP Filtering)
        if let Some(ref tools) = allowed_tools {
            let has_mcp = tools.iter().any(|t| t.starts_with("mcp_"));
            if !has_mcp {
                custom_config.mcp.enabled = false;
                custom_config.mcp.servers.clear();
            } else {
                custom_config.mcp.servers.retain(|server| {
                    let prefix = format!("mcp_{}", server.name);
                    tools.iter().any(|t| t.starts_with(&prefix))
                });
                if custom_config.mcp.servers.is_empty() {
                    custom_config.mcp.enabled = false;
                }
            }
        }

        let timeout_duration = match agent_name {
            "vision-agent" => std::time::Duration::from_secs(180),
            "research-agent" => std::time::Duration::from_secs(300),
            "openz-planagent" => std::time::Duration::from_secs(300),
            "coder" | "worker" => std::time::Duration::from_secs(900),
            "reviewer" => std::time::Duration::from_secs(600),
            "docs-agent" => std::time::Duration::from_secs(300),
            _ => std::time::Duration::from_secs(600),
        };

        let mut last_error = None;
        for provider in attempt_providers {
            let mut try_config = custom_config.clone();
            if let Some(cfg) = try_config.agents.get_mut(agent_name) {
                cfg.model_provider = provider.clone();
            }

            let overrides = AgentRunOverrides {
                is_subagent: true,
                ..Default::default()
            };

            println!(
                "Running subagent {} with model provider: {} (timeout: {:?})...",
                agent_name, provider, timeout_duration
            );

            let run_future = crate::agent::run_boxed(
                try_config,
                agent_name,
                Some(prompt.to_string()),
                None,
                None,
                None,
                vec![],
                false, // non-interactive
                None,
                allowed_tools.clone(),
                overrides,
            );

            match tokio::time::timeout(timeout_duration, Box::pin(run_future)).await {
                Ok(Ok(res)) => return Ok(res),
                Ok(Err(e)) => {
                    println!(
                        "{}",
                        console::style(format!(
                            "Attempt with {} failed: {:#}. Trying fallback...",
                            provider, e
                        ))
                        .yellow()
                    );
                    last_error = Some(e);
                }
                Err(_) => {
                    println!(
                        "{}",
                        console::style(format!(
                            "Attempt with {} timed out after {:?}. Trying fallback...",
                            provider, timeout_duration
                        ))
                        .yellow()
                    );
                    last_error = Some(anyhow::anyhow!("Timeout after {:?}", timeout_duration));
                }
            }
        }

        let err = last_error.unwrap_or_else(|| {
            anyhow::anyhow!("No model providers available for subagent {}", agent_name)
        });

        ::zeroclaw_log::record!(
            ERROR,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(::serde_json::json!({
                    "subagent": agent_name,
                    "error": format!("{}", err)
                })),
            &format!(
                "Subagent '{}' failed. Primary agent will execute the task itself.",
                agent_name
            )
        );

        println!(
            "{}",
            console::style(format!(
                "⚠️ Subagent '{}' failed: {:#}. Falling back to Primary Agent...",
                agent_name, err
            ))
            .red()
            .bold()
        );

        let primary_agent_name = if config.agents.contains_key("agentz") {
            "agentz"
        } else if config.agents.contains_key("oh-my-openagent") {
            "oh-my-openagent"
        } else if config.agents.contains_key("assistant") {
            "assistant"
        } else {
            config
                .agents
                .keys()
                .next()
                .map(|s| s.as_str())
                .unwrap_or("assistant")
        };

        let try_config = custom_config.clone();
        let overrides = AgentRunOverrides {
            is_subagent: false,
            ..Default::default()
        };

        let run_future = crate::agent::run_boxed(
            try_config,
            primary_agent_name,
            Some(prompt.to_string()),
            None,
            None,
            None,
            vec![],
            false,
            None,
            allowed_tools.clone(),
            overrides,
        );

        match tokio::time::timeout(timeout_duration, Box::pin(run_future)).await {
            Ok(Ok(res)) => Ok(res),
            Ok(Err(e)) => Err(anyhow::anyhow!(
                "Primary agent execution also failed: {:#}",
                e
            )),
            Err(_) => Err(anyhow::anyhow!(
                "Primary agent execution timed out after {:?}",
                timeout_duration
            )),
        }
    })
}

fn run_research_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        run_specialized_agent(
            config,
            agents_dir,
            "research-agent",
            prompt,
            Some(vec![
                "file_read".to_string(),
                "glob_search".to_string(),
                "content_search".to_string(),
                "web_search_tool".to_string(),
                "web_fetch".to_string(),
                "git_operations".to_string(),
            ]),
        )
        .await
    })
}

#[allow(clippy::type_complexity)]
fn run_planner_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<(String, String, String, String)>> + Send + 'a>,
> {
    Box::pin(async move {
        let raw_out = run_specialized_agent(
            config,
            agents_dir,
            "openz-planagent",
            prompt,
            Some(vec![
                "file_read".to_string(),
                "glob_search".to_string(),
                "content_search".to_string(),
            ]),
        )
        .await?;

        Ok(parse_planner_output(&raw_out))
    })
}

fn run_coder_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        run_specialized_agent(
            config,
            agents_dir,
            "coder",
            prompt,
            Some(vec![
                "file_read".to_string(),
                "file_write".to_string(),
                "file_edit".to_string(),
                "glob_search".to_string(),
                "content_search".to_string(),
            ]),
        )
        .await
    })
}

fn run_worker_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        run_specialized_agent(
            config,
            agents_dir,
            "worker",
            prompt,
            Some(vec![
                "file_read".to_string(),
                "file_write".to_string(),
                "file_edit".to_string(),
                "glob_search".to_string(),
                "content_search".to_string(),
                "shell".to_string(),
                "web_search_tool".to_string(),
                "web_fetch".to_string(),
            ]),
        )
        .await
    })
}

fn run_reviewer_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        run_specialized_agent(
            config,
            agents_dir,
            "reviewer",
            prompt,
            Some(vec![
                "file_read".to_string(),
                "shell".to_string(),
                "glob_search".to_string(),
                "content_search".to_string(),
            ]),
        )
        .await
    })
}

fn run_docs_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        run_specialized_agent(
            config,
            agents_dir,
            "docs-agent",
            prompt,
            Some(vec![
                "file_read".to_string(),
                "file_write".to_string(),
                "file_edit".to_string(),
                "glob_search".to_string(),
            ]),
        )
        .await
    })
}

fn run_vision_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt_with_image: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let instructions = "Analyze the visual contents of the target image(s) provided below. Describe what you see in extreme detail.";
        let vision_prompt = format!(
            "{}\n\nTarget image(s) containing visual data:\n{}",
            instructions, prompt_with_image
        );

        run_specialized_agent(
            config,
            agents_dir,
            "vision-agent",
            &vision_prompt,
            Some(vec!["file_read".to_string(), "image_info".to_string()]),
        )
        .await
    })
}

fn run_planner_agent_summary<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    prompt: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        run_specialized_agent(
            config,
            agents_dir,
            "openz-planagent",
            prompt,
            Some(vec!["file_read".to_string()]),
        )
        .await
    })
}

#[allow(clippy::type_complexity)]
fn run_primary_evaluation<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    primary_agent_alias: &'a str,
    user_request: &'a str,
    research_report: &'a str,
    spec: &'a str,
    plan: &'a str,
    files_to_change: &'a str,
    checklist: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(bool, String)>> + Send + 'a>> {
    Box::pin(async move {
        let eval_prompt = format!(
            "You are the Primary Lead Agent. Evaluate the proposed spec, plan, and files to change generated for the user request.\n\n\
            === USER REQUEST ===\n\
            {}\n\n\
            === RESEARCH REPORT ===\n\
            {}\n\n\
            === PROPOSED SPEC ===\n\
            {}\n\n\
            === PROPOSED PLAN ===\n\
            {}\n\n\
            === PROPOSED FILES TO CHANGE ===\n\
            {}\n\n\
            === PROPOSED ACCEPTANCE CHECKLIST ===\n\
            {}\n\n\
            Your job is to either APPROVE this plan, or REJECT it with detailed feedback.\n\
            Respond in one of the following formats:\n\n\
            APPROVED\n\n\
            or\n\n\
            REJECTED: <detailed explanation of why the plan is insufficient and what changes are required>",
            user_request, research_report, spec, plan, files_to_change, checklist
        );

        let raw_out = run_specialized_agent(
            config,
            agents_dir,
            primary_agent_alias,
            &eval_prompt,
            Some(vec!["file_read".to_string()]),
        )
        .await?;

        let trimmed = raw_out.trim();
        if trimmed.starts_with("APPROVED") || trimmed.to_uppercase().contains("APPROVED") {
            Ok((true, String::new()))
        } else if let Some(idx) = trimmed.to_uppercase().find("REJECTED:") {
            let feedback = trimmed[idx + 9..].trim().to_string();
            Ok((false, feedback))
        } else {
            Ok((false, trimmed.to_string()))
        }
    })
}

fn determine_execution_agent<'a>(
    config: &'a Config,
    agents_dir: &'a Path,
    primary_agent_alias: &'a str,
    user_request: &'a str,
    spec: &'a str,
    plan: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let route_prompt = format!(
            "You are the Primary Lead Agent. Analyze the user request, specification, and proposed plan, and decide whether a specialized Coder agent or a general Worker agent (for non-coding automations) should execute it.\n\n\
            === USER REQUEST ===\n\
            {}\n\n\
            === SPECIFICATION ===\n\
            {}\n\n\
            === PLAN ===\n\
            {}\n\n\
            Respond with exactly one word: CODER or WORKER. Do not include any other text.",
            user_request, spec, plan
        );

        let raw_out = run_specialized_agent(
            config,
            agents_dir,
            primary_agent_alias,
            &route_prompt,
            Some(vec!["file_read".to_string()]),
        )
        .await?;

        let upper = raw_out.to_uppercase();
        if upper.contains("WORKER") {
            Ok("worker".to_string())
        } else {
            Ok("coder".to_string())
        }
    })
}

fn parse_planner_output(output: &str) -> (String, String, String, String) {
    let mut spec = String::new();
    let mut plan = String::new();
    let mut files = String::new();
    let mut checklist = String::new();

    let mut current_section = 0; // 1 = spec, 2 = plan, 3 = files, 4 = checklist
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.contains("=== SPEC ===") {
            current_section = 1;
            continue;
        } else if trimmed.contains("=== PLAN ===") {
            current_section = 2;
            continue;
        } else if trimmed.contains("=== FILES TO CHANGE ===") || trimmed.contains("=== FILES ===") {
            current_section = 3;
            continue;
        } else if trimmed.contains("=== CHECKLIST ===")
            || trimmed.contains("=== ACCEPTANCE CHECKLIST ===")
        {
            current_section = 4;
            continue;
        }

        match current_section {
            1 => {
                spec.push_str(line);
                spec.push('\n');
            }
            2 => {
                plan.push_str(line);
                plan.push('\n');
            }
            3 => {
                files.push_str(line);
                files.push('\n');
            }
            4 => {
                checklist.push_str(line);
                checklist.push('\n');
            }
            _ => {
                spec.push_str(line);
                spec.push('\n');
            }
        }
    }

    if plan.is_empty() {
        plan = output.to_string();
    }

    (
        spec.trim().to_string(),
        plan.trim().to_string(),
        files.trim().to_string(),
        checklist.trim().to_string(),
    )
}

fn slugify(text: &str) -> String {
    let mut slug = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').chars().take(30).collect::<String>()
}

fn replace_image_markers(text: &str, description: &str) -> String {
    let mut result = String::new();
    let mut remaining = text;
    while let Some(start_idx) = remaining.find("[IMAGE:") {
        result.push_str(&remaining[..start_idx]);
        result.push_str("\n### [Image Description (vision-agent)] \n");
        result.push_str(description);
        result.push_str("\n###\n");

        let sub = &remaining[start_idx..];
        if let Some(end_idx) = sub.find(']') {
            remaining = &sub[end_idx + 1..];
        } else {
            remaining = &sub[7..]; // skip "[IMAGE:"
        }
    }
    result.push_str(remaining);
    result
}

async fn seed_default_agents(agents_dir: &Path) -> Result<()> {
    let agents = vec![
        (
            "openz-planagent",
            "You are the Plan Agent (`openz-planagent`) for OpenZ. Your job is to understand the user request, review the research findings, and construct a highly detailed technical specification and a step-by-step implementation plan.\n\nYou must format your output EXACTLY as follows:\n\n=== SPEC ===\n<detailed technical specification>\n\n=== PLAN ===\n<step-by-step implementation plan>\n\n=== FILES TO CHANGE ===\n<list of files to edit, one per line, e.g. src/main.rs>\n\n=== CHECKLIST ===\n<acceptance checklist/verification tests to run>",
        ),
        (
            "research-agent",
            "You are the Research Agent (`research-agent`) for OpenZ. Your job is to search the workspace files, git status, external documentation, or APIs to gather all technical facts, dependencies, codebase patterns, or external specs needed for the planning phase. Document your findings in a structured report.",
        ),
        (
            "coder",
            "You are the Coder Agent for OpenZ. Your job is to write high-quality, clean, and bug-free code matching the technical specification and implementation plan.\n\nYou must only edit files listed in files-to-change.md to avoid scope creep or unintended side effects. Document your modifications in implementation-log.md.",
        ),
        (
            "worker",
            "You are the Worker Agent for OpenZ. Your job is to perform non-coding automation tasks, command execution, or system operations described in the plan. Document your actions and results in implementation-log.md.",
        ),
        (
            "reviewer",
            "You are the Reviewer Agent for OpenZ. Your job is to review modifications, run tests/compiles/checks (e.g. cargo test, clippy, npm test), and debug errors. If any test or command fails, analyze the failure and output suggested code fixes to repair the issues. When testing, write all results to verification-report.md. If checks pass, write checks passed.",
        ),
        (
            "docs-agent",
            "You are the Docs Agent for OpenZ. Your job is to update or generate relevant documentation, including README.md, walkthrough.md, spec, or API documentation based on the completed task and implementation logs.",
        ),
        (
            "vision-agent",
            "You are the Vision Agent for OpenZ. Your job is to inspect images, UI mockups, or diagrams, and produce a highly detailed Markdown text description of the image so that a text-only assistant can fully understand what is in the image.",
        ),
    ];

    for (name, prompt) in agents {
        let path = agents_dir.join(format!("{}.md", name));
        if !path.exists() {
            fs::write(&path, prompt).await?;
        }
    }
    Ok(())
}
