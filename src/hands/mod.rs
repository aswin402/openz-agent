pub use zeroclaw_runtime::hands::*;

use crate::config::Config;
use anyhow::Result;

/// Bail with a clear error if the named agent isn't configured.
fn require_configured_agent(config: &Config, agent_alias: &str) -> Result<()> {
    if config.agent(agent_alias).is_none() {
        ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(::serde_json::json!({"agent_alias": agent_alias})),
            "hands CLI rejected: unknown agent alias"
        );
        anyhow::bail!("Unknown agent {agent_alias:?} (no [agents.{agent_alias}] entry configured)");
    }
    Ok(())
}

pub async fn handle_command(command: crate::HandsCommands, config: &Config) -> Result<()> {
    match command {
        crate::HandsCommands::List => {
            let blueprints = BlueprintRegistry::list();
            if blueprints.is_empty() {
                println!("No prebuilt blueprint templates available.");
                return Ok(());
            }

            println!(
                "🚀 Available Pre-built Task Blueprints ({}):",
                blueprints.len()
            );
            for bp in blueprints {
                println!("\n• ID: {}", bp.id);
                println!("  Name       : {}", bp.name);
                println!("  Description: {}", bp.description);
                println!("  Schedule   : {} (recommended)", bp.default_schedule);
                println!("  Required Tools: {}", bp.required_tools.join(", "));
            }
            println!();
            Ok(())
        }
        crate::HandsCommands::Bind {
            blueprint_id,
            agent_alias,
            schedule,
            model,
        } => {
            require_configured_agent(config, &agent_alias)?;
            let job = bind_blueprint_to_cron(
                config,
                &blueprint_id,
                &agent_alias,
                schedule.as_deref(),
                model,
                None, // default delivery config
            )?;

            println!(
                "✓ Successfully bound blueprint '{}' to agent '{}'!",
                blueprint_id, agent_alias
            );
            println!("  Cron Job ID: {}", job.id);
            println!("  Schedule   : {}", job.expression);
            Ok(())
        }
        crate::HandsCommands::Evolve { agent_alias } => {
            require_configured_agent(config, &agent_alias)?;
            let agent_workspace = config.agent_workspace_dir(&agent_alias);
            let (provider, model_name) = get_model_provider_for_agent(config, &agent_alias)?;

            println!(
                "Starting local skill evolution for agent '{}' using model '{}'...",
                agent_alias, model_name
            );
            let evolver =
                zeroclaw_runtime::skillforge::local_evolution::LocalEvolution::new(agent_workspace);
            match evolver
                .evolve(&agent_alias, config, provider.as_ref(), &model_name)
                .await
            {
                Ok(skills) => {
                    if skills.is_empty() {
                        println!(
                            "No new skills evolved. Make sure logs are enabled and contain shell command histories."
                        );
                    } else {
                        println!("✓ Successfully evolved {} new skill(s):", skills.len());
                        for s in skills {
                            println!("  • {}", s);
                        }
                    }
                }
                Err(e) => {
                    anyhow::bail!("Skill evolution failed: {}", e);
                }
            }
            Ok(())
        }
        crate::HandsCommands::OptimizePrompt {
            agent_alias,
            generations,
            eval_suite,
        } => {
            require_configured_agent(config, &agent_alias)?;
            let agent_workspace = config.agent_workspace_dir(&agent_alias);
            let (provider, model_name) = get_model_provider_for_agent(config, &agent_alias)?;

            println!(
                "Starting genetic prompt optimization for agent '{}' using model '{}' ({} generation(s))...",
                agent_alias, model_name, generations
            );
            let optimizer = zeroclaw_runtime::skillforge::prompt_optimizer::PromptOptimizer::new(
                agent_workspace,
            );
            match optimizer
                .optimize(
                    &agent_alias,
                    config,
                    provider.as_ref(),
                    &model_name,
                    generations,
                    &eval_suite,
                )
                .await
            {
                Ok(_) => {
                    println!(
                        "✓ Prompt optimization complete! Winning prompt promoted to IDENTITY.md."
                    );
                }
                Err(e) => {
                    anyhow::bail!("Prompt optimization failed: {}", e);
                }
            }
            Ok(())
        }
    }
}

fn get_model_provider_for_agent(
    config: &crate::config::Config,
    agent_alias: &str,
) -> Result<(Box<dyn zeroclaw_providers::ModelProvider>, String)> {
    let _agent_config = config
        .agent(agent_alias)
        .ok_or_else(|| anyhow::Error::msg(format!("Unknown agent {agent_alias}")))?;

    let agent_provider_resolved = config
        .resolved_model_provider_for_agent(agent_alias)
        .map(|(ty, alias, cfg)| (ty, alias.to_string(), cfg.clone()))
        .ok_or_else(|| {
            anyhow::Error::msg(format!(
                "Could not resolve model provider for agent {agent_alias}"
            ))
        })?;

    let (provider_type, _, agent_model_provider) = agent_provider_resolved;
    let provider_name = provider_type.to_string();

    let model_name = agent_model_provider.model.clone().ok_or_else(|| {
        anyhow::Error::msg(format!(
            "No model configured for model provider {provider_name}"
        ))
    })?;

    let provider_runtime_options = zeroclaw_providers::provider_runtime_options_from_config(config);

    let provider = zeroclaw_providers::create_routed_model_provider_with_options(
        config,
        &provider_name,
        agent_model_provider.api_key.as_deref(),
        agent_model_provider.uri.as_deref(),
        &config.reliability,
        &config.model_routes,
        &model_name,
        &provider_runtime_options,
    )?;

    Ok((provider, model_name))
}
