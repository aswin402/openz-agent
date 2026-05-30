#![recursion_limit = "256"]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::uninlined_format_args,
    clippy::disallowed_macros,
    clippy::large_futures,
    clippy::field_reassign_with_default,
    clippy::implicit_clone,
    clippy::needless_borrows_for_generic_args,
    clippy::unnecessary_map_or,
    clippy::map_unwrap_or,
    unused_variables,
    dead_code
)]

pub mod tui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dialoguer::Select;
use zeroclaw_config::providers::ModelProviderRef;
use zeroclaw_config::schema::Config;

#[derive(Parser, Debug)]
#[command(
    name = "openz",
    author = "theonlyhennygod",
    version = env!("CARGO_PKG_VERSION"),
    about = "openz - The minimal, self-improving, cutting-edge AI Agent CLI & TUI.",
    disable_help_flag = true,
    disable_version_flag = true
)]
struct Cli {
    #[arg(short, long)]
    help: bool,

    #[arg(short = 'V', long)]
    version: bool,

    /// Specify the agent alias to run (e.g. brain, planner, analysis, vision)
    #[arg(short, long)]
    agent: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show version, logo, and description
    Version,
    /// Run the interactive configuration wizard
    Configure {
        /// Mode to configure (e.g. "subagents")
        mode: Option<String>,
    },
    /// View the runtime event logs
    Logs {
        /// Verify the cryptographic integrity of the log file
        #[arg(long)]
        verify: bool,
    },
    /// Configure API keys/environment variables for MCP servers
    McpSetup,
    /// Configure and add a new subagent to the config
    AgentSetup,
    /// Manage hands task blueprints and self-evolution
    Hands {
        #[command(subcommand)]
        command: zeroclaw::HandsCommands,
    },
    /// Start the background daemon (gateway + channels + scheduler + heartbeat)
    Daemon {
        /// Host to bind the gateway to
        #[arg(long)]
        host: Option<String>,
        /// Port to bind the gateway to
        #[arg(short, long)]
        port: Option<u16>,
    },
    /// Manage the background system service
    Service {
        #[command(subcommand)]
        command: zeroclaw::ServiceCommands,
    },
    /// Manage messaging channels
    Channel {
        #[command(subcommand)]
        command: zeroclaw::ChannelCommands,
    },
    /// Manage background cron scheduler tasks
    Cron {
        #[command(subcommand)]
        command: zeroclaw::CronCommands,
    },
    /// Manage agent memory database
    Memory {
        #[command(subcommand)]
        command: zeroclaw::MemoryCommands,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install default crypto provider for Rustls TLS.
    if let Err(_) = rustls::crypto::ring::default_provider().install_default() {
        // Ignore if already installed
    }

    let args = std::env::args().collect::<Vec<String>>();

    // Check if --help or -h was passed anywhere
    let has_help = args.iter().any(|arg| arg == "--help" || arg == "-h");
    let has_version = args
        .iter()
        .any(|arg| arg == "--version" || arg == "-V" || arg == "version");

    if has_help {
        print_openz_help();
        return Ok(());
    }

    if has_version {
        print_openz_version();
        return Ok(());
    }

    let cli = Cli::parse();

    if let Some(cmd) = cli.command {
        match cmd {
            Commands::Version => {
                print_openz_version();
            }
            Commands::Configure { mode } => {
                let mut config = Config::load_or_init().await?;
                if let Some(ref m) = mode {
                    if m == "subagents" {
                        run_configure_subagents_wizard(&mut config).await?;
                    } else {
                        eprintln!(
                            "{}",
                            console::style(format!(
                                "Unknown mode '{}'. Running standard configure wizard.",
                                m
                            ))
                            .yellow()
                        );
                        run_configure_wizard(&mut config).await?;
                    }
                } else {
                    run_configure_wizard(&mut config).await?;
                }
            }
            Commands::Logs { verify } => {
                let config = Config::load_or_init().await?;
                if verify {
                    run_log_verification(&config)?;
                } else {
                    print_logs(&config)?;
                }
            }
            Commands::McpSetup => {
                let mut config = Config::load_or_init().await?;
                run_mcp_setup_wizard(&mut config).await?;
            }
            Commands::AgentSetup => {
                let mut config = Config::load_or_init().await?;
                run_agent_setup_wizard(&mut config).await?;
            }
            Commands::Hands { command } => {
                let config = Config::load_or_init().await?;
                zeroclaw::hands::handle_command(command, &config).await?;
            }
            Commands::Daemon { host, port } => {
                let config = Config::load_or_init().await?;
                let host = host.unwrap_or_else(|| config.gateway.host.clone());
                let port = port.unwrap_or(config.gateway.port);

                #[cfg(feature = "agent-runtime")]
                {
                    let canvas_store = zeroclaw_runtime::tools::CanvasStore::new();
                    let canvas_store_for_gateway = canvas_store.clone();
                    let canvas_store_for_channels = canvas_store.clone();

                    let subsystems = zeroclaw_runtime::daemon::DaemonSubsystems {
                        #[cfg(feature = "gateway")]
                        gateway_start: Some(Box::new(move |host, port, config, tx, reload_tx| {
                            let canvas_store = canvas_store_for_gateway.clone();
                            Box::pin(async move {
                                Box::pin(zeroclaw_gateway::run_gateway(
                                    &host,
                                    port,
                                    config,
                                    tx,
                                    reload_tx,
                                    Some(canvas_store),
                                ))
                                .await
                            })
                        })),
                        #[cfg(not(feature = "gateway"))]
                        gateway_start: None,

                        channels_start: Some(Box::new(move |config, cancel| {
                            let canvas_store = canvas_store_for_channels.clone();
                            Box::pin(async move {
                                Box::pin(zeroclaw_channels::orchestrator::start_channels(
                                    config,
                                    Some(canvas_store),
                                    cancel,
                                ))
                                .await
                            })
                        })),

                        mqtt_start: Some(Box::new(|mqtt_config| {
                            Box::pin(async move {
                                use std::sync::{Arc, Mutex};
                                use zeroclaw_config::schema::SopConfig;
                                use zeroclaw_memory::NoneMemory;
                                use zeroclaw_runtime::sop::{SopAuditLogger, SopEngine};

                                let engine =
                                    Arc::new(Mutex::new(SopEngine::new(SopConfig::default())));
                                let audit =
                                    Arc::new(SopAuditLogger::new(Arc::new(NoneMemory::default())));
                                zeroclaw_channels::orchestrator::mqtt::run_mqtt_sop_listener(
                                    &mqtt_config,
                                    engine,
                                    audit,
                                )
                                .await
                            })
                        })),
                    };

                    zeroclaw_runtime::daemon::run(config, host, port, subsystems).await?;
                }
                #[cfg(not(feature = "agent-runtime"))]
                {
                    anyhow::bail!("Daemon requires the agent-runtime feature to be enabled");
                }
            }
            Commands::Service { command } => {
                let config = Config::load_or_init().await?;
                #[cfg(feature = "agent-runtime")]
                {
                    let init_system = zeroclaw_runtime::service::InitSystem::Auto.resolve()?;
                    zeroclaw::service::handle_command(&command, &config, init_system)?;
                }
                #[cfg(not(feature = "agent-runtime"))]
                {
                    anyhow::bail!(
                        "Service management requires the agent-runtime feature to be enabled"
                    );
                }
            }
            Commands::Channel { command } => {
                let config = Config::load_or_init().await?;
                #[cfg(feature = "agent-runtime")]
                {
                    match command {
                        zeroclaw::ChannelCommands::Start => {
                            let canvas_store = zeroclaw_runtime::tools::CanvasStore::new();
                            let cancel = tokio_util::sync::CancellationToken::new();
                            Box::pin(zeroclaw_channels::orchestrator::start_channels(
                                config,
                                Some(canvas_store),
                                cancel,
                            ))
                            .await?;
                        }
                        zeroclaw::ChannelCommands::Doctor => {
                            Box::pin(zeroclaw_channels::orchestrator::doctor_channels(config))
                                .await?;
                        }
                        other => {
                            zeroclaw::channels::handle_command(other, &config).await?;
                        }
                    }
                }
                #[cfg(not(feature = "agent-runtime"))]
                {
                    anyhow::bail!(
                        "Channel management requires the agent-runtime feature to be enabled"
                    );
                }
            }
            Commands::Cron { command } => {
                let config = Config::load_or_init().await?;
                #[cfg(feature = "agent-runtime")]
                {
                    zeroclaw::cron::handle_command(command, &config)?;
                }
                #[cfg(not(feature = "agent-runtime"))]
                {
                    anyhow::bail!(
                        "Cron management requires the agent-runtime feature to be enabled"
                    );
                }
            }
            Commands::Memory { command } => {
                let config = Config::load_or_init().await?;
                zeroclaw::memory::cli::handle_command(command, &config).await?;
            }
        }
        return Ok(());
    }

    // Default: run interactive agent loop
    let mut config = Config::load_or_init().await?;

    // Wire CLI channel for interactive mode
    zeroclaw_runtime::agent::loop_::register_cli_channel_fn(Box::new(|| {
        Box::new(zeroclaw_channels::cli::CliChannel::new("cli"))
    }));

    // Find configured agent alias or run configure
    let agent_alias = if let Some(ref targeted) = cli.agent {
        if config.agents.contains_key(targeted) {
            targeted.clone()
        } else {
            let mut available: Vec<_> = config.agents.keys().cloned().collect();
            available.sort();
            eprintln!(
                "{}",
                console::style(format!("Error: Agent '{}' is not configured.", targeted))
                    .red()
                    .bold()
            );
            eprintln!("Available agents: {}", available.join(", "));
            std::process::exit(1);
        }
    } else if config.agents.is_empty() {
        println!("No agent configured yet. Let's run configuration first!");
        run_configure_wizard(&mut config).await?;
        "assistant".to_string()
    } else if config.agents.len() == 1 {
        config.agents.keys().next().unwrap().clone()
    } else {
        // If there are multiple agents, we prompt the user to choose
        let mut agents: Vec<_> = config.agents.keys().cloned().collect();
        agents.sort();

        let mut theme = dialoguer::theme::ColorfulTheme::default();
        let purple = console::Style::new().color256(99).bold();
        theme.active_item_style = purple;
        theme.prompt_style = console::Style::new().bold();

        let selection = dialoguer::Select::with_theme(&theme)
            .with_prompt("Select an agent to run")
            .items(&agents)
            .default(0)
            .interact()?;

        agents[selection].clone()
    };

    let final_temperature: Option<f64> = config
        .model_provider_for_agent(&agent_alias)
        .and_then(|e| e.temperature);

    // Session selection/resume wizard
    let sessions = list_sessions();
    let mut session_file = None;

    if !sessions.is_empty() && std::io::stdout().is_terminal() && std::io::stdin().is_terminal() {
        let mut default_model_name = "unknown-model".to_string();
        if let Some((_, _, model_cfg)) = config.resolved_model_provider_for_agent(&agent_alias) {
            default_model_name = model_cfg
                .model
                .clone()
                .unwrap_or_else(|| "unknown-model".to_string());
        }
        session_file = interactive_session_picker(&sessions, &default_model_name)?;
    }

    let session_state_file = if let Some(path) = session_file {
        Some(path)
    } else {
        let dir = get_sessions_dir();
        let _ = std::fs::create_dir_all(&dir);
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        Some(dir.join(format!("session_{timestamp}.json")))
    };

    // Run the agent loop
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        let system_prompt = "You are a helpful AI assistant.".to_string();
        let app = crate::tui::app::TuiApp::new(
            config,
            agent_alias,
            session_state_file,
            system_prompt,
            final_temperature,
        )?;
        app.run_loop().await?;
        Ok(())
    } else {
        Box::pin(zeroclaw_runtime::agent::run(
            config,
            &agent_alias,
            None, // message
            None, // provider override
            None, // model override
            final_temperature,
            Vec::new(),
            true, // interactive
            session_state_file,
            None, // allowed_tools
            zeroclaw_runtime::agent::loop_::AgentRunOverrides::default(),
        ))
        .await
        .map(|_| ())
    }
}

fn print_openz_help() {
    println!(
        "{}",
        console::style(
            "openz (ZeroClaw fork) - The minimal, self-improving, cutting-edge AI Agent CLI & TUI."
        )
        .cyan()
        .bold()
    );
    println!();
    println!("{}", console::style("Usage:").yellow().bold());
    println!("  openz                   Run the agent in TUI/CLI interactive mode");
    println!("  openz --help, -h        Show this help message");
    println!("  openz version           Show logo, version, and description");
    println!("  openz configure         Run the minimal configuration wizard");
    println!("  openz mcp-setup         Configure API keys for MCP servers");
    println!("  openz agent-setup       Configure and add a new subagent");
    println!("  openz logs              View full runtime logs");
    println!("  openz hands             Manage task blueprints and self-evolution");
    println!("  openz daemon            Start the background daemon (gateway + channels)");
    println!("  openz service           Manage the background system service");
    println!("  openz channel           Manage messaging channels");
    println!("  openz cron              Manage background scheduled tasks");
    println!("  openz memory            Manage agent memory database");
    println!();
}

fn print_openz_version() {
    println!(
        "{}",
        console::style("  ___  ____  _____ _   _ _____")
            .cyan()
            .bold()
    );
    println!(
        "{}",
        console::style(" / _ \\|  _ \\| ____| \\ | |__  /")
            .cyan()
            .bold()
    );
    println!(
        "{}",
        console::style("| | | | |_) |  _| |  \\| | / / ")
            .cyan()
            .bold()
    );
    println!(
        "{}",
        console::style("| |_| |  __/| |___| |\\  |/ /_ ")
            .cyan()
            .bold()
    );
    println!(
        "{}",
        console::style(" \\___/|_|   |_____|_| \\_/____|")
            .cyan()
            .bold()
    );
    println!();
    println!(
        "  {}{} v{}",
        console::style("open").white().bold(),
        console::style("z").color256(208).bold(),
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  {}",
        console::style(
            "openz (ZeroClaw fork) - The minimal, self-improving, cutting-edge AI Agent CLI & TUI."
        )
        .dim()
    );
    println!();
}

fn get_clean_theme() -> dialoguer::theme::ColorfulTheme {
    let mut theme = dialoguer::theme::ColorfulTheme::default();
    theme.active_item_style = console::Style::new().color256(99).bold(); // purple
    theme.prompt_style = console::Style::new().bold();
    theme
}

fn get_provider_status(config: &Config, provider: &str) -> String {
    let alias = "default";
    if let Some(p_cfg) = config.providers.models.find(provider, alias) {
        let has_key = p_cfg.api_key.as_ref().is_some_and(|k| !k.is_empty());
        let has_model = p_cfg.model.as_ref().is_some_and(|m| !m.is_empty());
        match provider {
            "ollama" | "lmstudio" => {
                if has_model {
                    format!(
                        "{} (model: {})",
                        console::style("Configured").green(),
                        p_cfg.model.as_ref().unwrap()
                    )
                } else {
                    console::style("Not configured").dim().to_string()
                }
            }
            _ => {
                if has_key && has_model {
                    format!(
                        "{} (model: {})",
                        console::style("Configured").green(),
                        p_cfg.model.as_ref().unwrap()
                    )
                } else if has_key {
                    console::style("Configured (no model)").yellow().to_string()
                } else {
                    console::style("Not configured").dim().to_string()
                }
            }
        }
    } else {
        console::style("Not configured").dim().to_string()
    }
}

fn find_api_key_for_family(config: &Config, family: &str) -> Option<String> {
    for (f, _alias, base) in config.providers.models.iter_entries() {
        if f == family {
            if let Some(ref key) = base.api_key {
                if !key.is_empty() {
                    return Some(key.clone());
                }
            }
        }
    }
    None
}

fn find_uri_for_family(config: &Config, family: &str) -> Option<String> {
    for (f, _alias, base) in config.providers.models.iter_entries() {
        if f == family {
            if let Some(ref uri) = base.uri {
                if !uri.is_empty() {
                    return Some(uri.clone());
                }
            }
        }
    }
    None
}

fn configure_subagent_provider_alias(
    config: &mut Config,
    family: &str,
    alias: &str,
    model: &str,
) -> Result<()> {
    config.providers.models.ensure(family, alias);
    let prefix = format!("providers.models.{family}.{alias}");
    if let Some(key) = find_api_key_for_family(config, family) {
        config.set_secret_persistent(&format!("{prefix}.api-key"), key)?;
    }
    if let Some(uri) = find_uri_for_family(config, family) {
        config.set_prop_persistent(&format!("{prefix}.uri"), &uri)?;
    }
    config.set_prop_persistent(&format!("{prefix}.model"), model)?;
    Ok(())
}

fn get_configured_families(config: &Config) -> Vec<String> {
    let mut families = Vec::new();
    for (family, _alias, base) in config.providers.models.iter_entries() {
        let has_key = base.api_key.as_ref().is_some_and(|k| !k.is_empty());
        let is_local = matches!(family, "ollama" | "lmstudio");
        if (has_key || is_local) && !families.contains(&family.to_string()) {
            families.push(family.to_string());
        }
    }
    families
}

fn get_recommended_model(subagent: &str, family: &str) -> &'static str {
    match (subagent, family) {
        ("coder", "anthropic") => "claude-3-5-sonnet-20241022",
        ("coder", "openai") => "gpt-4o",
        ("coder", "gemini") => "gemini-1.5-pro",
        ("coder", "deepseek") => "deepseek-chat",
        ("reviewer", "anthropic") => "claude-3-5-sonnet-20241022",
        ("reviewer", "openai") => "gpt-4o",
        ("reviewer", "gemini") => "gemini-1.5-pro",
        ("research-agent", "anthropic") => "claude-3-5-haiku-20241022",
        ("research-agent", "openai") => "gpt-4o-mini",
        ("research-agent", "gemini") => "gemini-1.5-flash",
        ("openz-planagent", "anthropic") => "claude-3-5-sonnet-20241022",
        ("openz-planagent", "openai") => "gpt-4o",
        ("openz-planagent", "gemini") => "gemini-1.5-pro",
        ("worker", "anthropic") => "claude-3-5-haiku-20241022",
        ("worker", "openai") => "gpt-4o-mini",
        ("worker", "gemini") => "gemini-1.5-flash",
        ("docs-agent", "anthropic") => "claude-3-5-sonnet-20241022",
        ("docs-agent", "openai") => "gpt-4o-mini",
        ("docs-agent", "gemini") => "gemini-1.5-flash",
        ("vision-agent", "anthropic") => "claude-3-5-sonnet-20241022",
        ("vision-agent", "openai") => "gpt-4o",
        ("vision-agent", "gemini") => "gemini-1.5-pro",
        _ => match family {
            "anthropic" => "claude-3-5-sonnet-20241022",
            "openai" => "gpt-4o",
            "gemini" => "gemini-1.5-pro",
            "groq" => "llama3-70b-8192",
            "deepseek" => "deepseek-chat",
            "ollama" => "llama3",
            "openrouter" => "anthropic/claude-3.5-sonnet",
            _ => "model-id",
        },
    }
}

fn get_available_models_for_families(
    subagent: &str,
    families: &[String],
) -> Vec<(String, String, String)> {
    let mut list = Vec::new();
    for family in families {
        let recommended = get_recommended_model(subagent, family);
        let models = match family.as_str() {
            "anthropic" => vec![
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ],
            "openai" => vec!["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"],
            "gemini" => vec!["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
            "groq" => vec!["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
            "deepseek" => vec!["deepseek-chat", "deepseek-coder"],
            "ollama" => vec!["llama3", "mistral", "phi3"],
            "openrouter" => vec![
                "anthropic/claude-3.5-sonnet",
                "google/gemini-flash-1.5",
                "meta-llama/llama-3-8b-instruct",
            ],
            "lmstudio" => vec!["model-id"],
            _ => vec![],
        };
        for m in models {
            let label = if m == recommended {
                format!("{}: {} (Recommended)", family, m)
            } else {
                format!("{}: {}", family, m)
            };
            list.push((label, family.clone(), m.to_string()));
        }
    }
    list
}

fn get_model_name_for_ref(config: &Config, provider_ref: &str) -> String {
    if provider_ref.is_empty() {
        return "None".to_string();
    }
    if let Some((family, alias)) = provider_ref.split_once('.') {
        if let Some(p_cfg) = config.providers.models.find(family, alias) {
            if let Some(ref m) = p_cfg.model {
                return format!("{} ({})", m, family);
            }
        }
    }
    format!("{} (unresolved)", provider_ref)
}

async fn run_configure_wizard(config: &mut Config) -> Result<()> {
    println!(
        "{}",
        console::style("=== openz Configuration Setup ===")
            .cyan()
            .bold()
    );
    println!("This will set up your model provider and default agent.");
    println!();

    let theme = get_clean_theme();

    loop {
        let choices = vec![
            "models - Configure AI Models & Providers",
            "telegram - Configure Telegram Integration",
            "exit - Exit Configuration Setup",
        ];

        let selection = Select::with_theme(&theme)
            .with_prompt("Select configuration menu")
            .items(&choices)
            .default(0)
            .interact()?;

        match selection {
            0 => {
                run_configure_models_submenu(config, &theme).await?;
            }
            1 => {
                run_configure_telegram(config).await?;
            }
            _ => {
                config.save_dirty().await?;
                println!(
                    "{}",
                    console::style("✓ Configuration successfully saved!")
                        .green()
                        .bold()
                );
                break;
            }
        }
    }

    Ok(())
}

async fn run_configure_models_submenu(
    config: &mut Config,
    theme: &dialoguer::theme::ColorfulTheme,
) -> Result<()> {
    let providers = vec![
        "anthropic",
        "openai",
        "gemini",
        "groq",
        "deepseek",
        "ollama",
        "openrouter",
        "lmstudio",
        "Other",
    ];

    loop {
        let mut items = Vec::new();
        for p in &providers {
            let status = get_provider_status(config, p);
            items.push(format!("{} - {}", p, status));
        }
        items.push("Exit".to_string());

        let selection = Select::with_theme(theme)
            .with_prompt("Select AI Model Provider to configure")
            .items(&items)
            .default(0)
            .interact()?;

        if selection == items.len() - 1 {
            println!(
                "{}",
                console::style("These all are configured, available models successfully saved")
                    .green()
            );
            config.save_dirty().await?;
            break;
        }

        let picked = if selection == providers.len() - 1 {
            let custom: String = dialoguer::Input::new()
                .with_prompt("Enter Custom Provider Name")
                .interact_text()?;
            custom.trim().to_lowercase()
        } else {
            providers[selection].to_string()
        };

        let needs_key = !matches!(picked.as_str(), "ollama" | "lmstudio");

        let api_key = if needs_key {
            let current_key = config
                .providers
                .models
                .find(&picked, "default")
                .and_then(|p| p.api_key.clone())
                .unwrap_or_default();

            let prompt = if !current_key.is_empty() {
                format!("Enter API Key for {} [keep existing: *********]", picked)
            } else {
                format!("Enter API Key for {}", picked)
            };

            let key = dialoguer::Password::new()
                .with_prompt(&prompt)
                .allow_empty_password(true)
                .interact()?;

            let key = key.trim().to_string();
            if key.is_empty() { current_key } else { key }
        } else {
            String::new()
        };

        let mut model_options = match picked.as_str() {
            "anthropic" => vec![
                "claude-3-5-sonnet-20241022 (Recommended)",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ],
            "openai" => vec![
                "gpt-4o (Recommended)",
                "gpt-4o-mini",
                "o1-preview",
                "o1-mini",
            ],
            "gemini" => vec![
                "gemini-1.5-pro (Recommended)",
                "gemini-1.5-flash",
                "gemini-2.0-flash-exp",
            ],
            "groq" => vec![
                "llama3-70b-8192 (Recommended)",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
            ],
            "deepseek" => vec!["deepseek-chat (Recommended)", "deepseek-coder"],
            "ollama" => vec!["llama3 (Recommended)", "mistral", "phi3"],
            "openrouter" => vec![
                "anthropic/claude-3.5-sonnet (Recommended)",
                "google/gemini-flash-1.5",
                "meta-llama/llama-3-8b-instruct",
            ],
            _ => vec![],
        };
        model_options.push("Custom Model ID");

        let model_selection = Select::with_theme(theme)
            .with_prompt("Select model")
            .items(&model_options)
            .default(0)
            .interact()?;

        let model = if model_selection == model_options.len() - 1 {
            let custom: String = dialoguer::Input::new()
                .with_prompt("Enter Custom Model ID")
                .interact_text()?;
            custom.trim().to_string()
        } else {
            let selected_raw = model_options[model_selection];
            if let Some(pos) = selected_raw.find(" (Recommended)") {
                selected_raw[..pos].to_string()
            } else {
                selected_raw.to_string()
            }
        };

        let alias = "default";
        config.providers.models.ensure(&picked, alias);

        let prefix = format!("providers.models.{picked}.{alias}");
        if !api_key.is_empty() {
            config.set_secret_persistent(&format!("{prefix}.api-key"), api_key.clone())?;
        }
        config.set_prop_persistent(&format!("{prefix}.model"), &model)?;

        if config.risk_profiles.get("default").is_none() {
            let mut default_profile = zeroclaw_config::schema::RiskProfileConfig::default();
            default_profile.ensure_default_auto_approve();
            config
                .risk_profiles
                .insert("default".to_string(), default_profile);
            config.mark_dirty("risk_profiles.default");
        }

        if config.runtime_profiles.get("default").is_none() {
            config.runtime_profiles.insert(
                "default".to_string(),
                zeroclaw_config::schema::RuntimeProfileConfig::default(),
            );
            config.mark_dirty("runtime_profiles.default");
        }

        let agent_prefix = "agents.assistant";
        config.set_prop_persistent(
            &format!("{agent_prefix}.model-provider"),
            &format!("{picked}.{alias}"),
        )?;
        config.set_prop_persistent(&format!("{agent_prefix}.risk-profile"), "default")?;
        config.set_prop_persistent(&format!("{agent_prefix}.runtime-profile"), "default")?;

        let subagent_names = vec![
            "coder",
            "reviewer",
            "research-agent",
            "openz-planagent",
            "worker",
            "docs-agent",
            "vision-agent",
        ];
        for sub_name in subagent_names {
            let sa_prefix = format!("agents.{}", sub_name);
            if config.agents.get(sub_name).is_none() {
                config.set_prop_persistent(
                    &format!("{sa_prefix}.model-provider"),
                    &format!("{picked}.{alias}"),
                )?;
                config.set_prop_persistent(&format!("{sa_prefix}.risk-profile"), "default")?;
                config.set_prop_persistent(&format!("{sa_prefix}.runtime-profile"), "default")?;
            }
        }
    }
    Ok(())
}

async fn run_configure_telegram(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "{}",
        console::style("=== Telegram Channel Configuration ===")
            .cyan()
            .bold()
    );
    println!("To connect to Telegram:");
    println!("  1. Open Telegram and search for @BotFather.");
    println!("  2. Send `/newbot` and follow the instructions to name your bot.");
    println!("  3. Copy the Bot Token provided at the end.");
    println!();

    let theme = get_clean_theme();

    loop {
        let configured: Vec<String> = config.channels.telegram.keys().cloned().collect();
        if configured.is_empty() {
            println!(
                "{}",
                console::style("Telegram gateway not setup yet.").yellow()
            );
            println!();
            let bot_token: String = dialoguer::Password::new()
                .with_prompt("Paste Telegram Bot Token")
                .interact()?;
            let bot_token = bot_token.trim().to_string();

            if bot_token.is_empty() {
                println!(
                    "{}",
                    console::style("No token provided. Skipping Telegram configuration.").yellow()
                );
                break;
            }

            println!("Validating token with Telegram API...");
            match validate_telegram_token(&bot_token).await {
                Ok(username) => {
                    let alias = "default";
                    let _ = config.create_map_key("channels.telegram", alias);
                    config.set_secret_persistent(
                        &format!("channels.telegram.{alias}.bot-token"),
                        bot_token,
                    )?;
                    config.set_prop_persistent(
                        &format!("channels.telegram.{alias}.enabled"),
                        "true",
                    )?;

                    println!();
                    let telegram_username: String = dialoguer::Input::new()
                        .with_prompt("Enter your Telegram username to authorize it (optional, press Enter to skip)")
                        .allow_empty(true)
                        .interact_text()?;
                    let telegram_username =
                        telegram_username.trim().trim_start_matches('@').to_string();
                    if !telegram_username.is_empty() {
                        let group_name = format!("telegram_{alias}");
                        let channel_ref = format!("telegram.{alias}");
                        use zeroclaw_config::multi_agent::{PeerGroupConfig, PeerUsername};
                        let channel_ref_clone = channel_ref.clone();
                        let matching_agents: Vec<String> = config
                            .agents
                            .iter()
                            .filter(|(_, agent)| {
                                agent
                                    .channels
                                    .iter()
                                    .any(|ch| ch.as_str() == channel_ref_clone)
                            })
                            .map(|(name, _)| name.clone())
                            .collect();
                        let group =
                            config
                                .peer_groups
                                .entry(group_name.clone())
                                .or_insert_with(|| PeerGroupConfig {
                                    channel: channel_ref,
                                    ..PeerGroupConfig::default()
                                });
                        for agent_name in matching_agents {
                            let agent_alias =
                                zeroclaw_config::multi_agent::AgentAlias::new(agent_name);
                            if !group.agents.contains(&agent_alias) {
                                group.agents.push(agent_alias);
                            }
                        }
                        let normalized = telegram_username.to_ascii_lowercase();
                        if !group
                            .external_peers
                            .iter()
                            .any(|p| p.as_str().to_ascii_lowercase() == normalized)
                        {
                            group
                                .external_peers
                                .push(PeerUsername::new(telegram_username));
                        }
                        config.mark_dirty(&format!("peer_groups.{group_name}"));
                        println!(
                            "{}",
                            console::style(format!(
                                "Username '@{normalized}' pre-authorized. No bind code required!"
                            ))
                            .green()
                        );
                    } else {
                        println!(
                            "No username entered. When you start the daemon ('openz daemon'), message your bot '/bind <code>' using the one-time bind code printed in the logs."
                        );
                    }
                    println!();

                    config.save_dirty().await?;
                    println!(
                        "{}",
                        console::style(format!(
                            "telegram gateway configured successfully ✔ (@{username})"
                        ))
                        .green()
                        .bold()
                    );
                    break;
                }
                Err(e) => {
                    println!(
                        "{}",
                        console::style(format!("Error: Invalid token or network error: {e}"))
                            .red()
                            .bold()
                    );
                    println!("Please check the token and try again.");
                    println!();
                }
            }
        } else {
            println!(
                "{}",
                console::style("Already setuped telegramgateway")
                    .green()
                    .bold()
            );
            println!("Total connected gateway: {}", configured.len());
            println!();

            let choices = vec![
                "1. Add new gateway",
                "2. Delete gateway",
                "3. Back to main menu",
            ];

            let selection = Select::with_theme(&theme)
                .with_prompt("Select option")
                .items(&choices)
                .default(0)
                .interact()?;

            match selection {
                0 => {
                    let alias: String = dialoguer::Input::new()
                        .with_prompt("Enter name (alias) for the new gateway (e.g., custom)")
                        .interact_text()?;
                    let alias = alias.trim().to_string();
                    if alias.is_empty() {
                        continue;
                    }
                    if config.channels.telegram.contains_key(&alias) {
                        println!(
                            "{}",
                            console::style(format!("Gateway '{alias}' already exists."))
                                .red()
                                .bold()
                        );
                        continue;
                    }

                    let bot_token: String = dialoguer::Password::new()
                        .with_prompt("Paste Telegram Bot Token")
                        .interact()?;
                    let bot_token = bot_token.trim().to_string();

                    if bot_token.is_empty() {
                        continue;
                    }

                    println!("Validating token with Telegram API...");
                    match validate_telegram_token(&bot_token).await {
                        Ok(username) => {
                            let _ = config.create_map_key("channels.telegram", &alias);
                            config.set_secret_persistent(
                                &format!("channels.telegram.{alias}.bot-token"),
                                bot_token,
                            )?;
                            config.set_prop_persistent(
                                &format!("channels.telegram.{alias}.enabled"),
                                "true",
                            )?;

                            println!();
                            let telegram_username: String = dialoguer::Input::new()
                                .with_prompt("Enter your Telegram username to authorize it (optional, press Enter to skip)")
                                .allow_empty(true)
                                .interact_text()?;
                            let telegram_username =
                                telegram_username.trim().trim_start_matches('@').to_string();
                            if !telegram_username.is_empty() {
                                let group_name = format!("telegram_{alias}");
                                let channel_ref = format!("telegram.{alias}");
                                use zeroclaw_config::multi_agent::{PeerGroupConfig, PeerUsername};
                                let channel_ref_clone = channel_ref.clone();
                                let matching_agents: Vec<String> = config
                                    .agents
                                    .iter()
                                    .filter(|(_, agent)| {
                                        agent
                                            .channels
                                            .iter()
                                            .any(|ch| ch.as_str() == channel_ref_clone)
                                    })
                                    .map(|(name, _)| name.clone())
                                    .collect();
                                let group = config
                                    .peer_groups
                                    .entry(group_name.clone())
                                    .or_insert_with(|| PeerGroupConfig {
                                        channel: channel_ref.clone(),
                                        ..PeerGroupConfig::default()
                                    });
                                for agent_name in matching_agents {
                                    let agent_alias =
                                        zeroclaw_config::multi_agent::AgentAlias::new(agent_name);
                                    if !group.agents.contains(&agent_alias) {
                                        group.agents.push(agent_alias);
                                    }
                                }
                                let normalized = telegram_username.to_ascii_lowercase();
                                if !group
                                    .external_peers
                                    .iter()
                                    .any(|p| p.as_str().to_ascii_lowercase() == normalized)
                                {
                                    group
                                        .external_peers
                                        .push(PeerUsername::new(telegram_username));
                                }
                                config.mark_dirty(&format!("peer_groups.{group_name}"));
                                println!(
                                    "{}",
                                    console::style(format!("Username '@{normalized}' pre-authorized. No bind code required!"))
                                        .green()
                                );
                            } else {
                                println!(
                                    "No username entered. When you start the daemon ('openz daemon'), message your bot '/bind <code>' using the one-time bind code printed in the logs."
                                );
                            }
                            println!();

                            config.save_dirty().await?;
                            println!(
                                "{}",
                                console::style(format!(
                                    "telegram gateway configured successfully ✔ (@{username})"
                                ))
                                .green()
                                .bold()
                            );
                        }
                        Err(e) => {
                            println!(
                                "{}",
                                console::style(format!(
                                    "Error: Invalid token or network error: {e}"
                                ))
                                .red()
                                .bold()
                            );
                            println!("Please check the token and try again.");
                            println!();
                        }
                    }
                }
                1 => {
                    let del_sel = Select::with_theme(&theme)
                        .with_prompt("Select gateway to delete")
                        .items(&configured)
                        .default(0)
                        .interact()?;
                    let alias_to_delete = &configured[del_sel];

                    let _ = config.delete_map_key("channels.telegram", alias_to_delete);
                    config.mark_dirty(&format!("channels.telegram.{}", alias_to_delete));
                    config.save_dirty().await?;
                    println!(
                        "{}",
                        console::style(format!(
                            "Deleted gateway '{alias_to_delete}' successfully ✔"
                        ))
                        .green()
                        .bold()
                    );
                }
                _ => {
                    break;
                }
            }
        }
    }

    println!();
    Ok(())
}

async fn validate_telegram_token(token: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;
    let url = format!("https://api.telegram.org/bot{token}/getMe");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Telegram API returned error status: {}",
            resp.status()
        ));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse response JSON: {e}"))?;

    if body["ok"].as_bool().unwrap_or(false) {
        if let Some(username) = body["result"]["username"].as_str() {
            Ok(username.to_string())
        } else {
            Ok("bot".to_string())
        }
    } else {
        let desc = body["description"].as_str().unwrap_or("Unknown error");
        Err(desc.to_string())
    }
}

async fn run_configure_subagents_wizard(config: &mut Config) -> Result<()> {
    println!(
        "{}",
        console::style("=== openz Subagents Configuration ===")
            .cyan()
            .bold()
    );
    println!("Configure custom models and fallback chains for default subagents.");
    println!();

    let theme = get_clean_theme();

    let subagents = vec![
        "coder",
        "reviewer",
        "research-agent",
        "openz-planagent",
        "worker",
        "docs-agent",
        "vision-agent",
    ];

    let descriptions = vec![
        (
            "coder",
            "Specialized in writing, refactoring, and fixing code across multiple files.",
        ),
        (
            "reviewer",
            "Responsible for reviewing code changes, security audits, and style adherence.",
        ),
        (
            "research-agent",
            "Gathers context, runs ripgrep searches, and explores codebases or documentation.",
        ),
        (
            "openz-planagent",
            "Generates high-level project milestones, plans, and architectural designs.",
        ),
        (
            "worker",
            "General background worker for running tasks, shell commands, and simple routines.",
        ),
        (
            "docs-agent",
            "Extracts documentation requirements and writes markdown guides/walkthroughs.",
        ),
        (
            "vision-agent",
            "Analyzes visual inputs/images and converts them to rich descriptive markdown.",
        ),
    ];

    loop {
        let mut subagent_items = Vec::new();
        for s in &subagents {
            let primary_model = if let Some(agent_cfg) = config.agents.get(*s) {
                get_model_name_for_ref(config, agent_cfg.model_provider.as_str())
            } else {
                "Not configured".to_string()
            };
            subagent_items.push(format!("{} (Current: {})", s, primary_model));
        }
        subagent_items.push("Exit".to_string());

        let selection = Select::with_theme(&theme)
            .with_prompt("Select a subagent to configure")
            .items(&subagent_items)
            .default(0)
            .interact()?;

        if selection == subagent_items.len() - 1 {
            break;
        }

        let subagent_name = subagents[selection];
        let description = descriptions
            .iter()
            .find(|(n, _)| *n == subagent_name)
            .map(|(_, d)| *d)
            .unwrap_or("");

        println!();
        println!(
            "{}",
            console::style(format!("=== Subagent: {} ===", subagent_name))
                .cyan()
                .bold()
        );
        println!("Description: {}", description);
        println!();

        loop {
            let agent_cfg = config.agents.get(subagent_name);
            let has_primary = agent_cfg.is_some_and(|c| !c.model_provider.is_empty());
            let fallbacks = agent_cfg
                .map(|c| c.model_fallbacks.clone())
                .unwrap_or_default();

            let has_fb1 = fallbacks.len() >= 1 && !fallbacks[0].is_empty();
            let has_fb2 = fallbacks.len() >= 2 && !fallbacks[1].is_empty();
            let has_fb3 = fallbacks.len() >= 3 && !fallbacks[2].is_empty();

            let primary_model_display = agent_cfg
                .map(|c| get_model_name_for_ref(config, c.model_provider.as_str()))
                .unwrap_or_else(|| "Not configured".to_string());

            let fb1_display = if has_fb1 {
                get_model_name_for_ref(config, &fallbacks[0])
            } else {
                "None".to_string()
            };
            let fb2_display = if has_fb2 {
                get_model_name_for_ref(config, &fallbacks[1])
            } else {
                "None".to_string()
            };
            let fb3_display = if has_fb3 {
                get_model_name_for_ref(config, &fallbacks[2])
            } else {
                "None".to_string()
            };

            let primary_tick = if has_primary { "✓" } else { " " };
            let fb1_tick = if has_fb1 { "✓" } else { " " };
            let fb2_tick = if has_fb2 { "✓" } else { " " };
            let fb3_tick = if has_fb3 { "✓" } else { " " };

            let options = vec![
                format!(
                    "[{}] Primary Model: {}",
                    primary_tick, primary_model_display
                ),
                format!("[{}] Fallback 1: {}", fb1_tick, fb1_display),
                format!("[{}] Fallback 2: {}", fb2_tick, fb2_display),
                format!("[{}] Fallback 3: {}", fb3_tick, fb3_display),
                "Default (Reset to defaults)".to_string(),
                "Done".to_string(),
            ];

            let opt_sel = Select::with_theme(&theme)
                .with_prompt("Configure settings")
                .items(&options)
                .default(0)
                .interact()?;

            match opt_sel {
                0 => {
                    if let Some((family, model_id)) =
                        select_subagent_model(subagent_name, config, &theme).await?
                    {
                        let alias = format!("{}", subagent_name);
                        configure_subagent_provider_alias(config, &family, &alias, &model_id)?;
                        let agent_cfg = config.agents.entry(subagent_name.to_string()).or_default();
                        agent_cfg.model_provider =
                            ModelProviderRef::new(format!("{family}.{alias}"));
                        agent_cfg.risk_profile = "default".to_string();
                        agent_cfg.runtime_profile = "default".to_string();
                        config.mark_dirty(&format!("agents.{}", subagent_name));
                        config.save_dirty().await?;
                    }
                }
                1 => {
                    if let Some((family, model_id)) =
                        select_subagent_model(subagent_name, config, &theme).await?
                    {
                        let alias = format!("{}_fallback_1", subagent_name);
                        configure_subagent_provider_alias(config, &family, &alias, &model_id)?;
                        let agent_cfg = config.agents.entry(subagent_name.to_string()).or_default();
                        if agent_cfg.model_fallbacks.is_empty() {
                            agent_cfg.model_fallbacks.push(format!("{family}.{alias}"));
                        } else {
                            agent_cfg.model_fallbacks[0] = format!("{family}.{alias}");
                        }
                        config.mark_dirty(&format!("agents.{}", subagent_name));
                        config.save_dirty().await?;
                    }
                }
                2 => {
                    if let Some((family, model_id)) =
                        select_subagent_model(subagent_name, config, &theme).await?
                    {
                        let alias = format!("{}_fallback_2", subagent_name);
                        configure_subagent_provider_alias(config, &family, &alias, &model_id)?;
                        let agent_cfg = config.agents.entry(subagent_name.to_string()).or_default();
                        while agent_cfg.model_fallbacks.len() < 2 {
                            agent_cfg.model_fallbacks.push(String::new());
                        }
                        agent_cfg.model_fallbacks[1] = format!("{family}.{alias}");
                        config.mark_dirty(&format!("agents.{}", subagent_name));
                        config.save_dirty().await?;
                    }
                }
                3 => {
                    if let Some((family, model_id)) =
                        select_subagent_model(subagent_name, config, &theme).await?
                    {
                        let alias = format!("{}_fallback_3", subagent_name);
                        configure_subagent_provider_alias(config, &family, &alias, &model_id)?;
                        let agent_cfg = config.agents.entry(subagent_name.to_string()).or_default();
                        while agent_cfg.model_fallbacks.len() < 3 {
                            agent_cfg.model_fallbacks.push(String::new());
                        }
                        agent_cfg.model_fallbacks[2] = format!("{family}.{alias}");
                        config.mark_dirty(&format!("agents.{}", subagent_name));
                        config.save_dirty().await?;
                    }
                }
                4 => {
                    let def_ref = config
                        .agents
                        .get("assistant")
                        .map(|a| a.model_provider.as_str().to_string())
                        .unwrap_or_else(|| {
                            let configured = get_configured_families(config);
                            if !configured.is_empty() {
                                format!("{}.default", configured[0])
                            } else {
                                "anthropic.default".to_string()
                            }
                        });
                    {
                        let agent_cfg = config.agents.entry(subagent_name.to_string()).or_default();
                        agent_cfg.model_provider = ModelProviderRef::new(def_ref);
                        agent_cfg.model_fallbacks.clear();
                        agent_cfg.risk_profile = "default".to_string();
                        agent_cfg.runtime_profile = "default".to_string();
                    }
                    config.mark_dirty(&format!("agents.{}", subagent_name));
                    config.save_dirty().await?;
                    println!("{}", console::style("✓ Reset to defaults").green());
                }
                _ => {
                    {
                        let agent_cfg = config.agents.entry(subagent_name.to_string()).or_default();
                        agent_cfg.model_fallbacks.retain(|x| !x.is_empty());
                    }
                    config.mark_dirty(&format!("agents.{}", subagent_name));
                    config.save_dirty().await?;

                    let (primary_provider_ref, fallbacks_snapshot) = {
                        let agent_cfg = config.agents.get(subagent_name).unwrap();
                        (
                            agent_cfg.model_provider.as_str().to_string(),
                            agent_cfg.model_fallbacks.clone(),
                        )
                    };

                    let primary_name = get_model_name_for_ref(config, &primary_provider_ref);
                    let mut fb_names = Vec::new();
                    for fb in &fallbacks_snapshot {
                        fb_names.push(get_model_name_for_ref(config, fb));
                    }
                    while fb_names.len() < 3 {
                        fb_names.push("None".to_string());
                    }

                    println!(
                        "{}",
                        console::style(format!(
                            "subagents configured {}: primary: {}, fallback 1: {}, fallback 2: {}, fallback 3: {}",
                            subagent_name,
                            primary_name,
                            fb_names[0],
                            fb_names[1],
                            fb_names[2]
                        ))
                        .green()
                        .bold()
                    );
                    println!();
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn select_subagent_model(
    subagent: &str,
    config: &Config,
    theme: &dialoguer::theme::ColorfulTheme,
) -> Result<Option<(String, String)>> {
    let configured_families = get_configured_families(config);
    if configured_families.is_empty() {
        println!(
            "{}",
            console::style("No AI Model Providers are configured yet. Please configure a provider via `openz configure` first.")
                .red()
                .bold()
        );
        return Ok(None);
    }

    let model_options = get_available_models_for_families(subagent, &configured_families);
    if model_options.is_empty() {
        return Ok(None);
    }

    let mut select_items: Vec<String> = model_options
        .iter()
        .map(|(label, _, _)| label.clone())
        .collect();
    select_items.push("Custom Model ID".to_string());

    let selection = Select::with_theme(theme)
        .with_prompt("Select model")
        .items(&select_items)
        .default(0)
        .interact()?;

    if selection == select_items.len() - 1 {
        let family_idx = Select::with_theme(theme)
            .with_prompt("Select Provider Family")
            .items(&configured_families)
            .default(0)
            .interact()?;
        let family = configured_families[family_idx].clone();

        let custom_id: String = dialoguer::Input::new()
            .with_prompt("Enter Custom Model ID")
            .interact_text()?;
        let model_id = custom_id.trim().to_string();
        if model_id.is_empty() {
            return Ok(None);
        }
        Ok(Some((family, model_id)))
    } else {
        let (_, family, model_id) = &model_options[selection];
        Ok(Some((family.clone(), model_id.clone())))
    }
}

fn print_logs(config: &Config) -> Result<()> {
    let log_path = config
        .config_path
        .parent()
        .context("Failed to get config path parent")?
        .join("state/runtime-trace.jsonl");

    if !log_path.exists() {
        println!("No logs found at {}", log_path.display());
        return Ok(());
    }

    let file = std::fs::File::open(&log_path)?;
    let reader = std::io::BufReader::new(file);

    for line in std::io::BufRead::lines(reader) {
        let line = line?;
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) {
            let timestamp = event["@timestamp"].as_str().unwrap_or("");
            let severity = event["severity_text"].as_str().unwrap_or("INFO");
            let category = event["event"]["category"].as_str().unwrap_or("");
            let action = event["event"]["action"].as_str().unwrap_or("");
            let message = event["message"].as_str().unwrap_or("");

            let severity_styled = match severity {
                "ERROR" => console::style(severity).red().bold(),
                "WARN" => console::style(severity).yellow().bold(),
                "INFO" => console::style(severity).green(),
                _ => console::style(severity).dim(),
            };

            let ts_display = if timestamp.len() >= 19 {
                format!("{} {}", &timestamp[0..10], &timestamp[11..19])
            } else {
                timestamp.to_string()
            };

            println!(
                "[{}] {} [{}:{}] {}",
                console::style(ts_display).dim(),
                severity_styled,
                console::style(category).cyan(),
                console::style(action).blue(),
                message
            );
        } else {
            println!("{}", line);
        }
    }

    Ok(())
}

fn run_log_verification(config: &Config) -> Result<()> {
    let log_path = config
        .config_path
        .parent()
        .context("Failed to get config path parent")?
        .join("state/runtime-trace.jsonl");

    if !log_path.exists() {
        println!("No logs found at {}", log_path.display());
        return Ok(());
    }

    println!(
        "Verifying cryptographic integrity of logs at {}...",
        log_path.display()
    );

    match zeroclaw_log::reader::verify_log_integrity(&log_path) {
        Ok(true) => {
            println!(
                "{}",
                console::style(
                    "✓ Log integrity verified: chain is continuous and signatures match."
                )
                .green()
                .bold()
            );
        }
        Ok(false) => {
            println!(
                "{}",
                console::style("✗ Log integrity check failed: potential tampering detected!")
                    .red()
                    .bold()
            );
            std::process::exit(1);
        }
        Err(err) => {
            println!(
                "{}",
                console::style(format!("✗ Verification error occurred: {err}"))
                    .red()
                    .bold()
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

struct SessionFile {
    path: std::path::PathBuf,
    modified: std::time::SystemTime,
    preview: String,
}

fn get_sessions_dir() -> std::path::PathBuf {
    let base = std::env::var("OPENZ_CONFIG_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            directories::BaseDirs::new()
                .map(|bd| bd.home_dir().join(".openz"))
                .unwrap_or_else(|| std::path::PathBuf::from(".openz"))
        });
    base.join("sessions")
}

fn list_sessions() -> Vec<SessionFile> {
    let mut sessions = Vec::new();
    let dirs = vec![
        get_sessions_dir(),
        // Fallback
        directories::BaseDirs::new()
            .map(|bd| bd.home_dir().join(".openz").join("sessions"))
            .unwrap_or_else(|| std::path::PathBuf::from(".openz/sessions")),
        directories::BaseDirs::new()
            .map(|bd| bd.home_dir().join(".zeroclaw").join("sessions"))
            .unwrap_or_else(|| std::path::PathBuf::from(".zeroclaw/sessions")),
    ];

    for dir in dirs {
        if dir.exists() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                        if let Ok(metadata) = entry.metadata() {
                            let modified = metadata
                                .modified()
                                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

                            // Try to get a short preview of the last message
                            let mut preview = "No messages".to_string();
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                if let Ok(state) =
                                    serde_json::from_str::<serde_json::Value>(&content)
                                {
                                    if let Some(history) = state["history"].as_array() {
                                        let mut last_msg = None;
                                        // First pass: find the last user message
                                        for msg in history.iter().rev() {
                                            let role = msg["role"].as_str().unwrap_or("");
                                            if role == "user" {
                                                if let Some(s) = msg["content"].as_str() {
                                                    last_msg = Some(s.to_string());
                                                    break;
                                                } else if let Some(arr) = msg["content"].as_array()
                                                {
                                                    for block in arr {
                                                        if let Some(text) = block["text"].as_str() {
                                                            last_msg = Some(text.to_string());
                                                            break;
                                                        }
                                                    }
                                                    if last_msg.is_some() {
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        // Fallback pass: find the last assistant message if no user message was found
                                        if last_msg.is_none() {
                                            for msg in history.iter().rev() {
                                                let role = msg["role"].as_str().unwrap_or("");
                                                if role == "assistant" {
                                                    if let Some(s) = msg["content"].as_str() {
                                                        // Strip think tags
                                                        let mut cleaned = String::new();
                                                        let mut remaining = s;
                                                        while let Some(start_idx) =
                                                            remaining.find("<think>")
                                                        {
                                                            cleaned
                                                                .push_str(&remaining[..start_idx]);
                                                            if let Some(end_idx) = remaining
                                                                [start_idx..]
                                                                .find("</think>")
                                                            {
                                                                remaining = &remaining
                                                                    [start_idx + end_idx + 8..];
                                                            } else {
                                                                remaining = "";
                                                                break;
                                                            }
                                                        }
                                                        cleaned.push_str(remaining);
                                                        last_msg = Some(cleaned);
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if let Some(msg) = last_msg {
                                            let mut trimmed_msg = msg.trim().replace('\n', " ");
                                            if trimmed_msg.starts_with('[') {
                                                if let Some(close_idx) = trimmed_msg.find(']') {
                                                    let ts_content = &trimmed_msg[1..close_idx];
                                                    if ts_content.len() >= 16 {
                                                        let clean_ts = &ts_content[..16];
                                                        trimmed_msg = format!(
                                                            "[{clean_ts}]{}",
                                                            &trimmed_msg[close_idx + 1..]
                                                        );
                                                    }
                                                }
                                            }
                                            if trimmed_msg.len() > 50 {
                                                preview = format!("{}...", &trimmed_msg[0..50]);
                                            } else {
                                                preview = trimmed_msg;
                                            }
                                        }
                                    }
                                }
                            }

                            sessions.push(SessionFile {
                                path,
                                modified,
                                preview,
                            });
                        }
                    }
                }
            }
        }
    }

    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    sessions
}

fn interactive_session_picker(
    sessions: &[SessionFile],
    default_model: &str,
) -> Result<Option<std::path::PathBuf>> {
    use crossterm::{
        event::{self, Event, KeyCode, KeyModifiers},
        terminal::{disable_raw_mode, enable_raw_mode},
    };
    use std::io::Write;

    if sessions.is_empty() {
        return Ok(None);
    }

    let mut selected_idx = 0;
    let items_len = sessions.len() + 1;
    let mut start_viewport = 0;
    let mut prev_lines_drawn = 0;
    let mut needs_redraw = true;

    enable_raw_mode().context("Failed to enable raw mode for session picker")?;
    let mut stdout = std::io::stdout();

    let _ = stdout.write_all(b"\x1B[?25l");
    let _ = stdout.flush();

    loop {
        if needs_redraw {
            if prev_lines_drawn > 0 {
                for _ in 0..prev_lines_drawn {
                    print!("\x1B[1A\x1B[K");
                }
                let _ = stdout.flush();
            }

            let mut lines_drawn = 0;
            let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
            let cols = cols as usize;
            let rows = rows as usize;

            let max_display_sessions = ((rows.saturating_sub(9)) / 3).max(1);

            if selected_idx > 0 {
                let s_idx = selected_idx - 1;
                if s_idx < start_viewport {
                    start_viewport = s_idx;
                } else if s_idx >= start_viewport + max_display_sessions {
                    start_viewport = s_idx + 1 - max_display_sessions;
                }
            } else {
                start_viewport = 0;
            }

            if start_viewport + max_display_sessions > sessions.len() {
                start_viewport = sessions.len().saturating_sub(max_display_sessions);
            }

            println!("\r\x1B[K");
            println!("\r\x1B[K\x1B[1m\x1B[38;2;139;92;246mOpenZ\x1B[0m");
            println!("\r\x1B[K");
            lines_drawn += 3;

            let is_selected_new = selected_idx == 0;
            let cursor_style_new = if is_selected_new {
                "\x1B[38;2;139;92;246m❯ \x1B[0m"
            } else {
                "  "
            };
            let label_new = if is_selected_new {
                "\x1B[1m\x1B[38;2;228;228;231mnew session\x1B[0m"
            } else {
                "\x1B[38;2;113;113;122mnew session\x1B[0m"
            };
            println!("\r\x1B[K{cursor_style_new}{label_new}");
            println!("\r\x1B[K");
            lines_drawn += 2;

            println!("\r\x1B[K\x1B[38;2;113;113;122mRecent sessions\x1B[0m");
            println!("\r\x1B[K");
            lines_drawn += 2;

            let end_viewport = (start_viewport + max_display_sessions).min(sessions.len());
            for idx in start_viewport..end_viewport {
                let is_selected = selected_idx > 0 && (idx == selected_idx - 1);
                let cursor_style = if is_selected {
                    "\x1B[38;2;139;92;246m❯ \x1B[0m"
                } else {
                    "  "
                };

                let s = &sessions[idx];
                let time_ago = match s.modified.elapsed() {
                    Ok(d) => {
                        let secs = d.as_secs();
                        if secs < 60 {
                            "just now".to_string()
                        } else if secs < 3600 {
                            format!("{}m ago", secs / 60)
                        } else if secs < 86400 {
                            format!("{}h ago", secs / 3600)
                        } else if secs < 172800 {
                            "yesterday".to_string()
                        } else {
                            format!("{} days ago", secs / 86400)
                        }
                    }
                    Err(_) => "some time ago".to_string(),
                };

                let max_preview_len = if cols > 6 { cols - 6 } else { 10 };
                let display_preview = if s.preview.len() > max_preview_len {
                    let mut char_idx = 0;
                    let mut byte_idx = 0;
                    for (b_idx, _) in s.preview.char_indices() {
                        if char_idx >= max_preview_len.saturating_sub(3) {
                            byte_idx = b_idx;
                            break;
                        }
                        char_idx += 1;
                    }
                    if byte_idx == 0 {
                        format!("{}...", s.preview)
                    } else {
                        format!("{}...", &s.preview[..byte_idx])
                    }
                } else {
                    s.preview.clone()
                };

                let preview_style = if is_selected {
                    format!("\x1B[1m\x1B[38;2;228;228;231m{}\x1B[0m", display_preview)
                } else {
                    format!("\x1B[38;2;161;161;170m{}\x1B[0m", display_preview)
                };

                println!("\r\x1B[K{cursor_style}{preview_style}");

                let meta_str = format!("{} · {}", time_ago, default_model);
                let max_meta_len = if cols > 8 { cols - 8 } else { 10 };
                let display_meta = if meta_str.len() > max_meta_len {
                    let mut char_idx = 0;
                    let mut byte_idx = 0;
                    for (b_idx, _) in meta_str.char_indices() {
                        if char_idx >= max_meta_len.saturating_sub(3) {
                            byte_idx = b_idx;
                            break;
                        }
                        char_idx += 1;
                    }
                    if byte_idx == 0 {
                        format!("{}...", meta_str)
                    } else {
                        format!("{}...", &meta_str[..byte_idx])
                    }
                } else {
                    meta_str
                };

                println!("\r\x1B[K   \x1B[38;2;113;113;122m{}\x1B[0m", display_meta);
                println!("\r\x1B[K");
                lines_drawn += 3;
            }

            prev_lines_drawn = lines_drawn;
            let _ = stdout.flush();
            needs_redraw = false;
        }

        match event::read()? {
            Event::Key(key_event) => {
                if key_event.kind == event::KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Up => {
                            if selected_idx > 0 {
                                selected_idx -= 1;
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Down => {
                            if selected_idx < items_len - 1 {
                                selected_idx += 1;
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Char('n') => {
                            if selected_idx != 0 {
                                selected_idx = 0;
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Enter => {
                            if prev_lines_drawn > 0 {
                                for _ in 0..prev_lines_drawn {
                                    print!("\x1B[1A\x1B[K");
                                }
                            }
                            let _ = stdout.flush();
                            let _ = stdout.write_all(b"\x1B[?25h");
                            let _ = stdout.flush();
                            let _ = disable_raw_mode();
                            if selected_idx == 0 {
                                return Ok(None);
                            } else {
                                return Ok(Some(sessions[selected_idx - 1].path.clone()));
                            }
                        }
                        KeyCode::Char('c')
                            if key_event.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            let _ = stdout.write_all(b"\x1B[?25h");
                            let _ = stdout.flush();
                            let _ = disable_raw_mode();
                            println!("ok byee....");
                            std::process::exit(130);
                        }
                        KeyCode::Esc => {
                            if prev_lines_drawn > 0 {
                                for _ in 0..prev_lines_drawn {
                                    print!("\x1B[1A\x1B[K");
                                }
                            }
                            let _ = stdout.flush();
                            let _ = stdout.write_all(b"\x1B[?25h");
                            let _ = stdout.flush();
                            let _ = disable_raw_mode();
                            return Ok(None);
                        }
                        _ => {}
                    }
                }
            }
            Event::Resize(_, _) => {
                needs_redraw = true;
            }
            _ => {}
        }
    }
}

async fn run_mcp_setup_wizard(config: &mut Config) -> Result<()> {
    println!(
        "{}",
        console::style("=== MCP Server API Configuration ===")
            .cyan()
            .bold()
    );
    if config.mcp.servers.is_empty() {
        println!("No MCP servers configured yet.");
        return Ok(());
    }

    let mut server_names = Vec::new();
    for s in &config.mcp.servers {
        let origin = if config.dynamic_mcp_servers.contains(&s.name) {
            "mcp.d"
        } else {
            "config.toml"
        };
        server_names.push(format!("{} ({})", s.name, origin));
    }

    let selection = Select::new()
        .with_prompt("Select an MCP server to configure")
        .items(&server_names)
        .interact()?;

    let selected_server = &config.mcp.servers[selection];
    let name = selected_server.name.clone();
    let server_command = selected_server.command.clone();
    let server_args = selected_server.args.clone();
    let server_transport = selected_server.transport.clone();
    let is_dynamic = config.dynamic_mcp_servers.contains(&name);

    println!();
    println!(
        "Configuring MCP Server: {}",
        console::style(&name).green().bold()
    );

    // Guess default environment variable name based on server name
    let default_env_name = match name.as_str() {
        "github" => "GITHUB_TOKEN",
        "opencode" => "OPENCODE_API_KEY",
        "exa" => "EXA_API_KEY",
        "gitlab" => "GITLAB_TOKEN",
        other => &format!("{}_API_KEY", other.to_uppercase().replace('-', "_")),
    };

    let env_name: String = dialoguer::Input::new()
        .with_prompt("Enter environment variable name")
        .default(default_env_name.to_string())
        .interact_text()?;

    let env_value: String = dialoguer::Password::new()
        .with_prompt(&format!("Enter value for {env_name}"))
        .interact()?;

    let env_name = env_name.trim().to_string();
    let env_value = env_value.trim().to_string();

    if is_dynamic {
        // Save to dynamic file: mcp.d/<name>.toml
        let config_dir = config
            .config_path
            .parent()
            .context("Failed to resolve config directory")?;
        let mcp_file_path = config_dir.join("mcp.d").join(format!("{}.toml", name));

        let mut content = if mcp_file_path.exists() {
            tokio::fs::read_to_string(&mcp_file_path).await?
        } else {
            String::new()
        };

        // If file was empty, initialize it with basic info from memory
        if content.trim().is_empty() {
            content = format!(
                "command = {:?}\nargs = {:?}\ntransport = {:?}\n",
                server_command, server_args, server_transport
            );
        }

        let mut doc: toml_edit::DocumentMut = content
            .parse()
            .context("Failed to parse dynamic MCP config file")?;

        // Add to [env] section
        if doc.get("env").is_none() {
            doc.insert("env", toml_edit::Item::Table(toml_edit::Table::new()));
        }

        if let Some(env_table) = doc.get_mut("env").and_then(|i| i.as_table_mut()) {
            env_table.insert(&env_name, toml_edit::value(env_value));
        }

        // Set enabled = true so the server is automatically activated
        doc.insert("enabled", toml_edit::value(true));

        tokio::fs::write(&mcp_file_path, doc.to_string()).await?;
        println!(
            "{}",
            console::style("Successfully saved credentials to dynamic configuration file:").green()
        );
        println!("  {}", mcp_file_path.display());
    } else {
        // Central config
        let mut full_config = config.clone();
        if let Some(server) = full_config.mcp.servers.iter_mut().find(|s| s.name == name) {
            server.env.insert(env_name.clone(), env_value.clone());
            server.enabled = true;
        }

        // Save central config
        full_config.save().await?;

        // Update in-memory copy
        if let Some(server) = config.mcp.servers.iter_mut().find(|s| s.name == name) {
            server.env.insert(env_name, env_value);
            server.enabled = true;
        }
        println!(
            "{}",
            console::style("Successfully saved credentials to central config.toml").green()
        );
    }

    Ok(())
}

async fn run_agent_setup_wizard(config: &mut Config) -> Result<()> {
    use dialoguer::{Input, Select};
    use zeroclaw_config::providers::ModelProviderRef;

    println!(
        "{}",
        console::style("=== Configure & Add Subagent ===")
            .cyan()
            .bold()
    );
    println!("This will add a new subagent to central config.toml");
    println!();

    let name: String = Input::new()
        .with_prompt("Enter agent name (e.g. coder, researcher)")
        .interact_text()?;
    let name = name.trim().to_lowercase();
    if name.is_empty() {
        return Err(anyhow::anyhow!("Agent name cannot be empty"));
    }

    let existing = config.agents.get(&name);

    let providers = vec![
        "anthropic.default",
        "openai.default",
        "gemini.default",
        "groq.default",
        "deepseek.default",
        "ollama.default",
        "openrouter.default",
        "lmstudio.default",
        "Other",
    ];

    let (default_sel, default_system, default_fallbacks, existing_custom) =
        if let Some(existing_agent) = existing {
            println!("Agent '{}' already exists. We will edit it.", name);

            let agent_ws = config.agent_workspace_dir(&name);
            let existing_prompt = if agent_ws.join("IDENTITY.md").exists() {
                tokio::fs::read_to_string(agent_ws.join("IDENTITY.md"))
                    .await
                    .unwrap_or_default()
            } else {
                String::new()
            };

            let p_str = existing_agent.model_provider.as_str();
            let idx = providers.iter().position(|&p| p == p_str);
            if let Some(selection) = idx {
                (
                    selection,
                    existing_prompt,
                    existing_agent.model_fallbacks.join(", "),
                    None,
                )
            } else {
                (
                    providers.len() - 1,
                    existing_prompt,
                    existing_agent.model_fallbacks.join(", "),
                    Some(p_str.to_string()),
                )
            }
        } else {
            (0, String::new(), String::new(), None)
        };

    let selection = Select::new()
        .with_prompt("Select model provider for this agent")
        .items(&providers)
        .default(default_sel)
        .interact()?;

    let picked_provider = if selection == providers.len() - 1 {
        let custom: String = Input::new()
            .with_prompt("Enter Custom Provider (e.g. compatible.myalias)")
            .default(existing_custom.unwrap_or_default())
            .interact_text()?;
        custom.trim().to_string()
    } else {
        providers[selection].to_string()
    };

    let system_prompt: String = Input::new()
        .with_prompt("Enter system prompt instructions (role/identity for this agent)")
        .default(default_system)
        .interact_text()?;
    let system_prompt = system_prompt.trim().to_string();

    let fallbacks_str: String = Input::new()
        .with_prompt("Enter fallback model providers (comma-separated, e.g. google.default, groq.default) [optional]")
        .allow_empty(true)
        .default(default_fallbacks)
        .interact_text()?;
    let mut fallbacks = Vec::new();
    for f in fallbacks_str.split(',') {
        let f_trimmed = f.trim();
        if !f_trimmed.is_empty() {
            fallbacks.push(f_trimmed.to_string());
        }
    }

    let mut new_agent = existing.cloned().unwrap_or_else(|| {
        let mut default_agent = zeroclaw_config::schema::AliasedAgentConfig::default();
        default_agent.risk_profile = "default".to_string();
        default_agent.runtime_profile = "default".to_string();
        default_agent
    });
    new_agent.model_provider = ModelProviderRef::new(picked_provider);
    new_agent.model_fallbacks = fallbacks;

    // Write system prompt to IDENTITY.md inside the agent workspace
    let agent_ws = config.agent_workspace_dir(&name);
    tokio::fs::create_dir_all(&agent_ws).await.ok();
    tokio::fs::write(agent_ws.join("IDENTITY.md"), system_prompt)
        .await
        .ok();

    config.agents.insert(name.clone(), new_agent);
    config.mark_dirty(&format!("agents.{}", name));
    config.save_dirty().await?;

    println!(
        "{}",
        console::style(format!("Successfully created/updated subagent: {}", name))
            .green()
            .bold()
    );
    println!(
        "Central config file updated: {}",
        config.config_path.display()
    );

    Ok(())
}
