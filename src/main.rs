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
    Configure,
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
            Commands::Configure => {
                let mut config = Config::load_or_init().await?;
                run_configure_wizard(&mut config).await?;
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

    if !sessions.is_empty() {
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
        "  {} v{}",
        console::style("openz").bold(),
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

async fn run_configure_wizard(config: &mut Config) -> Result<()> {
    println!(
        "{}",
        console::style("=== openz Configuration Setup ===")
            .cyan()
            .bold()
    );
    println!("This will set up your model provider and default agent.");
    println!();

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

    let selection = Select::new()
        .with_prompt("Select AI Model Provider")
        .items(&providers)
        .default(0)
        .interact()?;

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
        let key: String = dialoguer::Password::new()
            .with_prompt("Enter API Key")
            .interact()?;
        key.trim().to_string()
    } else {
        String::new()
    };

    let default_model = match picked.as_str() {
        "anthropic" => "claude-3-5-sonnet-20241022",
        "openai" => "gpt-4o",
        "gemini" => "gemini-1.5-pro",
        "groq" => "llama3-70b-8192",
        "deepseek" => "deepseek-chat",
        "ollama" => "llama3",
        "lmstudio" => "model-id",
        _ => "model-id",
    };

    let model: String = dialoguer::Input::new()
        .with_prompt("Enter Model ID")
        .default(default_model.to_string())
        .interact_text()?;
    let model = model.trim().to_string();

    let alias = "default";
    config.providers.models.ensure(&picked, alias);

    let prefix = format!("providers.models.{picked}.{alias}");
    if !api_key.is_empty() {
        config.set_secret_persistent(&format!("{prefix}.api_key"), api_key)?;
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
        &format!("{agent_prefix}.model_provider"),
        &format!("{picked}.{alias}"),
    )?;
    config.set_prop_persistent(&format!("{agent_prefix}.risk_profile"), "default")?;
    config.set_prop_persistent(&format!("{agent_prefix}.runtime_profile"), "default")?;

    config.save_dirty().await?;
    println!();
    println!(
        "{}",
        console::style("✓ Configuration saved successfully!")
            .green()
            .bold()
    );
    println!("Config file: {}", config.config_path.display());
    println!();

    Ok(())
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

    enable_raw_mode().context("Failed to enable raw mode for session picker")?;
    let mut stdout = std::io::stdout();

    let _ = stdout.write_all(b"\x1B[?25l");
    let _ = stdout.flush();

    loop {
        let mut lines_drawn = 0;
        let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
        let cols = cols as usize;
        let rows = rows as usize;

        // Allocate 7 lines for header/spacing and 2 lines of safety buffer.
        let max_display_sessions = ((rows.saturating_sub(9)) / 3).max(1);

        // Adjust start_viewport to keep selected_idx visible (0 is new session, 1..=sessions.len() are sessions)
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

        // Safety bound: start_viewport should not overshoot
        if start_viewport + max_display_sessions > sessions.len() {
            start_viewport = sessions.len().saturating_sub(max_display_sessions);
        }

        println!("\r\x1B[K");
        println!("\r\x1B[K\x1B[1m\x1B[38;2;139;92;246mOpenZ\x1B[0m");
        println!("\r\x1B[K");
        lines_drawn += 3;

        // Render "new session" at index 0
        let is_selected_new = selected_idx == 0;
        let cursor_str_new = if is_selected_new { "❯ " } else { "  " };
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

        // Render "Recent sessions" header
        println!("\r\x1B[K\x1B[38;2;113;113;122mRecent sessions\x1B[0m");
        println!("\r\x1B[K");
        lines_drawn += 2;

        let end_viewport = (start_viewport + max_display_sessions).min(sessions.len());
        for idx in start_viewport..end_viewport {
            let is_selected = selected_idx > 0 && (idx == selected_idx - 1);
            let cursor_str = if is_selected { "❯ " } else { "  " };
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

        let _ = stdout.flush();

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key_event) = event::read()? {
                if key_event.kind == event::KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Up => {
                            if selected_idx > 0 {
                                selected_idx -= 1;
                            }
                        }
                        KeyCode::Down => {
                            if selected_idx < items_len - 1 {
                                selected_idx += 1;
                            }
                        }
                        KeyCode::Char('n') => {
                            selected_idx = 0;
                        }
                        KeyCode::Enter => {
                            for _ in 0..lines_drawn {
                                print!("\x1B[1A\x1B[K");
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
                            for _ in 0..lines_drawn {
                                print!("\x1B[1A\x1B[K");
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
        }

        print!("\x1B[{}A", lines_drawn);
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
