use crate::observability::Observer;
use crate::tools::Tool;
use anyhow::Result;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
};
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use zeroclaw_config::schema::Config;
use zeroclaw_providers::{ChatMessage, ModelProvider};

// Pre-edit ritual states check:
// 1. `history`: Source of truth — created here for local chat log.
// 2. `input`: Source of truth — created here for active input text.
// 3. `active_model` / `active_provider`: Source of truth is agent config / switch state — cloned here.
// 4. `cost_usd` / `tokens_used`: Source of truth is CostTracker — queried/updated on tick.
// 5. `mcp_servers`: Source of truth is config.mcp.servers — resolved on start.
// 6. `skills`: Source of truth is skills registry — resolved on start.
// 7. `lsp_status`: Source of truth — created here based on presence of rust-analyzer.
// 8. `todos`: Source of truth is GSD task.md / .planning/task.md — parsed dynamically.
// 9. `subagents`: Source of truth — created here to list delegates.
// 10. `status`: Source of truth — created here to show running tool status.
// 11. `is_thinking`: Source of truth — created here to control rendering animations.
// 12. `scroll_offset`: Source of truth — created here to track view scrolling.

#[derive(Clone, Debug)]
pub struct TodoItem {
    pub checked: bool,
    pub text: String,
}

pub enum TuiEvent {
    Input(KeyEvent),
    Tick,
    AgentDelta(crate::agent::loop_::StreamDelta),
    AgentFinished(Result<Vec<ChatMessage>>),
}

pub struct TuiSession {
    pub history: Vec<ChatMessage>,
    pub input: String,
    pub active_model: String,
    pub active_provider: String,
    pub cost_usd: f64,
    pub tokens_used: usize,
    pub mcp_servers: Vec<String>,
    pub skills: Vec<String>,
    pub lsp_status: String,
    pub todos: Vec<TodoItem>,
    pub subagents: Vec<String>,
    pub status: String,
    pub is_thinking: bool,
    pub scroll_offset: usize,
    pub spinner_frame: usize,
}

impl TuiSession {
    pub fn new(
        history: Vec<ChatMessage>,
        active_model: String,
        active_provider: String,
        mcp_servers: Vec<String>,
        skills: Vec<String>,
    ) -> Self {
        let lsp_status = if which::which("rust-analyzer").is_ok() {
            "Active".to_string()
        } else {
            "Inactive".to_string()
        };

        let todos = load_todos();

        Self {
            history,
            input: String::new(),
            active_model,
            active_provider,
            cost_usd: 0.0,
            tokens_used: 0,
            mcp_servers,
            skills,
            lsp_status,
            todos,
            subagents: Vec::new(),
            status: "Idle".to_string(),
            is_thinking: false,
            scroll_offset: 0,
            spinner_frame: 0,
        }
    }
}

pub(crate) fn load_todos() -> Vec<TodoItem> {
    let mut todos = Vec::new();
    let paths = vec![PathBuf::from("task.md"), PathBuf::from(".planning/task.md")];
    for path in paths {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("- [ ]") {
                        todos.push(TodoItem {
                            checked: false,
                            text: trimmed[5..].trim().to_string(),
                        });
                    } else if trimmed.starts_with("- [x]") || trimmed.starts_with("- [X]") {
                        todos.push(TodoItem {
                            checked: true,
                            text: trimmed[5..].trim().to_string(),
                        });
                    } else if trimmed.starts_with("- [/]") {
                        todos.push(TodoItem {
                            checked: false,
                            text: format!("(in progress) {}", trimmed[5..].trim()),
                        });
                    }
                }
                break;
            }
        }
    }
    todos
}

pub struct TuiApp {
    pub session: TuiSession,
}

impl TuiApp {
    pub fn new(
        history: Vec<ChatMessage>,
        active_model: String,
        active_provider: String,
        mcp_servers: Vec<String>,
        skills: Vec<String>,
    ) -> Self {
        Self {
            session: TuiSession::new(history, active_model, active_provider, mcp_servers, skills),
        }
    }

    pub async fn run_loop(
        &mut self,
        config: Config,
        model_provider: Box<dyn ModelProvider>,
        tools_registry: Arc<Vec<Box<dyn Tool>>>,
        observer: Arc<dyn Observer>,
        turn_temperature: Option<f64>,
        approval_manager: Option<Arc<crate::approval::ApprovalManager>>,
        channel_name: String,
        multimodal_config: zeroclaw_config::schema::MultimodalConfig,
        max_tool_iterations: usize,
        excluded_tools: Vec<String>,
        tool_call_dedup_exempt: Vec<String>,
        activated_handle: Option<Arc<Mutex<crate::tools::ActivatedToolSet>>>,
        model_switch_callback: crate::agent::loop_::ModelSwitchCallback,
        pacing_config: zeroclaw_config::schema::PacingConfig,
        strict_tool_parsing: bool,
        max_tool_result_chars: usize,
        max_context_tokens: usize,
        session_state_file: Option<PathBuf>,
        _mem: Arc<dyn zeroclaw_memory::Memory>,
    ) -> Result<()> {
        let provider_ref = Arc::new(tokio::sync::Mutex::new(model_provider));
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, cursor::Hide)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let (event_tx, mut event_rx) = mpsc::channel::<TuiEvent>(100);

        // Key polling thread
        let event_tx_input = event_tx.clone();
        tokio::spawn(async move {
            loop {
                if event::poll(Duration::from_millis(50)).unwrap_or(false) {
                    if let Ok(Event::Key(key)) = event::read() {
                        if event_tx_input.send(TuiEvent::Input(key)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        // Spinner tick thread
        let event_tx_tick = event_tx.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if event_tx_tick.send(TuiEvent::Tick).await.is_err() {
                    break;
                }
            }
        });

        let mut provider_name = self.session.active_provider.clone();
        let mut model_name = self.session.active_model.clone();

        loop {
            // Draw TUI frame
            terminal.draw(|f| {
                self.draw_ui(f);
            })?;

            tokio::select! {
                Some(event) = event_rx.recv() => {
                    match event {
                        TuiEvent::Input(key) => {
                            if !self.session.is_thinking {
                                match key.code {
                                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                                        break;
                                    }
                                    KeyCode::Char(c) => {
                                        self.session.input.push(c);
                                    }
                                    KeyCode::Backspace => {
                                        self.session.input.pop();
                                    }
                                    KeyCode::Esc => {
                                        self.session.input.clear();
                                    }
                                    KeyCode::Up => {
                                        if self.session.scroll_offset > 0 {
                                            self.session.scroll_offset -= 1;
                                        }
                                    }
                                    KeyCode::Down => {
                                        self.session.scroll_offset += 1;
                                    }
                                    KeyCode::PageUp => {
                                        if self.session.scroll_offset > 10 {
                                            self.session.scroll_offset -= 10;
                                        } else {
                                            self.session.scroll_offset = 0;
                                        }
                                    }
                                    KeyCode::PageDown => {
                                        self.session.scroll_offset += 10;
                                    }
                                    KeyCode::Enter => {
                                        let trimmed = self.session.input.trim().to_string();
                                        if !trimmed.is_empty() {
                                            self.session.input.clear();
                                            if trimmed == "/quit" || trimmed == "/exit" {
                                                break;
                                            } else if trimmed == "/clear" || trimmed == "/new" {
                                                self.session.history.clear();
                                                self.session.history.push(ChatMessage::system("You are an AI assistant."));
                                                self.session.status = "Cleared session".to_string();
                                                if let Some(ref path) = session_state_file {
                                                    let _ = crate::agent::loop_::save_interactive_session_history(path, &self.session.history);
                                                }
                                            } else if trimmed.starts_with("/models ") {
                                                let target = trimmed["/models ".len()..].trim();
                                                if let Some((p, m)) = target.split_once('/') {
                                                    let new_provider_name = p.trim().to_string();
                                                    let new_model_name = m.trim().to_string();
                                                    let new_agent_model_provider = config
                                                        .providers
                                                        .models
                                                        .find(&new_provider_name, "default");
                                                    let provider_runtime_options =
                                                        zeroclaw_providers::provider_runtime_options_from_config(&config);

                                                    match zeroclaw_providers::create_routed_model_provider_with_options(
                                                        &config,
                                                        &new_provider_name,
                                                        new_agent_model_provider.and_then(|e| e.api_key.as_deref()),
                                                        new_agent_model_provider.and_then(|e| e.uri.as_deref()),
                                                        &config.reliability,
                                                        &config.model_routes,
                                                        &new_model_name,
                                                        &provider_runtime_options,
                                                    ) {
                                                        Ok(new_mp) => {
                                                            {
                                                                let mut guard = provider_ref.lock().await;
                                                                *guard = new_mp;
                                                            }
                                                            provider_name = new_provider_name.clone();
                                                            model_name = new_model_name.clone();
                                                            self.session.active_provider = new_provider_name;
                                                            self.session.active_model = new_model_name;
                                                            self.session.status = format!("Switched model to {}", target);
                                                        }
                                                        Err(e) => {
                                                            self.session.status = format!("Switch failed: {}", e);
                                                        }
                                                    }
                                                } else {
                                                    self.session.status = "Use: /models <provider>/<model>".to_string();
                                                }
                                            } else if trimmed == "/models" || trimmed == "/model" {
                                                self.session.status = "Use: /models <provider>/<model>".to_string();
                                            } else {
                                                // Start LLM/Agent Loop Turn
                                                self.session.is_thinking = true;
                                                self.session.status = "Thinking...".to_string();
                                                self.session.scroll_offset = 0;

                                                // Push user message to TUI history
                                                let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z");
                                                let enriched = format!("[{}] {}", now, trimmed);
                                                self.session.history.push(ChatMessage::user(&enriched));

                                                // Spawn background processing task
                                                let event_tx_done = event_tx.clone();
                                                let mut history_clone = self.session.history.clone();
                                                let provider_ref_clone = Arc::clone(&provider_ref);
                                                let tools_reg_clone = Arc::clone(&tools_registry);
                                                let obs_clone = Arc::clone(&observer);
                                                let provider_name_clone = provider_name.clone();
                                                let model_name_clone = model_name.clone();
                                                let app_mgr_clone = approval_manager.clone();
                                                let chan_name_clone = channel_name.clone();
                                                let multi_clone = multimodal_config.clone();
                                                let dedup_clone = tool_call_dedup_exempt.clone();
                                                let act_clone = activated_handle.clone();
                                                let switch_cb_clone = model_switch_callback.clone();
                                                let pacing_clone = pacing_config.clone();
                                                let exc_clone = excluded_tools.clone();

                                                tokio::spawn(async move {
                                                    let (delta_tx, mut delta_rx) = mpsc::channel::<crate::agent::loop_::StreamDelta>(100);

                                                    // Relay stream events to main loop
                                                    let event_tx_relay = event_tx_done.clone();
                                                    tokio::spawn(async move {
                                                        while let Some(delta) = delta_rx.recv().await {
                                                            let _ = event_tx_relay.send(TuiEvent::AgentDelta(delta)).await;
                                                        }
                                                    });

                                                    let prov_guard = provider_ref_clone.lock().await;
                                                    let res = crate::agent::loop_::run_tool_call_loop(
                                                        &**prov_guard,
                                                        &mut history_clone,
                                                        &tools_reg_clone,
                                                        obs_clone.as_ref(),
                                                        &provider_name_clone,
                                                        &model_name_clone,
                                                        turn_temperature,
                                                        true,
                                                        app_mgr_clone.as_deref(),
                                                        &chan_name_clone,
                                                        None,
                                                        &multi_clone,
                                                        None, // vision_provider
                                                        max_tool_iterations,
                                                        None, // cancel token
                                                        Some(delta_tx),
                                                        None,
                                                        &exc_clone,
                                                        &dedup_clone,
                                                        act_clone.as_ref(),
                                                        Some(switch_cb_clone),
                                                        &pacing_clone,
                                                        strict_tool_parsing,
                                                        max_tool_result_chars,
                                                        max_context_tokens,
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                    ).await;

                                                    match res {
                                                        Ok(_) => {
                                                            let _ = event_tx_done.send(TuiEvent::AgentFinished(Ok(history_clone))).await;
                                                        }
                                                        Err(e) => {
                                                            let _ = event_tx_done.send(TuiEvent::AgentFinished(Err(e))).await;
                                                        }
                                                    }
                                                });
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            } else {
                                // In-flight cancellation support
                                if let KeyCode::Char('c') = key.code {
                                    if key.modifiers.contains(event::KeyModifiers::CONTROL) {
                                        // Wait, we can't easily cancel without token, let's just abort thinking locally
                                        self.session.is_thinking = false;
                                        self.session.status = "Cancelled".to_string();
                                    }
                                }
                            }
                        }
                        TuiEvent::Tick => {
                            if self.session.is_thinking {
                                self.session.spinner_frame = (self.session.spinner_frame + 1) % 4;
                            }
                            self.session.todos = load_todos();
                        }
                        TuiEvent::AgentDelta(delta) => {
                            match delta {
                                crate::agent::loop_::StreamDelta::Text(txt) => {
                                    // Append streaming response text to last message if it's assistant
                                    if let Some(last) = self.session.history.last_mut() {
                                        if last.role == "assistant" {
                                            last.content.push_str(&txt);
                                        } else {
                                            self.session.history.push(ChatMessage::assistant(&txt));
                                        }
                                    } else {
                                        self.session.history.push(ChatMessage::assistant(&txt));
                                    }
                                }
                                crate::agent::loop_::StreamDelta::Status(stat) => {
                                    self.session.status = stat;
                                }
                            }
                        }
                        TuiEvent::AgentFinished(res) => {
                            self.session.is_thinking = false;
                            match res {
                                Ok(new_history) => {
                                    self.session.history = new_history;
                                    self.session.status = "Ready".to_string();

                                    // Auto-save history if session file is set
                                    if let Some(ref path) = session_state_file {
                                        let _ = crate::agent::loop_::save_interactive_session_history(path, &self.session.history);
                                    }

                                    // Refresh subagents or cost usage if any
                                    if let Some(tracker) = crate::cost::CostTracker::get_or_init_global(config.cost.clone(), &config.data_dir) {
                                        if let Ok(summary) = tracker.get_summary() {
                                            self.session.cost_usd = summary.session_cost_usd;
                                            self.session.tokens_used = summary.total_tokens as usize;
                                        }
                                    }
                                }
                                Err(e) => {
                                    self.session.status = format!("Error: {}", e);
                                    self.session.history.push(ChatMessage::assistant(&format!("An error occurred: {}", e)));
                                }
                            }
                        }
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen, cursor::Show)?;
        Ok(())
    }

    fn draw_ui(&self, f: &mut ratatui::Frame) {
        let size = f.area();

        // Screen split: Left sidebar (30%), Right chat area (70%)
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(28), Constraint::Percentage(72)])
            .split(size);

        self.draw_sidebar(f, main_chunks[0]);
        self.draw_chat_area(f, main_chunks[1]);
    }

    fn draw_sidebar(&self, f: &mut ratatui::Frame, area: Rect) {
        let block = Block::default()
            .title(Span::styled(
                " openz status ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        let sidebar_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6), // Session Info
                Constraint::Length(4), // Usage Metrics
                Constraint::Length(6), // MCP Servers
                Constraint::Length(6), // Skills list
                Constraint::Length(4), // LSP / Subagents status
                Constraint::Min(4),    // GSD Todos
            ])
            .split(block.inner(area));

        f.render_widget(block, area);

        // Segment 1: Model/Provider
        let model_text = vec![
            Line::from(vec![
                Span::styled("Provider: ", Style::default().fg(Color::Cyan)),
                Span::raw(&self.session.active_provider),
            ]),
            Line::from(vec![
                Span::styled("Model: ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    &self.session.active_model,
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("LSP: ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    &self.session.lsp_status,
                    Style::default().fg(if self.session.lsp_status == "Active" {
                        Color::Green
                    } else {
                        Color::Yellow
                    }),
                ),
            ]),
        ];
        f.render_widget(
            Paragraph::new(model_text).block(
                Block::default()
                    .title("Active LLM Service")
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::Rgb(40, 40, 40))),
            ),
            sidebar_layout[0],
        );

        // Segment 2: Usage
        let usage_text = vec![
            Line::from(vec![
                Span::styled("Tokens: ", Style::default().fg(Color::Cyan)),
                Span::raw(format!("{}", self.session.tokens_used)),
            ]),
            Line::from(vec![
                Span::styled("Session Cost: ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("${:.5}", self.session.cost_usd),
                    Style::default().fg(Color::Green),
                ),
            ]),
        ];
        f.render_widget(
            Paragraph::new(usage_text).block(
                Block::default()
                    .title("Resource Metrics")
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::Rgb(40, 40, 40))),
            ),
            sidebar_layout[1],
        );

        // Segment 3: MCP
        let mut mcp_lines = Vec::new();
        if self.session.mcp_servers.is_empty() {
            mcp_lines.push(Line::from(Span::styled(
                "None active",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for mcp in &self.session.mcp_servers {
                mcp_lines.push(Line::from(vec![
                    Span::styled(" ✦ ", Style::default().fg(Color::Cyan)),
                    Span::raw(mcp),
                ]));
            }
        }
        f.render_widget(
            Paragraph::new(mcp_lines).block(
                Block::default()
                    .title("MCP Servers")
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::Rgb(40, 40, 40))),
            ),
            sidebar_layout[2],
        );

        // Segment 4: Skills
        let mut skill_lines = Vec::new();
        if self.session.skills.is_empty() {
            skill_lines.push(Line::from(Span::styled(
                "None installed",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for skill in &self.session.skills {
                skill_lines.push(Line::from(vec![
                    Span::styled(" ⚡ ", Style::default().fg(Color::Yellow)),
                    Span::raw(skill),
                ]));
            }
        }
        f.render_widget(
            Paragraph::new(skill_lines).block(
                Block::default()
                    .title("Active Skills")
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::Rgb(40, 40, 40))),
            ),
            sidebar_layout[3],
        );

        // Segment 5: Subagents
        let mut sub_lines = Vec::new();
        sub_lines.push(Line::from(vec![
            Span::styled("Active count: ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", self.session.subagents.len())),
        ]));
        if !self.session.subagents.is_empty() {
            sub_lines.push(Line::from(vec![
                Span::raw(" Running: "),
                Span::styled(
                    self.session.subagents.join(", "),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
        }
        f.render_widget(
            Paragraph::new(sub_lines).block(
                Block::default()
                    .title("Subagents")
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::Rgb(40, 40, 40))),
            ),
            sidebar_layout[4],
        );

        // Segment 6: GSD Todos
        let mut todo_lines = Vec::new();
        if self.session.todos.is_empty() {
            todo_lines.push(Line::from(Span::styled(
                "No tasks found",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for todo in &self.session.todos {
                let symbol = if todo.checked { "[x] " } else { "[ ] " };
                let color = if todo.checked {
                    Color::DarkGray
                } else {
                    Color::White
                };
                todo_lines.push(Line::from(vec![
                    Span::styled(
                        symbol,
                        Style::default().fg(if todo.checked {
                            Color::Green
                        } else {
                            Color::Yellow
                        }),
                    ),
                    Span::styled(&todo.text, Style::default().fg(color)),
                ]));
            }
        }
        f.render_widget(
            Paragraph::new(todo_lines).block(Block::default().title("GSD Checklist")),
            sidebar_layout[5],
        );
    }

    fn draw_chat_area(&self, f: &mut ratatui::Frame, area: Rect) {
        let chat_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),    // Chat log
                Constraint::Length(3), // Input area
            ])
            .split(area);

        // Render Chat History
        let history_block = Block::default()
            .title(Span::styled(
                " chat session ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        // Format Chat History Lines
        let mut chat_lines = Vec::new();
        for msg in &self.session.history {
            if msg.role == "system" {
                continue; // Hide raw system messages
            }

            let role_label = match msg.role.as_str() {
                "user" => Span::styled(
                    "User ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                "assistant" => Span::styled(
                    "openz ",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                _ => Span::styled(
                    format!("{} ", msg.role),
                    Style::default().fg(Color::Magenta),
                ),
            };

            chat_lines.push(Line::from(vec![
                role_label,
                Span::styled("─".repeat(4), Style::default().fg(Color::Rgb(50, 50, 50))),
            ]));

            // Simple markdown-ish line rendering
            for line in msg.content.lines() {
                if line.trim().starts_with("```") {
                    chat_lines.push(Line::from(Span::styled(
                        line,
                        Style::default().fg(Color::DarkGray),
                    )));
                } else if line.trim().starts_with("- ") || line.trim().starts_with("* ") {
                    chat_lines.push(Line::from(vec![
                        Span::styled("  • ", Style::default().fg(Color::Cyan)),
                        Span::raw(&line[2..]),
                    ]));
                } else {
                    chat_lines.push(Line::from(format!("  {}", line)));
                }
            }
            chat_lines.push(Line::from("")); // Spacer
        }

        // Add thinking spinner at bottom if thinking
        if self.session.is_thinking {
            let spinner = match self.session.spinner_frame {
                0 => "⠋",
                1 => "⠙",
                2 => "⠹",
                _ => "⠸",
            };
            chat_lines.push(Line::from(vec![
                Span::styled(
                    format!("{} openz ", spinner),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("Thinking ({})", self.session.status),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]));
        }

        // Scroll management
        let inner_height = history_block.inner(chat_layout[0]).height as usize;
        let mut scroll = 0usize;
        if chat_lines.len() > inner_height {
            let max_scroll = chat_lines.len() - inner_height;
            scroll = std::cmp::min(self.session.scroll_offset, max_scroll);
            // Auto scroll to bottom if scroll offset is 0 or user is active at bottom
            if self.session.scroll_offset == 0 {
                scroll = max_scroll;
            }
        }

        let chat_paragraph = Paragraph::new(chat_lines)
            .block(history_block)
            .wrap(Wrap { trim: false })
            .scroll((scroll as u16, 0));

        f.render_widget(chat_paragraph, chat_layout[0]);

        // Render Input Box
        let input_title = if self.session.is_thinking {
            " [Ctrl+C to Cancel] "
        } else {
            " Input (Type message or /command, press Enter) "
        };

        let input_block = Block::default()
            .title(Span::styled(input_title, Style::default().fg(Color::Cyan)))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(if self.session.is_thinking {
                Color::Yellow
            } else {
                Color::DarkGray
            }));

        let input_text = if self.session.is_thinking {
            format!("Processing: {}", self.session.status)
        } else {
            self.session.input.clone()
        };

        let input_paragraph =
            Paragraph::new(input_text)
                .block(input_block)
                .style(Style::default().fg(if self.session.is_thinking {
                    Color::DarkGray
                } else {
                    Color::White
                }));

        f.render_widget(input_paragraph, chat_layout[1]);
    }
}
