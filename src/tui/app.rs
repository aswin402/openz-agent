use std::path::PathBuf;
use tokio::sync::mpsc;
use zeroclaw_config::schema::Config;
use zeroclaw_runtime::agent::loop_::AgentRunOverrides;
use zeroclaw_runtime::agent::tui_events::RuntimeEvent;

pub struct TuiApp {
    config: Config,
    agent_alias: String,
    session_state_file: Option<PathBuf>,
    final_temperature: Option<f64>,
    tui_sender: mpsc::Sender<RuntimeEvent>,
    _tui_receiver: mpsc::Receiver<RuntimeEvent>,
}

impl TuiApp {
    pub fn new(
        config: Config,
        agent_alias: String,
        session_state_file: Option<PathBuf>,
        _system_prompt: String,
        final_temperature: Option<f64>,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::channel(100);
        Ok(Self {
            config,
            agent_alias,
            session_state_file,
            final_temperature,
            tui_sender: tx,
            _tui_receiver: rx,
        })
    }

    pub async fn run_loop(self) -> anyhow::Result<()> {
        let tui_sender = self.tui_sender.clone();
        let mut overrides = AgentRunOverrides::default();
        overrides.tui_sender = Some(tui_sender);

        let res = Box::pin(zeroclaw_runtime::agent::run(
            self.config,
            &self.agent_alias,
            None, // message
            None, // provider override
            None, // model override
            self.final_temperature,
            Vec::new(),
            true, // interactive
            self.session_state_file,
            None, // allowed_tools
            overrides,
        ))
        .await;

        res.map(|_| ())
    }
}
