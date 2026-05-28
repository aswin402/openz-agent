use crate::schema::SshRuntimeConfig;
use std::path::{Path, PathBuf};
use zeroclaw_api::runtime_traits::RuntimeAdapter;

/// SSH remote execution runtime adapter.
#[derive(Debug, Clone)]
pub struct SshRuntime {
    config: SshRuntimeConfig,
}

impl SshRuntime {
    pub fn new(config: SshRuntimeConfig) -> Self {
        Self { config }
    }
}

impl RuntimeAdapter for SshRuntime {
    fn name(&self) -> &str {
        "ssh"
    }

    fn has_shell_access(&self) -> bool {
        true
    }

    fn has_filesystem_access(&self) -> bool {
        true
    }

    fn storage_path(&self) -> PathBuf {
        let home = directories::UserDirs::new().map(|u| u.home_dir().to_path_buf());
        if let Some(h) = home {
            let openz = h.join(".openz");
            let zeroclaw = h.join(".zeroclaw");
            if openz.exists() {
                openz
            } else if zeroclaw.exists() {
                zeroclaw
            } else {
                openz
            }
        } else {
            let openz = PathBuf::from(".openz");
            let zeroclaw = PathBuf::from(".zeroclaw");
            if openz.exists() {
                openz
            } else if zeroclaw.exists() {
                zeroclaw
            } else {
                openz
            }
        }
    }

    fn supports_long_running(&self) -> bool {
        false
    }

    fn build_shell_command(
        &self,
        command: &str,
        _workspace_dir: &Path,
    ) -> anyhow::Result<tokio::process::Command> {
        let mut process = tokio::process::Command::new("ssh");

        // Basic settings for SSH stability and security
        process.arg("-p").arg(self.config.port.to_string());
        process.arg("-o").arg("BatchMode=yes");
        process.arg("-o").arg("StrictHostKeyChecking=accept-new");

        if let Some(ref key) = self.config.key_path {
            process.arg("-i").arg(key);
        }

        // Add any configured extra CLI arguments
        for arg in &self.config.extra_args {
            process.arg(arg);
        }

        // Target: username@host
        let target = format!("{}@{}", self.config.username, self.config.host);
        process.arg(target);

        // Command to run remotely: cd to workspace if configured, then execute command
        let remote_cmd = if let Some(ref remote_dir) = self.config.workspace_dir {
            format!("cd {} && {}", remote_dir, command)
        } else {
            command.to_string()
        };

        process.arg(remote_cmd);

        Ok(process)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SshRuntimeConfig;

    #[test]
    fn test_ssh_runtime_name() {
        let cfg = SshRuntimeConfig::default();
        let runtime = SshRuntime::new(cfg);
        assert_eq!(runtime.name(), "ssh");
        assert!(runtime.has_shell_access());
        assert!(runtime.has_filesystem_access());
        assert!(!runtime.supports_long_running());
    }

    #[test]
    fn test_build_shell_command_basic() {
        let cfg = SshRuntimeConfig {
            host: "1.2.3.4".to_string(),
            port: 2222,
            username: "testuser".to_string(),
            key_path: None,
            workspace_dir: None,
            extra_args: vec!["-v".to_string()],
        };
        let runtime = SshRuntime::new(cfg);
        let cwd = std::env::temp_dir();
        let cmd = runtime.build_shell_command("echo hello", &cwd).unwrap();

        let debug = format!("{cmd:?}");
        assert!(debug.contains("ssh"));
        assert!(debug.contains("-p"));
        assert!(debug.contains("2222"));
        assert!(debug.contains("testuser@1.2.3.4"));
        assert!(debug.contains("-o"));
        assert!(debug.contains("BatchMode=yes"));
        assert!(debug.contains("-v"));
        assert!(debug.contains("echo hello"));
    }

    #[test]
    fn test_build_shell_command_with_key_and_workspace() {
        let cfg = SshRuntimeConfig {
            host: "remote.host".to_string(),
            port: 22,
            username: "root".to_string(),
            key_path: Some("/home/user/.ssh/id_ed25519".to_string()),
            workspace_dir: Some("/opt/workspace".to_string()),
            extra_args: Vec::new(),
        };
        let runtime = SshRuntime::new(cfg);
        let cwd = std::env::temp_dir();
        let cmd = runtime.build_shell_command("ls -la", &cwd).unwrap();

        let debug = format!("{cmd:?}");
        assert!(debug.contains("ssh"));
        assert!(debug.contains("-i"));
        assert!(debug.contains("/home/user/.ssh/id_ed25519"));
        assert!(debug.contains("root@remote.host"));
        assert!(debug.contains("cd /opt/workspace && ls -la"));
    }
}
