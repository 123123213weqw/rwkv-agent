use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::time::timeout;

#[derive(Clone, Debug)]
pub struct CommandPolicy {
    pub enabled: bool,
    pub workspace: Option<PathBuf>,
    pub timeout: Duration,
    pub max_output_bytes: usize,
}

impl Default for CommandPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            workspace: None,
            timeout: Duration::from_secs(10),
            max_output_bytes: 64 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SandboxedCommand {
    policy: CommandPolicy,
}

impl SandboxedCommand {
    pub fn new(policy: CommandPolicy) -> Self {
        Self { policy }
    }

    pub fn available(&self) -> bool {
        self.policy.enabled && self.policy.workspace.is_some() && cfg!(target_os = "linux")
    }

    pub async fn execute(&self, command: &str) -> Result<Value, String> {
        if !self.policy.enabled {
            return Ok(json!({"status":"rejected","message":"run_command is disabled"}));
        }
        if !cfg!(target_os = "linux") {
            return Ok(
                json!({"status":"rejected","message":"run_command requires Linux bubblewrap isolation"}),
            );
        }
        let workspace = self
            .policy
            .workspace
            .as_ref()
            .ok_or_else(|| "run_command workspace is not configured".to_string())?;
        let workspace = tokio::fs::canonicalize(workspace)
            .await
            .map_err(|e| format!("invalid command workspace: {e}"))?;
        let probe = Command::new("bwrap")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        if !matches!(probe, Ok(status) if status.success()) {
            return Ok(
                json!({"status":"rejected","message":"bubblewrap is unavailable; unsafe fallback is forbidden"}),
            );
        }
        let mut process = Command::new("bwrap");
        process
            .kill_on_drop(true)
            .env_clear()
            .env("PATH", "/usr/local/bin:/usr/bin:/bin")
            .args(["--die-with-parent", "--unshare-all", "--new-session"])
            .args(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]);
        for path in ["/usr", "/bin", "/lib", "/lib64"] {
            if std::path::Path::new(path).exists() {
                process.args(["--ro-bind", path, path]);
            }
        }
        process
            .arg("--bind")
            .arg(&workspace)
            .arg("/workspace")
            .args(["--chdir", "/workspace", "/bin/sh", "-lc"])
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = process
            .spawn()
            .map_err(|e| format!("sandbox launch failed: {e}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "sandbox stdout pipe is unavailable".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "sandbox stderr pipe is unavailable".to_string())?;
        let output_limit = self.policy.max_output_bytes;
        let stdout_reader = tokio::spawn(read_bounded(stdout, output_limit));
        let stderr_reader = tokio::spawn(read_bounded(stderr, output_limit));
        let status = match timeout(self.policy.timeout, child.wait()).await {
            Ok(result) => result.map_err(|e| format!("sandbox wait failed: {e}"))?,
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                let _ = stdout_reader.await;
                let _ = stderr_reader.await;
                return Ok(
                    json!({"status":"timeout","message":"command exceeded its time budget"}),
                );
            }
        };
        let (stdout, stdout_truncated) = stdout_reader
            .await
            .map_err(|e| format!("stdout reader failed: {e}"))?
            .map_err(|e| format!("read sandbox stdout: {e}"))?;
        let (stderr, stderr_truncated) = stderr_reader
            .await
            .map_err(|e| format!("stderr reader failed: {e}"))?
            .map_err(|e| format!("read sandbox stderr: {e}"))?;
        Ok(json!({
            "status": if status.success() {"ok"} else {"error"},
            "exit_code": status.code(),
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_truncated || stderr_truncated,
        }))
    }
}

async fn read_bounded(
    mut reader: impl AsyncRead + Unpin,
    limit: usize,
) -> std::io::Result<(String, bool)> {
    let mut retained = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0u8; 8192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        let available = limit.saturating_sub(retained.len());
        let keep = read.min(available);
        retained.extend_from_slice(&buffer[..keep]);
        truncated |= keep < read;
    }
    Ok((String::from_utf8_lossy(&retained).into_owned(), truncated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn command_is_closed_by_default() {
        let command = SandboxedCommand::new(CommandPolicy::default());
        let result = command.execute("echo unsafe").await.unwrap();
        assert_eq!(result["status"], "rejected");
    }

    #[tokio::test]
    async fn output_budget_is_hard() {
        let (value, truncated) = read_bounded(&b"abcdef"[..], 3).await.unwrap();
        assert_eq!(value, "abc");
        assert!(truncated);
    }
}
