use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
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
    workspace_alias: Option<String>,
    backend: SandboxBackend,
}

#[derive(Clone, Debug)]
enum SandboxBackend {
    Bubblewrap(Option<String>),
    ProotUserNet,
    Invalid(String),
}

impl SandboxBackend {
    fn from_env() -> Self {
        match env::var("RWKV_AGENT_SANDBOX_BACKEND")
            .unwrap_or_else(|_| "bubblewrap".into())
            .trim()
        {
            "bubblewrap" => match bubblewrap_apparmor_profile_from_env() {
                Ok(profile) => Self::Bubblewrap(profile),
                Err(message) => Self::Invalid(message),
            },
            "proot_usernet" => Self::ProotUserNet,
            value => Self::Invalid(value.to_string()),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Bubblewrap(Some(_)) => "bubblewrap_apparmor_userns_no_network_no_unsafe_fallback",
            Self::Bubblewrap(None) => "bubblewrap_no_network_no_unsafe_fallback",
            Self::ProotUserNet => "proot_unprivileged_usernet_no_unsafe_fallback",
            Self::Invalid(_) => "invalid_sandbox_backend",
        }
    }
}

impl SandboxedCommand {
    pub fn new(policy: CommandPolicy) -> Self {
        Self {
            policy,
            workspace_alias: None,
            backend: SandboxBackend::from_env(),
        }
    }

    pub fn available(&self) -> bool {
        let backend_ready = match &self.backend {
            SandboxBackend::Bubblewrap(profile) => {
                executable_exists("bwrap") && (profile.is_none() || executable_exists("aa-exec"))
            }
            SandboxBackend::ProotUserNet => ["proot", "setpriv", "unshare"]
                .into_iter()
                .all(executable_exists),
            SandboxBackend::Invalid(_) => false,
        };
        self.policy.enabled
            && self.policy.workspace.is_some()
            && cfg!(target_os = "linux")
            && backend_ready
    }

    pub fn sandbox_mode(&self) -> &'static str {
        self.backend.label()
    }

    /// Restrict a command executor to one existing directory below the configured
    /// workspace. The selected directory is mounted as `/workspace`, so commands
    /// cannot accidentally read or write sibling jobs and may use task-relative
    /// paths without repeating a host-side prefix.
    pub async fn scoped(&self, relative: &str) -> Result<Self, String> {
        if !self.policy.enabled {
            return Err("run_command is disabled".into());
        }
        let relative = relative.trim();
        let path = Path::new(relative);
        if relative.is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err("working_directory must be a safe relative directory".into());
        }
        let configured = self
            .policy
            .workspace
            .as_ref()
            .ok_or_else(|| "run_command workspace is not configured".to_string())?;
        let root = tokio::fs::canonicalize(configured)
            .await
            .map_err(|error| format!("invalid command workspace: {error}"))?;
        let target = tokio::fs::canonicalize(root.join(path))
            .await
            .map_err(|error| format!("invalid working_directory: {error}"))?;
        if !target.starts_with(&root) {
            return Err("working_directory escapes the configured workspace".into());
        }
        let metadata = tokio::fs::metadata(&target)
            .await
            .map_err(|error| format!("invalid working_directory: {error}"))?;
        if !metadata.is_dir() {
            return Err("working_directory must select a directory".into());
        }
        let mut policy = self.policy.clone();
        policy.workspace = Some(target);
        Ok(Self {
            policy,
            workspace_alias: Some(relative.to_string()),
            backend: self.backend.clone(),
        })
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
        let (command, workspace_alias_rewritten) = self.normalize_workspace_alias(command);
        let (_sandbox_temp, mut process) = match &self.backend {
            SandboxBackend::Bubblewrap(profile) => {
                let probe = bubblewrap_command(profile.as_deref())
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
                (
                    None,
                    bubblewrap_process(&workspace, &command, profile.as_deref()),
                )
            }
            SandboxBackend::ProotUserNet => {
                if !["proot", "setpriv", "unshare"]
                    .into_iter()
                    .all(executable_exists)
                {
                    return Ok(
                        json!({"status":"rejected","message":"proot_usernet prerequisites are unavailable; unsafe fallback is forbidden"}),
                    );
                }
                let temp = SandboxTemp::create()?;
                let process = proot_usernet_process(&workspace, temp.path(), &command);
                (Some(temp), process)
            }
            SandboxBackend::Invalid(value) => {
                return Ok(
                    json!({"status":"rejected","message":format!("invalid sandbox backend: {value}")}),
                );
            }
        };
        process
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
            "workspace_alias_rewritten": workspace_alias_rewritten,
            "standalone_test_runner_rewritten": false,
        }))
    }

    pub async fn workspace_fingerprint(&self) -> Result<u64, String> {
        let configured = self
            .policy
            .workspace
            .as_ref()
            .ok_or_else(|| "run_command workspace is not configured".to_string())?;
        let root = tokio::fs::canonicalize(configured)
            .await
            .map_err(|error| format!("invalid command workspace: {error}"))?;
        tokio::task::spawn_blocking(move || fingerprint_tree(&root))
            .await
            .map_err(|error| format!("workspace fingerprint task failed: {error}"))?
            .map_err(|error| format!("workspace fingerprint failed: {error}"))
    }

    /// Return a bounded, deterministic inventory of files visible to this
    /// scoped workspace. This is controller-supplied orientation, not model
    /// speculation: it lets an agent distinguish existing inputs from target
    /// artifacts that still need to be created without spending its first
    /// turns guessing paths.
    pub async fn workspace_inventory(&self) -> Result<Vec<String>, String> {
        let root = self.workspace_root().await?;
        tokio::task::spawn_blocking(move || collect_workspace_inventory(&root, 256, 12_000))
            .await
            .map_err(|error| format!("workspace inventory task failed: {error}"))?
            .map_err(|error| format!("workspace inventory failed: {error}"))
    }

    pub async fn read_file(&self, path: &str) -> Result<Value, String> {
        let (relative, target) = self.resolve_existing_file(path).await?;
        let metadata = tokio::fs::metadata(&target)
            .await
            .map_err(|error| format!("cannot inspect file: {error}"))?;
        if metadata.len() > self.policy.max_output_bytes as u64 {
            return Ok(json!({
                "status":"rejected",
                "message":"file exceeds the workspace text-tool output limit",
                "path":relative,
            }));
        }
        let bytes = tokio::fs::read(&target)
            .await
            .map_err(|error| format!("cannot read file: {error}"))?;
        let content = String::from_utf8(bytes)
            .map_err(|_| "read_file accepts UTF-8 text files only".to_string())?;
        Ok(json!({
            "status":"ok",
            "path":relative,
            "content":content,
            "stdout":content,
            "bytes":metadata.len(),
        }))
    }

    pub async fn write_file(&self, path: &str, content: &str) -> Result<Value, String> {
        if content.len() > self.policy.max_output_bytes {
            return Ok(json!({
                "status":"rejected",
                "message":"content exceeds the workspace text-tool limit",
            }));
        }
        if content.contains('\0') {
            return Ok(json!({"status":"rejected","message":"content contains NUL"}));
        }
        let relative = self.relative_workspace_path(path)?;
        let root = self.workspace_root().await?;
        let parent_relative = relative.parent().unwrap_or_else(|| Path::new(""));
        let parent = tokio::fs::canonicalize(root.join(parent_relative))
            .await
            .map_err(|error| format!("write_file parent must already exist: {error}"))?;
        if !parent.starts_with(&root) {
            return Err("workspace file path escapes the configured root".into());
        }
        let name = relative
            .file_name()
            .ok_or_else(|| "workspace file path must name a file".to_string())?;
        let target = parent.join(name);
        if let Ok(metadata) = tokio::fs::symlink_metadata(&target).await
            && (metadata.file_type().is_symlink() || !metadata.is_file())
        {
            return Ok(json!({
                "status":"rejected",
                "message":"write_file refuses symlinks and non-files",
                "path":relative,
            }));
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let temporary = parent.join(format!(
            ".rwkv-agent-write-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .await
            .map_err(|error| format!("cannot create atomic workspace file: {error}"))?;
        if let Err(error) = file.write_all(content.as_bytes()).await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(format!("cannot write workspace file: {error}"));
        }
        if let Err(error) = file.sync_all().await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(format!("cannot sync workspace file: {error}"));
        }
        drop(file);
        if let Err(error) = tokio::fs::rename(&temporary, &target).await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(format!("cannot commit workspace file: {error}"));
        }
        Ok(json!({
            "status":"ok",
            "path":relative,
            "bytes_written":content.len(),
        }))
    }

    pub async fn edit_file(
        &self,
        path: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<Value, String> {
        if old_text.is_empty() {
            return Ok(json!({"status":"rejected","message":"old_text must not be empty"}));
        }
        let (relative, target) = self.resolve_existing_file(path).await?;
        let bytes = tokio::fs::read(&target)
            .await
            .map_err(|error| format!("cannot read file for edit: {error}"))?;
        if bytes.len() > self.policy.max_output_bytes {
            return Ok(json!({
                "status":"rejected",
                "message":"file exceeds the workspace text-tool limit",
                "path":relative,
            }));
        }
        let content = String::from_utf8(bytes)
            .map_err(|_| "edit_file accepts UTF-8 text files only".to_string())?;
        let matches = content.match_indices(old_text).count();
        if matches != 1 {
            return Ok(json!({
                "status":"rejected",
                "message":format!("old_text must match exactly once; observed {matches}"),
                "path":relative,
            }));
        }
        let updated = content.replacen(old_text, new_text, 1);
        let result = self.write_file(&relative, &updated).await?;
        if result.get("status").and_then(Value::as_str) != Some("ok") {
            return Ok(result);
        }
        Ok(json!({
            "status":"ok",
            "path":relative,
            "replacements":1,
            "bytes_written":updated.len(),
        }))
    }

    async fn workspace_root(&self) -> Result<PathBuf, String> {
        if !self.policy.enabled {
            return Err("workspace tools are disabled".into());
        }
        let configured = self
            .policy
            .workspace
            .as_ref()
            .ok_or_else(|| "run_command workspace is not configured".to_string())?;
        tokio::fs::canonicalize(configured)
            .await
            .map_err(|error| format!("invalid command workspace: {error}"))
    }

    fn relative_workspace_path(&self, path: &str) -> Result<PathBuf, String> {
        let value = path.trim();
        if value.is_empty() || value.chars().count() > 4096 || value.contains('\0') {
            return Err("workspace file path is invalid".into());
        }
        let value = value.strip_prefix("/workspace/").unwrap_or(value);
        let relative = Path::new(value);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err("workspace file path must be relative and cannot escape".into());
        }
        Ok(relative.to_path_buf())
    }

    async fn resolve_existing_file(&self, path: &str) -> Result<(String, PathBuf), String> {
        let relative = self.relative_workspace_path(path)?;
        let root = self.workspace_root().await?;
        let target = tokio::fs::canonicalize(root.join(&relative))
            .await
            .map_err(|error| format!("workspace file does not exist: {error}"))?;
        if !target.starts_with(&root) {
            return Err("workspace file path escapes the configured root".into());
        }
        let metadata = tokio::fs::metadata(&target)
            .await
            .map_err(|error| format!("cannot inspect workspace file: {error}"))?;
        if !metadata.is_file() {
            return Err("workspace file path must select a regular file".into());
        }
        Ok((relative.to_string_lossy().into_owned(), target))
    }

    fn normalize_workspace_alias(&self, command: &str) -> (String, bool) {
        let Some(alias) = self.workspace_alias.as_deref() else {
            return (command.to_string(), false);
        };
        let absolute = format!("/workspace/{alias}");
        let after_absolute = command.replace(&absolute, "/workspace");
        let normalized = after_absolute.replace(alias, ".");
        let changed = normalized != command;
        (normalized, changed)
    }
}

fn executable_exists(name: &str) -> bool {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .any(|directory| directory.join(name).is_file())
        || ["/usr/local/bin", "/usr/bin", "/bin"]
            .into_iter()
            .any(|directory| Path::new(directory).join(name).is_file())
}

fn bubblewrap_apparmor_profile_from_env() -> Result<Option<String>, String> {
    parse_bubblewrap_apparmor_profile(
        env::var("RWKV_AGENT_BWRAP_APPARMOR_PROFILE")
            .ok()
            .as_deref(),
    )
}

fn parse_bubblewrap_apparmor_profile(value: Option<&str>) -> Result<Option<String>, String> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if value.len() > 128
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/'))
    {
        return Err("invalid RWKV_AGENT_BWRAP_APPARMOR_PROFILE".into());
    }
    Ok(Some(value.to_string()))
}

fn bubblewrap_command(apparmor_profile: Option<&str>) -> Command {
    if let Some(profile) = apparmor_profile {
        let mut process = Command::new("aa-exec");
        process.args(["-p", profile, "--", "bwrap"]);
        process
    } else {
        Command::new("bwrap")
    }
}

fn bubblewrap_process(workspace: &Path, command: &str, apparmor_profile: Option<&str>) -> Command {
    let mut process = bubblewrap_command(apparmor_profile);
    process
        .kill_on_drop(true)
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .args(["--die-with-parent", "--unshare-all", "--new-session"])
        .args(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]);
    for path in ["/usr", "/bin", "/lib", "/lib64", "/etc/alternatives"] {
        if Path::new(path).exists() {
            process.args(["--ro-bind", path, path]);
        }
    }
    process
        .arg("--bind")
        .arg(workspace)
        .arg("/workspace")
        .args(["--chdir", "/workspace", "/bin/sh", "-lc"])
        .arg(command);
    process
}

fn proot_usernet_process(workspace: &Path, temp: &Path, command: &str) -> Command {
    let mut process = Command::new("setpriv");
    process
        .kill_on_drop(true)
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .args(["--reuid=65534", "--regid=65534", "--clear-groups"])
        .args([
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--fork",
            "--kill-child=KILL",
            "proot",
            "-R",
            "/",
            "-b",
        ])
        .arg(format!("{}:/workspace", workspace.display()))
        .args(["-b"])
        .arg(format!("{}:/tmp", temp.display()))
        .args(["-w", "/workspace", "/bin/sh", "-lc"])
        .arg(command);
    process
}

struct SandboxTemp(PathBuf);

impl SandboxTemp {
    fn create() -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = env::temp_dir().join(format!(
            "rwkv-agent-sandbox-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&path)
            .map_err(|error| format!("cannot create sandbox temp directory: {error}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o777))
                .map_err(|error| format!("cannot configure sandbox temp directory: {error}"))?;
        }
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for SandboxTemp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fingerprint_tree(root: &Path) -> std::io::Result<u64> {
    fn ignored(path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                matches!(name, "__pycache__" | ".pytest_cache" | ".mypy_cache")
                    || name.ends_with(".pyc")
                    || name.ends_with(".pyo")
            })
    }

    fn visit(root: &Path, path: &Path, hasher: &mut DefaultHasher) -> std::io::Result<()> {
        let mut entries = std::fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let path = entry.path();
            if ignored(&path) {
                continue;
            }
            path.strip_prefix(root).unwrap_or(&path).hash(hasher);
            let metadata = std::fs::symlink_metadata(&path)?;
            metadata.file_type().is_dir().hash(hasher);
            metadata.file_type().is_file().hash(hasher);
            metadata.file_type().is_symlink().hash(hasher);
            if metadata.is_dir() {
                visit(root, &path, hasher)?;
            } else if metadata.is_file() {
                std::fs::read(&path)?.hash(hasher);
            } else if metadata.file_type().is_symlink() {
                std::fs::read_link(&path)?.hash(hasher);
            }
        }
        Ok(())
    }

    let mut hasher = DefaultHasher::new();
    visit(root, root, &mut hasher)?;
    Ok(hasher.finish())
}

fn collect_workspace_inventory(
    root: &Path,
    max_entries: usize,
    max_chars: usize,
) -> std::io::Result<Vec<String>> {
    fn ignored(path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                matches!(name, "__pycache__" | ".pytest_cache" | ".mypy_cache")
                    || name.ends_with(".pyc")
                    || name.ends_with(".pyo")
                    || name.starts_with(".rwkv-agent-write-")
            })
    }

    let mut pending = vec![root.to_path_buf()];
    let mut rows = Vec::new();
    while let Some(directory) = pending.pop() {
        let mut entries = std::fs::read_dir(&directory)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries.into_iter().rev() {
            let path = entry.path();
            if ignored(&path) {
                continue;
            }
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                pending.push(path);
            } else if metadata.is_file() {
                let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy();
                rows.push(format!("{relative} ({} bytes)", metadata.len()));
            }
        }
    }
    rows.sort();
    let mut used = 0usize;
    let mut bounded = Vec::new();
    for row in rows {
        if bounded.len() >= max_entries || used.saturating_add(row.chars().count()) > max_chars {
            bounded.push("... inventory truncated by controller ...".into());
            break;
        }
        used += row.chars().count();
        bounded.push(row);
    }
    Ok(bounded)
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

    #[test]
    fn apparmor_profile_name_is_strict_and_optional() {
        assert_eq!(parse_bubblewrap_apparmor_profile(None).unwrap(), None);
        assert_eq!(
            parse_bubblewrap_apparmor_profile(Some(" lxc-usernsexec ")).unwrap(),
            Some("lxc-usernsexec".into())
        );
        assert!(parse_bubblewrap_apparmor_profile(Some("bad profile")).is_err());
        assert!(parse_bubblewrap_apparmor_profile(Some("bad&profile")).is_err());
    }

    #[test]
    fn apparmor_bubblewrap_uses_an_explicit_aa_exec_transition() {
        let process =
            bubblewrap_process(Path::new("/tmp/workspace"), "true", Some("lxc-usernsexec"));
        let command = process.as_std();
        assert_eq!(command.get_program(), "aa-exec");
        let args = command
            .get_args()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(&args[..4], ["-p", "lxc-usernsexec", "--", "bwrap"]);
        assert!(args.iter().any(|value| value == "--unshare-all"));
        assert!(args.iter().any(|value| value == "/workspace"));
    }

    #[tokio::test]
    async fn scoped_workspace_rejects_escape_and_selects_existing_directory() {
        let root = tempfile::tempdir().unwrap();
        tokio::fs::create_dir(root.path().join("job"))
            .await
            .unwrap();
        let command = SandboxedCommand::new(CommandPolicy {
            enabled: true,
            workspace: Some(root.path().to_path_buf()),
            ..CommandPolicy::default()
        });

        let scoped = command.scoped("job").await.unwrap();
        assert_eq!(scoped.policy.workspace, Some(root.path().join("job")));
        assert_eq!(
            scoped.normalize_workspace_alias("cd /workspace/job && cat job/input.txt"),
            ("cd /workspace && cat ./input.txt".into(), true)
        );
        assert!(command.scoped("../escape").await.is_err());
        assert!(command.scoped("/absolute").await.is_err());
        assert!(command.scoped("missing").await.is_err());

        let before = scoped.workspace_fingerprint().await.unwrap();
        tokio::fs::write(root.path().join("job/output.txt"), b"changed")
            .await
            .unwrap();
        let after = scoped.workspace_fingerprint().await.unwrap();
        assert_ne!(before, after);

        tokio::fs::create_dir(root.path().join("job/__pycache__"))
            .await
            .unwrap();
        tokio::fs::write(
            root.path().join("job/__pycache__/module.cpython.pyc"),
            b"cache",
        )
        .await
        .unwrap();
        assert_eq!(after, scoped.workspace_fingerprint().await.unwrap());
    }

    #[tokio::test]
    async fn workspace_file_tools_are_confined_atomic_and_exact() {
        let root = tempfile::tempdir().unwrap();
        tokio::fs::create_dir(root.path().join("job"))
            .await
            .unwrap();
        tokio::fs::write(root.path().join("job/input.txt"), "mode=debug\n")
            .await
            .unwrap();
        tokio::fs::create_dir(root.path().join("job/pkg"))
            .await
            .unwrap();
        tokio::fs::write(root.path().join("job/pkg/code.py"), "pass\n")
            .await
            .unwrap();
        let command = SandboxedCommand::new(CommandPolicy {
            enabled: true,
            workspace: Some(root.path().join("job")),
            ..CommandPolicy::default()
        });

        let read = command.read_file("/workspace/input.txt").await.unwrap();
        assert_eq!(read["status"], "ok");
        assert_eq!(read["content"], "mode=debug\n");
        assert_eq!(
            command.workspace_inventory().await.unwrap(),
            vec!["input.txt (11 bytes)", "pkg/code.py (5 bytes)"]
        );

        let edit = command
            .edit_file("input.txt", "mode=debug", "mode=release")
            .await
            .unwrap();
        assert_eq!(edit["status"], "ok");
        assert_eq!(
            tokio::fs::read_to_string(root.path().join("job/input.txt"))
                .await
                .unwrap(),
            "mode=release\n"
        );

        let write = command
            .write_file("result.txt", "status=pass\n")
            .await
            .unwrap();
        assert_eq!(write["status"], "ok");
        assert_eq!(
            tokio::fs::read_to_string(root.path().join("job/result.txt"))
                .await
                .unwrap(),
            "status=pass\n"
        );
        assert!(command.read_file("../escape.txt").await.is_err());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("input.txt", root.path().join("job/link.txt")).unwrap();
            let rejected = command.write_file("link.txt", "unsafe").await.unwrap();
            assert_eq!(rejected["status"], "rejected");
        }
    }
}
