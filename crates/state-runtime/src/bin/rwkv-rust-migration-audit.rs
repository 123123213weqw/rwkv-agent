use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use rwkv_state_runtime::sha256_file;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct FileRecord {
    path: String,
    extension: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct MigrationAudit {
    schema_version: String,
    rust_files: usize,
    rust_bytes: u64,
    legacy_by_extension: BTreeMap<String, usize>,
    legacy_runtime_files: Vec<FileRecord>,
    migration_complete: bool,
    policy: String,
}

fn excluded(relative: &Path) -> bool {
    let text = relative.to_string_lossy();
    [
        ".git/",
        "target/",
        ".venv/",
        "node_modules/",
        "bench/baselines/",
        "bench/artifacts/",
        "evidence/",
        "oracle-validation-pr1483/",
    ]
    .iter()
    .any(|prefix| text == prefix.trim_end_matches('/') || text.starts_with(prefix))
}

fn runtime_scope(relative: &Path) -> bool {
    matches!(
        relative
            .components()
            .next()
            .and_then(|value| value.as_os_str().to_str()),
        Some("src" | "bench" | "benchmarks" | "scripts" | "demos" | "tests" | "web")
    )
}

fn collect(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let relative = path.strip_prefix(root)?;
            if excluded(relative) {
                continue;
            }
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let root = arguments
        .windows(2)
        .find(|pair| pair[0] == "--root")
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or(env::current_dir()?);
    let output = arguments
        .windows(2)
        .find(|pair| pair[0] == "--output")
        .map(|pair| PathBuf::from(&pair[1]));
    let files = collect(&root)?;
    let mut rust_files = 0;
    let mut rust_bytes = 0_u64;
    let mut legacy_by_extension = BTreeMap::new();
    let mut legacy_runtime_files = Vec::new();
    for path in files {
        let relative = path.strip_prefix(&root)?;
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let bytes = fs::metadata(&path)?.len();
        if extension == "rs" {
            rust_files += 1;
            rust_bytes = rust_bytes.saturating_add(bytes);
            continue;
        }
        if matches!(extension.as_str(), "py" | "js" | "ts" | "sh") {
            *legacy_by_extension.entry(extension.clone()).or_insert(0) += 1;
            if runtime_scope(relative) {
                legacy_runtime_files.push(FileRecord {
                    path: relative.to_string_lossy().to_string(),
                    extension,
                    bytes,
                    sha256: sha256_file(&path)?,
                });
            }
        }
    }
    let audit = MigrationAudit {
        schema_version: "rust-migration-audit.v1".into(),
        rust_files,
        rust_bytes,
        legacy_by_extension,
        migration_complete: legacy_runtime_files.is_empty(),
        legacy_runtime_files,
        policy: "No new Python/JavaScript/Shell runtime code; replace frozen legacy one module at a time after parity gates."
            .into(),
    };
    if let Some(path) = output {
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), &audit)?;
    }
    println!("{}", serde_json::to_string(&audit)?);
    Ok(())
}
