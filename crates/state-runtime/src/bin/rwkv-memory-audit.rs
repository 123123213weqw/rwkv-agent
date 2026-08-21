use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
struct Args {
    gpu_index: u32,
    qwen_path: PathBuf,
    rwkv_path: PathBuf,
    reserve_bytes: u64,
    output: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct GpuInfo {
    host: String,
    index: u32,
    name: String,
    driver: String,
    total_bytes: u64,
    free_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ArmAudit {
    arm_id: String,
    model_path: String,
    dtype: String,
    quantization: String,
    provider_mode: String,
    weight_file_bytes: u64,
    reserve_bytes: u64,
    minimum_required_bytes: u64,
    safe_to_attempt_load: bool,
    status: String,
    reason: String,
}

#[derive(Clone, Debug, Serialize)]
struct AuditResult {
    schema_version: String,
    gpu: GpuInfo,
    primary_comparison_same_hardware: bool,
    primary_arms: Vec<ArmAudit>,
    historical_reference: serde_json::Value,
    limitation: String,
}

fn value(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn parse_args() -> Result<Args, String> {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.iter().any(|value| value == "--help") {
        println!(
            "rwkv-memory-audit --qwen-path PATH --rwkv-path PATH \
             [--gpu-index N] [--reserve-gib N] [--output PATH]"
        );
        std::process::exit(0);
    }
    Ok(Args {
        gpu_index: value(&arguments, "--gpu-index")
            .as_deref()
            .unwrap_or("0")
            .parse()
            .map_err(|_| "--gpu-index must be an integer")?,
        qwen_path: PathBuf::from(
            value(&arguments, "--qwen-path").ok_or("--qwen-path is required")?,
        ),
        rwkv_path: PathBuf::from(
            value(&arguments, "--rwkv-path").ok_or("--rwkv-path is required")?,
        ),
        reserve_bytes: value(&arguments, "--reserve-gib")
            .as_deref()
            .unwrap_or("2")
            .parse::<u64>()
            .map_err(|_| "--reserve-gib must be an integer")?
            .saturating_mul(GIB),
        output: value(&arguments, "--output").map(PathBuf::from),
    })
}

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn gpu_info(index: u32) -> Result<GpuInfo, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            &index.to_string(),
        ])
        .output()
        .map_err(|error| format!("nvidia-smi failed to start: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let line = String::from_utf8_lossy(&output.stdout);
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err("nvidia-smi returned an unexpected row".into());
    }
    let mib = 1024_u64 * 1024;
    Ok(GpuInfo {
        host: hostname(),
        index,
        name: fields[0].into(),
        total_bytes: fields[1]
            .parse::<u64>()
            .map_err(|_| "invalid total memory")?
            .saturating_mul(mib),
        free_bytes: fields[2]
            .parse::<u64>()
            .map_err(|_| "invalid free memory")?
            .saturating_mul(mib),
        driver: fields[3].into(),
    })
}

fn weight_bytes(path: &Path) -> Result<u64, String> {
    if path.is_file() {
        return fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|error| format!("{}: {error}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("model path does not exist: {}", path.display()));
    }
    let mut pending = vec![path.to_path_buf()];
    let mut total = 0_u64;
    while let Some(directory) = pending.pop() {
        for entry in
            fs::read_dir(&directory).map_err(|error| format!("{}: {error}", directory.display()))?
        {
            let entry = entry.map_err(|error| error.to_string())?;
            let file_type = entry.file_type().map_err(|error| error.to_string())?;
            if file_type.is_dir() {
                pending.push(entry.path());
                continue;
            }
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            if name.ends_with(".safetensors")
                || name.ends_with(".pth")
                || name.ends_with("pytorch_model.bin")
            {
                total = total
                    .saturating_add(entry.metadata().map_err(|error| error.to_string())?.len());
            }
        }
    }
    if total == 0 {
        return Err(format!("no model weight files under {}", path.display()));
    }
    Ok(total)
}

fn arm(
    arm_id: &str,
    path: &Path,
    dtype: &str,
    provider_mode: &str,
    reserve_bytes: u64,
    gpu: &GpuInfo,
) -> ArmAudit {
    match weight_bytes(path) {
        Ok(weights) => {
            let required = weights.saturating_add(reserve_bytes);
            let safe =
                required <= gpu.free_bytes && required <= gpu.total_bytes.saturating_mul(95) / 100;
            ArmAudit {
                arm_id: arm_id.into(),
                model_path: path.display().to_string(),
                dtype: dtype.into(),
                quantization: "none".into(),
                provider_mode: provider_mode.into(),
                weight_file_bytes: weights,
                reserve_bytes,
                minimum_required_bytes: required,
                safe_to_attempt_load: safe,
                status: if safe { "ready_for_live_probe" } else { "blocked_unsupported" }.into(),
                reason: if safe {
                    "weight bytes plus reserve fit the currently free VRAM; live load is still required"
                } else {
                    "weight bytes plus reserve exceed the safe same-GPU envelope; do not OOM-probe"
                }
                .into(),
            }
        }
        Err(error) => ArmAudit {
            arm_id: arm_id.into(),
            model_path: path.display().to_string(),
            dtype: dtype.into(),
            quantization: "none".into(),
            provider_mode: provider_mode.into(),
            weight_file_bytes: 0,
            reserve_bytes,
            minimum_required_bytes: 0,
            safe_to_attempt_load: false,
            status: "missing_artifact".into(),
            reason: error,
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(std::io::Error::other)?;
    let gpu = gpu_info(args.gpu_index).map_err(std::io::Error::other)?;
    let arms = vec![
        arm(
            "qwen3.5-9b-fp16-primary",
            &args.qwen_path,
            "fp16",
            "qwen_transcript_reprefill",
            args.reserve_bytes,
            &gpu,
        ),
        arm(
            "rwkv-7.2b-fp32io16-primary",
            &args.rwkv_path,
            "fp32io16",
            "rwkv_recurrent",
            args.reserve_bytes,
            &gpu,
        ),
    ];
    let result = AuditResult {
        schema_version: "stateful-memory-audit.v1".into(),
        gpu,
        primary_comparison_same_hardware: arms.iter().all(|arm| arm.safe_to_attempt_load),
        primary_arms: arms,
        historical_reference: serde_json::json!({
            "arm_id":"qwen3.5-9b-nf4-historical",
            "gpu":"RTX 4080 16GB",
            "peak_vram_mib":8429,
            "primary_comparator":false
        }),
        limitation:
            "This is an OOM-prevention lower-bound audit, not a live VRAM or capability result."
                .into(),
    };
    if let Some(path) = args.output {
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), &result)?;
    }
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
