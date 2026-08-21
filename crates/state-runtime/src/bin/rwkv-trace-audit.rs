use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use rwkv_state_runtime::{duplicate_report, load_trace_jsonl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.is_empty() || arguments.iter().any(|value| value == "--help") {
        println!("rwkv-trace-audit [--output PATH] TRACE.jsonl TRACE.jsonl [...]");
        return Ok(());
    }
    let mut output = None;
    let mut paths = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--output" {
            index += 1;
            output = Some(PathBuf::from(
                arguments.get(index).ok_or("--output requires a path")?,
            ));
        } else {
            paths.push(PathBuf::from(&arguments[index]));
        }
        index += 1;
    }
    if paths.len() < 2 {
        return Err("at least two traces are required".into());
    }
    let traces = paths
        .iter()
        .map(|path| load_trace_jsonl(path))
        .collect::<Result<Vec<_>, _>>()?;
    let references = traces.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let report = duplicate_report(&references);
    if let Some(path) = output {
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), &report)?;
    }
    println!("{}", serde_json::to_string(&report)?);
    if !report.clean {
        std::process::exit(2);
    }
    Ok(())
}
