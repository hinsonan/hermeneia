use clap::Parser;
use hermeneia_lib::speaker::{
    diarize_audio_with_progress, DiarizeParams, DiarizationResult, SpeakerDevice, SpeakerModel,
    SpeakerModelManager,
};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "speaker")]
#[command(about = "Speaker diarization for audio files")]
struct Args {
    /// Input audio file to diarize
    #[arg(short, long)]
    input: Option<String>,

    /// Output file path (stdout if not specified)
    #[arg(short, long)]
    output: Option<String>,

    /// Model bundle: english, multilingual
    #[arg(short, long, default_value = "english")]
    model: String,

    /// Expected number of speakers (auto-detect if not specified)
    #[arg(long)]
    num_speakers: Option<i32>,

    /// Clustering threshold (0.0–1.0, default 0.5; lower = more speakers)
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Inference device: cpu, cuda, coreml
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Output format: text, json
    #[arg(short, long, default_value = "text")]
    format: String,

    /// Assign names to speakers by position or key=value pairs.
    /// Examples:
    ///   --names "Alice,Bob"       (Speaker 0=Alice, Speaker 1=Bob)
    ///   --names "0=Alice,1=Bob"   (explicit key=value)
    #[arg(long)]
    names: Option<String>,

    /// List available model bundles with cache status, then exit
    #[arg(long)]
    list_models: bool,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if args.list_models {
        print_model_list();
        return Ok(());
    }

    let input = args
        .input
        .ok_or_else(|| anyhow::anyhow!("--input is required unless --list-models is used"))?;

    let model = parse_model(&args.model)?;
    let device = parse_device(&args.device)?;

    tracing::info!("Diarizing: {}", input);
    tracing::info!(
        "Model: {}, Device: {}, Threshold: {}",
        model.display_name(),
        device.provider_string(),
        args.threshold
    );
    if let Some(n) = args.num_speakers {
        tracing::info!("Expected speakers: {}", n);
    }

    let params = DiarizeParams {
        model,
        num_speakers: args.num_speakers,
        threshold: args.threshold,
        device,
    };

    let cancel = Arc::new(AtomicBool::new(false));

    // Progress callback that prints to stderr
    let progress_cb: hermeneia_lib::speaker::DiarizeProgressCallback =
        Box::new(|processed, total| {
            if total > 0 {
                let pct = processed * 100 / total;
                eprint!("\r  Progress: {}% ({}/{})", pct, processed, total);
                let _ = std::io::stderr().flush();
            }
        });

    let result = diarize_audio_with_progress(&input, params, Some(progress_cb), Some(cancel))?;

    eprintln!(); // newline after progress

    let names = args
        .names
        .as_deref()
        .map(parse_speaker_names)
        .unwrap_or_default();

    match args.format.to_lowercase().as_str() {
        "json" => {
            let json = format_as_json(&result, &names)?;
            write_output(json, args.output.as_deref())?;
        }
        "text" | _ => {
            let text = format_as_text(&result, &names);
            write_output(text, args.output.as_deref())?;
        }
    }

    Ok(())
}

fn parse_model(s: &str) -> anyhow::Result<SpeakerModel> {
    match s.to_lowercase().as_str() {
        "english" => Ok(SpeakerModel::English),
        "multilingual" => Ok(SpeakerModel::Multilingual),
        _ => anyhow::bail!(
            "Invalid model '{}'. Use: english, multilingual",
            s
        ),
    }
}

fn parse_device(s: &str) -> anyhow::Result<SpeakerDevice> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(SpeakerDevice::Cpu),
        "cuda" => Ok(SpeakerDevice::Cuda),
        "coreml" => Ok(SpeakerDevice::CoreMl),
        _ => anyhow::bail!(
            "Invalid device '{}'. Use: cpu, cuda, coreml",
            s
        ),
    }
}

fn print_model_list() {
    let models = [SpeakerModel::English, SpeakerModel::Multilingual];
    println!("Available speaker diarization model bundles:\n");
    for model in &models {
        let cached = SpeakerModelManager::is_cached(model);
        let status = if cached { "✓ cached" } else { "not downloaded" };
        println!(
            "  {:15} {:.1} MB   [{}]",
            model.cli_key(),
            model.approx_size_mb(),
            status
        );
        println!("             {}", model.display_name());
        let (seg_repo, seg_file) = model.segmentation_source();
        let (emb_repo, emb_file) = model.embedding_source();
        println!("             Segmentation: {}/{}", seg_repo, seg_file);
        println!("             Embedding:    {}/{}", emb_repo, emb_file);
        println!();
    }
}

/// Parse --names string into a HashMap<speaker_id, name>.
/// Supports two formats:
///   positional: "Alice,Bob"     → {0: "Alice", 1: "Bob"}
///   key=value:  "0=Alice,1=Bob" → {0: "Alice", 1: "Bob"}
fn parse_speaker_names(names_str: &str) -> HashMap<i32, String> {
    let mut map = HashMap::new();
    let parts: Vec<&str> = names_str.split(',').collect();
    let is_kv = parts.iter().any(|p| p.contains('='));
    if is_kv {
        for part in parts {
            if let Some((k, v)) = part.split_once('=') {
                if let Ok(id) = k.trim().parse::<i32>() {
                    let name = v.trim().to_string();
                    if !name.is_empty() {
                        map.insert(id, name);
                    }
                }
            }
        }
    } else {
        for (i, name) in parts.iter().enumerate() {
            let name = name.trim().to_string();
            if !name.is_empty() {
                map.insert(i as i32, name);
            }
        }
    }
    map
}

fn speaker_label(id: i32, names: &HashMap<i32, String>) -> String {
    names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("Speaker {}", id))
}

fn format_as_text(result: &DiarizationResult, names: &HashMap<i32, String>) -> String {
    let mut out = String::new();
    for seg in &result.segments {
        let start_min = (seg.start / 60.0) as u32;
        let start_sec = seg.start % 60.0;
        let end_min = (seg.end / 60.0) as u32;
        let end_sec = seg.end % 60.0;
        out.push_str(&format!(
            "[{:02}:{:04.1} - {:02}:{:04.1}] {}\n",
            start_min,
            start_sec,
            end_min,
            end_sec,
            speaker_label(seg.speaker, names),
        ));
    }
    out.push_str(&format!(
        "\nDetected {} speaker(s) in {:.1}s audio (processed in {:.2}s using {} on {})",
        result.num_speakers,
        result.audio_duration,
        result.inference_time,
        result.model,
        result.device,
    ));
    out
}

fn format_as_json(
    result: &DiarizationResult,
    names: &HashMap<i32, String>,
) -> anyhow::Result<String> {
    let segments: Vec<serde_json::Value> = result
        .segments
        .iter()
        .map(|seg| {
            serde_json::json!({
                "speaker": seg.speaker,
                "name": speaker_label(seg.speaker, names),
                "start": seg.start,
                "end": seg.end,
            })
        })
        .collect();

    // Build speaker_names map with string keys for JSON compatibility
    let speaker_names: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .map(|(id, name)| (id.to_string(), serde_json::Value::String(name.clone())))
        .collect();

    let output = serde_json::json!({
        "segments": segments,
        "speaker_names": speaker_names,
        "num_speakers": result.num_speakers,
        "audio_duration": result.audio_duration,
        "inference_time": result.inference_time,
        "model": result.model,
        "device": result.device,
    });

    Ok(serde_json::to_string_pretty(&output)?)
}

fn write_output(content: String, path: Option<&str>) -> anyhow::Result<()> {
    match path {
        Some(p) => {
            std::fs::write(p, content)?;
            tracing::info!("Output written to: {}", p);
        }
        None => println!("{}", content),
    }
    Ok(())
}
