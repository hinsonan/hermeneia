use clap::Parser;
use hermeneia_lib::transcribe::{
    transcribe_audio_with_reporter, ModelValidator, TranscribeParams,
    TranscriptionTask, ValidationResult, WhisperModel
};
use tracing::info;

mod progress;
use progress::TranscriptionProgress;

#[derive(Parser, Debug)]
#[command(name = "transcribe")]
#[command(about = "Transcribe audio files using Whisper")]
struct Args {
    /// Input audio file
    #[arg(short, long)]
    input: String,

    /// Output text file (optional, prints to stdout if not specified)
    #[arg(short, long)]
    output: Option<String>,

    /// Whisper model size
    #[arg(short, long, default_value = "tiny")]
    model: String,

    /// Task type: transcribe or translate
    #[arg(short, long, default_value = "transcribe")]
    task: String,

    /// Language code (e.g., "en", "es"), auto-detect if not specified
    #[arg(short, long)]
    language: Option<String>,

    /// Include timestamps
    #[arg(long)]
    timestamps: bool,

    /// Force CPU (disable GPU)
    #[arg(long)]
    cpu: bool,

    /// Output format: text, json, srt
    #[arg(short, long, default_value = "text")]
    format: String,

    /// Check system compatibility without transcribing
    #[arg(long)]
    check_only: bool,

    /// Treat warnings as errors
    #[arg(long)]
    strict: bool,

    /// Force execution despite warnings/errors
    #[arg(long)]
    force: bool,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Parse model
    let model = match args.model.to_lowercase().as_str() {
        "tiny" => WhisperModel::Tiny,
        "tiny.en" => WhisperModel::TinyEn,
        "base" => WhisperModel::Base,
        "base.en" => WhisperModel::BaseEn,
        "small" => WhisperModel::Small,
        "small.en" => WhisperModel::SmallEn,
        "medium" => WhisperModel::Medium,
        "medium.en" => WhisperModel::MediumEn,
        "large" => WhisperModel::Large,
        "large-v2" => WhisperModel::LargeV2,
        "large-v3" => WhisperModel::LargeV3,
        _ => anyhow::bail!("Invalid model: {}", args.model),
    };

    // Parse task
    let task = match args.task.to_lowercase().as_str() {
        "transcribe" => TranscriptionTask::Transcribe,
        "translate" => TranscriptionTask::Translate,
        _ => anyhow::bail!("Invalid task: {}", args.task),
    };

    info!("Transcribing: {}", args.input);
    info!("Model: {:?}, Task: {:?}", model, task);

    // Validate model compatibility with system
    let validator = ModelValidator::new()
        .map_err(|e| anyhow::anyhow!("Failed to detect system capabilities: {}", e))?;

    let validation_result = validator.validate_model(model, args.cpu);

    match &validation_result {
        ValidationResult::Error(msg) => {
            if !args.force {
                eprintln!("❌ Cannot run model: {}", msg);
                if let Some(recommended) = Some(validator.recommend_model()) {
                    eprintln!("💡 Recommended model: {:?}", recommended);
                }
                eprintln!("\nUse --force to attempt anyway (not recommended)");
                std::process::exit(1);
            } else {
                eprintln!("⚠️  Forcing despite error: {}", msg);
            }
        }
        ValidationResult::Warning(warnings) => {
            if args.strict && !args.force {
                eprintln!("⚠️  System compatibility warnings:");
                for w in warnings {
                    eprintln!("  - {}", w);
                }
                if let Some(recommended) = Some(validator.recommend_model()) {
                    eprintln!("💡 Recommended model: {:?}", recommended);
                }
                eprintln!("\nUse --force to proceed anyway");
                std::process::exit(1);
            } else {
                for w in warnings {
                    eprintln!("⚠️  {}", w);
                }
            }
        }
        ValidationResult::Ok => {}
    }

    // Handle --check-only flag
    if args.check_only {
        println!("\n=== System Capabilities ===");
        let caps = validator.capabilities();
        println!("RAM: {:.1}GB total, {:.1}GB available", caps.total_ram_gb, caps.available_ram_gb);
        if let Some(gpu) = &caps.gpu_info {
            println!("GPU: {:?}", gpu.device_type);
            if let Some(vram_total) = gpu.vram_total_gb {
                println!("VRAM: {:.1}GB total", vram_total);
            }
            if let Some(vram_avail) = gpu.vram_available_gb {
                println!("      {:.1}GB available", vram_avail);
            }
            if let Some((major, minor)) = gpu.compute_capability {
                println!("Compute Capability: {}.{}", major, minor);
            }
        } else {
            println!("GPU: None detected (will use CPU)");
        }

        println!("\n=== Model Validation ===");
        println!("Model: {:?}", model);
        println!("Status: {:?}", match validation_result {
            ValidationResult::Ok => "✅ OK",
            ValidationResult::Warning(_) => "⚠️  WARNING",
            ValidationResult::Error(_) => "❌ ERROR",
        });

        println!("\n=== Recommended Model ===");
        println!("{:?}", validator.recommend_model());

        return Ok(());
    }

    let params = TranscribeParams {
        model,
        task,
        language: args.language,
        timestamps: args.timestamps,
        force_cpu: args.cpu,
        use_quantized: false,
    };

    let progress = TranscriptionProgress::new();

    let result = transcribe_audio_with_reporter(&args.input, params, &progress)?;

    println!();
    info!(
        "Transcription complete: {:.2}s audio, {:.2}s processing",
        result.duration, result.inference_time
    );

    // Output result
    match args.format.to_lowercase().as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&result)?;
            if let Some(output_path) = args.output {
                std::fs::write(output_path, json)?;
            } else {
                println!("{}", json);
            }
        }
        "srt" => {
            let srt = format_as_srt(&result);
            if let Some(output_path) = args.output {
                std::fs::write(output_path, srt)?;
            } else {
                println!("{}", srt);
            }
        }
        "text" | _ => {
            if let Some(output_path) = args.output {
                std::fs::write(output_path, &result.text)?;
            } else {
                println!("{}", result.text);
            }
        }
    }

    Ok(())
}

fn format_as_srt(result: &hermeneia_lib::transcribe::TranscriptResult) -> String {
    result
        .segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            format!(
                "{}\n{} --> {}\n{}\n",
                i + 1,
                format_timestamp(seg.start.unwrap_or(0.0)),
                format_timestamp(seg.end.unwrap_or(0.0)),
                seg.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0).floor() as u32;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
    let secs = (seconds % 60.0).floor() as u32;
    let millis = ((seconds % 1.0) * 1000.0).floor() as u32;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}
