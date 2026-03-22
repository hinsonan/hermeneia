use clap::Parser;
use hermeneia_lib::annotate::{
    annotate_audio_with_reporter, format_as_json, format_as_srt, format_as_text,
    parse_speaker_device, parse_speaker_model, parse_speaker_names, parse_task,
    parse_whisper_model, AnnotateParams, AnnotationPhase, AnnotationProgress,
    AnnotationProgressReporter,
};
use hermeneia_lib::speaker::DiarizeParams;
use hermeneia_lib::transcribe::TranscribeParams;
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "annotate", about = "Transcribe audio with speaker labels")]
struct Args {
    /// Input audio file
    #[arg(short, long)]
    input: String,

    /// Output file path (stdout if not specified)
    #[arg(short, long)]
    output: Option<String>,

    /// Whisper model size: tiny, base, small, medium, large, large-v2, large-v3
    #[arg(long, default_value = "tiny")]
    transcribe_model: String,

    /// Speaker model bundle: english, multilingual
    #[arg(long, default_value = "english")]
    speaker_model: String,

    /// Task type: transcribe or translate
    #[arg(short, long, default_value = "transcribe")]
    task: String,

    /// Language code (e.g. "en", "es"), auto-detect if not specified
    #[arg(short, long)]
    language: Option<String>,

    /// Expected number of speakers (auto-detect if not specified)
    #[arg(long)]
    num_speakers: Option<i32>,

    /// Clustering threshold (0.0–1.0, default 0.5; lower = more speakers)
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Inference device for speaker diarization: cpu, cuda, coreml
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Assign names to speakers by position or key=value pairs.
    /// Examples:
    ///   --names "Alice,Bob"       (Speaker 0=Alice, Speaker 1=Bob)
    ///   --names "0=Alice,1=Bob"   (explicit key=value)
    #[arg(long)]
    names: Option<String>,

    /// Output format: srt, json, text
    #[arg(short, long, default_value = "srt")]
    format: String,

    /// Omit timestamps from transcription (incompatible with --format srt)
    #[arg(long)]
    no_timestamps: bool,
}

struct CliAnnotationReporter;

impl AnnotationProgressReporter for CliAnnotationReporter {
    fn report(&self, progress: AnnotationProgress) {
        match progress.phase {
            AnnotationPhase::Starting => {
                eprint!("  Starting annotation...");
            }
            AnnotationPhase::DecodingAudio => {
                eprint!("\n  Decoding audio...");
            }
            AnnotationPhase::PreparingAudio => {
                eprint!("\n  Preparing audio...");
            }
            AnnotationPhase::LoadingSpeakerModel => {
                eprint!("\n  [1/2] Loading speaker model...");
            }
            AnnotationPhase::Diarizing => {
                if let (Some(current), Some(total)) = (progress.current, progress.total) {
                    let pct = if total > 0 { current * 100 / total } else { 0 };
                    eprint!("\r  [1/2] Diarization: {}% ({}/{})", pct, current, total);
                }
            }
            AnnotationPhase::LoadingTranscriptionModel => {
                eprint!("\n  [2/2] Loading transcription model...");
            }
            AnnotationPhase::Transcribing => {
                if let (Some(current), Some(total)) = (progress.current, progress.total) {
                    let pct = if total > 0 { current * 100 / total } else { 0 };
                    eprint!("\r  [2/2] Transcribing: {}% ({}/{})", pct, current, total);
                }
            }
            AnnotationPhase::Merging => {
                eprint!("\n  Merging speaker diarization + transcript...");
            }
            AnnotationPhase::Completed => {
                eprintln!("\n  Annotation complete.");
            }
        }
        let _ = std::io::stderr().flush();
    }
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if args.no_timestamps && args.format.eq_ignore_ascii_case("srt") {
        anyhow::bail!("--no-timestamps cannot be used with --format srt: SRT requires timestamps");
    }

    if args.no_timestamps {
        tracing::warn!(
            "--no-timestamps is set; speaker assignment quality is degraded because alignment timestamps are missing."
        );
    }

    let transcribe_model = parse_whisper_model(&args.transcribe_model)?;
    let speaker_model = parse_speaker_model(&args.speaker_model)?;
    let task = parse_task(&args.task)?;
    let device = parse_speaker_device(&args.device)?;

    let params = AnnotateParams {
        transcribe: TranscribeParams {
            model: transcribe_model,
            task,
            language: args.language,
            timestamps: !args.no_timestamps,
            force_cpu: false,
            use_quantized: false,
        },
        diarize: DiarizeParams {
            model: speaker_model,
            num_speakers: args.num_speakers,
            threshold: args.threshold,
            device,
        },
        speaker_names: args
            .names
            .as_deref()
            .map(parse_speaker_names)
            .unwrap_or_default(),
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let reporter = Arc::new(CliAnnotationReporter);
    let result = annotate_audio_with_reporter(&args.input, params, reporter, Some(cancel))?;

    let content = match args.format.to_lowercase().as_str() {
        "json" => format_as_json(&result)?,
        "text" => format_as_text(&result),
        "srt" => format_as_srt(&result),
        other => {
            tracing::warn!("Unrecognized format '{}', defaulting to SRT", other);
            format_as_srt(&result)
        }
    };

    write_output(content, args.output.as_deref())?;

    Ok(())
}
