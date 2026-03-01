use clap::Parser;
use hermeneia_lib::speaker::{
    diarize_audio_with_progress, DiarizeParams, DiarizationResult, SpeakerDevice, SpeakerModel,
};
use hermeneia_lib::transcribe::{
    transcribe_audio_with_reporter, ProgressReporter, TranscribeParams, TranscriptResult,
    TranscriptSegment, TranscriptionTask, WhisperModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

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

// --- Progress reporter for transcription step ---

struct AnnotateProgress {
    progress_bar: ProgressBar,
    first_call: Arc<Mutex<bool>>,
}

impl AnnotateProgress {
    fn new() -> Self {
        let progress_bar = ProgressBar::new_spinner();
        progress_bar.set_style(
            ProgressStyle::default_spinner()
                .template("[{elapsed_precise}] {spinner:.cyan} {msg}")
                .expect("Invalid spinner template"),
        );
        progress_bar.set_message("[2/2] Loading model and detecting language...");
        progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));
        Self {
            progress_bar,
            first_call: Arc::new(Mutex::new(true)),
        }
    }
}

impl ProgressReporter for AnnotateProgress {
    fn report(&self, current: usize, total: usize) {
        if total == 0 {
            return;
        }
        if let Ok(mut is_first) = self.first_call.lock() {
            if *is_first {
                *is_first = false;
                self.progress_bar.disable_steady_tick();
                self.progress_bar.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>3}% {msg}")
                        .expect("Invalid progress bar template")
                        .progress_chars("█▓░"),
                );
                self.progress_bar.set_length(100);
                self.progress_bar.set_message("[2/2] Transcribing...");
            }
        }
        let percentage = (current as f64 / total as f64 * 100.0) as u64;
        self.progress_bar.set_position(percentage);
    }

    fn finish(&self) {
        self.progress_bar.finish_with_message("Transcription complete!");
    }
}

impl Drop for AnnotateProgress {
    fn drop(&mut self) {
        self.progress_bar.finish();
    }
}

// --- Merge logic ---

struct AnnotatedSegment<'a> {
    transcript: &'a TranscriptSegment,
    speaker_id: i32,
}

/// For each transcript segment, find the diarization segment with the most overlap.
fn assign_speakers<'a>(
    transcript: &'a TranscriptResult,
    diarization: &DiarizationResult,
) -> Vec<AnnotatedSegment<'a>> {
    transcript
        .segments
        .iter()
        .map(|seg| {
            let seg_start = seg.start.unwrap_or(0.0);
            let seg_end = seg.end.unwrap_or(seg_start);

            let speaker_id = diarization
                .segments
                .iter()
                .map(|s| {
                    let overlap = (f64::min(seg_end, s.end as f64)
                        - f64::max(seg_start, s.start as f64))
                    .max(0.0);
                    (s.speaker, overlap)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(id, _)| id)
                .unwrap_or(0);

            AnnotatedSegment {
                transcript: seg,
                speaker_id,
            }
        })
        .collect()
}

// --- Output formatters ---

fn format_as_srt(segments: &[AnnotatedSegment], names: &HashMap<i32, String>) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let start = seg.transcript.start.unwrap_or(0.0);
            let end = seg.transcript.end.unwrap_or(start);
            let label = speaker_label(seg.speaker_id, names);
            format!(
                "{}\n{} --> {}\n[{}] {}\n",
                i + 1,
                format_timestamp(start),
                format_timestamp(end),
                label,
                seg.transcript.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_as_json(
    segments: &[AnnotatedSegment],
    names: &HashMap<i32, String>,
    transcript: &TranscriptResult,
    diarization: &DiarizationResult,
) -> anyhow::Result<String> {
    let seg_values: Vec<serde_json::Value> = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let name = speaker_label(seg.speaker_id, names);
            serde_json::json!({
                "index": i + 1,
                "start": seg.transcript.start.unwrap_or(0.0),
                "end": seg.transcript.end.unwrap_or(0.0),
                "speaker": seg.speaker_id,
                "speaker_name": name,
                "text": seg.transcript.text.trim(),
            })
        })
        .collect();

    let speaker_names_map: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .map(|(id, name)| (id.to_string(), serde_json::Value::String(name.clone())))
        .collect();

    let output = serde_json::json!({
        "segments": seg_values,
        "speaker_names": speaker_names_map,
        "num_speakers": diarization.num_speakers,
        "language": transcript.language,
        "audio_duration": diarization.audio_duration,
    });

    Ok(serde_json::to_string_pretty(&output)?)
}

fn format_as_text(segments: &[AnnotatedSegment], names: &HashMap<i32, String>) -> String {
    let mut out = String::new();
    for seg in segments {
        let start = seg.transcript.start.unwrap_or(0.0);
        let start_min = (start / 60.0).floor() as u32;
        let start_sec = (start % 60.0).floor() as u32;
        let label = speaker_label(seg.speaker_id, names);
        out.push_str(&format!(
            "[{:02}:{:02}] {}: {}\n",
            start_min,
            start_sec,
            label,
            seg.transcript.text.trim()
        ));
    }
    out
}

// --- Helpers (same logic as transcribe/speaker binaries) ---

fn format_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0).floor() as u32;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
    let secs = (seconds % 60.0).floor() as u32;
    let millis = ((seconds % 1.0) * 1000.0).floor() as u32;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

fn parse_whisper_model(s: &str) -> anyhow::Result<WhisperModel> {
    match s.to_lowercase().as_str() {
        "tiny" => Ok(WhisperModel::Tiny),
        "tiny.en" => Ok(WhisperModel::TinyEn),
        "base" => Ok(WhisperModel::Base),
        "base.en" => Ok(WhisperModel::BaseEn),
        "small" => Ok(WhisperModel::Small),
        "small.en" => Ok(WhisperModel::SmallEn),
        "medium" => Ok(WhisperModel::Medium),
        "medium.en" => Ok(WhisperModel::MediumEn),
        "large" => Ok(WhisperModel::Large),
        "large-v2" => Ok(WhisperModel::LargeV2),
        "large-v3" => Ok(WhisperModel::LargeV3),
        _ => anyhow::bail!(
            "Invalid transcribe model '{}'. Use: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, large-v2, large-v3",
            s
        ),
    }
}

fn parse_speaker_model(s: &str) -> anyhow::Result<SpeakerModel> {
    match s.to_lowercase().as_str() {
        "english" => Ok(SpeakerModel::English),
        "multilingual" => Ok(SpeakerModel::Multilingual),
        _ => anyhow::bail!(
            "Invalid speaker model '{}'. Use: english, multilingual",
            s
        ),
    }
}

fn parse_device(s: &str) -> anyhow::Result<SpeakerDevice> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(SpeakerDevice::Cpu),
        "cuda" => Ok(SpeakerDevice::Cuda),
        "coreml" => Ok(SpeakerDevice::CoreMl),
        _ => anyhow::bail!("Invalid device '{}'. Use: cpu, cuda, coreml", s),
    }
}

fn parse_speaker_names(names_str: &str) -> HashMap<i32, String> {
    names_str
        .split(',')
        .enumerate()
        .filter_map(|(i, part)| {
            let (id, name) = match part.split_once('=') {
                Some((k, v)) => (k.trim().parse::<i32>().ok()?, v.trim()),
                None => (i as i32, part.trim()),
            };
            if name.is_empty() { None } else { Some((id, name.to_string())) }
        })
        .collect()
}

fn speaker_label(id: i32, names: &HashMap<i32, String>) -> String {
    names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("Speaker {}", id))
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

// --- Entry point ---

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if args.no_timestamps && args.format.to_lowercase() == "srt" {
        anyhow::bail!(
            "--no-timestamps cannot be used with --format srt: SRT requires timestamps"
        );
    }

    if args.no_timestamps {
        tracing::warn!(
            "--no-timestamps is set: transcript segments will have no timing information, \
             so all segments will be assigned to whichever speaker owns t=0. \
             Speaker assignment will be unreliable."
        );
    }

    let whisper_model = parse_whisper_model(&args.transcribe_model)?;
    let speaker_model = parse_speaker_model(&args.speaker_model)?;
    let device = parse_device(&args.device)?;

    let task = match args.task.to_lowercase().as_str() {
        "transcribe" => TranscriptionTask::Transcribe,
        "translate" => TranscriptionTask::Translate,
        _ => anyhow::bail!("Invalid task '{}'. Use: transcribe, translate", args.task),
    };

    tracing::info!("Annotating: {}", args.input);
    tracing::info!(
        "Transcription model: {:?}, Speaker model: {}, Device: {}",
        whisper_model,
        speaker_model.display_name(),
        device.provider_string()
    );

    // Step 1/2: Speaker diarization
    tracing::info!("[1/2] Running speaker diarization...");
    let diarize_params = DiarizeParams {
        model: speaker_model,
        num_speakers: args.num_speakers,
        threshold: args.threshold,
        device,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let progress_cb: hermeneia_lib::speaker::DiarizeProgressCallback =
        Box::new(|processed, total| {
            if total > 0 {
                let pct = processed * 100 / total;
                eprint!("\r  [1/2] Diarization: {}% ({}/{})", pct, processed, total);
                let _ = std::io::stderr().flush();
            }
        });

    let diarization =
        diarize_audio_with_progress(&args.input, diarize_params, Some(progress_cb), Some(cancel))?;

    eprintln!();
    tracing::info!(
        "[1/2] Diarization complete: {} speaker(s), {:.1}s audio",
        diarization.num_speakers,
        diarization.audio_duration
    );

    // Step 2/2: Transcription
    tracing::info!("[2/2] Running transcription...");
    let transcribe_params = TranscribeParams {
        model: whisper_model,
        task,
        language: args.language,
        timestamps: !args.no_timestamps,
        force_cpu: false,
        use_quantized: false,
    };

    let progress = AnnotateProgress::new();
    let transcript =
        transcribe_audio_with_reporter(&args.input, transcribe_params, &progress, None)?;
    drop(progress);

    println!();
    tracing::info!(
        "[2/2] Transcription complete: {:.2}s audio, {:.2}s processing",
        transcript.duration,
        transcript.inference_time
    );

    // Merge: align transcript segments to speaker segments by overlap
    let names = args
        .names
        .as_deref()
        .map(parse_speaker_names)
        .unwrap_or_default();

    let annotated = assign_speakers(&transcript, &diarization);

    let content = match args.format.to_lowercase().as_str() {
        "json" => format_as_json(&annotated, &names, &transcript, &diarization)?,
        "text" => format_as_text(&annotated, &names),
        "srt" | _ => format_as_srt(&annotated, &names),
    };

    write_output(content, args.output.as_deref())?;

    Ok(())
}
