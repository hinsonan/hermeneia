use clap::Parser;
use hermeneia_lib::audio::{decode_audio_file, encode_wav, get_audio_info, trim_audio, TrimParams};
use tracing::{info, debug, error};

/// Command-line tool for trimming audio files
#[derive(Parser, Debug)]
#[command(name = "audio-trim")]
#[command(about = "Trim audio files to a specific time range", long_about = None)]
struct Args {
    /// Input audio file (MP3, FLAC, WAV, OGG, etc.)
    #[arg(short, long)]
    input: String,

    /// Output WAV file
    #[arg(short, long)]
    output: String,

    /// Start time in seconds
    #[arg(short, long)]
    start: f64,

    /// End time in seconds
    #[arg(short, long)]
    end: f64,

    /// Show detailed information
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    let args = Args::parse();

    // Step 1: Get audio info
    debug!("Getting audio info");
    let info = get_audio_info(&args.input)?;

    info!(
        file = %args.input,
        duration_sec = info.duration_seconds,
        duration_min = info.duration_seconds / 60.0,
        sample_rate = info.sample_rate,
        channels = info.channels,
        format = %info.format,
        "Input audio file info"
    );

    // Step 2: Validate trim parameters
    let params = TrimParams::new(args.start, args.end)?;

    info!(
        start_sec = params.start_seconds,
        end_sec = params.end_seconds,
        trim_duration_sec = params.trim_duration(),
        "Trim range"
    );

    if params.end_seconds > info.duration_seconds {
        error!(
            trim_end = params.end_seconds,
            audio_duration = info.duration_seconds,
            "Trim end time exceeds audio duration"
        );
        std::process::exit(1);
    }

    // Step 3: Decode audio
    info!("Decoding audio");
    let start_time = std::time::Instant::now();
    let audio = decode_audio_file(&args.input)?;

    debug!(
        samples = audio.samples.len(),
        size_mb = (audio.samples.len() * 4) as f64 / 1_048_576.0,
        decode_time_sec = start_time.elapsed().as_secs_f64(),
        "Audio decoded"
    );

    // Step 4: Trim audio
    info!("Trimming audio");
    let trimmed = trim_audio(&audio, &params)?;

    debug!(
        samples = trimmed.samples.len(),
        duration_sec = trimmed.duration_seconds(),
        "Audio trimmed"
    );

    // Step 5: Encode to WAV
    info!("Encoding to WAV");
    let encode_start = std::time::Instant::now();
    encode_wav(&trimmed, &args.output)?;

    debug!(
        encode_time_sec = encode_start.elapsed().as_secs_f64(),
        "WAV encoding complete"
    );

    info!(
        output = %args.output,
        total_time_sec = start_time.elapsed().as_secs_f64(),
        "Done! Output saved"
    );

    Ok(())
}