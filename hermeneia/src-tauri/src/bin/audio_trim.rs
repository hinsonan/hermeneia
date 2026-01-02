use clap::Parser;
use hermeneia_lib::audio::{get_audio_info, TrimParams};
use hermeneia_lib::audio::trim::trim_audio_file;
use tracing::{info, error};

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

    // Step 3: Trim audio (optimized - uses streaming or direct byte copy)
    info!("Trimming audio using optimized method");
    let start_time = std::time::Instant::now();

    trim_audio_file(&args.input, &args.output, &params)?;

    info!(
        output = %args.output,
        total_time_sec = start_time.elapsed().as_secs_f64(),
        "Done! Output saved"
    );

    Ok(())
}