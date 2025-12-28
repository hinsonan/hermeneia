// src-tauri/src/bin/audio_trim.rs

use clap::Parser;
use hermeneia_lib::audio::{decode_audio_file, encode_wav, get_audio_info, trim_audio, TrimParams};

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
    let args = Args::parse();

    println!("🎵 Audio Trimmer");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Step 1: Get audio info
    if args.verbose {
        println!("\n📋 Getting audio info...");
    }
    let info = get_audio_info(&args.input)?;
    
    println!("\n📊 Input File: {}", args.input);
    println!("   Duration: {:.2} seconds ({:.2} minutes)", 
        info.duration_seconds, info.duration_seconds / 60.0);
    println!("   Sample Rate: {} Hz", info.sample_rate);
    println!("   Channels: {}", info.channels);
    println!("   Format: {}", info.format);

    // Step 2: Validate trim parameters
    let params = TrimParams::new(args.start, args.end)?;
    
    println!("\n✂️  Trim Range:");
    println!("   Start: {:.2}s", params.start_seconds);
    println!("   End: {:.2}s", params.end_seconds);
    println!("   Duration: {:.2}s", params.trim_duration());

    if params.end_seconds > info.duration_seconds {
        eprintln!("\n❌ Error: Trim end time ({:.2}s) exceeds audio duration ({:.2}s)", 
            params.end_seconds, info.duration_seconds);
        std::process::exit(1);
    }

    // Step 3: Decode audio
    println!("\n🔊 Decoding audio...");
    let start_time = std::time::Instant::now();
    let audio = decode_audio_file(&args.input)?;
    
    if args.verbose {
        println!("   Loaded {} samples ({:.2} MB)", 
            audio.samples.len(),
            (audio.samples.len() * 4) as f64 / 1_048_576.0);
        println!("   Decode time: {:.2}s", start_time.elapsed().as_secs_f64());
    }

    // Step 4: Trim audio
    println!("\n✂️  Trimming audio...");
    let trimmed = trim_audio(&audio, &params)?;
    
    if args.verbose {
        println!("   Trimmed to {} samples", trimmed.samples.len());
        println!("   New duration: {:.2}s", trimmed.duration_seconds());
    }

    // Step 5: Encode to WAV
    println!("\n💾 Encoding to WAV...");
    let encode_start = std::time::Instant::now();
    encode_wav(&trimmed, &args.output)?;
    
    if args.verbose {
        println!("   Encode time: {:.2}s", encode_start.elapsed().as_secs_f64());
    }

    println!("\n✅ Done! Output saved to: {}", args.output);
    println!("   Total time: {:.2}s", start_time.elapsed().as_secs_f64());

    Ok(())
}