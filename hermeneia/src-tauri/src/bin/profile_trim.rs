// src-tauri/src/bin/profile_trim.rs
//
// Profile audio trim operations with detailed timing breakdowns

use hermeneia_lib::audio::trim::trim_audio_file;
use hermeneia_lib::audio::TrimParams;
use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "profile-trim")]
#[command(about = "Profile audio trimming performance")]
struct Args {
    /// Input audio file
    #[arg(short, long)]
    input: PathBuf,

    /// Output audio file
    #[arg(short, long)]
    output: PathBuf,

    /// Start time in seconds
    #[arg(short, long, default_value = "10.0")]
    start: f64,

    /// End time in seconds
    #[arg(short, long, default_value = "20.0")]
    end: f64,

    /// Number of iterations to average
    #[arg(short, long, default_value = "5")]
    iterations: usize,
}

fn format_duration(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.2} μs", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.2} ms", secs * 1000.0)
    } else {
        format!("{:.2} s", secs)
    }
}

fn get_file_size_mb(path: &PathBuf) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / 1_000_000.0)
        .unwrap_or(0.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Audio Trim Performance Profiler ===\n");

    // Validate input
    if !args.input.exists() {
        eprintln!("Error: Input file does not exist: {}", args.input.display());
        std::process::exit(1);
    }

    let input_size = get_file_size_mb(&args.input);
    let input_ext = args.input.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown");

    println!("Configuration:");
    println!("  Input file: {}", args.input.display());
    println!("  Input size: {:.2} MB", input_size);
    println!("  Format: {}", input_ext.to_uppercase());
    println!("  Output file: {}", args.output.display());
    println!("  Trim range: {:.2}s to {:.2}s ({:.2}s duration)",
             args.start, args.end, args.end - args.start);
    println!("  Iterations: {}\n", args.iterations);

    let params = TrimParams::new(args.start, args.end)?;

    let mut timings = Vec::new();

    for i in 1..=args.iterations {
        println!("--- Iteration {}/{} ---", i, args.iterations);

        // Remove output file if it exists
        if args.output.exists() {
            std::fs::remove_file(&args.output)?;
        }

        let start = Instant::now();

        trim_audio_file(&args.input, &args.output, &params)?;

        let elapsed = start.elapsed().as_secs_f64();
        timings.push(elapsed);

        let output_size = get_file_size_mb(&args.output);
        let throughput = input_size / elapsed;

        println!("  Time: {}", format_duration(elapsed));
        println!("  Output size: {:.2} MB", output_size);
        println!("  Throughput: {:.2} MB/s\n", throughput);
    }

    // Calculate statistics
    println!("=== Performance Summary ===");

    let min = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg = timings.iter().sum::<f64>() / timings.len() as f64;

    // Calculate standard deviation
    let variance = timings.iter()
        .map(|&t| {
            let diff = t - avg;
            diff * diff
        })
        .sum::<f64>() / timings.len() as f64;
    let std_dev = variance.sqrt();

    println!("  Min time: {}", format_duration(min));
    println!("  Max time: {}", format_duration(max));
    println!("  Avg time: {}", format_duration(avg));
    println!("  Std dev: {}", format_duration(std_dev));
    println!("  Avg throughput: {:.2} MB/s", input_size / avg);

    // Estimate processing rate
    let trim_duration = args.end - args.start;
    let realtime_factor = trim_duration / avg;
    println!("  Realtime factor: {:.2}x (trimming {:.2}s audio in {:.2}s)",
             realtime_factor, trim_duration, avg);

    println!("\nAll timings: {:?}", timings.iter().map(|&t| format_duration(t)).collect::<Vec<_>>());

    Ok(())
}
