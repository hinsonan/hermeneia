// src-tauri/src/bin/generate_test_audio.rs
//
// Generate test audio files for profiling trim operations
// Supports WAV (direct), MP3, and FLAC (via ffmpeg conversion)

use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::f32::consts::PI;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "generate-test-audio")]
#[command(about = "Generate test audio files for profiling (WAV/MP3/FLAC)")]
struct Args {
    /// Output file path (supports .wav, .mp3, .flac extensions)
    #[arg(short, long)]
    output: PathBuf,

    /// Duration in seconds
    #[arg(short, long, default_value = "60")]
    duration: f64,

    /// Sample rate in Hz
    #[arg(short, long, default_value = "44100")]
    sample_rate: u32,

    /// Number of channels (1=mono, 2=stereo)
    #[arg(short, long, default_value = "2")]
    channels: u16,

    /// Frequency of test tone in Hz
    #[arg(short, long, default_value = "440")]
    frequency: f32,

    /// Bit depth (16 or 32, only for WAV output)
    #[arg(short, long, default_value = "16")]
    bits: u16,

    /// MP3 bitrate in kbps (e.g., 128, 192, 320)
    #[arg(long, default_value = "320")]
    mp3_bitrate: u32,
}

#[derive(Debug, PartialEq)]
enum OutputFormat {
    Wav,
    Mp3,
    Flac,
}

fn detect_format(path: &PathBuf) -> OutputFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("mp3") => OutputFormat::Mp3,
        Some("flac") => OutputFormat::Flac,
        _ => OutputFormat::Wav,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let output_format = detect_format(&args.output);
    let needs_conversion = output_format != OutputFormat::Wav;

    // Determine actual output path (use temp WAV if conversion needed)
    let wav_path = if needs_conversion {
        let mut temp = args.output.clone();
        temp.set_extension("temp.wav");
        temp
    } else {
        args.output.clone()
    };

    println!("Generating test audio file:");
    println!("  Output: {}", args.output.display());
    println!("  Format: {:?}", output_format);
    println!("  Duration: {:.2} seconds", args.duration);
    println!("  Sample rate: {} Hz", args.sample_rate);
    println!("  Channels: {}", args.channels);
    println!("  Frequency: {} Hz", args.frequency);
    if output_format == OutputFormat::Wav {
        println!("  Bit depth: {} bits", args.bits);
    }
    if output_format == OutputFormat::Mp3 {
        println!("  MP3 bitrate: {} kbps", args.mp3_bitrate);
    }

    let spec = WavSpec {
        channels: args.channels,
        sample_rate: args.sample_rate,
        bits_per_sample: args.bits,
        sample_format: if args.bits == 32 {
            SampleFormat::Float
        } else {
            SampleFormat::Int
        },
    };

    let mut writer = WavWriter::create(&wav_path, spec)?;

    let total_samples = (args.duration * args.sample_rate as f64) as usize;
    let total_frames = total_samples;

    println!("  Total frames: {}", total_frames);
    println!("  Total samples: {}", total_frames * args.channels as usize);

    let estimated_size = total_frames * args.channels as usize * (args.bits / 8) as usize;
    println!(
        "  Estimated size: {:.2} MB",
        estimated_size as f64 / 1_000_000.0
    );

    println!("\nGenerating samples...");

    let start = std::time::Instant::now();

    // Generate a sine wave test tone
    for frame in 0..total_frames {
        let t = frame as f32 / args.sample_rate as f32;
        let sample = (t * args.frequency * 2.0 * PI).sin() * 0.5; // 50% amplitude

        // Write same sample to all channels
        for _ in 0..args.channels {
            if args.bits == 32 {
                writer.write_sample(sample)?;
            } else {
                // Convert to i16
                let sample_i16 = (sample * 32767.0) as i16;
                writer.write_sample(sample_i16)?;
            }
        }

        // Progress indicator
        if frame % (args.sample_rate as usize * 10) == 0 {
            let progress = (frame as f64 / total_frames as f64) * 100.0;
            println!("  Progress: {:.1}%", progress);
        }
    }

    writer.finalize()?;

    let elapsed = start.elapsed();
    println!("\nWAV generation complete!");
    println!("  Time taken: {:.2} seconds", elapsed.as_secs_f64());
    println!(
        "  WAV size: {:.2} MB",
        std::fs::metadata(&wav_path)?.len() as f64 / 1_000_000.0
    );

    // Convert to MP3/FLAC if needed
    if needs_conversion {
        println!("\nConverting to {:?} format...", output_format);
        let convert_start = std::time::Instant::now();

        let success = match output_format {
            OutputFormat::Mp3 => {
                let bitrate = format!("{}k", args.mp3_bitrate);
                Command::new("ffmpeg")
                    .arg("-i")
                    .arg(&wav_path)
                    .arg("-b:a")
                    .arg(&bitrate)
                    .arg("-y") // Overwrite output file
                    .arg(&args.output)
                    .output()?
                    .status
                    .success()
            }
            OutputFormat::Flac => {
                Command::new("ffmpeg")
                    .arg("-i")
                    .arg(&wav_path)
                    .arg("-y") // Overwrite output file
                    .arg(&args.output)
                    .output()?
                    .status
                    .success()
            }
            OutputFormat::Wav => unreachable!(),
        };

        if !success {
            eprintln!("\nError: ffmpeg conversion failed!");
            eprintln!("Make sure ffmpeg is installed and in your PATH.");
            std::fs::remove_file(&wav_path)?;
            return Err("Conversion failed".into());
        }

        let convert_elapsed = convert_start.elapsed();
        println!(
            "  Conversion time: {:.2} seconds",
            convert_elapsed.as_secs_f64()
        );

        // Remove temporary WAV file
        std::fs::remove_file(&wav_path)?;
        println!("  Removed temporary WAV file");
    }

    println!("\nFinal output:");
    println!("  File: {}", args.output.display());
    println!(
        "  Size: {:.2} MB",
        std::fs::metadata(&args.output)?.len() as f64 / 1_000_000.0
    );

    Ok(())
}
