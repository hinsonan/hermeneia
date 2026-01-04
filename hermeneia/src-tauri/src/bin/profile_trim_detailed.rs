// src-tauri/src/bin/profile_trim_detailed.rs
//
// Profile audio trim with internal timing instrumentation

use hound::{WavReader, WavWriter, SampleFormat};
use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use hermeneia_lib::audio::types::TrimParams;

#[derive(Parser, Debug)]
#[command(name = "profile-trim-detailed")]
#[command(about = "Profile audio trimming with detailed breakdown")]
struct Args {
    /// Input WAV file
    #[arg(short, long)]
    input: PathBuf,

    /// Output WAV file
    #[arg(short, long)]
    output: PathBuf,

    /// Start time in seconds
    #[arg(short, long, default_value = "10.0")]
    start: f64,

    /// End time in seconds
    #[arg(short, long, default_value = "20.0")]
    end: f64,
}

fn trim_wav_detailed(
    input_path: &PathBuf,
    output_path: &PathBuf,
    params: &TrimParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Detailed Timing Breakdown ===\n");

    let t_total = Instant::now();

    // 1. Open reader
    let t_open = Instant::now();
    let mut reader = WavReader::open(input_path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    println!("1. Open reader:     {:>10.3} ms", t_open.elapsed().as_secs_f64() * 1000.0);

    // Validate
    if params.end_seconds > duration {
        return Err(format!("End time {} exceeds duration {}", params.end_seconds, duration).into());
    }

    // 2. Calculate positions
    let t_calc = Instant::now();
    let start_frame = (params.start_seconds * spec.sample_rate as f64) as u32;
    let end_frame = (params.end_seconds * spec.sample_rate as f64) as u32;
    let total_frames = reader.duration();
    let start_frame = start_frame.min(total_frames);
    let end_frame = end_frame.min(total_frames);
    let frames_to_read = end_frame - start_frame;
    println!("2. Calculate pos:   {:>10.3} ms", t_calc.elapsed().as_secs_f64() * 1000.0);

    // 3. Seek
    let t_seek = Instant::now();
    let start_sample_idx = start_frame * spec.channels as u32;
    reader.seek(start_sample_idx)?;
    println!("3. Seek to start:   {:>10.3} ms", t_seek.elapsed().as_secs_f64() * 1000.0);

    // 4. Create writer
    let t_writer = Instant::now();
    let mut writer = WavWriter::create(output_path, spec)?;
    println!("4. Create writer:   {:>10.3} ms", t_writer.elapsed().as_secs_f64() * 1000.0);

    // 5. Read and write loop
    const BUFFER_SIZE: usize = 4096;
    let channels = spec.channels as usize;
    let mut frames_remaining = frames_to_read;

    let mut total_read_time = 0.0;
    let mut total_write_time = 0.0;
    let mut iteration_count = 0;

    match spec.sample_format {
        SampleFormat::Float => {
            let mut iter = reader.samples::<f32>();
            while frames_remaining > 0 {
                let chunk_frames = frames_remaining.min(BUFFER_SIZE as u32);
                let samples_to_process = chunk_frames as usize * channels;

                let t_read = Instant::now();
                let mut samples = Vec::with_capacity(samples_to_process);
                for _ in 0..samples_to_process {
                    if let Some(sample) = iter.next() {
                        samples.push(sample?);
                    }
                }
                total_read_time += t_read.elapsed().as_secs_f64();

                let t_write = Instant::now();
                for sample in samples {
                    writer.write_sample(sample)?;
                }
                total_write_time += t_write.elapsed().as_secs_f64();

                frames_remaining -= chunk_frames;
                iteration_count += 1;
            }
        }
        SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => {
                    let mut iter = reader.samples::<i16>();
                    while frames_remaining > 0 {
                        let chunk_frames = frames_remaining.min(BUFFER_SIZE as u32);
                        let samples_to_process = chunk_frames as usize * channels;

                        let t_read = Instant::now();
                        let mut samples = Vec::with_capacity(samples_to_process);
                        for _ in 0..samples_to_process {
                            if let Some(sample) = iter.next() {
                                samples.push(sample?);
                            }
                        }
                        total_read_time += t_read.elapsed().as_secs_f64();

                        let t_write = Instant::now();
                        for sample in samples {
                            writer.write_sample(sample)?;
                        }
                        total_write_time += t_write.elapsed().as_secs_f64();

                        frames_remaining -= chunk_frames;
                        iteration_count += 1;
                    }
                }
                24 | 32 => {
                    let mut iter = reader.samples::<i32>();
                    while frames_remaining > 0 {
                        let chunk_frames = frames_remaining.min(BUFFER_SIZE as u32);
                        let samples_to_process = chunk_frames as usize * channels;

                        let t_read = Instant::now();
                        let mut samples = Vec::with_capacity(samples_to_process);
                        for _ in 0..samples_to_process {
                            if let Some(sample) = iter.next() {
                                samples.push(sample?);
                            }
                        }
                        total_read_time += t_read.elapsed().as_secs_f64();

                        let t_write = Instant::now();
                        for sample in samples {
                            writer.write_sample(sample)?;
                        }
                        total_write_time += t_write.elapsed().as_secs_f64();

                        frames_remaining -= chunk_frames;
                        iteration_count += 1;
                    }
                }
                _ => return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into()),
            }
        }
    }

    println!("5. Read samples:    {:>10.3} ms ({} iterations)", total_read_time * 1000.0, iteration_count);
    println!("6. Write samples:   {:>10.3} ms", total_write_time * 1000.0);

    // 7. Finalize
    let t_finalize = Instant::now();
    writer.finalize()?;
    println!("7. Finalize writer: {:>10.3} ms", t_finalize.elapsed().as_secs_f64() * 1000.0);

    let total_time = t_total.elapsed().as_secs_f64();
    println!("\n--- Total Time:     {:>10.3} ms ---", total_time * 1000.0);

    // Calculate percentages
    let overhead = total_time - (total_read_time + total_write_time);
    println!("\nTime Distribution:");
    println!("  Read:      {:.1}%", (total_read_time / total_time) * 100.0);
    println!("  Write:     {:.1}%", (total_write_time / total_time) * 100.0);
    println!("  Overhead:  {:.1}%", (overhead / total_time) * 100.0);

    let samples_processed = frames_to_read as usize * channels;
    println!("\nPerformance:");
    println!("  Samples/sec:  {:.2} M", (samples_processed as f64 / total_time) / 1_000_000.0);
    println!("  Frames/sec:   {:.2} M", (frames_to_read as f64 / total_time) / 1_000_000.0);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Detailed WAV Trim Profiling ===");
    println!("Input:  {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!("Range:  {:.2}s to {:.2}s", args.start, args.end);

    let params = TrimParams::new(args.start, args.end)?;

    trim_wav_detailed(&args.input, &args.output, &params)?;

    Ok(())
}
