// src-tauri/src/bin/profile_compressed_detailed.rs
//
// Profile compressed audio trim with detailed timing breakdown

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use hound::{WavWriter, WavSpec, SampleFormat};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::audio::AudioBufferRef;
use hermeneia_lib::audio::types::TrimParams;

#[derive(Parser, Debug)]
#[command(name = "profile-compressed-detailed")]
#[command(about = "Profile compressed audio trimming with detailed breakdown")]
struct Args {
    /// Input audio file (MP3, FLAC, etc.)
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

fn convert_to_f32(buffer: &AudioBufferRef) -> Vec<f32> {
    let spec = buffer.spec();
    let channels = spec.channels.count();
    let frames = buffer.frames();
    let mut output = Vec::with_capacity(frames * channels);

    match buffer {
        AudioBufferRef::F32(buf) => {
            let planes = buf.planes();
            for i in 0..frames {
                for plane in planes.planes() {
                    output.push(plane[i]);
                }
            }
        }
        AudioBufferRef::F64(buf) => {
            let planes = buf.planes();
            for i in 0..frames {
                for plane in planes.planes() {
                    output.push(plane[i] as f32);
                }
            }
        }
        AudioBufferRef::S16(buf) => {
            let planes = buf.planes();
            for i in 0..frames {
                for plane in planes.planes() {
                    output.push(plane[i] as f32 / 32768.0);
                }
            }
        }
        AudioBufferRef::S32(buf) => {
            let planes = buf.planes();
            for i in 0..frames {
                for plane in planes.planes() {
                    output.push(plane[i] as f32 / 2147483648.0);
                }
            }
        }
        _ => {}
    }

    output
}

fn trim_compressed_detailed(
    input_path: &PathBuf,
    output_path: &PathBuf,
    params: &TrimParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Detailed Timing Breakdown ===\n");

    let t_total = Instant::now();

    // 1. Open file
    let t_open = Instant::now();
    let file = File::open(input_path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    println!("1. Open file:           {:>10.3} ms", t_open.elapsed().as_secs_f64() * 1000.0);

    // 2. Probe format
    let t_probe = Instant::now();
    let mut hint = Hint::new();
    if let Some(extension) = input_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())?;
    let mut format = probed.format;
    println!("2. Probe format:        {:>10.3} ms", t_probe.elapsed().as_secs_f64() * 1000.0);

    // 3. Get track info
    let t_track = Instant::now();
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.ok_or("Sample rate not found")?;
    let mut channels_opt = track.codec_params.channels.map(|c| c.count() as u16);
    println!("3. Get track info:      {:>10.3} ms", t_track.elapsed().as_secs_f64() * 1000.0);

    // 4. Create decoder
    let t_decoder = Instant::now();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;
    println!("4. Create decoder:      {:>10.3} ms", t_decoder.elapsed().as_secs_f64() * 1000.0);

    // 5. Seek to start
    let t_seek = Instant::now();
    let target_ts = (params.start_seconds * sample_rate as f64) as u64;
    let seek_result = format.seek(
        SeekMode::Accurate,
        SeekTo::TimeStamp { ts: target_ts, track_id }
    )?;
    let mut current_sample = seek_result.actual_ts;
    println!("5. Seek to start:       {:>10.3} ms (target: {}, actual: {})",
             t_seek.elapsed().as_secs_f64() * 1000.0, target_ts, current_sample);

    // 6. Determine channels if unknown
    let t_channels = Instant::now();
    if channels_opt.is_none() {
        let first_packet = format.next_packet()?;
        let decoded = decoder.decode(&first_packet)?;
        channels_opt = Some(decoded.spec().channels.count() as u16);
        let re_seek = format.seek(
            SeekMode::Accurate,
            SeekTo::TimeStamp { ts: target_ts, track_id }
        )?;
        current_sample = re_seek.actual_ts;
    }
    let channels = channels_opt.unwrap();
    println!("6. Determine channels:  {:>10.3} ms (channels: {})",
             t_channels.elapsed().as_secs_f64() * 1000.0, channels);

    // 7. Create writer
    let t_writer = Instant::now();
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(output_path, spec)?;
    println!("7. Create writer:       {:>10.3} ms", t_writer.elapsed().as_secs_f64() * 1000.0);

    let end_sample = (params.end_seconds * sample_rate as f64) as u64;

    // 8. Decode loop
    let mut total_read_time = 0.0;
    let mut total_decode_time = 0.0;
    let mut total_convert_time = 0.0;
    let mut total_write_time = 0.0;
    let mut packet_count = 0;
    let mut total_frames = 0;

    loop {
        if current_sample >= end_sample {
            break;
        }

        let t_read = Instant::now();
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };
        total_read_time += t_read.elapsed().as_secs_f64();

        if packet.track_id() != track_id {
            continue;
        }

        let t_decode = Instant::now();
        let decoded = decoder.decode(&packet)?;
        total_decode_time += t_decode.elapsed().as_secs_f64();

        let t_convert = Instant::now();
        let samples = convert_to_f32(&decoded);
        let frames_in_packet = samples.len() / channels as usize;
        total_convert_time += t_convert.elapsed().as_secs_f64();

        let mut start_offset_frames = 0;
        if current_sample < target_ts {
            if current_sample + (frames_in_packet as u64) > target_ts {
                start_offset_frames = (target_ts - current_sample) as usize;
            } else {
                current_sample += frames_in_packet as u64;
                continue;
            }
        }

        let valid_frames_in_packet = frames_in_packet - start_offset_frames;
        let frames_until_end = end_sample.saturating_sub(current_sample + start_offset_frames as u64);
        let frames_to_write = (valid_frames_in_packet as u64).min(frames_until_end) as usize;

        if frames_to_write > 0 {
            let t_write = Instant::now();
            let start_index = start_offset_frames * channels as usize;
            let end_index = start_index + (frames_to_write * channels as usize);

            for &sample in &samples[start_index..end_index] {
                writer.write_sample(sample)?;
            }
            total_write_time += t_write.elapsed().as_secs_f64();
            total_frames += frames_to_write;
        }

        current_sample += frames_in_packet as u64;
        packet_count += 1;
    }

    println!("8. Read packets:        {:>10.3} ms ({} packets)",
             total_read_time * 1000.0, packet_count);
    println!("9. Decode packets:      {:>10.3} ms", total_decode_time * 1000.0);
    println!("10. Convert samples:    {:>10.3} ms", total_convert_time * 1000.0);
    println!("11. Write samples:      {:>10.3} ms ({} frames)",
             total_write_time * 1000.0, total_frames);

    // 12. Finalize
    let t_finalize = Instant::now();
    writer.finalize()?;
    println!("12. Finalize writer:    {:>10.3} ms", t_finalize.elapsed().as_secs_f64() * 1000.0);

    let total_time = t_total.elapsed().as_secs_f64();
    println!("\n--- Total Time:         {:>10.3} ms ---", total_time * 1000.0);

    // Calculate percentages
    let processing_time = total_read_time + total_decode_time + total_convert_time + total_write_time;
    let overhead = total_time - processing_time;

    println!("\nTime Distribution:");
    println!("  Read packets:   {:.1}%", (total_read_time / total_time) * 100.0);
    println!("  Decode:         {:.1}%", (total_decode_time / total_time) * 100.0);
    println!("  Convert:        {:.1}%", (total_convert_time / total_time) * 100.0);
    println!("  Write:          {:.1}%", (total_write_time / total_time) * 100.0);
    println!("  Overhead:       {:.1}%", (overhead / total_time) * 100.0);

    println!("\nPerformance:");
    println!("  Packets/sec:    {:.2} K", (packet_count as f64 / total_time) / 1000.0);
    println!("  Frames/sec:     {:.2} M", (total_frames as f64 / total_time) / 1_000_000.0);
    println!("  Avg packet:     {:.2} frames", total_frames as f64 / packet_count as f64);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Detailed Compressed Audio Trim Profiling ===");
    println!("Input:  {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!("Range:  {:.2}s to {:.2}s", args.start, args.end);

    let params = TrimParams::new(args.start, args.end)?;

    trim_compressed_detailed(&args.input, &args.output, &params)?;

    Ok(())
}
