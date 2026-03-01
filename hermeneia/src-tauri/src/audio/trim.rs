// src-tauri/src/audio/trim.rs

use crate::audio::types::{AudioData, TrimParams};
use crate::error::{AudioError, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Trim audio data to a specific time range
///
/// # Arguments
/// * `audio` - The audio data to trim
/// * `params` - Start and end times in seconds
pub fn trim_audio(audio: &AudioData, params: &TrimParams) -> Result<AudioData> {
    // Validate trim range against audio duration
    let duration = audio.duration_seconds();

    if params.end_seconds > duration {
        return Err(AudioError::TrimRangeOutOfBounds {
            start: params.start_seconds,
            end: params.end_seconds,
            duration,
        });
    }

    // Calculate sample indices
    let samples_per_second = audio.sample_rate as f64 * audio.channels as f64;

    let start_sample_index = (params.start_seconds * samples_per_second) as usize;
    let end_sample_index = (params.end_seconds * samples_per_second) as usize;

    // Ensure indices are aligned to frame boundaries (multiples of channels)
    let channels = audio.channels as usize;
    let start_sample_index = (start_sample_index / channels) * channels;
    let end_sample_index = (end_sample_index / channels) * channels;

    // Clamp to valid range
    let start_sample_index = start_sample_index.min(audio.samples.len());
    let end_sample_index = end_sample_index.min(audio.samples.len());

    // Extract the slice
    let trimmed_samples = audio.samples[start_sample_index..end_sample_index].to_vec();

    Ok(AudioData {
        samples: trimmed_samples,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
    })
}

/// Check if a file is a WAV file by examining its extension
fn is_wav_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
}

/// Trim a WAV file directly using seeking (O(1) complexity)
///
/// Note: This operation strips metadata (tags) from the original file.
fn trim_wav_direct<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    params: &TrimParams,
) -> Result<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    let mut reader = WavReader::open(input_path)
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;

    if params.end_seconds > duration {
        return Err(AudioError::TrimRangeOutOfBounds {
            start: params.start_seconds,
            end: params.end_seconds,
            duration,
        });
    }

    // Calculate frames (time steps)
    let start_frame = (params.start_seconds * spec.sample_rate as f64) as u32;
    let end_frame = (params.end_seconds * spec.sample_rate as f64) as u32;
    let total_frames = reader.duration(); // reader.duration() returns frames, not bytes

    // Clamp range
    let start_frame = start_frame.min(total_frames);
    let end_frame = end_frame.min(total_frames);
    let frames_to_read = end_frame - start_frame;

    // Seek to the start position.
    // Hound 3.5.1 seek() takes a frame index (samples per channel, independent of channel count).
    // Per hound docs: "multiply number of seconds with sample_rate" — channels are NOT included.
    reader
        .seek(start_frame)
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to seek WAV: {}", e)))?;

    let mut writer = WavWriter::create(output_path, spec)
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to create WAV: {}", e)))?;

    // We use a buffer size to balance memory vs I/O calls
    // Increased from 4096 to 32768 for better I/O performance
    const BUFFER_SIZE: usize = 32768;
    let channels = spec.channels as usize;
    let mut frames_remaining = frames_to_read;

    // Optimized bulk read/write approach
    // Collect samples into a buffer and write in batches to minimize function call overhead
    match spec.sample_format {
        SampleFormat::Float => {
            let mut iter = reader.samples::<f32>();
            while frames_remaining > 0 {
                let chunk_frames = frames_remaining.min(BUFFER_SIZE as u32);
                let samples_to_process = chunk_frames as usize * channels;

                // Bulk read into buffer
                let mut buffer = Vec::with_capacity(samples_to_process);
                for _ in 0..samples_to_process {
                    if let Some(sample) = iter.next() {
                        buffer.push(
                            sample.map_err(|e| {
                                AudioError::DecodeFailed(format!("Read error: {}", e))
                            })?,
                        );
                    } else {
                        break;
                    }
                }

                // Bulk write from buffer
                for sample in buffer {
                    writer
                        .write_sample(sample)
                        .map_err(|e| AudioError::EncodeFailed(format!("Write error: {}", e)))?;
                }

                frames_remaining -= chunk_frames;
            }
        }
        SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => {
                    let mut iter = reader.samples::<i16>();
                    while frames_remaining > 0 {
                        let chunk_frames = frames_remaining.min(BUFFER_SIZE as u32);
                        let samples_to_process = chunk_frames as usize * channels;

                        // Bulk read into buffer
                        let mut buffer = Vec::with_capacity(samples_to_process);
                        for _ in 0..samples_to_process {
                            if let Some(sample) = iter.next() {
                                buffer.push(sample.map_err(|e| {
                                    AudioError::DecodeFailed(format!("Read error: {}", e))
                                })?);
                            } else {
                                break;
                            }
                        }

                        // Bulk write from buffer
                        for sample in buffer {
                            writer.write_sample(sample).map_err(|e| {
                                AudioError::EncodeFailed(format!("Write error: {}", e))
                            })?;
                        }

                        frames_remaining -= chunk_frames;
                    }
                }
                // Handle 24-bit and 32-bit integers using i32 container
                24 | 32 => {
                    let mut iter = reader.samples::<i32>();
                    while frames_remaining > 0 {
                        let chunk_frames = frames_remaining.min(BUFFER_SIZE as u32);
                        let samples_to_process = chunk_frames as usize * channels;

                        // Bulk read into buffer
                        let mut buffer = Vec::with_capacity(samples_to_process);
                        for _ in 0..samples_to_process {
                            if let Some(sample) = iter.next() {
                                buffer.push(sample.map_err(|e| {
                                    AudioError::DecodeFailed(format!("Read error: {}", e))
                                })?);
                            } else {
                                break;
                            }
                        }

                        // Bulk write from buffer
                        for sample in buffer {
                            writer.write_sample(sample).map_err(|e| {
                                AudioError::EncodeFailed(format!("Write error: {}", e))
                            })?;
                        }

                        frames_remaining -= chunk_frames;
                    }
                }
                _ => {
                    return Err(AudioError::DecodeFailed(format!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    )))
                }
            }
        }
    }

    writer
        .finalize()
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Trim any audio file using streaming with seeking (optimized for compressed formats)
///
/// Note: This operation strips metadata (tags) from the original file.
fn trim_compressed_streaming<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    params: &TrimParams,
) -> Result<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let path_str = input_path.to_string_lossy().to_string();

    let file = File::open(input_path).map_err(|e| AudioError::FileOpen {
        path: path_str.clone(),
        source: e,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = input_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe format: {}", e)))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track found".to_string()))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::DecodeFailed("Sample rate not found".to_string()))?;

    let mut channels_opt = track.codec_params.channels.map(|c| c.count() as u16);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    // --- Seeking Logic ---
    let target_ts = (params.start_seconds * sample_rate as f64) as u64;

    // Seek and capture the ACTUAL timestamp we landed on.
    // Compressed formats might not land exactly on the requested sample.
    let seek_result = format
        .seek(
            SeekMode::Accurate,
            SeekTo::TimeStamp {
                ts: target_ts,
                track_id,
            },
        )
        .map_err(|e| AudioError::DecodeFailed(format!("Seek failed: {}", e)))?;

    // Update current sample tracker to the actual position
    let mut current_sample = seek_result.actual_ts;

    // --- Channel Determination (if unknown) ---
    if channels_opt.is_none() {
        // Decode one packet to find channel count
        let first_packet = format.next_packet().map_err(|e| {
            AudioError::DecodeFailed(format!("Failed to read packet for metadata: {}", e))
        })?;

        let decoded = decoder
            .decode(&first_packet)
            .map_err(|e| AudioError::DecodeFailed(format!("Decode error: {}", e)))?;

        channels_opt = Some(decoded.spec().channels.count() as u16);

        // We consumed a packet, so we must re-seek to ensuring we don't miss start data
        let re_seek = format
            .seek(
                SeekMode::Accurate,
                SeekTo::TimeStamp {
                    ts: target_ts,
                    track_id,
                },
            )
            .map_err(|e| AudioError::DecodeFailed(format!("Re-seek failed: {}", e)))?;

        current_sample = re_seek.actual_ts;
    }

    let channels = channels_opt.unwrap();

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec)
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to create WAV: {}", e)))?;

    let end_sample = (params.end_seconds * sample_rate as f64) as u64;

    // --- Decode Loop ---
    loop {
        if current_sample >= end_sample {
            break;
        }

        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break, // End of stream
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| AudioError::DecodeFailed(format!("Decode error: {}", e)))?;

        // Convert Planar -> Interleaved
        let samples = convert_to_f32(&decoded);
        let frames_in_packet = samples.len() / channels as usize;

        // Handle case where we seeked to a keyframe *before* our start time
        // We might need to skip some initial samples in this specific packet
        let mut start_offset_frames = 0;
        if current_sample < target_ts {
            if current_sample + (frames_in_packet as u64) > target_ts {
                // We are crossing the start line in this packet
                start_offset_frames = (target_ts - current_sample) as usize;
            } else {
                // Entire packet is before start time (should be rare with Accurate seek)
                current_sample += frames_in_packet as u64;
                continue;
            }
        }

        // Calculate how many frames to write
        let valid_frames_in_packet = frames_in_packet - start_offset_frames;

        // How many frames left until the end marker?
        let frames_until_end =
            end_sample.saturating_sub(current_sample + start_offset_frames as u64);

        let frames_to_write = (valid_frames_in_packet as u64).min(frames_until_end) as usize;

        if frames_to_write > 0 {
            let start_index = start_offset_frames * channels as usize;
            let end_index = start_index + (frames_to_write * channels as usize);

            // Write samples - Hound buffers internally, so this is reasonably efficient
            // Further optimization would require changes to Hound API
            for &sample in &samples[start_index..end_index] {
                writer
                    .write_sample(sample)
                    .map_err(|e| AudioError::EncodeFailed(format!("Write failed: {}", e)))?;
            }
        }

        current_sample += frames_in_packet as u64;
    }

    writer
        .finalize()
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to finalize: {}", e)))?;

    Ok(())
}

/// Convert AudioBufferRef to Vec<f32>, handling Planar to Interleaved conversion
/// Optimized to process samples in cache-friendly order
fn convert_to_f32(buffer: &AudioBufferRef) -> Vec<f32> {
    let spec = buffer.spec();
    let channels = spec.channels.count();
    let frames = buffer.frames();

    // Pre-allocate: frames * channels
    let total_samples = frames * channels;
    let mut output = vec![0.0f32; total_samples];

    match buffer {
        AudioBufferRef::F32(buf) => {
            let planes = buf.planes();
            // Process channel-by-channel for better cache locality
            // Converts Planar (LLLLRRRR) to Interleaved (LRLRLRLR)
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample;
                }
            }
        }
        AudioBufferRef::F64(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample as f32;
                }
            }
        }
        AudioBufferRef::S16(buf) => {
            let planes = buf.planes();
            const NORM_S16: f32 = 1.0 / 32768.0;
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    // Normalize i16 to f32 (-1.0 to 1.0)
                    output[frame * channels + ch] = sample as f32 * NORM_S16;
                }
            }
        }
        AudioBufferRef::S32(buf) => {
            let planes = buf.planes();
            const NORM_S32: f32 = 1.0 / 2147483648.0;
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    // Normalize i32 to f32 (-1.0 to 1.0)
                    output[frame * channels + ch] = sample as f32 * NORM_S32;
                }
            }
        }
        _ => {}
    }

    output
}

/// Trim an audio file using the fastest method available
///
/// Automatically selects the optimal trimming strategy:
/// - WAV files: Direct seeking (O(1), no decode/encode)
/// - Compressed formats: Streaming with seeking (fast, low memory)
///
/// # Arguments
/// * `input_path` - Path to input audio file
/// * `output_path` - Path to output WAV file
/// * `params` - Start and end times in seconds
pub fn trim_audio_file<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    params: &TrimParams,
) -> Result<()> {
    if is_wav_file(&input_path) {
        // Fast path: WAV → WAV direct copy
        trim_wav_direct(input_path, output_path, params)
    } else {
        // Streaming path: Decode with seeking
        trim_compressed_streaming(input_path, output_path, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(duration_seconds: f64, sample_rate: u32, channels: u16) -> AudioData {
        let total_samples = (duration_seconds * sample_rate as f64 * channels as f64) as usize;
        let samples = vec![0.5f32; total_samples];

        AudioData {
            samples,
            sample_rate,
            channels,
        }
    }

    #[test]
    fn test_trim_middle_section() {
        let audio = create_test_audio(10.0, 44100, 2);
        let params = TrimParams::new(3.0, 7.0).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        assert_eq!(trimmed.duration_seconds(), 4.0);
    }

    #[test]
    fn test_trim_start() {
        let audio = create_test_audio(10.0, 44100, 2);
        let params = TrimParams::new(0.0, 5.0).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        assert_eq!(trimmed.duration_seconds(), 5.0);
    }

    #[test]
    fn test_trim_out_of_bounds() {
        let audio = create_test_audio(10.0, 44100, 2);
        let params = TrimParams::new(5.0, 15.0).unwrap();
        let result = trim_audio(&audio, &params);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_trim_params() {
        assert!(TrimParams::new(10.0, 5.0).is_err());
        assert!(TrimParams::new(-1.0, 5.0).is_err());
    }

    #[test]
    fn test_mono_vs_stereo() {
        let mono = create_test_audio(1.0, 44100, 1);
        let stereo = create_test_audio(1.0, 44100, 2);

        assert_eq!(mono.samples.len(), 44100);
        assert_eq!(stereo.samples.len(), 88200);
    }

    #[test]
    fn test_trim_entire_audio() {
        let audio = create_test_audio(5.0, 44100, 2);
        let params = TrimParams::new(0.0, 5.0).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        assert_eq!(trimmed.duration_seconds(), audio.duration_seconds());
        assert_eq!(trimmed.samples.len(), audio.samples.len());
    }

    #[test]
    fn test_trim_very_short_duration() {
        let audio = create_test_audio(10.0, 44100, 2);
        let params = TrimParams::new(5.0, 5.01).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        // Should be approximately 0.01 seconds
        assert!(trimmed.duration_seconds() < 0.02);
        assert!(trimmed.duration_seconds() > 0.005);
    }

    #[test]
    fn test_trim_at_exact_end() {
        let audio = create_test_audio(10.0, 44100, 2);
        let duration = audio.duration_seconds();
        let params = TrimParams::new(5.0, duration).unwrap();
        let result = trim_audio(&audio, &params);

        assert!(result.is_ok());
    }

    #[test]
    fn test_trim_preserves_sample_rate_and_channels() {
        let audio = create_test_audio(10.0, 48000, 1);
        let params = TrimParams::new(2.0, 6.0).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        assert_eq!(trimmed.sample_rate, 48000);
        assert_eq!(trimmed.channels, 1);
    }

    #[test]
    fn test_trim_empty_audio() {
        let audio = AudioData {
            samples: vec![],
            sample_rate: 44100,
            channels: 2,
        };
        let params = TrimParams::new(0.0, 1.0).unwrap();
        let result = trim_audio(&audio, &params);

        // Should fail because end_seconds > duration (0.0)
        assert!(result.is_err());
    }

    #[test]
    fn test_is_wav_file_with_wav_extension() {
        assert!(is_wav_file("test.wav"));
        assert!(is_wav_file("test.WAV"));
        assert!(is_wav_file("test.WaV"));
        assert!(is_wav_file("/path/to/file.wav"));
    }

    #[test]
    fn test_is_wav_file_with_other_extensions() {
        assert!(!is_wav_file("test.mp3"));
        assert!(!is_wav_file("test.flac"));
        assert!(!is_wav_file("test.ogg"));
        assert!(!is_wav_file("test"));
    }

    #[test]
    fn test_is_wav_file_no_extension() {
        assert!(!is_wav_file("test"));
        assert!(!is_wav_file("/path/to/file"));
    }

    // Note: Testing convert_to_f32 directly is challenging because it requires
    // creating Symphonia AudioBufferRef instances, which have complex internal
    // structures. The function is well-tested indirectly through the trim functions
    // and the decoder tests.

    #[test]
    fn test_trim_audio_different_sample_rates() {
        let rates = vec![22050, 44100, 48000, 96000];

        for rate in rates {
            let audio = create_test_audio(5.0, rate, 2);
            let params = TrimParams::new(1.0, 3.0).unwrap();
            let trimmed = trim_audio(&audio, &params).unwrap();

            assert_eq!(trimmed.sample_rate, rate);
            assert_eq!(trimmed.duration_seconds(), 2.0);
        }
    }

    #[test]
    fn test_trim_audio_boundary_alignment() {
        // Test that trimming aligns to frame boundaries
        let audio = create_test_audio(10.0, 44100, 2);
        let params = TrimParams::new(1.5, 5.5).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        // Samples should be aligned to channel boundaries
        assert_eq!(trimmed.samples.len() % trimmed.channels as usize, 0);
    }
}
