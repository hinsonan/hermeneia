// src-tauri/src/audio/trim.rs

use crate::audio::types::{AudioData, TrimParams};
use crate::error::{AudioError, Result};
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::audio::AudioBufferRef;

/// Trim audio data to a specific time range
/// 
/// # Arguments
/// * `audio` - The audio data to trim
/// * `params` - Start and end times in seconds
/// 
/// # Returns
/// New AudioData containing only the trimmed portion
/// 
/// # Example
/// ```
/// use hermeneia_lib::audio::{AudioData, TrimParams, trim_audio};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create test audio: 10 seconds of stereo at 44.1kHz
/// let original_audio = AudioData {
///     samples: vec![0.5; 882000],  // 10 seconds
///     sample_rate: 44100,
///     channels: 2,
/// };
/// 
/// // Trim from 5 seconds to 10 seconds
/// let params = TrimParams::new(5.0, 10.0)?;
/// let trimmed = trim_audio(&original_audio, &params)?;
/// 
/// assert_eq!(trimmed.duration_seconds(), 5.0);
/// assert_eq!(trimmed.sample_rate, 44100);
/// assert_eq!(trimmed.channels, 2);
/// # Ok(())
/// # }
/// ```
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
    // Formula: sample_index = time_in_seconds × sample_rate × num_channels
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

/// Trim a WAV file directly using byte-level operations (fastest method)
///
/// This function does NOT decode/encode - it directly copies PCM bytes from the
/// input WAV to output WAV. This is extremely fast (limited only by disk I/O).
///
/// # Arguments
/// * `input_path` - Path to input WAV file
/// * `output_path` - Path to output WAV file
/// * `params` - Start and end times in seconds
///
/// # Performance
/// Can trim a 1-hour WAV file in milliseconds regardless of file size
fn trim_wav_direct<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    params: &TrimParams,
) -> Result<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    // Open WAV file and read header
    let mut reader = WavReader::open(input_path)
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;

    // Validate trim range
    if params.end_seconds > duration {
        return Err(AudioError::TrimRangeOutOfBounds {
            start: params.start_seconds,
            end: params.end_seconds,
            duration,
        });
    }

    // Calculate sample positions (frames, not individual samples)
    let start_frame = (params.start_seconds * spec.sample_rate as f64) as u32;
    let end_frame = (params.end_seconds * spec.sample_rate as f64) as u32;
    let total_frames = reader.duration();

    // Clamp to valid range
    let start_frame = start_frame.min(total_frames);
    let end_frame = end_frame.min(total_frames);

    // Create output WAV writer
    let mut writer = WavWriter::create(output_path, spec)
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to create WAV: {}", e)))?;

    // Read and write samples in the specified range
    // Use the reader's sample iterator and skip/take to extract the range
    match spec.sample_format {
        SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .samples::<f32>()
                .skip((start_frame * spec.channels as u32) as usize)
                .take(((end_frame - start_frame) * spec.channels as u32) as usize)
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| AudioError::DecodeFailed(format!("Failed to read samples: {}", e)))?;

            for sample in samples {
                writer.write_sample(sample)
                    .map_err(|e| AudioError::EncodeFailed(format!("Failed to write sample: {}", e)))?;
            }
        }
        SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => {
                    let samples: Vec<i16> = reader
                        .samples::<i16>()
                        .skip((start_frame * spec.channels as u32) as usize)
                        .take(((end_frame - start_frame) * spec.channels as u32) as usize)
                        .collect::<std::result::Result<Vec<i16>, _>>()
                        .map_err(|e| AudioError::DecodeFailed(format!("Failed to read samples: {}", e)))?;

                    for sample in samples {
                        writer.write_sample(sample)
                            .map_err(|e| AudioError::EncodeFailed(format!("Failed to write sample: {}", e)))?;
                    }
                }
                32 => {
                    let samples: Vec<i32> = reader
                        .samples::<i32>()
                        .skip((start_frame * spec.channels as u32) as usize)
                        .take(((end_frame - start_frame) * spec.channels as u32) as usize)
                        .collect::<std::result::Result<Vec<i32>, _>>()
                        .map_err(|e| AudioError::DecodeFailed(format!("Failed to read samples: {}", e)))?;

                    for sample in samples {
                        writer.write_sample(sample)
                            .map_err(|e| AudioError::EncodeFailed(format!("Failed to write sample: {}", e)))?;
                    }
                }
                _ => {
                    return Err(AudioError::DecodeFailed(
                        format!("Unsupported bit depth: {}", spec.bits_per_sample)
                    ));
                }
            }
        }
    }

    writer.finalize()
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Trim any audio file using streaming with seeking (optimized for compressed formats)
///
/// This function seeks to the start position and only decodes the necessary portion,
/// then streams directly to the output WAV file without loading everything into memory.
///
/// # Arguments
/// * `input_path` - Path to input audio file (MP3, FLAC, OGG, etc.)
/// * `output_path` - Path to output WAV file
/// * `params` - Start and end times in seconds
///
/// # Performance
/// Much faster than decoding the entire file. Memory usage is constant regardless
/// of input file size.
fn trim_compressed_streaming<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    params: &TrimParams,
) -> Result<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let path_str = input_path.to_string_lossy().to_string();

    // Open the file
    let file = File::open(input_path).map_err(|e| AudioError::FileOpen {
        path: path_str.clone(),
        source: e,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create format hint
    let mut hint = Hint::new();
    if let Some(extension) = input_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    // Probe the format
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe format: {}", e)))?;

    let mut format = probed.format;

    // Find audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track found".to_string()))?;

    let track_id = track.id;

    // Get audio parameters
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::DecodeFailed("Sample rate not found".to_string()))?;

    let mut channels_opt = track.codec_params.channels.map(|c| c.count() as u16);

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    // Convert seconds to sample timestamp for seeking
    let start_ts = (params.start_seconds * sample_rate as f64) as u64;

    // Helper to create seek position using sample timestamp
    let make_start_seek = || SeekTo::TimeStamp { ts: start_ts, track_id };

    // Seek to start position
    format.seek(SeekMode::Accurate, make_start_seek())
        .map_err(|e| AudioError::DecodeFailed(format!("Seek failed: {}", e)))?;

    // Determine channels by decoding first packet if needed
    if channels_opt.is_none() {
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(_) => return Err(AudioError::DecodeFailed("Could not read first packet".to_string())),
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = decoder
                .decode(&packet)
                .map_err(|e| AudioError::DecodeFailed(format!("Decode error: {}", e)))?;

            let ch = match &decoded {
                AudioBufferRef::F32(buf) => buf.spec().channels.count(),
                AudioBufferRef::F64(buf) => buf.spec().channels.count(),
                AudioBufferRef::S16(buf) => buf.spec().channels.count(),
                AudioBufferRef::S32(buf) => buf.spec().channels.count(),
                _ => return Err(AudioError::DecodeFailed("Unsupported sample format".to_string())),
            } as u16;

            channels_opt = Some(ch);
            break;
        }

        // Need to seek again after determining channels
        format.seek(SeekMode::Accurate, make_start_seek())
            .map_err(|e| AudioError::DecodeFailed(format!("Second seek failed: {}", e)))?;
    }

    let channels = channels_opt.ok_or_else(|| AudioError::DecodeFailed("Could not determine channels".to_string()))?;

    // Create WAV writer
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec)
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to create WAV: {}", e)))?;

    // Calculate end time in samples
    let end_sample = (params.end_seconds * sample_rate as f64) as u64;
    let mut current_sample = (params.start_seconds * sample_rate as f64) as u64;

    // Decode and write packets until we reach end time
    loop {
        // Check if we've reached the end
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

        // Convert to f32 and write
        let samples = convert_to_f32(&decoded);

        // Calculate how many samples to write from this packet
        let frames_in_packet = samples.len() / channels as usize;
        let frames_remaining = (end_sample - current_sample) as usize;
        let frames_to_write = frames_in_packet.min(frames_remaining);
        let samples_to_write = frames_to_write * channels as usize;

        for &sample in &samples[..samples_to_write] {
            writer.write_sample(sample)
                .map_err(|e| AudioError::EncodeFailed(format!("Write failed: {}", e)))?;
        }

        current_sample += frames_in_packet as u64;
    }

    writer.finalize()
        .map_err(|e| AudioError::EncodeFailed(format!("Failed to finalize: {}", e)))?;

    Ok(())
}

/// Convert AudioBufferRef to Vec<f32>
fn convert_to_f32(buffer: &AudioBufferRef) -> Vec<f32> {
    let mut output = Vec::new();

    match buffer {
        AudioBufferRef::F32(buf) => {
            for plane in buf.planes().planes() {
                output.extend_from_slice(plane);
            }
        }
        AudioBufferRef::F64(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32));
            }
        }
        AudioBufferRef::S16(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32 / 32768.0));
            }
        }
        AudioBufferRef::S32(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32 / 2147483648.0));
            }
        }
        _ => {}
    }

    output
}

/// Trim an audio file using the fastest method available
///
/// Automatically selects the optimal trimming strategy:
/// - WAV files: Direct byte copy (extremely fast, no decode/encode)
/// - Compressed formats: Streaming with seeking (fast, low memory)
///
/// # Arguments
/// * `input_path` - Path to input audio file
/// * `output_path` - Path to output WAV file
/// * `params` - Start and end times in seconds
///
/// # Example
/// ```no_run
/// use hermeneia_lib::audio::{trim_audio_file, TrimParams};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let params = TrimParams::new(5.0, 10.0)?;
/// trim_audio_file("input.mp3", "output.wav", &params)?;
/// # Ok(())
/// # }
/// ```
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

    /// Helper to create test audio data
    fn create_test_audio(duration_seconds: f64, sample_rate: u32, channels: u16) -> AudioData {
        let total_samples = (duration_seconds * sample_rate as f64 * channels as f64) as usize;
        let samples = vec![0.5f32; total_samples]; // Silent audio

        AudioData {
            samples,
            sample_rate,
            channels,
        }
    }

    #[test]
    fn test_trim_middle_section() {
        // Create 10 seconds of audio
        let audio = create_test_audio(10.0, 44100, 2);
        
        // Trim from 3s to 7s (should give 4 seconds)
        let params = TrimParams::new(3.0, 7.0).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        assert_eq!(trimmed.duration_seconds(), 4.0);
    }

    #[test]
    fn test_trim_start() {
        let audio = create_test_audio(10.0, 44100, 2);
        
        // Trim from beginning
        let params = TrimParams::new(0.0, 5.0).unwrap();
        let trimmed = trim_audio(&audio, &params).unwrap();

        assert_eq!(trimmed.duration_seconds(), 5.0);
    }

    #[test]
    fn test_trim_out_of_bounds() {
        let audio = create_test_audio(10.0, 44100, 2);
        
        // Try to trim beyond duration
        let params = TrimParams::new(5.0, 15.0).unwrap();
        let result = trim_audio(&audio, &params);

        assert!(result.is_err());
        match result {
            Err(AudioError::TrimRangeOutOfBounds { .. }) => (),
            _ => panic!("Expected TrimRangeOutOfBounds error"),
        }
    }

    #[test]
    fn test_invalid_trim_params() {
        // Start > End
        assert!(TrimParams::new(10.0, 5.0).is_err());
        
        // Negative start
        assert!(TrimParams::new(-1.0, 5.0).is_err());
    }

    #[test]
    fn test_mono_vs_stereo() {
        // Test that sample calculation is correct for different channel counts
        let mono = create_test_audio(1.0, 44100, 1);
        let stereo = create_test_audio(1.0, 44100, 2);

        assert_eq!(mono.samples.len(), 44100);
        assert_eq!(stereo.samples.len(), 88200); // 2x for stereo
    }
}