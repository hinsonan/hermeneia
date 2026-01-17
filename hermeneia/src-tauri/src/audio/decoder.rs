// src-tauri/src/audio/decoder.rs

use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;

use crate::audio::types::{AudioData, AudioInfo};
use crate::error::{AudioError, Result};

/// Decodes an audio file to PCM samples in memory
/// 
/// Supports: MP3, FLAC, WAV, OGG Vorbis, AAC, and more via symphonia
/// 
/// # Arguments
/// * `path` - Path to the audio file
/// 
/// # Returns
/// AudioData containing all decoded PCM samples
/// 
/// # Example
/// ```no_run
/// use hermeneia_lib::audio::{decode_audio_file, AudioData};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let audio = decode_audio_file("sermon.mp3")?;
/// println!("Loaded {} seconds of audio", audio.duration_seconds());
/// println!("Sample rate: {} Hz", audio.sample_rate);
/// println!("Channels: {}", audio.channels);
/// # Ok(())
/// # }
/// ```
pub fn decode_audio_file<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let path = path.as_ref();
    let path_str = path.to_string_lossy().to_string();

    // Open the file
    let file = File::open(path).map_err(|e| AudioError::FileOpen {
        path: path_str.clone(),
        source: e,
    })?;

    // Create a media source stream (buffered reader)
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create a hint to help symphonia detect the format
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    // Probe the media source to detect format
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe format: {}", e)))?;

    let mut format = probed.format;

    // Find the default audio track (skip video/subtitle tracks)
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track found in file".to_string()))?;

    let track_id = track.id;

    // Extract audio parameters
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::DecodeFailed("Sample rate not found".to_string()))?;

    // Try to get channels from metadata, but it may not be available for some MP3s
    let mut channels_opt = track.codec_params.channels.map(|c| c.count() as u16);

    // Create decoder for this track
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    // Decode all packets into a sample buffer
    let mut samples = Vec::new();

    // If channels not in metadata, decode first packet to get channel info
    if channels_opt.is_none() {
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(_) => return Err(AudioError::DecodeFailed("Could not read first packet to determine channels".to_string())),
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = decoder
                .decode(&packet)
                .map_err(|e| AudioError::DecodeFailed(format!("Decode error on first packet: {}", e)))?;

            // Get channel count from decoded audio
            let ch = match &decoded {
                AudioBufferRef::F32(buf) => buf.spec().channels.count(),
                AudioBufferRef::F64(buf) => buf.spec().channels.count(),
                AudioBufferRef::S8(buf) => buf.spec().channels.count(),
                AudioBufferRef::S16(buf) => buf.spec().channels.count(),
                AudioBufferRef::S24(buf) => buf.spec().channels.count(),
                AudioBufferRef::S32(buf) => buf.spec().channels.count(),
                AudioBufferRef::U8(buf) => buf.spec().channels.count(),
                AudioBufferRef::U16(buf) => buf.spec().channels.count(),
                AudioBufferRef::U24(buf) => buf.spec().channels.count(),
                AudioBufferRef::U32(buf) => buf.spec().channels.count(),
            } as u16;

            channels_opt = Some(ch);

            // Convert this first packet to samples
            convert_audio_buffer_to_f32(&decoded, &mut samples);
            break;
        }
    }

    let channels = channels_opt.ok_or_else(|| AudioError::DecodeFailed("Could not determine channel count".to_string()))?;

    loop {
        // Get next packet
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break, // End of stream
        };

        // Skip packets from other tracks (e.g., video, album art)
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| AudioError::DecodeFailed(format!("Decode error: {}", e)))?;

        // Convert decoded audio to f32 samples
        convert_audio_buffer_to_f32(&decoded, &mut samples);
    }

    Ok(AudioData {
        samples,
        sample_rate,
        channels,
    })
}

/// Get audio file metadata without decoding all samples
/// 
/// Much faster than decode_audio_file() for just getting duration/info
/// 
/// # Example
/// ```no_run
/// use hermeneia_lib::audio::get_audio_info;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let info = get_audio_info("sermon.mp3")?;
/// println!("Duration: {:.2} minutes", info.duration_seconds / 60.0);
/// println!("Format: {}", info.format);
/// # Ok(())
/// # }
/// ```
pub fn get_audio_info<P: AsRef<Path>>(path: P) -> Result<AudioInfo> {
    let path = path.as_ref();
    let path_str = path.to_string_lossy().to_string();

    let file = File::open(path).map_err(|e| AudioError::FileOpen {
        path: path_str.clone(),
        source: e,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe: {}", e)))?;

    let format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track".to_string()))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(0);

    // Calculate duration from frame count
    let duration_seconds = if let (Some(n_frames), Some(sr)) =
        (track.codec_params.n_frames, track.codec_params.sample_rate)
    {
        n_frames as f64 / sr as f64
    } else {
        0.0
    };

    Ok(AudioInfo {
        duration_seconds,
        sample_rate,
        channels,
        format: format!("{:?}", track.codec_params.codec),
        bit_depth: track.codec_params.bits_per_sample.map(|b| b as u16),
    })
}

/// Convert symphonia's AudioBufferRef to Vec<f32> in interleaved format
///
/// Handles all sample formats (u8, i16, i32, f32, f64) and converts to f32.
/// Symphonia stores audio in planar format (separate array per channel),
/// but we need interleaved format [L, R, L, R, ...] for downstream processing.
fn convert_audio_buffer_to_f32(buffer: &AudioBufferRef, output: &mut Vec<f32>) {
    match buffer {
        // Already f32 - interleave from planes
        AudioBufferRef::F32(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| s);
        }

        // Convert f64 → f32
        AudioBufferRef::F64(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| s as f32);
        }

        // Convert signed integers to f32 in range [-1.0, 1.0]
        AudioBufferRef::S8(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| s as f32 / 128.0);
        }
        AudioBufferRef::S16(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| s as f32 / 32768.0);
        }
        AudioBufferRef::S24(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| s.inner() as f32 / 8388608.0);
        }
        AudioBufferRef::S32(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| s as f32 / 2147483648.0);
        }

        // Convert unsigned integers to f32
        AudioBufferRef::U8(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| (s as f32 - 128.0) / 128.0);
        }
        AudioBufferRef::U16(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| (s as f32 - 32768.0) / 32768.0);
        }
        AudioBufferRef::U24(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| (s.inner() as f32 - 8388608.0) / 8388608.0);
        }
        AudioBufferRef::U32(buf) => {
            interleave_planes(buf.planes().planes(), output, |&s| (s as f32 - 2147483648.0) / 2147483648.0);
        }
    }
}

/// Interleave samples from planar format to interleaved format
///
/// Converts from [L0, L1, L2, ...], [R0, R1, R2, ...] (planar)
/// to [L0, R0, L1, R1, L2, R2, ...] (interleaved)
fn interleave_planes<T, F>(planes: &[&[T]], output: &mut Vec<f32>, convert: F)
where
    F: Fn(&T) -> f32,
{
    if planes.is_empty() {
        return;
    }

    // For mono, just convert directly
    if planes.len() == 1 {
        output.extend(planes[0].iter().map(&convert));
        return;
    }

    // For multi-channel, interleave the samples
    let num_samples = planes[0].len();
    output.reserve(num_samples * planes.len());

    for i in 0..num_samples {
        for plane in planes {
            if i < plane.len() {
                output.push(convert(&plane[i]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavWriter, WavSpec, SampleFormat};
    use std::io::Write;

    /// Helper function to create a simple WAV file for testing
    fn create_test_wav_file(
        path: &str,
        duration_seconds: f64,
        sample_rate: u32,
        channels: u16,
    ) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec).expect("Failed to create WAV writer");
        let num_samples = (duration_seconds * sample_rate as f64 * channels as f64) as usize;

        // Write a simple sine wave
        for i in 0..num_samples {
            let sample = ((i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 16384.0) as i16;
            writer.write_sample(sample).expect("Failed to write sample");
        }

        writer.finalize().expect("Failed to finalize WAV");
    }

    #[test]
    fn test_decode_audio_file_wav() {
        let temp_file = "/tmp/test_decode.wav";
        create_test_wav_file(temp_file, 1.0, 44100, 2);

        let audio = decode_audio_file(temp_file).unwrap();

        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
        assert!((audio.duration_seconds() - 1.0).abs() < 0.01);
        assert!(!audio.samples.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_decode_audio_file_mono() {
        let temp_file = "/tmp/test_decode_mono.wav";
        create_test_wav_file(temp_file, 0.5, 48000, 1);

        let audio = decode_audio_file(temp_file).unwrap();

        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels, 1);
        assert!((audio.duration_seconds() - 0.5).abs() < 0.01);

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_decode_audio_file_not_found() {
        let result = decode_audio_file("/nonexistent/file.wav");
        assert!(result.is_err());

        match result.unwrap_err() {
            AudioError::FileOpen { .. } => {}, // Expected error
            _ => panic!("Expected FileOpen error"),
        }
    }

    #[test]
    fn test_decode_audio_file_invalid_format() {
        // Create a file with invalid content
        let temp_file = "/tmp/test_invalid.wav";
        let mut file = std::fs::File::create(temp_file).unwrap();
        file.write_all(b"This is not a valid audio file").unwrap();
        drop(file);

        let result = decode_audio_file(temp_file);
        assert!(result.is_err());

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_get_audio_info_wav() {
        let temp_file = "/tmp/test_info.wav";
        create_test_wav_file(temp_file, 2.5, 44100, 2);

        let info = get_audio_info(temp_file).unwrap();

        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert!((info.duration_seconds - 2.5).abs() < 0.1);
        assert_eq!(info.bit_depth, Some(16));

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_get_audio_info_mono() {
        let temp_file = "/tmp/test_info_mono.wav";
        create_test_wav_file(temp_file, 1.0, 48000, 1);

        let info = get_audio_info(temp_file).unwrap();

        assert_eq!(info.sample_rate, 48000);
        assert_eq!(info.channels, 1);

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_get_audio_info_file_not_found() {
        let result = get_audio_info("/nonexistent/file.wav");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_different_sample_rates() {
        let rates = vec![(22050, "/tmp/test_22k.wav"), (44100, "/tmp/test_44k.wav"), (48000, "/tmp/test_48k.wav")];

        for (rate, path) in rates {
            create_test_wav_file(path, 0.5, rate, 2);
            let audio = decode_audio_file(path).unwrap();
            assert_eq!(audio.sample_rate, rate);

            // Cleanup
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn test_decode_and_verify_samples_in_range() {
        let temp_file = "/tmp/test_sample_range.wav";
        create_test_wav_file(temp_file, 0.1, 44100, 2);

        let audio = decode_audio_file(temp_file).unwrap();

        // All samples should be in valid f32 range [-1.0, 1.0]
        for &sample in &audio.samples {
            assert!(sample >= -1.0 && sample <= 1.0, "Sample {} out of range", sample);
        }

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_audio_data_calculations() {
        let temp_file = "/tmp/test_calculations.wav";
        create_test_wav_file(temp_file, 3.0, 44100, 2);

        let audio = decode_audio_file(temp_file).unwrap();

        // Verify calculations
        let expected_samples = 3.0 * 44100.0 * 2.0; // duration * rate * channels
        assert!((audio.samples.len() as f64 - expected_samples).abs() < 1000.0);

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_decode_very_short_audio() {
        let temp_file = "/tmp/test_short.wav";
        create_test_wav_file(temp_file, 0.01, 44100, 2); // 10ms

        let audio = decode_audio_file(temp_file).unwrap();

        assert!(audio.duration_seconds() < 0.02);
        assert!(!audio.samples.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_get_audio_info_vs_decode_consistency() {
        let temp_file = "/tmp/test_consistency.wav";
        create_test_wav_file(temp_file, 1.5, 44100, 2);

        let info = get_audio_info(temp_file).unwrap();
        let audio = decode_audio_file(temp_file).unwrap();

        // Info and decoded data should match
        assert_eq!(info.sample_rate, audio.sample_rate);
        assert_eq!(info.channels, audio.channels);
        assert!((info.duration_seconds - audio.duration_seconds()).abs() < 0.1);

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}