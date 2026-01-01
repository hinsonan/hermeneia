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

/// Convert symphonia's AudioBufferRef to Vec<f32>
/// 
/// Handles all sample formats (u8, i16, i32, f32, f64) and converts to f32
fn convert_audio_buffer_to_f32(buffer: &AudioBufferRef, output: &mut Vec<f32>) {
    match buffer {
        // Already f32 - just copy
        AudioBufferRef::F32(buf) => {
            for plane in buf.planes().planes() {
                output.extend_from_slice(plane);
            }
        }
        
        // Convert f64 → f32
        AudioBufferRef::F64(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32));
            }
        }
        
        // Convert signed integers to f32 in range [-1.0, 1.0]
        AudioBufferRef::S8(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32 / 128.0));
            }
        }
        AudioBufferRef::S16(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32 / 32768.0));
            }
        }
        AudioBufferRef::S24(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s.inner() as f32 / 8388608.0));
            }
        }
        AudioBufferRef::S32(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| s as f32 / 2147483648.0));
            }
        }
        
        // Convert unsigned integers to f32
        AudioBufferRef::U8(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| (s as f32 - 128.0) / 128.0));
            }
        }
        AudioBufferRef::U16(buf) => {
            for plane in buf.planes().planes() {
                output.extend(plane.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
            }
        }
        AudioBufferRef::U24(buf) => {
            for plane in buf.planes().planes() {
                output.extend(
                    plane
                        .iter()
                        .map(|&s| (s.inner() as f32 - 8388608.0) / 8388608.0),
                );
            }
        }
        AudioBufferRef::U32(buf) => {
            for plane in buf.planes().planes() {
                output.extend(
                    plane
                        .iter()
                        .map(|&s| (s as f32 - 2147483648.0) / 2147483648.0),
                );
            }
        }
    }
}