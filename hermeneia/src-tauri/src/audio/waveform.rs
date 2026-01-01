// src-tauri/src/audio/waveform.rs

use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;

use crate::audio::types::WaveformPeaks;
use crate::error::{AudioError, Result};

/// Extract waveform peaks from an audio file for visualization
///
/// This function efficiently processes large audio files (up to 4+ hours)
/// by streaming through the file and calculating min/max peaks for
/// a fixed number of segments.
///
/// # Arguments
/// * `path` - Path to the audio file
/// * `num_peaks` - Number of peak pairs to extract (default: 2000 if None)
///
/// # Returns
/// WaveformPeaks containing min/max amplitude data for visualization
///
/// # Performance
/// - Memory efficient: Only stores peak data, not all samples
/// - 4-hour audio: ~16KB of peak data vs ~1.27GB of raw samples
/// - Processes files in a single streaming pass
///
/// # Example
/// ```no_run
/// use hermeneia_lib::audio::extract_waveform_peaks;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Extract 2000 peaks for waveform display
/// let peaks = extract_waveform_peaks("long_sermon.mp3", Some(2000))?;
/// println!("Peaks: {}, Duration: {:.1}s", peaks.num_peaks, peaks.duration_seconds);
/// # Ok(())
/// # }
/// ```
pub fn extract_waveform_peaks<P: AsRef<Path>>(
    path: P,
    num_peaks: Option<usize>,
) -> Result<WaveformPeaks> {
    let path = path.as_ref();
    let path_str = path.to_string_lossy().to_string();
    let num_peaks = num_peaks.unwrap_or(2000);

    if num_peaks == 0 {
        return Err(AudioError::InvalidTrimParams(
            "num_peaks must be greater than 0".to_string(),
        ));
    }

    // Open the file
    let file = File::open(path).map_err(|e| AudioError::FileOpen {
        path: path_str.clone(),
        source: e,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create format hint
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    // Probe the file
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe: {}", e)))?;

    let mut format = probed.format;

    // Find audio track and extract needed parameters
    let (track_id, sample_rate, channels_opt, codec_params, total_frames_opt) = {
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

        // Try to get channels from metadata, but it may not be available for some MP3s
        let channels_opt = track.codec_params.channels.map(|c| c.count() as u16);
        let total_frames_opt = track.codec_params.n_frames;
        let codec_params = track.codec_params.clone();

        (track_id, sample_rate, channels_opt, codec_params, total_frames_opt)
    };

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    // If channels not in metadata, decode first packet to get channel info
    let (channels, first_packet_decoded) = if let Some(ch) = channels_opt {
        (ch, None)
    } else {
        // Decode first packet to determine channels
        let (ch, decoded) = loop {
            let packet = format
                .next_packet()
                .map_err(|e| AudioError::DecodeFailed(format!("Failed to read first packet: {}", e)))?;

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = decoder
                .decode(&packet)
                .map_err(|e| AudioError::DecodeFailed(format!("Failed to decode first packet: {}", e)))?;

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

            break (ch, decoded);
        };
        (ch, Some(decoded))
    };

    // Calculate total frames and duration
    // If we don't have frame count, we need to count during streaming
    // For now, return a better error message
    let total_frames = total_frames_opt.ok_or_else(|| {
        AudioError::DecodeFailed(
            "Frame count not available in metadata. This file may need a full decode to determine length.".to_string()
        )
    })?;

    let duration_seconds = total_frames as f64 / sample_rate as f64;

    // Initialize peak buffers
    let mut min_peaks = vec![f32::MAX; num_peaks];
    let mut max_peaks = vec![f32::MIN; num_peaks];

    // Calculate how many frames belong to each peak segment
    let frames_per_peak = total_frames as f64 / num_peaks as f64;

    // Track current frame position
    let mut current_frame: u64 = 0;

    // If we decoded the first packet to get channel info, process it for peaks now
    if let Some(ref decoded) = first_packet_decoded {
        process_packet_peaks(
            decoded,
            channels,
            &mut current_frame,
            frames_per_peak,
            &mut min_peaks,
            &mut max_peaks,
        );
    }

    // Stream through packets and calculate peaks
    loop {
        // Get next packet
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break, // End of stream
        };

        // Skip non-audio tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode packet
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| AudioError::DecodeFailed(format!("Decode error: {}", e)))?;

        // Process samples from this packet
        process_packet_peaks(
            &decoded,
            channels,
            &mut current_frame,
            frames_per_peak,
            &mut min_peaks,
            &mut max_peaks,
        );
    }

    // Handle any peaks that didn't get set (shouldn't happen, but safety)
    for i in 0..num_peaks {
        if min_peaks[i] == f32::MAX {
            min_peaks[i] = 0.0;
        }
        if max_peaks[i] == f32::MIN {
            max_peaks[i] = 0.0;
        }
    }

    Ok(WaveformPeaks {
        min_peaks,
        max_peaks,
        num_peaks,
        duration_seconds,
        channels,
        sample_rate,
    })
}

/// Process a decoded packet and update peak values
///
/// Handles all sample formats and updates min/max peaks for the appropriate segments
fn process_packet_peaks(
    buffer: &AudioBufferRef,
    channels: u16,
    current_frame: &mut u64,
    frames_per_peak: f64,
    min_peaks: &mut [f32],
    max_peaks: &mut [f32],
) {
    // Convert buffer to f32 samples and process
    match buffer {
        AudioBufferRef::F32(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| s,
            );
        }
        AudioBufferRef::F64(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| s as f32,
            );
        }
        AudioBufferRef::S16(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| s as f32 / 32768.0,
            );
        }
        AudioBufferRef::S32(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| s as f32 / 2147483648.0,
            );
        }
        AudioBufferRef::S8(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| s as f32 / 128.0,
            );
        }
        AudioBufferRef::S24(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| s.inner() as f32 / 8388608.0,
            );
        }
        AudioBufferRef::U8(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| (s as f32 - 128.0) / 128.0,
            );
        }
        AudioBufferRef::U16(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| (s as f32 - 32768.0) / 32768.0,
            );
        }
        AudioBufferRef::U24(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| (s.inner() as f32 - 8388608.0) / 8388608.0,
            );
        }
        AudioBufferRef::U32(buf) => {
            process_samples_generic(
                buf.planes().planes(),
                channels,
                current_frame,
                frames_per_peak,
                min_peaks,
                max_peaks,
                |&s| (s as f32 - 2147483648.0) / 2147483648.0,
            );
        }
    }
}

/// Generic sample processor for any sample type
///
/// Processes planar audio data (separate channel planes) and updates peaks
fn process_samples_generic<T, F>(
    planes: &[&[T]],
    channels: u16,
    current_frame: &mut u64,
    frames_per_peak: f64,
    min_peaks: &mut [f32],
    max_peaks: &mut [f32],
    convert: F,
) where
    F: Fn(&T) -> f32,
{
    if planes.is_empty() {
        return;
    }

    let num_peaks = min_peaks.len();
    let frame_count = planes[0].len();

    // Iterate through frames (one sample per channel)
    for frame_idx in 0..frame_count {
        // Determine which peak segment this frame belongs to
        let peak_idx = (*current_frame as f64 / frames_per_peak) as usize;

        if peak_idx >= num_peaks {
            break; // Safety: don't overflow peak buffer
        }

        // Calculate min/max across all channels for this frame
        let mut frame_min = f32::MAX;
        let mut frame_max = f32::MIN;

        for channel in 0..channels as usize {
            if channel < planes.len() {
                let sample = convert(&planes[channel][frame_idx]);
                frame_min = frame_min.min(sample);
                frame_max = frame_max.max(sample);
            }
        }

        // Update peak values for this segment
        min_peaks[peak_idx] = min_peaks[peak_idx].min(frame_min);
        max_peaks[peak_idx] = max_peaks[peak_idx].max(frame_max);

        *current_frame += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::types::AudioData;
    use crate::audio::encoder::encode_wav;
    use std::path::PathBuf;

    /// Helper to create synthetic audio data for testing
    fn create_test_audio(duration_seconds: f64, sample_rate: u32, channels: u16) -> AudioData {
        let total_samples = (duration_seconds * sample_rate as f64 * channels as f64) as usize;
        let mut samples = Vec::with_capacity(total_samples);

        // Create a simple sine wave pattern for testing
        for i in 0..total_samples {
            let t = i as f32 / (sample_rate as f32 * channels as f32);
            let value = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5;
            samples.push(value);
        }

        AudioData {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Helper to create a temporary WAV file for testing
    /// Returns (path, cleanup_guard)
    fn create_test_wav_file(audio: &AudioData, name: &str) -> PathBuf {
        let temp_path = std::env::temp_dir().join(format!("hermeneia_test_{}.wav", name));

        // Encode to WAV
        encode_wav(audio, &temp_path).expect("Failed to encode WAV");

        temp_path
    }

    /// Helper to cleanup test file
    fn cleanup_test_file(path: &PathBuf) {
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_extract_peaks_validates_num_peaks() {
        let result = extract_waveform_peaks("nonexistent.mp3", Some(0));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("greater than 0"));
    }

    #[test]
    fn test_extract_peaks_basic() {
        // Create 1 second of test audio
        let audio = create_test_audio(1.0, 44100, 2);
        let temp_file = create_test_wav_file(&audio, "basic");

        // Extract 100 peaks
        let peaks = extract_waveform_peaks(&temp_file, Some(100))
            .expect("Failed to extract peaks");

        // Verify basic properties
        assert_eq!(peaks.num_peaks, 100);
        assert_eq!(peaks.min_peaks.len(), 100);
        assert_eq!(peaks.max_peaks.len(), 100);
        assert_eq!(peaks.channels, 2);
        assert_eq!(peaks.sample_rate, 44100);
        assert!((peaks.duration_seconds - 1.0).abs() < 0.01); // ~1 second

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_peaks_are_in_valid_range() {
        let audio = create_test_audio(2.0, 44100, 2);
        let temp_file = create_test_wav_file(&audio, "valid_range");

        let peaks = extract_waveform_peaks(&temp_file, Some(200))
            .expect("Failed to extract peaks");

        // All peaks should be in valid amplitude range [-1.0, 1.0]
        for &min in &peaks.min_peaks {
            assert!(min >= -1.0 && min <= 1.0, "Min peak out of range: {}", min);
        }

        for &max in &peaks.max_peaks {
            assert!(max >= -1.0 && max <= 1.0, "Max peak out of range: {}", max);
        }

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_min_always_less_than_or_equal_max() {
        let audio = create_test_audio(1.5, 44100, 2);
        let temp_file = create_test_wav_file(&audio, "min_max");

        let peaks = extract_waveform_peaks(&temp_file, Some(150))
            .expect("Failed to extract peaks");

        // For each segment, min should be <= max
        for i in 0..peaks.num_peaks {
            assert!(
                peaks.min_peaks[i] <= peaks.max_peaks[i],
                "Peak {}: min ({}) > max ({})",
                i,
                peaks.min_peaks[i],
                peaks.max_peaks[i]
            );
        }

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_different_num_peaks() {
        let audio = create_test_audio(3.0, 44100, 2);
        let temp_file = create_test_wav_file(&audio, "different_peaks");

        // Test with different peak counts
        for num_peaks in [10, 100, 500, 1000, 2000] {
            let peaks = extract_waveform_peaks(&temp_file, Some(num_peaks))
                .expect(&format!("Failed with {} peaks", num_peaks));

            assert_eq!(peaks.num_peaks, num_peaks);
            assert_eq!(peaks.min_peaks.len(), num_peaks);
            assert_eq!(peaks.max_peaks.len(), num_peaks);
        }

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_mono_audio() {
        // Test with mono audio instead of stereo
        let audio = create_test_audio(1.0, 44100, 1);
        let temp_file = create_test_wav_file(&audio, "mono");

        let peaks = extract_waveform_peaks(&temp_file, Some(100))
            .expect("Failed to extract peaks from mono audio");

        assert_eq!(peaks.channels, 1);
        assert_eq!(peaks.num_peaks, 100);

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_long_audio() {
        // Test with 30 seconds (simulating longer files)
        let audio = create_test_audio(30.0, 44100, 2);
        let temp_file = create_test_wav_file(&audio, "long");

        let peaks = extract_waveform_peaks(&temp_file, Some(2000))
            .expect("Failed to extract peaks from long audio");

        assert_eq!(peaks.num_peaks, 2000);
        assert!((peaks.duration_seconds - 30.0).abs() < 0.1);

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_default_num_peaks() {
        let audio = create_test_audio(1.0, 44100, 2);
        let temp_file = create_test_wav_file(&audio, "default");

        // Test with None (should default to 2000)
        let peaks = extract_waveform_peaks(&temp_file, None)
            .expect("Failed with default peaks");

        assert_eq!(peaks.num_peaks, 2000);

        cleanup_test_file(&temp_file);
    }

    #[test]
    fn test_file_not_found() {
        let result = extract_waveform_peaks("/nonexistent/path/audio.mp3", Some(100));
        assert!(result.is_err());
    }

    #[test]
    fn test_peaks_capture_amplitude_variation() {
        // Create audio with known amplitude pattern: silence -> loud -> silence
        let sample_rate = 44100;
        let channels = 1;
        let duration = 3.0;
        let total_samples = (duration * sample_rate as f64 * channels as f64) as usize;

        let mut samples = vec![0.0; total_samples];

        // Middle third is loud (0.8 amplitude)
        let third = total_samples / 3;
        for i in third..(2 * third) {
            samples[i] = 0.8;
        }

        let audio = AudioData {
            samples,
            sample_rate,
            channels,
        };

        let temp_file = create_test_wav_file(&audio, "amplitude_variation");

        // Extract 30 peaks (10 per second)
        let peaks = extract_waveform_peaks(&temp_file, Some(30))
            .expect("Failed to extract peaks");

        // First 10 peaks should be near 0 (silence)
        for i in 0..10 {
            assert!(
                peaks.max_peaks[i].abs() < 0.1,
                "Expected silence in first section, got {}",
                peaks.max_peaks[i]
            );
        }

        // Middle 10 peaks should be around 0.8 (loud)
        for i in 10..20 {
            assert!(
                peaks.max_peaks[i] > 0.7,
                "Expected loud signal in middle section, got {}",
                peaks.max_peaks[i]
            );
        }

        // Last 10 peaks should be near 0 (silence)
        for i in 20..30 {
            assert!(
                peaks.max_peaks[i].abs() < 0.1,
                "Expected silence in last section, got {}",
                peaks.max_peaks[i]
            );
        }

        cleanup_test_file(&temp_file);
    }
}
