// src-tauri/src/audio/waveform.rs

use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::Time;
use std::fs::File;
use std::path::Path;

use crate::audio::types::WaveformPeaks;
use crate::error::{AudioError, Result};

/// Calculate optimal frame skip for adaptive sampling
///
/// For files with many frames per peak, we can skip frames while still
/// maintaining quality. This dramatically speeds up processing of long files.
///
/// Returns a skip value between 1 (no skip) and 50 (aggressive sampling)
fn calculate_optimal_skip(total_frames: u64, num_peaks: usize) -> usize {
    let frames_per_peak = total_frames as f64 / num_peaks as f64;

    // We want at least ~100 samples per peak for quality
    // If frames_per_peak is 10,000, we can skip every 100th frame
    const MIN_SAMPLES_PER_PEAK: f64 = 100.0;
    const MAX_SKIP: usize = 50;

    let optimal_skip = (frames_per_peak / MIN_SAMPLES_PER_PEAK).floor() as usize;

    // Clamp between 1 (no skip) and MAX_SKIP
    optimal_skip.clamp(1, MAX_SKIP)
}

/// Calculate optimal number of seek groups based on file duration
///
/// Balances I/O efficiency (fewer seeks) with processing speed (skip unneeded data).
/// Returns number of groups to divide the file into for seeking.
fn calculate_seek_groups(duration_seconds: f64, num_peaks: usize) -> usize {
    // Base calculation: aim for ~20 peaks per group as a starting point
    let base_groups = (num_peaks / 20).max(1);

    // Adjust based on duration:
    // - Very short files (< 30s): 1 group (pure sequential, no seeks needed)
    // - Short files (< 5min): fewer groups (seeking overhead not worth it)
    // - Long files: more groups to skip more of the file
    let duration_factor = if duration_seconds < 30.0 {
        0.0 // Force single group for very short files
    } else if duration_seconds < 300.0 {
        0.5 // Fewer groups for short files
    } else if duration_seconds < 1800.0 {
        1.0 // Normal grouping for medium files
    } else {
        1.5 // More groups for very long files
    };

    let adjusted_groups = (base_groups as f64 * duration_factor) as usize;

    // Clamp to reasonable bounds:
    // - Min 1 (pure sequential)
    // - Max 200 (avoid excessive seeking even for very long files)
    adjusted_groups.clamp(1, 200)
}

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

    // Calculate total frames and duration first (needed for processing parameters)
    let total_frames = total_frames_opt.ok_or_else(|| {
        AudioError::DecodeFailed(
            "Frame count not available in metadata. This file may need a full decode to determine length.".to_string()
        )
    })?;
    let duration_seconds = total_frames as f64 / sample_rate as f64;

    // Initialize peak buffers
    let mut min_peaks = vec![f32::MAX; num_peaks];
    let mut max_peaks = vec![f32::MIN; num_peaks];

    // Calculate processing parameters
    let frames_per_peak = total_frames as f64 / num_peaks as f64;
    let frame_skip = calculate_optimal_skip(total_frames, num_peaks);

    // Track current frame position (may be updated if we decode first packet for channel detection)
    let mut initial_frame: u64 = 0;

    // If channels not in metadata, decode first packet to get channel info
    // We process this packet immediately to avoid holding a reference to the decoder
    let channels = if let Some(ch) = channels_opt {
        ch
    } else {
        // Decode first packet to determine channels
        let ch = loop {
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

            // Process this first packet immediately (don't store the reference)
            process_packet_peaks(
                &decoded,
                ch,
                &mut initial_frame,
                frames_per_peak,
                frame_skip,
                &mut min_peaks,
                &mut max_peaks,
            );

            break ch;
        };
        ch
    };

    // UNIFIED GROUPED-SEEK APPROACH
    //
    // Strategy: Divide the file into seek groups, seek to each group's start,
    // then decode sequentially within the group to fill multiple peaks.
    //
    // This balances:
    // - I/O efficiency: Fewer seeks than per-peak seeking (good for HDDs, OS caching)
    // - Speed: Only decode packets needed for each group (skip most of the file)
    // - Quality: Use process_packet_peaks for proper frame boundary tracking
    //
    // For all file lengths, we use the same algorithm with adaptive group sizing.

    // Calculate optimal number of seek groups based on duration
    // - Very short files (< 30s): 1 group (pure sequential)
    // - Short files (< 5min): ~10 groups
    // - Medium files (5-30min): ~30-50 groups
    // - Long files (30min+): ~100 groups (cap to avoid too many seeks)
    let num_seek_groups = calculate_seek_groups(duration_seconds, num_peaks);
    let peaks_per_group = (num_peaks + num_seek_groups - 1) / num_seek_groups;
    let time_per_group = duration_seconds / num_seek_groups as f64;

    for group_idx in 0..num_seek_groups {
        let group_start_peak = group_idx * peaks_per_group;
        let group_end_peak = ((group_idx + 1) * peaks_per_group).min(num_peaks);

        // Skip if we've already filled all peaks
        if group_start_peak >= num_peaks {
            break;
        }

        // For first group, continue from where we left off (may have decoded for channel detection)
        // For subsequent groups, seek to the group's start position
        let mut current_frame = if group_idx == 0 {
            initial_frame
        } else {
            let group_start_time = group_idx as f64 * time_per_group;
            let group_start_frame = (group_start_time * sample_rate as f64) as u64;

            // Seek to group start
            let seek_result = format.seek(
                SeekMode::Coarse,
                SeekTo::Time {
                    time: Time::from(group_start_time),
                    track_id: Some(track_id),
                },
            );

            if seek_result.is_err() {
                // If seek fails, skip this group
                continue;
            }

            decoder.reset();
            group_start_frame
        };

        // Decode packets until we've filled all peaks for this group
        // Use a packet limit to prevent runaway decoding if something goes wrong
        let max_packets_per_group = (peaks_per_group * 10).max(50);
        let mut packets_in_group = 0;

        loop {
            // Check if we've filled all peaks for this group
            let current_peak_idx = (current_frame as f64 / frames_per_peak) as usize;
            if current_peak_idx >= group_end_peak {
                break;
            }

            // Safety limit on packets per group
            if packets_in_group >= max_packets_per_group {
                break;
            }

            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(_) => break,
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(d) => d,
                Err(_) => continue,
            };

            process_packet_peaks(
                &decoded,
                channels,
                &mut current_frame,
                frames_per_peak,
                frame_skip,
                &mut min_peaks,
                &mut max_peaks,
            );

            packets_in_group += 1;
        }
    }

    // Handle any peaks that didn't get set
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
    frame_skip: usize,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
                frame_skip,
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
///
/// OPTIMIZED:
/// - Uses incremental boundary tracking instead of per-frame division (635M divisions → ~few hundred)
/// - Batch processes frames in peak segments for better cache locality
/// - Supports frame skipping for very long files (adaptive sampling)
fn process_samples_generic<T, F>(
    planes: &[&[T]],
    channels: u16,
    current_frame: &mut u64,
    frames_per_peak: f64,
    frame_skip: usize,
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

    if frame_count == 0 {
        return;
    }

    // Calculate which peak bucket we start in (only ONE division for entire packet)
    let mut peak_idx = (*current_frame as f64 / frames_per_peak) as usize;

    if peak_idx >= num_peaks {
        return;
    }

    // Calculate the frame index where we'll move to the next peak
    // Use comparison instead of division for each frame
    let mut next_peak_boundary = ((peak_idx + 1) as f64 * frames_per_peak) as u64;

    let mut frame_idx = 0;

    // BATCH PROCESSING + ADAPTIVE SAMPLING
    // Process frames in peak-sized batches, optionally skipping frames for very long files
    while frame_idx < frame_count && peak_idx < num_peaks {
        // Calculate how many frames belong to this peak segment
        let frames_until_boundary = (next_peak_boundary - *current_frame) as usize;
        let frames_to_process = frames_until_boundary.min(frame_count - frame_idx);

        // Process frames in this peak segment, potentially skipping some
        let mut i = 0;
        while i < frames_to_process {
            let idx = frame_idx + i;

            // Calculate min/max across all channels for this frame
            let mut frame_min = f32::MAX;
            let mut frame_max = f32::MIN;

            for channel in 0..channels as usize {
                if channel < planes.len() {
                    let sample = convert(&planes[channel][idx]);
                    frame_min = frame_min.min(sample);
                    frame_max = frame_max.max(sample);
                }
            }

            // Update peak values for this segment
            min_peaks[peak_idx] = min_peaks[peak_idx].min(frame_min);
            max_peaks[peak_idx] = max_peaks[peak_idx].max(frame_max);

            // ADAPTIVE SAMPLING: Skip frames based on file duration
            i += frame_skip;
        }

        // Update counters for next batch
        *current_frame += frames_to_process as u64;
        frame_idx += frames_to_process;

        // Move to next peak if we've filled this one
        if *current_frame >= next_peak_boundary {
            peak_idx += 1;
            if peak_idx < num_peaks {
                next_peak_boundary = ((peak_idx + 1) as f64 * frames_per_peak) as u64;
            }
        }
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
