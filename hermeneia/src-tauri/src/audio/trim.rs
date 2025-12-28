// src-tauri/src/audio/trim.rs

use crate::audio::types::{AudioData, TrimParams};
use crate::error::{AudioError, Result};

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