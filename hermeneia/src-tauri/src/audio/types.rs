use serde::{Deserialize, Serialize};

/// Represents decoded audio data in memory as PCM samples
///
/// Samples are stored interleaved: [L, R, L, R, ...] for stereo
/// or [M, M, M, ...] for mono, where each sample is a 32-bit float
/// in the range [-1.0, 1.0]
#[derive(Debug, Clone)]
pub struct AudioData {
    /// PCM audio samples as 32-bit floats, interleaved by channel
    /// Example for stereo: [left_0, right_0, left_1, right_1, ...]
    pub samples: Vec<f32>,

    /// Sample rate in Hz (e.g., 44100, 48000)
    pub sample_rate: u32,

    /// Number of audio channels (1 = mono, 2 = stereo)
    pub channels: u16,
}

/// Shared speech-prepared audio representation used by both
/// transcription and speaker diarization.
///
/// - mono
/// - 16kHz
/// - f32 PCM in [-1.0, 1.0]
#[derive(Debug, Clone)]
pub struct SpeechAudio {
    /// Mono PCM at 16kHz, ready for speech models
    pub samples_16k_mono: Vec<f32>,

    /// Original decoded audio duration in seconds
    pub duration_seconds: f64,
}

impl AudioData {
    /// Calculate the total duration of the audio in seconds
    ///
    /// Duration = total_samples / (sample_rate * channels)
    pub fn duration_seconds(&self) -> f64 {
        let total_frames = self.samples.len() as f64 / self.channels as f64;
        total_frames / self.sample_rate as f64
    }

    /// Get the number of audio frames (one sample per channel)
    ///
    /// For stereo: 1000 samples = 500 frames
    /// For mono: 1000 samples = 1000 frames
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }
}

/// Metadata about an audio file without loading all samples
///
/// Use this for quick info queries without decoding the entire file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    /// Total duration in seconds
    pub duration_seconds: f64,

    /// Sample rate in Hz
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Audio format/codec name (e.g., "MP3", "FLAC", "Vorbis")
    pub format: String,

    /// Bit depth if available (e.g., 16, 24)
    pub bit_depth: Option<u16>,
}

/// Parameters for trimming an audio file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrimParams {
    /// Start time in seconds (must be >= 0)
    pub start_seconds: f64,

    /// End time in seconds (must be > start_seconds)
    pub end_seconds: f64,
}

/// Waveform peak data for visualization
///
/// Contains min/max peak values for efficient waveform rendering.
/// Each peak represents a segment of the audio file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveformPeaks {
    /// Minimum amplitude values for each segment (range: -1.0 to 1.0)
    pub min_peaks: Vec<f32>,

    /// Maximum amplitude values for each segment (range: -1.0 to 1.0)
    pub max_peaks: Vec<f32>,

    /// Number of peaks/segments
    pub num_peaks: usize,

    /// Duration of audio in seconds
    pub duration_seconds: f64,

    /// Number of audio channels (for reference)
    pub channels: u16,

    /// Sample rate (for reference)
    pub sample_rate: u32,
}

impl TrimParams {
    /// Create new trim parameters with validation
    pub fn new(start_seconds: f64, end_seconds: f64) -> crate::error::Result<Self> {
        use crate::error::AudioError;

        if start_seconds < 0.0 {
            return Err(AudioError::InvalidTrimParams(format!(
                "Start time cannot be negative: {}",
                start_seconds
            )));
        }

        if end_seconds <= start_seconds {
            return Err(AudioError::InvalidTrimParams(format!(
                "End time ({}) must be greater than start time ({})",
                end_seconds, start_seconds
            )));
        }

        Ok(Self {
            start_seconds,
            end_seconds,
        })
    }

    /// Get the duration of the trimmed audio
    pub fn trim_duration(&self) -> f64 {
        self.end_seconds - self.start_seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_data_duration_stereo() {
        let audio = AudioData {
            samples: vec![0.0; 88200], // 1 second stereo at 44.1kHz
            sample_rate: 44100,
            channels: 2,
        };
        assert_eq!(audio.duration_seconds(), 1.0);
    }

    #[test]
    fn test_audio_data_duration_mono() {
        let audio = AudioData {
            samples: vec![0.0; 48000], // 1 second mono at 48kHz
            sample_rate: 48000,
            channels: 1,
        };
        assert_eq!(audio.duration_seconds(), 1.0);
    }

    #[test]
    fn test_audio_data_frame_count_stereo() {
        let audio = AudioData {
            samples: vec![0.0; 1000], // 500 frames stereo
            sample_rate: 44100,
            channels: 2,
        };
        assert_eq!(audio.frame_count(), 500);
    }

    #[test]
    fn test_audio_data_frame_count_mono() {
        let audio = AudioData {
            samples: vec![0.0; 1000], // 1000 frames mono
            sample_rate: 44100,
            channels: 1,
        };
        assert_eq!(audio.frame_count(), 1000);
    }

    #[test]
    fn test_audio_data_empty() {
        let audio = AudioData {
            samples: vec![],
            sample_rate: 44100,
            channels: 2,
        };
        assert_eq!(audio.duration_seconds(), 0.0);
        assert_eq!(audio.frame_count(), 0);
    }

    #[test]
    fn test_trim_params_valid() {
        let params = TrimParams::new(1.0, 5.0).unwrap();
        assert_eq!(params.start_seconds, 1.0);
        assert_eq!(params.end_seconds, 5.0);
        assert_eq!(params.trim_duration(), 4.0);
    }

    #[test]
    fn test_trim_params_negative_start() {
        let result = TrimParams::new(-1.0, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_trim_params_end_before_start() {
        let result = TrimParams::new(5.0, 3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_trim_params_equal_start_end() {
        let result = TrimParams::new(5.0, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_trim_params_zero_start() {
        let params = TrimParams::new(0.0, 1.0).unwrap();
        assert_eq!(params.start_seconds, 0.0);
        assert_eq!(params.trim_duration(), 1.0);
    }

    #[test]
    fn test_trim_params_large_values() {
        let params = TrimParams::new(100.0, 3600.0).unwrap();
        assert_eq!(params.trim_duration(), 3500.0);
    }

    #[test]
    fn test_trim_params_small_duration() {
        let params = TrimParams::new(0.0, 0.001).unwrap();
        assert_eq!(params.trim_duration(), 0.001);
    }

    #[test]
    fn test_waveform_peaks_clone() {
        let peaks = WaveformPeaks {
            min_peaks: vec![-0.5, -0.3],
            max_peaks: vec![0.5, 0.3],
            num_peaks: 2,
            duration_seconds: 10.0,
            channels: 2,
            sample_rate: 44100,
        };
        let cloned = peaks.clone();
        assert_eq!(cloned.num_peaks, 2);
        assert_eq!(cloned.duration_seconds, 10.0);
    }

    #[test]
    fn test_audio_info_serialization() {
        let info = AudioInfo {
            duration_seconds: 120.5,
            sample_rate: 44100,
            channels: 2,
            format: "MP3".to_string(),
            bit_depth: Some(16),
        };

        // Test that it can be serialized (will fail if Serialize isn't properly derived)
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("120.5"));
        assert!(json.contains("MP3"));

        // Test deserialization
        let deserialized: AudioInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.duration_seconds, 120.5);
        assert_eq!(deserialized.format, "MP3");
    }

    #[test]
    fn test_audio_info_without_bit_depth() {
        let info = AudioInfo {
            duration_seconds: 60.0,
            sample_rate: 48000,
            channels: 1,
            format: "Vorbis".to_string(),
            bit_depth: None,
        };
        assert_eq!(info.bit_depth, None);
    }
}
