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

impl TrimParams {
    /// Create new trim parameters with validation
    pub fn new(start_seconds: f64, end_seconds: f64) -> crate::error::Result<Self> {
        use crate::error::AudioError;

        if start_seconds < 0.0 {
            return Err(AudioError::InvalidTrimParams(
                format!("Start time cannot be negative: {}", start_seconds)
            ));
        }

        if end_seconds <= start_seconds {
            return Err(AudioError::InvalidTrimParams(
                format!("End time ({}) must be greater than start time ({})", 
                    end_seconds, start_seconds)
            ));
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