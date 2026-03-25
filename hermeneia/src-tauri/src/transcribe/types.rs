use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Progress callback function type (legacy, for backward compatibility)
/// Parameters: (current_frames, total_frames)
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Transcription phase for progress reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionPhase {
    DecodingAudio,
    PreparingAudio,
    LoadingModel,
    Transcribing,
    Completed,
}

/// Progress event payload for Tauri events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionProgress {
    pub phase: TranscriptionPhase,
    pub current: Option<usize>,
    pub total: Option<usize>,
    pub message: String,
}

impl TranscriptionProgress {
    /// Create a "decoding audio" progress event (indeterminate).
    pub fn decoding_audio() -> Self {
        Self {
            phase: TranscriptionPhase::DecodingAudio,
            current: None,
            total: None,
            message: "Decoding audio...".to_string(),
        }
    }

    /// Create a "decoding audio" progress event with known progress.
    pub fn decoding_audio_progress(current: usize, total: usize) -> Self {
        if total > 0 {
            Self {
                phase: TranscriptionPhase::DecodingAudio,
                current: Some(current.min(total)),
                total: Some(total),
                message: "Decoding audio...".to_string(),
            }
        } else {
            Self::decoding_audio()
        }
    }

    /// Create a "preparing audio" progress event.
    pub fn preparing_audio() -> Self {
        Self {
            phase: TranscriptionPhase::PreparingAudio,
            current: None,
            total: None,
            message: "Preparing mono 16kHz audio...".to_string(),
        }
    }

    /// Create a "loading model" progress event
    pub fn loading_model() -> Self {
        Self {
            phase: TranscriptionPhase::LoadingModel,
            current: None,
            total: None,
            message: "Loading model...".to_string(),
        }
    }

    /// Create a "transcribing" progress event
    pub fn transcribing(current: usize, total: usize) -> Self {
        let percentage = if total > 0 {
            (current as f64 / total as f64 * 100.0) as usize
        } else {
            0
        };
        Self {
            phase: TranscriptionPhase::Transcribing,
            current: Some(current),
            total: Some(total),
            message: format!("Transcribing... {}%", percentage),
        }
    }

    /// Create a "completed" progress event
    pub fn completed() -> Self {
        Self {
            phase: TranscriptionPhase::Completed,
            current: Some(100),
            total: Some(100),
            message: "Transcription complete".to_string(),
        }
    }
}

/// Trait for reporting transcription progress
/// Allows decoupling of progress reporting from transcription logic
pub trait ProgressReporter: Send + Sync {
    /// Report progress (current_frame, total_frames)
    fn report(&self, current: usize, total: usize);

    /// Called when transcription starts
    fn start(&self) {}

    /// Called when transcription completes
    fn finish(&self) {}
}

/// Null implementation for testing/no progress reporting
#[derive(Debug, Clone)]
pub struct NoProgress;

impl ProgressReporter for NoProgress {
    fn report(&self, _current: usize, _total: usize) {}
}

/// Whisper model size variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WhisperModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
}

impl WhisperModel {
    /// Get the HuggingFace model ID for this model
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::TinyEn => "openai/whisper-tiny.en",
            Self::Base => "openai/whisper-base",
            Self::BaseEn => "openai/whisper-base.en",
            Self::Small => "openai/whisper-small",
            Self::SmallEn => "openai/whisper-small.en",
            Self::Medium => "openai/whisper-medium",
            Self::MediumEn => "openai/whisper-medium.en",
            Self::Large => "openai/whisper-large",
            Self::LargeV2 => "openai/whisper-large-v2",
            Self::LargeV3 => "openai/whisper-large-v3",
        }
    }

    /// Check if this model is multilingual
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn => false,
        }
    }
}

/// Task type for Whisper inference
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TranscriptionTask {
    Transcribe, // Speech to text in original language
    Translate,  // Speech to English
}

/// Parameters for transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeParams {
    pub model: WhisperModel,
    pub task: TranscriptionTask,
    pub language: Option<String>,
    pub timestamps: bool,
    pub force_cpu: bool,
    pub use_quantized: bool,
}

impl Default for TranscribeParams {
    fn default() -> Self {
        Self {
            model: WhisperModel::Tiny,
            task: TranscriptionTask::Transcribe,
            language: None,
            timestamps: true,
            force_cpu: false,
            use_quantized: false,
        }
    }
}

/// A single transcribed segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub id: usize,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub text: String,
}

/// Complete transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptResult {
    pub segments: Vec<TranscriptSegment>,
    pub text: String,
    pub language: Option<String>,
    pub duration: f64,
    pub model: WhisperModel,
    pub inference_time: f64,
}

/// Paths to model files
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
    pub is_quantized: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_ids() {
        assert_eq!(WhisperModel::Tiny.model_id(), "openai/whisper-tiny");
        assert_eq!(WhisperModel::LargeV3.model_id(), "openai/whisper-large-v3");
    }

    #[test]
    fn test_default_params() {
        let params = TranscribeParams::default();
        assert!(matches!(params.model, WhisperModel::Tiny));
        assert!(params.timestamps);
        assert!(!params.force_cpu);
    }

    #[test]
    fn test_no_progress_trait() {
        let reporter = NoProgress;
        // Should not panic and do nothing
        reporter.report(50, 100);
        reporter.start();
        reporter.finish();
    }
}
