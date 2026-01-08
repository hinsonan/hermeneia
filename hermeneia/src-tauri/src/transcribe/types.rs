use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
}
