use serde::{Deserialize, Serialize};

use crate::error::{AudioError, Result};

/// Available speaker diarization model bundles.
/// Each bundle pairs the pyannote-segmentation-3.0 segmentation model
/// with a language-appropriate 3DSpeaker embedding model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SpeakerModel {
    /// English-optimized: 3DSpeaker ERes2Net trained on VoxCeleb (~26.5 MB embedding)
    #[default]
    English,
    /// Multilingual: 3DSpeaker ERes2Net base (~39.6 MB embedding)
    Multilingual,
}

impl SpeakerModel {
    pub fn display_name(&self) -> &str {
        match self {
            Self::English => "English (3DSpeaker ERes2Net VoxCeleb, 26.5 MB)",
            Self::Multilingual => "Multilingual (3DSpeaker ERes2Net base, 39.6 MB)",
        }
    }

    /// HuggingFace repo and filename for the segmentation model (shared).
    /// Uses the sherpa-onnx re-exported version which includes required ONNX metadata.
    pub fn segmentation_source(&self) -> (&str, &str) {
        (
            "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            "model.onnx",
        )
    }

    /// HuggingFace repo and filename for the embedding model.
    pub fn embedding_source(&self) -> (&str, &str) {
        match self {
            Self::English => (
                "csukuangfj/speaker-embedding-models",
                "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
            ),
            Self::Multilingual => (
                "csukuangfj/speaker-embedding-models",
                "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            ),
        }
    }

    /// CLI key used with --model flag
    pub fn cli_key(&self) -> &str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
        }
    }

    /// Approximate total download size (segmentation + embedding) in MB
    pub fn approx_size_mb(&self) -> f32 {
        match self {
            Self::English => 32.5,
            Self::Multilingual => 45.6,
        }
    }
}

/// Inference device / execution provider for ONNX Runtime
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SpeakerDevice {
    /// CPU inference (stable, default)
    #[default]
    Cpu,
    /// NVIDIA CUDA (requires cuda feature at build time)
    Cuda,
    /// Apple CoreML / Metal (macOS only)
    CoreMl,
}

impl SpeakerDevice {
    /// Maps to the provider string accepted by sherpa-rs DiarizeConfig
    pub fn provider_string(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
        }
    }
}

/// Parameters controlling the diarization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizeParams {
    pub model: SpeakerModel,
    /// Expected number of speakers (None = auto-detect via threshold)
    pub num_speakers: Option<i32>,
    /// Clustering threshold — higher merges more speakers (0.0–1.0, default 0.5)
    pub threshold: f32,
    /// Inference device (default: CPU for stability)
    pub device: SpeakerDevice,
}

impl Default for DiarizeParams {
    fn default() -> Self {
        Self {
            model: SpeakerModel::English,
            num_speakers: None,
            threshold: 0.5,
            device: SpeakerDevice::Cpu,
        }
    }
}

/// Validate speaker diarization parameters.
pub fn validate_diarize_params(params: &DiarizeParams) -> Result<()> {
    if !(0.0..=1.0).contains(&params.threshold) {
        return Err(AudioError::InvalidDiarizeParams(format!(
            "Threshold must be between 0.0 and 1.0, got {}",
            params.threshold
        )));
    }

    if let Some(num) = params.num_speakers {
        if num < 1 {
            return Err(AudioError::InvalidDiarizeParams(format!(
                "Expected number of speakers must be >= 1, got {}",
                num
            )));
        }
    }

    Ok(())
}

/// A single speaker-labeled time range in the output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    /// 0-indexed speaker ID (consistent within a single run)
    pub speaker: i32,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
}

/// Full diarization output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationResult {
    pub segments: Vec<SpeakerSegment>,
    /// Number of unique speakers detected
    pub num_speakers: usize,
    /// Total audio duration in seconds
    pub audio_duration: f32,
    /// Inference wall-clock time in seconds
    pub inference_time: f64,
    /// Display name of the model bundle used
    pub model: String,
    /// Device that ran inference
    pub device: String,
}
