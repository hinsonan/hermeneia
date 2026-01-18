mod types;
pub mod model;
pub mod preprocessing;
pub mod inference;
mod decoder;
pub mod language;
pub mod tauri_progress;
pub mod requirements;
pub mod validator;

pub use types::{
    WhisperModel, TranscribeParams, TranscriptResult,
    TranscriptSegment, TranscriptionTask, ModelFiles, ProgressCallback,
    ProgressReporter, NoProgress, TranscriptionPhase, TranscriptionProgress,
};
pub use model::ModelManager;
pub use inference::{transcribe_audio, transcribe_audio_with_progress, transcribe_audio_with_reporter};
pub use tauri_progress::TauriProgressReporter;
pub use requirements::ModelRequirements;
pub use validator::{ModelValidator, ValidationResult};
