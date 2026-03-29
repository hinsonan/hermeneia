mod decoder;
pub mod inference;
pub mod language;
pub mod model;
pub mod preprocessing;
pub mod requirements;
pub mod tauri_progress;
mod types;
pub mod validator;

pub use inference::{
    transcribe_audio, transcribe_audio_with_progress, transcribe_audio_with_reporter,
    transcribe_prepared_audio_with_progress, transcribe_prepared_audio_with_reporter,
};
pub use model::ModelManager;
pub use requirements::ModelRequirements;
pub use tauri_progress::TauriProgressReporter;
pub use types::{
    ModelFiles, NoProgress, ProgressCallback, ProgressReporter, TranscribeParams, TranscriptResult,
    TranscriptSegment, TranscriptionPhase, TranscriptionProgress, TranscriptionTask, WhisperModel,
};
pub use validator::{ModelValidator, ValidationResult};
