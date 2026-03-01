mod inference;
mod model;
pub mod types;

pub use inference::{diarize_audio, diarize_audio_with_progress, DiarizeProgressCallback};
pub use model::SpeakerModelManager;
pub use types::{
    validate_diarize_params, DiarizationResult, DiarizeParams, SpeakerDevice, SpeakerModel,
    SpeakerSegment,
};
