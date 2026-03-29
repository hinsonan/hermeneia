mod inference;
mod model;
pub mod types;

pub use inference::{
    diarize_audio, diarize_audio_with_progress, diarize_prepared_audio,
    diarize_prepared_audio_with_callbacks, diarize_prepared_audio_with_progress, DiarizeCallbacks,
    DiarizeProgressCallback, DiarizeStage, DiarizeStageProgress, DiarizeStageProgressCallback,
};
pub use model::SpeakerModelManager;
pub use types::{
    validate_diarize_params, DiarizationResult, DiarizeParams, SpeakerDevice, SpeakerModel,
    SpeakerSegment,
};
