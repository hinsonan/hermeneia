// src-tauri/src/audio/mod.rs

mod decoder;
mod encoder;
pub mod playback;
mod preprocessing;
pub mod trim;
mod types;
pub mod waveform;

// Re-export commonly used items
pub use decoder::{
    decode_audio_file, decode_audio_file_with_progress, get_audio_info, DecodeProgressCallback,
};
pub use encoder::encode_wav;
pub use playback::AudioPlayer;
pub use preprocessing::{
    convert_to_mono, convert_to_mono_owned, prepare_speech_audio, prepare_speech_audio_owned,
    resample_to_16khz, resample_to_16khz_owned,
};
pub use trim::trim_audio;
pub use types::{AudioData, AudioInfo, SpeechAudio, TrimParams, WaveformPeaks};
pub use waveform::extract_waveform_peaks;
