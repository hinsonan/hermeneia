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
pub use preprocessing::{convert_to_mono, prepare_speech_audio, resample_to_16khz};
pub use trim::trim_audio;
pub use types::{AudioData, AudioInfo, SpeechAudio, TrimParams, WaveformPeaks};
pub use waveform::extract_waveform_peaks;
