// src-tauri/src/audio/mod.rs

pub mod decoder;
pub mod encoder;
pub mod trim;
pub mod types;

// Re-export commonly used items
pub use decoder::{decode_audio_file, get_audio_info};
pub use encoder::encode_wav;
pub use trim::trim_audio;
pub use types::{AudioData, AudioInfo, TrimParams};