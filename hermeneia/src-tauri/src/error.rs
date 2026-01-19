use thiserror::Error;

/// All possible errors that can occur during audio processing
#[derive(Debug, Error)]
pub enum AudioError {
    /// Failed to open or read the audio file from disk
    #[error("Failed to open audio file '{path}': {source}")]
    FileOpen {
        path: String,
        source: std::io::Error,
    },

    /// The audio format is not supported by symphonia
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Error occurred while decoding the audio data
    #[error("Audio decoding failed: {0}")]
    DecodeFailed(String),

    /// Error occurred while encoding to WAV
    #[error("WAV encoding failed: {0}")]
    EncodeFailed(String),

    /// Invalid trim parameters (e.g., start > end, negative values)
    #[error("Invalid trim parameters: {0}")]
    InvalidTrimParams(String),

    /// Trim range is outside the audio file's duration
    #[error("Trim range ({start}s to {end}s) exceeds audio duration ({duration}s)")]
    TrimRangeOutOfBounds {
        start: f64,
        end: f64,
        duration: f64,
    },

    /// Generic I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error from symphonia decoder
    #[error("Symphonia error: {0}")]
    Symphonia(String),

    /// Error from hound WAV encoder
    #[error("Hound WAV error: {0}")]
    Hound(#[from] hound::Error),

    /// Failed to download model from HuggingFace
    #[error("Failed to download model '{model}': {details}")]
    ModelDownload { model: String, details: String },

    /// Failed to load model files
    #[error("Failed to load model '{model}': {details}")]
    ModelLoad { model: String, details: String },

    /// Transcription inference failed
    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    /// Audio preprocessing failed
    #[error("Audio preprocessing failed: {0}")]
    AudioPreprocessing(String),

    /// Invalid transcription parameters
    #[error("Invalid transcription parameters: {0}")]
    InvalidTranscribeParams(String),

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Out of memory error with specific context and suggestions
    #[error("Out of memory: {message}")]
    OutOfMemory {
        message: String,
        device: String, // "RAM" or "VRAM"
        required_gb: f32,
        model_name: String,
    },
}

/// Convenient Result type that uses our AudioError
pub type Result<T> = std::result::Result<T, AudioError>;