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
}

/// Convenient Result type that uses our AudioError
pub type Result<T> = std::result::Result<T, AudioError>;