pub mod audio;
pub mod error;
pub mod gpu;

// Re-export for convenience
pub use audio::*;
pub use error::{AudioError, Result};

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Extract waveform peaks from an audio file for visualization
///
/// Tauri command that processes audio files and returns peak data
/// for displaying waveforms in the frontend.
///
/// # Arguments
/// * `file_path` - Path to the audio file
/// * `num_peaks` - Optional number of peaks (default: 2000)
///
/// # Returns
/// WaveformPeaks as JSON with min/max peak arrays
#[tauri::command]
fn get_waveform_peaks(
    file_path: String,
    num_peaks: Option<usize>,
) -> std::result::Result<WaveformPeaks, String> {
    audio::extract_waveform_peaks(&file_path, num_peaks)
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {

    gpu::apply_optimizations();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, get_waveform_peaks])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
