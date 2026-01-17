use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter};

use crate::transcribe::types::{ProgressReporter, TranscriptionProgress};

/// Event name for transcription progress events
pub const TRANSCRIPTION_PROGRESS_EVENT: &str = "transcription-progress";

/// Tauri-based progress reporter that emits events to the frontend
pub struct TauriProgressReporter {
    app_handle: AppHandle,
    first_report: AtomicBool,
}

impl TauriProgressReporter {
    /// Create a new Tauri progress reporter
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            first_report: AtomicBool::new(true),
        }
    }

    /// Emit a progress event to the frontend
    fn emit(&self, progress: TranscriptionProgress) {
        if let Err(e) = self.app_handle.emit(TRANSCRIPTION_PROGRESS_EVENT, progress) {
            tracing::warn!("Failed to emit progress event: {}", e);
        }
    }
}

impl ProgressReporter for TauriProgressReporter {
    fn start(&self) {
        self.emit(TranscriptionProgress::loading_model());
    }

    fn report(&self, current: usize, total: usize) {
        // On first report, we're now in transcribing phase
        if self.first_report.swap(false, Ordering::Relaxed) {
            tracing::info!("Transcription started: {}/{} frames", current, total);
        }
        self.emit(TranscriptionProgress::transcribing(current, total));
    }

    fn finish(&self) {
        self.emit(TranscriptionProgress::completed());
    }
}
