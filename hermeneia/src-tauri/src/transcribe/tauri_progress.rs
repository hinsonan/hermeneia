use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter};

use crate::transcribe::types::{ProgressReporter, TranscriptionProgress};

/// Event name for transcription progress events
pub const TRANSCRIPTION_PROGRESS_EVENT: &str = "transcription-progress";

/// Tauri-based progress reporter that emits events to the frontend
pub struct TauriProgressReporter {
    app_handle: AppHandle,
    first_report: AtomicBool,
    last_logged_pct: AtomicUsize,
}

impl TauriProgressReporter {
    /// Create a new Tauri progress reporter
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            first_report: AtomicBool::new(true),
            last_logged_pct: AtomicUsize::new(0),
        }
    }

    /// Emit a progress event to the frontend
    fn emit(&self, progress: TranscriptionProgress) {
        if let Err(e) = self.app_handle.emit(TRANSCRIPTION_PROGRESS_EVENT, progress) {
            tracing::warn!("Failed to emit progress event: {}", e);
        }
    }

    /// Emit decode progress events.
    pub fn emit_decoding_audio(&self) {
        self.emit(TranscriptionProgress::decoding_audio());
    }

    /// Emit decode progress with frame counts.
    pub fn emit_decoding_audio_progress(&self, current: usize, total: usize) {
        self.emit(TranscriptionProgress::decoding_audio_progress(
            current, total,
        ));
    }

    /// Emit preparing-audio stage event.
    pub fn emit_preparing_audio(&self) {
        self.emit(TranscriptionProgress::preparing_audio());
    }

    /// Emit loading-model stage event.
    pub fn emit_loading_model(&self) {
        self.emit(TranscriptionProgress::loading_model());
    }
}

impl ProgressReporter for TauriProgressReporter {
    fn start(&self) {
        tracing::info!("Transcription starting: loading model...");
        self.emit_loading_model();
    }

    fn report(&self, current: usize, total: usize) {
        if self.first_report.swap(false, Ordering::Relaxed) {
            tracing::info!("Transcription started: {} frames total", total);
        }

        // Log every 10% of progress
        let pct = if total > 0 { current * 100 / total } else { 0 };
        let last = self.last_logged_pct.load(Ordering::Relaxed);
        if pct / 10 > last / 10 {
            self.last_logged_pct.store(pct, Ordering::Relaxed);
            tracing::info!("Transcription progress: {}% ({}/{})", pct, current, total);
        }

        self.emit(TranscriptionProgress::transcribing(current, total));
    }

    fn finish(&self) {
        tracing::info!("Transcription complete");
        self.emit(TranscriptionProgress::completed());
    }
}
