use hermeneia_lib::transcribe::ProgressReporter;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};

pub struct TranscriptionProgress {
    progress_bar: ProgressBar,
    first_call: Arc<Mutex<bool>>,
}

impl TranscriptionProgress {
    pub fn new() -> Self {
        let progress_bar = ProgressBar::new_spinner();
        progress_bar.set_style(
            ProgressStyle::default_spinner()
                .template("[{elapsed_precise}] {spinner:.cyan} {msg}")
                .expect("Invalid spinner template"),
        );
        progress_bar.set_message("Loading model and detecting language...");
        progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));

        Self {
            progress_bar,
            first_call: Arc::new(Mutex::new(true)),
        }
    }
}

impl ProgressReporter for TranscriptionProgress {
    fn report(&self, current: usize, total: usize) {
        if total == 0 {
            return;
        }

        // Switch from spinner to progress bar on first call
        if let Ok(mut is_first) = self.first_call.lock() {
            if *is_first {
                *is_first = false;
                self.progress_bar.disable_steady_tick();
                self.progress_bar.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>3}% {msg}")
                        .expect("Invalid progress bar template")
                        .progress_chars("█▓░"),
                );
                self.progress_bar.set_length(100);
                self.progress_bar.set_message("Transcribing...");
            }
        }

        let percentage = (current as f64 / total as f64 * 100.0) as u64;
        self.progress_bar.set_position(percentage);
    }

    fn finish(&self) {
        self.progress_bar.finish_with_message("Complete!");
    }
}

impl Drop for TranscriptionProgress {
    fn drop(&mut self) {
        self.progress_bar.finish();
    }
}
