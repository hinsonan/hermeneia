// src-tauri/src/audio/playback.rs
//! Audio playback module using cpal and symphonia
//!
//! Provides audio playback functionality that runs entirely in Rust,
//! bypassing WebKit's audio system for maximum compatibility.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{AudioError, Result};

/// Shared state for audio playback - all atomic for thread safety
struct SharedPlaybackState {
    is_playing: AtomicBool,
    current_frame: AtomicU64,
    total_frames: AtomicU64,
    sample_rate: AtomicU64,
    should_stop: AtomicBool,
    seek_to_frame: AtomicU64,
    seek_pending: AtomicBool,
}

impl SharedPlaybackState {
    fn new() -> Self {
        Self {
            is_playing: AtomicBool::new(false),
            current_frame: AtomicU64::new(0),
            total_frames: AtomicU64::new(0),
            sample_rate: AtomicU64::new(44100),
            should_stop: AtomicBool::new(false),
            seek_to_frame: AtomicU64::new(0),
            seek_pending: AtomicBool::new(false),
        }
    }
}

/// Audio player that manages playback in a background thread
///
/// The cpal Stream lives entirely in the background thread,
/// so AudioPlayer itself is Send + Sync.
pub struct AudioPlayer {
    state: Arc<SharedPlaybackState>,
    playback_thread: Mutex<Option<JoinHandle<()>>>,
    loaded_file: Mutex<Option<PathBuf>>,
}

// AudioPlayer is Send + Sync because:
// - state uses Arc<T> where T contains only atomic types
// - playback_thread is Mutex<Option<JoinHandle>> which is Send+Sync
// - loaded_file is Mutex<Option<PathBuf>> which is Send+Sync
unsafe impl Send for AudioPlayer {}
unsafe impl Sync for AudioPlayer {}

impl AudioPlayer {
    pub fn new() -> Self {
        Self {
            state: Arc::new(SharedPlaybackState::new()),
            playback_thread: Mutex::new(None),
            loaded_file: Mutex::new(None),
        }
    }

    /// Load and start playing an audio file
    pub fn play_file<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        // Stop any existing playback
        self.stop();

        let path = path.as_ref().to_path_buf();
        let state = Arc::clone(&self.state);

        // Store the loaded file path
        *self.loaded_file.lock().unwrap() = Some(path.clone());

        // Reset state
        state.should_stop.store(false, Ordering::SeqCst);
        state.current_frame.store(0, Ordering::SeqCst);
        state.seek_pending.store(false, Ordering::SeqCst);
        state.is_playing.store(true, Ordering::SeqCst);

        // Start playback in a background thread
        let handle = thread::spawn(move || {
            if let Err(e) = run_playback(path, state) {
                tracing::error!("Playback error: {}", e);
            }
        });

        *self.playback_thread.lock().unwrap() = Some(handle);
        Ok(())
    }

    /// Pause playback
    pub fn pause(&self) {
        self.state.is_playing.store(false, Ordering::SeqCst);
    }

    /// Resume playback
    pub fn resume(&self) {
        self.state.is_playing.store(true, Ordering::SeqCst);
    }

    /// Toggle play/pause
    pub fn toggle(&self) {
        let current = self.state.is_playing.load(Ordering::SeqCst);
        self.state.is_playing.store(!current, Ordering::SeqCst);
    }

    /// Seek to a specific time in seconds
    pub fn seek(&self, time_seconds: f64) {
        let rate = self.state.sample_rate.load(Ordering::SeqCst);
        let frame = (time_seconds * rate as f64) as u64;
        self.state.seek_to_frame.store(frame, Ordering::SeqCst);
        self.state.seek_pending.store(true, Ordering::SeqCst);
    }

    /// Stop playback completely
    pub fn stop(&mut self) {
        self.state.should_stop.store(true, Ordering::SeqCst);
        self.state.is_playing.store(false, Ordering::SeqCst);

        // Wait for playback thread to finish
        if let Some(handle) = self.playback_thread.lock().unwrap().take() {
            let _ = handle.join();
        }

        self.state.current_frame.store(0, Ordering::SeqCst);
    }

    /// Get the current playback state
    pub fn get_state(&self) -> (bool, f64, f64) {
        let is_playing = self.state.is_playing.load(Ordering::SeqCst);
        let frame = self.state.current_frame.load(Ordering::SeqCst);
        let total = self.state.total_frames.load(Ordering::SeqCst);
        let rate = self.state.sample_rate.load(Ordering::SeqCst);

        let current_time = if rate > 0 {
            frame as f64 / rate as f64
        } else {
            0.0
        };

        let duration = if rate > 0 {
            total as f64 / rate as f64
        } else {
            0.0
        };

        (is_playing, current_time, duration)
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Run the audio playback loop in a dedicated thread
fn run_playback(path: PathBuf, state: Arc<SharedPlaybackState>) -> Result<()> {
    // Open and probe the audio file
    let file = std::fs::File::open(&path).map_err(|e| AudioError::FileOpen {
        path: path.to_string_lossy().to_string(),
        source: e,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe: {}", e)))?;

    let mut format = probed.format;

    // Find the audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track found".to_string()))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::DecodeFailed("No sample rate".to_string()))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2);
    let total_frames = track.codec_params.n_frames.unwrap_or(0);

    // Update state
    state.sample_rate.store(sample_rate as u64, Ordering::SeqCst);
    state.total_frames.store(total_frames, Ordering::SeqCst);

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    // Decode the entire file first (for simplicity - streaming would be better for large files)
    let mut all_samples: Vec<f32> = Vec::new();
    let spec = SignalSpec::new(
        sample_rate,
        symphonia::core::audio::Channels::FRONT_LEFT
            | symphonia::core::audio::Channels::FRONT_RIGHT,
    );

    loop {
        if state.should_stop.load(Ordering::SeqCst) {
            return Ok(());
        }

        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let frames = decoded.frames();
        let mut sample_buf = SampleBuffer::<f32>::new(frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    if all_samples.is_empty() {
        return Err(AudioError::DecodeFailed("No audio samples decoded".to_string()));
    }

    // Set up cpal audio output
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| AudioError::DecodeFailed("No output device available".to_string()))?;

    let config = StreamConfig {
        channels: channels as u16,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Shared sample buffer and read position
    let samples = Arc::new(all_samples);
    let samples_clone = Arc::clone(&samples);
    let state_clone = Arc::clone(&state);
    let read_pos = Arc::new(AtomicU64::new(0));
    let read_pos_clone = Arc::clone(&read_pos);
    let channels_count = channels;

    // Build the output stream
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let is_playing = state_clone.is_playing.load(Ordering::SeqCst);

                if !is_playing {
                    // Output silence when paused
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    return;
                }

                let pos = read_pos_clone.load(Ordering::SeqCst) as usize;

                for (i, sample) in data.iter_mut().enumerate() {
                    let idx = pos + i;
                    if idx < samples_clone.len() {
                        *sample = samples_clone[idx];
                    } else {
                        *sample = 0.0;
                    }
                }

                let new_pos = (pos + data.len()).min(samples_clone.len());
                read_pos_clone.store(new_pos as u64, Ordering::SeqCst);

                // Update current frame
                let frame = new_pos / channels_count;
                state_clone.current_frame.store(frame as u64, Ordering::SeqCst);
            },
            |err| {
                tracing::error!("Audio stream error: {}", err);
            },
            None,
        )
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to build stream: {}", e)))?;

    stream
        .play()
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to start stream: {}", e)))?;

    // Main loop - handle seek and wait for completion
    loop {
        if state.should_stop.load(Ordering::SeqCst) {
            break;
        }

        // Handle seek
        if state.seek_pending.load(Ordering::SeqCst) {
            let seek_frame = state.seek_to_frame.load(Ordering::SeqCst);
            let seek_sample = (seek_frame as usize * channels).min(samples.len());
            read_pos.store(seek_sample as u64, Ordering::SeqCst);
            state.current_frame.store(seek_frame, Ordering::SeqCst);
            state.seek_pending.store(false, Ordering::SeqCst);
        }

        // Check if playback finished
        let current_pos = read_pos.load(Ordering::SeqCst) as usize;
        if current_pos >= samples.len() {
            state.is_playing.store(false, Ordering::SeqCst);
            state.current_frame.store(0, Ordering::SeqCst);
            read_pos.store(0, Ordering::SeqCst);
            break;
        }

        thread::sleep(Duration::from_millis(50));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_player_creation() {
        let player = AudioPlayer::new();
        let (is_playing, current_time, _duration) = player.get_state();
        assert!(!is_playing);
        assert_eq!(current_time, 0.0);
    }
}
