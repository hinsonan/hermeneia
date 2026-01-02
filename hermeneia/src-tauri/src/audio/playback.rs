// src-tauri/src/audio/playback.rs
//! Audio playback module using cpal and symphonia with STREAMING support
//!
//! Uses a ring buffer to stream audio from disk, enabling playback of files
//! of any size without loading everything into memory.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use ringbuf::{traits::*, HeapRb};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{AudioError, Result};

// Ring buffer size: 5 seconds of audio at 48kHz stereo = 480k samples
const RING_BUFFER_SIZE: usize = 48000 * 2 * 5;

/// Shared state for audio playback - all atomic for thread safety
struct SharedPlaybackState {
    is_playing: AtomicBool,
    current_frame: AtomicU64,
    total_frames: AtomicU64,
    sample_rate: AtomicU64,
    channels: AtomicU64,
    should_stop: AtomicBool,
    seek_to_frame: AtomicU64,
    seek_pending: AtomicBool,
    buffer_flush_pending: AtomicBool,
}

impl SharedPlaybackState {
    fn new() -> Self {
        Self {
            is_playing: AtomicBool::new(false),
            current_frame: AtomicU64::new(0),
            total_frames: AtomicU64::new(0),
            sample_rate: AtomicU64::new(44100),
            channels: AtomicU64::new(2),
            should_stop: AtomicBool::new(false),
            seek_to_frame: AtomicU64::new(0),
            seek_pending: AtomicBool::new(false),
            buffer_flush_pending: AtomicBool::new(false),
        }
    }
}

/// Audio player that manages playback in a background thread
///
/// Uses a ring buffer for streaming, so large files don't need to be
/// loaded entirely into memory.
pub struct AudioPlayer {
    state: Arc<SharedPlaybackState>,
    decoder_thread: Mutex<Option<JoinHandle<()>>>,
    playback_thread: Mutex<Option<JoinHandle<()>>>,
    loaded_file: Mutex<Option<PathBuf>>,
}

// AudioPlayer is Send + Sync because all fields are thread-safe
unsafe impl Send for AudioPlayer {}
unsafe impl Sync for AudioPlayer {}

impl AudioPlayer {
    pub fn new() -> Self {
        Self {
            state: Arc::new(SharedPlaybackState::new()),
            decoder_thread: Mutex::new(None),
            playback_thread: Mutex::new(None),
            loaded_file: Mutex::new(None),
        }
    }

    /// Load and start playing an audio file
    pub fn play_file<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        tracing::debug!("AudioPlayer::play_file - Stopping existing playback");
        // Stop any existing playback
        self.stop();

        let path = path.as_ref().to_path_buf();
        let state = Arc::clone(&self.state);

        // Store the loaded file path
        *self.loaded_file.lock().unwrap() = Some(path.clone());

        // Probe the file FIRST to get metadata before starting threads
        // This prevents race conditions where playback thread reads stale values
        tracing::debug!("AudioPlayer::play_file - Probing file for metadata");
        let (sample_rate, channels, total_frames) = probe_audio_file(&path)?;

        // Store metadata in state BEFORE starting threads
        state.sample_rate.store(sample_rate as u64, Ordering::SeqCst);
        state.channels.store(channels, Ordering::SeqCst);
        state.total_frames.store(total_frames, Ordering::SeqCst);

        tracing::debug!("AudioPlayer::play_file - File metadata: {}Hz, {} channels, {} frames",
            sample_rate, channels, total_frames);

        // Reset state
        state.should_stop.store(false, Ordering::SeqCst);
        state.current_frame.store(0, Ordering::SeqCst);
        state.seek_pending.store(false, Ordering::SeqCst);
        state.is_playing.store(true, Ordering::SeqCst);

        // Create ring buffer for streaming audio
        let ring = HeapRb::<f32>::new(RING_BUFFER_SIZE);
        let (producer, consumer) = ring.split();

        // Wrap consumer in Arc<Mutex> for sharing with audio callback
        let consumer_arc = Arc::new(Mutex::new(consumer));
        let consumer_clone = Arc::clone(&consumer_arc);

        // Start decoder thread
        let decoder_state = Arc::clone(&state);
        let decoder_path = path.clone();
        let decoder_handle = thread::spawn(move || {
            if let Err(e) = run_decoder(decoder_path, decoder_state, producer) {
                tracing::error!("Decoder error: {}", e);
            }
        });

        // Start playback thread
        let playback_state = Arc::clone(&state);
        let playback_handle = thread::spawn(move || {
            if let Err(e) = run_playback_stream(playback_state, consumer_clone) {
                tracing::error!("Playback error: {}", e);
            }
        });

        *self.decoder_thread.lock().unwrap() = Some(decoder_handle);
        *self.playback_thread.lock().unwrap() = Some(playback_handle);

        Ok(())
    }

    /// Pause playback
    pub fn pause(&self) {
        tracing::debug!("AudioPlayer::pause");
        self.state.is_playing.store(false, Ordering::SeqCst);
    }

    /// Resume playback
    pub fn resume(&self) {
        tracing::debug!("AudioPlayer::resume");
        self.state.is_playing.store(true, Ordering::SeqCst);
    }

    /// Toggle play/pause
    pub fn toggle(&self) {
        let current = self.state.is_playing.load(Ordering::SeqCst);
        tracing::debug!("AudioPlayer::toggle (was: {}, now: {})", current, !current);
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
        tracing::debug!("AudioPlayer::stop - Setting should_stop flag and waiting for threads");
        self.state.should_stop.store(true, Ordering::SeqCst);
        self.state.is_playing.store(false, Ordering::SeqCst);

        // Wait for threads to finish
        if let Some(handle) = self.decoder_thread.lock().unwrap().take() {
            tracing::debug!("Waiting for decoder thread to finish");
            let _ = handle.join();
            tracing::debug!("Decoder thread finished");
        }
        if let Some(handle) = self.playback_thread.lock().unwrap().take() {
            tracing::debug!("Waiting for playback thread to finish");
            let _ = handle.join();
            tracing::debug!("Playback thread finished");
        }

        self.state.current_frame.store(0, Ordering::SeqCst);
        tracing::debug!("AudioPlayer::stop - Complete");
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

/// Probe audio file to get metadata (sample rate, channels, total frames)
/// This is called BEFORE starting threads to avoid race conditions
fn probe_audio_file(path: &std::path::Path) -> Result<(u32, u64, u64)> {
    let file = std::fs::File::open(path).map_err(|e| AudioError::FileOpen {
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

    let format = probed.format;

    // Find the audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track found".to_string()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::DecodeFailed("No sample rate".to_string()))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2) as u64;
    let total_frames = track.codec_params.n_frames.unwrap_or(0);

    Ok((sample_rate, channels, total_frames))
}

/// Decoder thread: reads audio file and streams samples to ring buffer
fn run_decoder(
    path: PathBuf,
    state: Arc<SharedPlaybackState>,
    mut producer: ringbuf::HeapProd<f32>,
) -> Result<()> {
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
        .unwrap_or(2) as u64;
    let total_frames = track.codec_params.n_frames.unwrap_or(0);

    // Note: Metadata is already set in play_file() before threads start
    // This avoids race conditions with playback thread reading stale values

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    let spec = SignalSpec::new(
        sample_rate,
        symphonia::core::audio::Channels::FRONT_LEFT
            | symphonia::core::audio::Channels::FRONT_RIGHT,
    );

    tracing::debug!("Decoder thread started - sample_rate: {}Hz, channels: {}, total_frames: {}",
        sample_rate, channels, total_frames);

    // Main decoding loop - streams samples to ring buffer
    let mut packets_decoded = 0;
    let mut samples_written = 0;

    loop {
        // Check for stop signal
        if state.should_stop.load(Ordering::SeqCst) {
            tracing::debug!("Decoder stopping - decoded {} packets, wrote {} samples",
                packets_decoded, samples_written);
            break;
        }

        // Handle seek requests
        if state.seek_pending.load(Ordering::SeqCst) {
            let seek_frame = state.seek_to_frame.load(Ordering::SeqCst);

            // Convert frame to time: seconds = frame / sample_rate
            let seek_seconds = seek_frame as f64 / sample_rate as f64;
            let seconds_whole = seek_seconds.floor() as u64;
            let seconds_frac = seek_seconds - seconds_whole as f64;

            let seek_time = SeekTo::Time {
                time: symphonia::core::units::Time::new(
                    seconds_whole,
                    seconds_frac,
                ),
                track_id: Some(track_id),
            };

            // Attempt to seek
            if let Ok(seeked) = format.seek(SeekMode::Accurate, seek_time) {
                // Update current position to seek target
                state.current_frame.store(seeked.actual_ts, Ordering::SeqCst);

                // Signal consumer to flush ring buffer (discard old audio)
                state.buffer_flush_pending.store(true, Ordering::SeqCst);

                // Reset decoder state to start fresh from seek position
                decoder.reset();
            }

            state.seek_pending.store(false, Ordering::SeqCst);
        }

        // Decode next packet
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => {
                // End of stream
                state.is_playing.store(false, Ordering::SeqCst);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Decode error: {}", e);
                continue;
            }
        };

        packets_decoded += 1;

        // Convert to f32 samples
        let frames = decoded.frames();
        let mut sample_buf = SampleBuffer::<f32>::new(frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        if packets_decoded == 1 {
            tracing::debug!("First packet decoded: {} frames, {} samples", frames, samples.len());
        }

        // Write samples to ring buffer (blocking if buffer is full)
        let mut written = 0;
        while written < samples.len() {
            // Check for stop/seek while waiting
            if state.should_stop.load(Ordering::SeqCst) || state.seek_pending.load(Ordering::SeqCst) {
                break;
            }

            // Try to write remaining samples
            let chunk = &samples[written..];
            let n = producer.push_slice(chunk);
            written += n;
            samples_written += n;

            // If buffer is full and we're paused, sleep briefly
            if n == 0 {
                thread::sleep(Duration::from_millis(10));
            }
        }

        if packets_decoded % 100 == 0 {
            tracing::debug!("Decoded {} packets, {} samples written to buffer",
                packets_decoded, samples_written);
        }
    }

    Ok(())
}

/// Playback thread: reads from ring buffer and outputs to audio device
fn run_playback_stream(
    state: Arc<SharedPlaybackState>,
    consumer: Arc<Mutex<ringbuf::HeapCons<f32>>>,
) -> Result<()> {
    // Get audio device
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| AudioError::DecodeFailed("No output device available".to_string()))?;

    // Use the file's sample rate and channel count (from state)
    let file_sample_rate = state.sample_rate.load(Ordering::SeqCst) as u32;
    let file_channels = state.channels.load(Ordering::SeqCst) as u16;

    // Try to use file's sample rate, but fall back to common rates if unsupported
    // Some devices don't support unusual rates like 24kHz
    let supported_rates = [
        file_sample_rate, // Try file's rate first
        48000,            // Common rate
        44100,            // CD quality
        96000,            // High quality
        22050,            // Lower quality
    ];

    let mut stream_config = StreamConfig {
        channels: file_channels,
        sample_rate: cpal::SampleRate(file_sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Test if we can build a stream with this config
    // Try each sample rate until one works
    let mut successful_rate = file_sample_rate;
    for &rate in &supported_rates {
        stream_config.sample_rate = cpal::SampleRate(rate);

        // Try to build a test stream to see if this config is supported
        match device.build_output_stream(
            &stream_config,
            |_data: &mut [f32], _: &cpal::OutputCallbackInfo| {},
            |_err| {},
            None,
        ) {
            Ok(test_stream) => {
                // Stream creation succeeded - this rate is supported
                successful_rate = rate;
                tracing::debug!("Using sample rate: {}Hz (file: {}Hz)", rate, file_sample_rate);
                drop(test_stream); // Clean up test stream
                break;
            }
            Err(e) => {
                tracing::debug!("Sample rate {}Hz not supported: {}", rate, e);
                continue;
            }
        }
    }

    // Update config with successful rate
    stream_config.sample_rate = cpal::SampleRate(successful_rate);

    // If sample rate differs from file, warn user
    if successful_rate != file_sample_rate {
        tracing::warn!(
            "Device doesn't support {}Hz, using {}Hz instead. Audio may play at different speed.",
            file_sample_rate, successful_rate
        );
    }

    // Clone Arc references for sharing with callback
    let consumer_clone = Arc::clone(&consumer);
    let state_clone = Arc::clone(&state);

    // Track samples consumed for position tracking
    let samples_consumed = Arc::new(AtomicU64::new(0));
    let samples_consumed_clone = Arc::clone(&samples_consumed);

    // Track callback invocations for debugging
    let callback_count = Arc::new(AtomicU64::new(0));
    let callback_count_clone = Arc::clone(&callback_count);

    // Build actual output stream
    let stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let count = callback_count_clone.fetch_add(1, Ordering::SeqCst);
                let is_playing = state_clone.is_playing.load(Ordering::SeqCst);
                let channels = state_clone.channels.load(Ordering::SeqCst);

                if count == 0 {
                    tracing::debug!("Audio callback invoked for first time - buffer size: {}", data.len());
                }

                if !is_playing {
                    // Output silence when paused
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    return;
                }

                // Check if we need to flush the buffer (after seek)
                if state_clone.buffer_flush_pending.load(Ordering::SeqCst) {
                    let mut consumer_guard = consumer_clone.lock().unwrap();

                    // Drain all old audio from ring buffer by skipping everything
                    let to_skip = consumer_guard.occupied_len();
                    consumer_guard.skip(to_skip);

                    // Reset position tracking to match seek target
                    let current_frame = state_clone.current_frame.load(Ordering::SeqCst);
                    samples_consumed_clone.store(current_frame * channels, Ordering::SeqCst);

                    // Clear the flush flag
                    state_clone.buffer_flush_pending.store(false, Ordering::SeqCst);

                    // Output silence this cycle
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    return;
                }

                // Check if seek is in progress
                if state_clone.seek_pending.load(Ordering::SeqCst) {
                    // Seek in progress, output silence
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    return;
                }

                // Read from ring buffer
                let mut consumer_guard = consumer_clone.lock().unwrap();
                let available = consumer_guard.occupied_len();
                let to_read = data.len().min(available);

                if count < 5 || count % 100 == 0 {
                    tracing::debug!("Callback #{}: playing={}, available={}, to_read={}",
                        count, is_playing, available, to_read);
                }

                if to_read > 0 {
                    // Read available samples
                    let read = consumer_guard.pop_slice(&mut data[..to_read]);

                    // Update consumed count and position
                    let consumed = samples_consumed_clone.fetch_add(read as u64, Ordering::SeqCst) + read as u64;
                    let frame = consumed / channels;
                    state_clone.current_frame.store(frame, Ordering::SeqCst);

                    // Fill rest with silence if buffer underrun
                    for sample in data[to_read..].iter_mut() {
                        *sample = 0.0;
                    }
                } else {
                    // Buffer underrun - output silence
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                }
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

    tracing::debug!("Playback stream started successfully");

    // Keep stream alive until stop signal
    loop {
        if state.should_stop.load(Ordering::SeqCst) {
            tracing::debug!("Playback thread stopping");
            break;
        }
        thread::sleep(Duration::from_millis(100));
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
