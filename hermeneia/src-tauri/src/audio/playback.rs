// src-tauri/src/audio/playback.rs
//! Audio playback module using cpal and symphonia with STREAMING support
//!
//! Uses a ring buffer to stream audio from disk, enabling playback of files
//! of any size without loading everything into memory.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use ringbuf::{traits::*, HeapRb};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use symphonia::core::audio::AudioBufferRef;
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
    sample_rate: AtomicU64,        // File's native sample rate
    device_sample_rate: AtomicU64, // Device's actual sample rate (for resampling)
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
            device_sample_rate: AtomicU64::new(44100),
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

        // Detect device sample rate (may differ from file rate)
        let device_sample_rate = detect_device_sample_rate(sample_rate)?;

        // Store metadata in state BEFORE starting threads
        state
            .sample_rate
            .store(sample_rate as u64, Ordering::SeqCst);
        state
            .device_sample_rate
            .store(device_sample_rate as u64, Ordering::SeqCst);
        state.channels.store(channels, Ordering::SeqCst);
        state.total_frames.store(total_frames, Ordering::SeqCst);

        tracing::debug!(
            "AudioPlayer::play_file - File: {}Hz, {} ch, {} frames | Device: {}Hz",
            sample_rate,
            channels,
            total_frames,
            device_sample_rate
        );

        if sample_rate != device_sample_rate {
            tracing::info!(
                "Resampling enabled: {}Hz -> {}Hz",
                sample_rate,
                device_sample_rate
            );
        }

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

        // Reset all state so get_state() returns duration=0
        // This signals to frontend that threads are dead and need restart
        self.state.current_frame.store(0, Ordering::SeqCst);
        self.state.total_frames.store(0, Ordering::SeqCst);
        self.state.sample_rate.store(44100, Ordering::SeqCst); // Reset to default
        self.state.device_sample_rate.store(44100, Ordering::SeqCst); // Reset to default
        self.state.channels.store(2, Ordering::SeqCst); // Reset to default

        // Clear loaded file
        *self.loaded_file.lock().unwrap() = None;

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
/// For formats like AAC/M4A where channel info isn't in metadata, we decode a packet to detect it
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
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe: {}", e)))?;

    let mut format = probed.format;

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

    // Try to get channels from metadata
    let mut channels = track.codec_params.channels.map(|c| c.count() as u64);
    let track_id = track.id;
    let total_frames = track.codec_params.n_frames.unwrap_or(0);

    // If channels not in metadata (common for AAC/M4A), decode a packet to detect
    if channels.is_none() || channels == Some(0) {
        tracing::debug!("Channel count not in metadata, decoding packet to detect...");

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

        // Find and decode first audio packet
        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(_) => {
                    tracing::warn!("Could not read packet to detect channels, defaulting to 2");
                    channels = Some(2);
                    break;
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    channels = Some(decoded.spec().channels.count() as u64);
                    tracing::debug!("Detected {} channels from decoded audio", channels.unwrap());
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to decode packet for channel detection: {}, defaulting to 2",
                        e
                    );
                    channels = Some(2);
                    break;
                }
            }
        }
    }

    Ok((sample_rate, channels.unwrap_or(2), total_frames))
}

/// Detect the best sample rate supported by the audio device
/// Returns the device's preferred rate, trying file_rate first then common rates
fn detect_device_sample_rate(file_sample_rate: u32) -> Result<u32> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| AudioError::DecodeFailed("No output device available".to_string()))?;

    // Rates to try, in order of preference
    let rates_to_try = [
        file_sample_rate, // Try file's rate first
        48000,            // Common rate
        44100,            // CD quality
        96000,            // High quality
        22050,            // Lower quality
    ];

    for &rate in &rates_to_try {
        let config = StreamConfig {
            channels: 2, // Most devices support stereo
            sample_rate: cpal::SampleRate(rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Try to build a test stream to see if this config is supported
        match device.build_output_stream(
            &config,
            |_data: &mut [f32], _: &cpal::OutputCallbackInfo| {},
            |_err| {},
            None,
        ) {
            Ok(test_stream) => {
                drop(test_stream);
                tracing::debug!("Device supports sample rate: {}Hz", rate);
                return Ok(rate);
            }
            Err(_) => continue,
        }
    }

    // Fallback to 48kHz if nothing else works
    tracing::warn!("Could not detect supported sample rate, defaulting to 48000Hz");
    Ok(48000)
}

/// Decoder thread: reads audio file and streams samples to ring buffer
fn run_decoder(
    path: PathBuf,
    state: Arc<SharedPlaybackState>,
    mut producer: ringbuf::HeapProd<f32>,
) -> Result<()> {
    // Get sample rates and channels from state (already probed in play_file)
    let file_sample_rate = state.sample_rate.load(Ordering::SeqCst) as u32;
    let device_sample_rate = state.device_sample_rate.load(Ordering::SeqCst) as u32;
    let file_channels = state.channels.load(Ordering::SeqCst) as usize;

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
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to probe: {}", e)))?;

    let mut format = probed.format;

    // Find the audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::DecodeFailed("No audio track found".to_string()))?;

    let track_id = track.id;
    let total_frames = track.codec_params.n_frames.unwrap_or(0);

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeFailed(format!("Failed to create decoder: {}", e)))?;

    // Initialize resampler if needed
    let needs_resampling = file_sample_rate != device_sample_rate;
    let mut resampler: Option<SincFixedIn<f32>> = None;
    let mut resample_buffer: Vec<f32> = Vec::new();

    if needs_resampling {
        tracing::info!(
            "Initializing resampler: {}Hz -> {}Hz",
            file_sample_rate,
            device_sample_rate
        );
    }

    tracing::debug!(
        "Decoder thread started - file: {}Hz {}ch, device: {}Hz, frames: {}",
        file_sample_rate,
        file_channels,
        device_sample_rate,
        total_frames
    );

    // Main decoding loop - streams samples to ring buffer
    let mut packets_decoded = 0;
    let mut samples_written = 0;
    let mut just_seeked = false;

    loop {
        // Check for stop signal
        if state.should_stop.load(Ordering::SeqCst) {
            tracing::debug!(
                "Decoder stopping - decoded {} packets, wrote {} samples",
                packets_decoded,
                samples_written
            );
            break;
        }

        // Handle seek requests
        if state.seek_pending.load(Ordering::SeqCst) {
            let seek_frame = state.seek_to_frame.load(Ordering::SeqCst);
            let seek_seconds = seek_frame as f64 / file_sample_rate as f64;

            tracing::debug!(
                "Seek request: frame={}, time={:.3}s",
                seek_frame,
                seek_seconds
            );

            // Try multiple seek strategies for maximum compatibility
            // Strategy 1: SeekTo::TimeStamp (sample-based) - best for PCM/WAV
            // Strategy 2: SeekTo::Time - better for compressed formats (MP3, OGG, etc.)
            // For each, try Accurate mode first, then Coarse as fallback

            // Helper to create TimeStamp seek target
            let make_seek_ts = || SeekTo::TimeStamp {
                ts: seek_frame,
                track_id: track_id,
            };

            // Helper to create Time seek target
            let make_seek_time = || {
                let seconds_whole = seek_seconds.floor() as u64;
                let seconds_frac = seek_seconds - seconds_whole as f64;
                SeekTo::Time {
                    time: symphonia::core::units::Time::new(seconds_whole, seconds_frac),
                    track_id: Some(track_id),
                }
            };

            // Try all strategies in order of preference
            let seek_result = format
                .seek(SeekMode::Accurate, make_seek_ts())
                .or_else(|_| format.seek(SeekMode::Coarse, make_seek_ts()))
                .or_else(|_| format.seek(SeekMode::Accurate, make_seek_time()))
                .or_else(|_| format.seek(SeekMode::Coarse, make_seek_time()));

            match seek_result {
                Ok(seeked) => {
                    tracing::info!(
                        "Seek successful: target={}, actual={}",
                        seek_frame,
                        seeked.actual_ts
                    );

                    // Update current position to seek target
                    state
                        .current_frame
                        .store(seeked.actual_ts, Ordering::SeqCst);

                    // Signal consumer to flush and wait for it to complete
                    state.buffer_flush_pending.store(true, Ordering::SeqCst);

                    // Wait for consumer to clear the flag (indicating flush complete)
                    // Timeout after 100ms to avoid deadlock
                    let flush_start = std::time::Instant::now();
                    while state.buffer_flush_pending.load(Ordering::SeqCst) {
                        if flush_start.elapsed().as_millis() > 100 {
                            tracing::warn!("Buffer flush timeout after seek");
                            break;
                        }
                        thread::sleep(Duration::from_micros(100));
                    }
                    tracing::debug!("Buffer flush completed in {:?}", flush_start.elapsed());

                    // Reset decoder state to start fresh from seek position
                    decoder.reset();
                    just_seeked = true;
                }
                Err(e) => {
                    tracing::error!(
                        "Seek failed, all strategies failed for frame {}: {}",
                        seek_frame,
                        e
                    );
                }
            }

            state.seek_pending.store(false, Ordering::SeqCst);
        }

        // Decode next packet
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(e) => {
                // Check what kind of error - could be EOF or actual error
                let is_eof = matches!(e, symphonia::core::errors::Error::IoError(ref io_err)
                    if io_err.kind() == std::io::ErrorKind::UnexpectedEof)
                    || matches!(e, symphonia::core::errors::Error::ResetRequired);

                tracing::debug!("next_packet error: {:?}, is_eof={}", e, is_eof);

                // End of stream - but don't exit! We need to stay alive for seek requests.
                // Wait for either a seek request or stop signal.
                tracing::debug!("Decoder reached end of stream, waiting for seek or stop");

                loop {
                    if state.should_stop.load(Ordering::SeqCst) {
                        tracing::debug!("Decoder stopping after end of stream");
                        return Ok(());
                    }

                    if state.seek_pending.load(Ordering::SeqCst) {
                        tracing::debug!("Seek requested after end of stream, breaking to handle");
                        break; // Break inner loop to process seek in outer loop
                    }

                    thread::sleep(Duration::from_millis(50));
                }
                continue; // Continue outer loop to handle the seek
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        if just_seeked {
            tracing::info!(
                "First packet after seek: ts={}, dur={}",
                packet.ts(),
                packet.dur()
            );
            just_seeked = false;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Decode error: {}", e);
                continue;
            }
        };

        packets_decoded += 1;

        // Convert decoded audio to interleaved f32 (handles all sample formats)
        let mut samples = convert_audio_buffer_to_f32_interleaved(&decoded);
        let actual_channels = decoded.spec().channels.count();

        if packets_decoded == 1 {
            tracing::debug!(
                "First packet decoded: {} frames, {} samples, {} channels",
                decoded.frames(),
                samples.len(),
                actual_channels
            );
        }

        // Convert mono to stereo for playback (most devices expect stereo)
        if actual_channels == 1 {
            samples = mono_to_stereo(&samples);
        }

        // Apply resampling if needed
        let output_samples = if needs_resampling {
            // Lazy init resampler with first packet's frame count
            if resampler.is_none() {
                let params = SincInterpolationParameters {
                    sinc_len: 128,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 128,
                    window: rubato::WindowFunction::BlackmanHarris2,
                };

                // Output channels is always 2 (stereo) after mono-to-stereo conversion
                let output_channels = 2;
                resampler = Some(
                    SincFixedIn::<f32>::new(
                        device_sample_rate as f64 / file_sample_rate as f64,
                        2.0,
                        params,
                        samples.len() / output_channels,
                        output_channels,
                    )
                    .map_err(|e| AudioError::DecodeFailed(format!("Resampler init: {}", e)))?,
                );
            }

            // De-interleave stereo for rubato (expects separate channel vectors)
            let (left, right): (Vec<f32>, Vec<f32>) = samples
                .chunks(2)
                .map(|chunk| (chunk[0], chunk.get(1).copied().unwrap_or(chunk[0])))
                .unzip();

            let resampler = resampler.as_mut().unwrap();

            // Resize input to match resampler's expected chunk size
            let chunk_size = resampler.input_frames_max();
            let mut left_chunk = left;
            let mut right_chunk = right;

            // Pad or truncate to chunk size
            left_chunk.resize(chunk_size, 0.0);
            right_chunk.resize(chunk_size, 0.0);

            let waves_in = vec![left_chunk, right_chunk];

            match resampler.process(&waves_in, None) {
                Ok(waves_out) => {
                    // Re-interleave the output
                    resample_buffer.clear();
                    let out_frames = waves_out[0].len();
                    resample_buffer.reserve(out_frames * 2);
                    for i in 0..out_frames {
                        resample_buffer.push(waves_out[0][i]);
                        resample_buffer.push(waves_out[1][i]);
                    }
                    &resample_buffer[..]
                }
                Err(e) => {
                    tracing::warn!("Resample error: {}, using original samples", e);
                    &samples[..]
                }
            }
        } else {
            &samples[..]
        };

        // Write samples to ring buffer (blocking if buffer is full)
        let mut written = 0;
        while written < output_samples.len() {
            // Check for stop/seek while waiting
            if state.should_stop.load(Ordering::SeqCst) || state.seek_pending.load(Ordering::SeqCst)
            {
                break;
            }

            // Try to write remaining samples
            let chunk = &output_samples[written..];
            let n = producer.push_slice(chunk);
            written += n;
            samples_written += n;

            // If buffer is full and we're paused, sleep briefly
            if n == 0 {
                thread::sleep(Duration::from_millis(10));
            }
        }

        if packets_decoded % 100 == 0 {
            tracing::debug!(
                "Decoded {} packets, {} samples written to buffer",
                packets_decoded,
                samples_written
            );
        }
    }

    Ok(())
}

/// Convert symphonia AudioBufferRef to interleaved f32 samples
/// Handles all sample formats and properly interleaves channels
fn convert_audio_buffer_to_f32_interleaved(buffer: &AudioBufferRef) -> Vec<f32> {
    let spec = buffer.spec();
    let channels = spec.channels.count();
    let frames = buffer.frames();
    let total_samples = frames * channels;
    let mut output = vec![0.0f32; total_samples];

    match buffer {
        AudioBufferRef::F32(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample;
                }
            }
        }
        AudioBufferRef::F64(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample as f32;
                }
            }
        }
        AudioBufferRef::S16(buf) => {
            let planes = buf.planes();
            const NORM: f32 = 1.0 / 32768.0;
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample as f32 * NORM;
                }
            }
        }
        AudioBufferRef::S32(buf) => {
            let planes = buf.planes();
            const NORM: f32 = 1.0 / 2147483648.0;
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample as f32 * NORM;
                }
            }
        }
        AudioBufferRef::S24(buf) => {
            let planes = buf.planes();
            const NORM: f32 = 1.0 / 8388608.0;
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample.inner() as f32 * NORM;
                }
            }
        }
        AudioBufferRef::S8(buf) => {
            let planes = buf.planes();
            const NORM: f32 = 1.0 / 128.0;
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = sample as f32 * NORM;
                }
            }
        }
        AudioBufferRef::U8(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = (sample as f32 - 128.0) / 128.0;
                }
            }
        }
        AudioBufferRef::U16(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = (sample as f32 - 32768.0) / 32768.0;
                }
            }
        }
        AudioBufferRef::U24(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = (sample.inner() as f32 - 8388608.0) / 8388608.0;
                }
            }
        }
        AudioBufferRef::U32(buf) => {
            let planes = buf.planes();
            for (ch, plane) in planes.planes().iter().enumerate() {
                for (frame, &sample) in plane.iter().enumerate() {
                    output[frame * channels + ch] = (sample as f32 - 2147483648.0) / 2147483648.0;
                }
            }
        }
    }

    output
}

/// Convert mono samples to stereo by duplicating to both channels
fn mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &sample in mono {
        stereo.push(sample);
        stereo.push(sample);
    }
    stereo
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

    // Use the device sample rate determined during play_file()
    // The decoder resamples to this rate, so we use it directly
    let device_sample_rate = state.device_sample_rate.load(Ordering::SeqCst) as u32;

    // Always use stereo output (decoder converts mono to stereo)
    let stream_config = StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(device_sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    tracing::debug!(
        "Playback stream config: {}Hz, 2 channels",
        device_sample_rate
    );

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
                // Always stereo output (decoder converts mono to stereo)
                let channels: u64 = 2;

                if count == 0 {
                    tracing::debug!(
                        "Audio callback invoked for first time - buffer size: {}",
                        data.len()
                    );
                }

                // Check if we need to flush buffer and reset position after seek
                if state_clone.buffer_flush_pending.load(Ordering::SeqCst) {
                    let mut consumer_guard = consumer_clone.lock().unwrap();

                    // Drain all buffered audio
                    let to_skip = consumer_guard.occupied_len();
                    consumer_guard.skip(to_skip);
                    tracing::debug!("Flushed {} samples from ring buffer", to_skip);

                    // Reset position tracking to match seek target
                    let current_frame = state_clone.current_frame.load(Ordering::SeqCst);
                    samples_consumed_clone.store(current_frame * channels, Ordering::SeqCst);

                    // Clear the flush flag to signal decoder it can proceed
                    state_clone
                        .buffer_flush_pending
                        .store(false, Ordering::SeqCst);

                    // Output silence while decoder refills with new position
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    return;
                }

                if !is_playing {
                    // Output silence when paused
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
                    tracing::debug!(
                        "Callback #{}: playing={}, available={}, to_read={}",
                        count,
                        is_playing,
                        available,
                        to_read
                    );
                }

                if to_read > 0 {
                    // Read available samples
                    let read = consumer_guard.pop_slice(&mut data[..to_read]);

                    // Update consumed count and position
                    let consumed = samples_consumed_clone.fetch_add(read as u64, Ordering::SeqCst)
                        + read as u64;
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

    #[test]
    fn test_shared_playback_state_new() {
        let state = SharedPlaybackState::new();
        assert!(!state.is_playing.load(Ordering::SeqCst));
        assert_eq!(state.current_frame.load(Ordering::SeqCst), 0);
        assert_eq!(state.total_frames.load(Ordering::SeqCst), 0);
        assert_eq!(state.sample_rate.load(Ordering::SeqCst), 44100);
        assert_eq!(state.device_sample_rate.load(Ordering::SeqCst), 44100);
        assert_eq!(state.channels.load(Ordering::SeqCst), 2);
        assert!(!state.should_stop.load(Ordering::SeqCst));
        assert!(!state.seek_pending.load(Ordering::SeqCst));
        assert_eq!(state.seek_to_frame.load(Ordering::SeqCst), 0);
        assert!(!state.buffer_flush_pending.load(Ordering::SeqCst));
    }

    #[test]
    fn test_audio_player_pause_resume() {
        let player = AudioPlayer::new();

        // Initially should be paused
        let (is_playing, _, _) = player.get_state();
        assert!(!is_playing);

        // Simulate playing state
        player.state.is_playing.store(true, Ordering::SeqCst);
        let (is_playing, _, _) = player.get_state();
        assert!(is_playing);

        // Pause
        player.pause();
        let (is_playing, _, _) = player.get_state();
        assert!(!is_playing);

        // Resume
        player.resume();
        let (is_playing, _, _) = player.get_state();
        assert!(is_playing);
    }

    #[test]
    fn test_audio_player_toggle() {
        let player = AudioPlayer::new();

        // Initially false
        let (is_playing, _, _) = player.get_state();
        assert!(!is_playing);

        // Toggle to true
        player.toggle();
        let (is_playing, _, _) = player.get_state();
        assert!(is_playing);

        // Toggle back to false
        player.toggle();
        let (is_playing, _, _) = player.get_state();
        assert!(!is_playing);
    }

    #[test]
    fn test_audio_player_seek() {
        let player = AudioPlayer::new();

        // Set sample rate
        player.state.sample_rate.store(48000, Ordering::SeqCst);

        // Seek to 5 seconds
        player.seek(5.0);

        // Check seek parameters
        assert!(player.state.seek_pending.load(Ordering::SeqCst));
        let seek_frame = player.state.seek_to_frame.load(Ordering::SeqCst);
        assert_eq!(seek_frame, 5 * 48000); // 5 seconds * 48000 Hz
    }

    #[test]
    fn test_audio_player_get_state_calculations() {
        let player = AudioPlayer::new();

        // Set up state
        player.state.sample_rate.store(44100, Ordering::SeqCst);
        player.state.channels.store(2, Ordering::SeqCst);
        player.state.total_frames.store(441000, Ordering::SeqCst); // 10 seconds
        player.state.current_frame.store(220500, Ordering::SeqCst); // 5 seconds
        player.state.is_playing.store(true, Ordering::SeqCst);

        let (is_playing, current_time, duration) = player.get_state();

        assert!(is_playing);
        assert_eq!(current_time, 5.0);
        assert_eq!(duration, 10.0);
    }

    #[test]
    fn test_audio_player_get_state_with_zero_sample_rate() {
        let player = AudioPlayer::new();

        // Set sample rate to 0 (edge case)
        player.state.sample_rate.store(0, Ordering::SeqCst);
        player.state.current_frame.store(1000, Ordering::SeqCst);
        player.state.total_frames.store(5000, Ordering::SeqCst);

        let (_, current_time, duration) = player.get_state();

        // Should return 0.0 when sample rate is 0 (avoid division by zero)
        assert_eq!(current_time, 0.0);
        assert_eq!(duration, 0.0);
    }

    #[test]
    fn test_audio_player_stop_sets_flags() {
        let mut player = AudioPlayer::new();

        // Set playing state
        player.state.is_playing.store(true, Ordering::SeqCst);
        player.state.current_frame.store(1000, Ordering::SeqCst);

        // Stop
        player.stop();

        // Check flags
        assert!(!player.state.is_playing.load(Ordering::SeqCst));
        assert!(player.state.should_stop.load(Ordering::SeqCst));
        assert_eq!(player.state.current_frame.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_audio_player_multiple_operations() {
        let player = AudioPlayer::new();

        // Set initial state
        player.state.sample_rate.store(44100, Ordering::SeqCst);
        player.state.total_frames.store(441000, Ordering::SeqCst);

        // Play
        player.resume();
        assert!(player.state.is_playing.load(Ordering::SeqCst));

        // Seek
        player.seek(3.0);
        assert!(player.state.seek_pending.load(Ordering::SeqCst));

        // Pause
        player.pause();
        assert!(!player.state.is_playing.load(Ordering::SeqCst));

        // Resume again
        player.resume();
        assert!(player.state.is_playing.load(Ordering::SeqCst));
    }

    #[test]
    fn test_ring_buffer_size_constant() {
        // Ensure ring buffer size is reasonable (5 seconds at 48kHz stereo)
        assert_eq!(RING_BUFFER_SIZE, 48000 * 2 * 5);
        assert_eq!(RING_BUFFER_SIZE, 480000);
    }

    #[test]
    fn test_audio_player_default_state_values() {
        let player = AudioPlayer::new();

        // Verify default values match SharedPlaybackState::new()
        assert_eq!(player.state.sample_rate.load(Ordering::SeqCst), 44100);
        assert_eq!(player.state.channels.load(Ordering::SeqCst), 2);
        assert_eq!(player.state.current_frame.load(Ordering::SeqCst), 0);
        assert_eq!(player.state.total_frames.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_seek_with_different_sample_rates() {
        let player = AudioPlayer::new();

        // Test with 48kHz
        player.state.sample_rate.store(48000, Ordering::SeqCst);
        player.seek(10.0);
        assert_eq!(player.state.seek_to_frame.load(Ordering::SeqCst), 480000);

        // Test with 44.1kHz
        player.state.sample_rate.store(44100, Ordering::SeqCst);
        player.seek(10.0);
        assert_eq!(player.state.seek_to_frame.load(Ordering::SeqCst), 441000);

        // Test with 96kHz
        player.state.sample_rate.store(96000, Ordering::SeqCst);
        player.seek(5.0);
        assert_eq!(player.state.seek_to_frame.load(Ordering::SeqCst), 480000);
    }

    #[test]
    fn test_audio_player_thread_safety() {
        // Test that AudioPlayer can be used across threads (Send + Sync)
        let player = Arc::new(Mutex::new(AudioPlayer::new()));

        let player_clone = Arc::clone(&player);
        let handle = std::thread::spawn(move || {
            let p = player_clone.lock().unwrap();
            let (_, _, _) = p.get_state();
        });

        handle.join().unwrap();
    }
}
