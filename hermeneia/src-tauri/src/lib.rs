pub mod audio;
pub mod download;
pub mod error;
pub mod gpu;
pub mod gpu_cleanup;
pub mod system_info;
pub mod transcribe;
pub mod translate;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// Re-export for convenience
pub use audio::*;
pub use error::{AudioError, Result};
pub use transcribe::*;



/// Global audio player state managed by Tauri
pub struct AppState {
    pub player: Mutex<AudioPlayer>,
    pub cancel_inference: Mutex<Arc<AtomicBool>>,
    pub cancel_download: Mutex<Arc<AtomicBool>>,
    pub is_downloading: Arc<AtomicBool>,
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Cancel a running inference operation (transcription or translation)
#[tauri::command]
fn cancel_inference(state: tauri::State<AppState>) {
    let flag = state.cancel_inference.lock().unwrap();
    flag.store(true, Ordering::SeqCst);
    tracing::info!("Inference cancellation requested");
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
async fn get_waveform_peaks(
    file_path: String,
    num_peaks: Option<usize>,
) -> std::result::Result<WaveformPeaks, String> {
    // Run blocking audio processing in a dedicated thread pool
    tokio::task::spawn_blocking(move || {
        audio::extract_waveform_peaks(&file_path, num_peaks)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Trim an audio file and save to a new location
///
/// Tauri command that decodes, trims, and re-encodes audio to WAV format.
///
/// # Arguments
/// * `input_path` - Path to source audio file
/// * `output_path` - Path where trimmed audio will be saved
/// * `start_seconds` - Start time in seconds
/// * `end_seconds` - End time in seconds
///
/// # Returns
/// Ok(()) on success, error message string on failure
#[tauri::command]
async fn trim_audio_file(
    input_path: String,
    output_path: String,
    start_seconds: f64,
    end_seconds: f64,
) -> std::result::Result<(), String> {
    // Run blocking audio processing in a dedicated thread pool
    tokio::task::spawn_blocking(move || {
        // Validate parameters
        let params = TrimParams::new(start_seconds, end_seconds)
            .map_err(|e| e.to_string())?;

        // Use optimized trim function (WAV direct copy or streaming)
        audio::trim::trim_audio_file(&input_path, &output_path, &params)
            .map_err(|e| e.to_string())?;

        Ok(())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Transcribe an audio file using Whisper
///
/// # Arguments
/// * `app_handle` - Tauri app handle for emitting progress events
/// * `file_path` - Path to audio file
/// * `model` - Whisper model size (e.g., "tiny", "base")
/// * `task` - "transcribe" or "translate"
/// * `language` - Language code (optional)
/// * `timestamps` - Include timestamp information
#[tauri::command]
async fn transcribe_audio_file(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    file_path: String,
    model: String,
    task: String,
    language: Option<String>,
    timestamps: bool,
) -> std::result::Result<TranscriptResult, String> {
    // Swap in a fresh cancel flag so any previous job's flag stays true
    let cancel_flag = {
        let mut guard = state.cancel_inference.lock().unwrap();
        let new_flag = Arc::new(AtomicBool::new(false));
        *guard = new_flag.clone();
        new_flag
    };

    tokio::task::spawn_blocking(move || {
        let model_enum = parse_whisper_model(&model)
            .ok_or_else(|| format!("Invalid model: {}", model))?;

        let task_enum = match task.as_str() {
            "transcribe" => TranscriptionTask::Transcribe,
            "translate" => TranscriptionTask::Translate,
            _ => return Err(format!("Invalid task: {}", task)),
        };

        let params = TranscribeParams {
            model: model_enum,
            task: task_enum,
            language,
            timestamps,
            force_cpu: false,
            use_quantized: false,
        };

        // Create progress reporter and signal start
        let reporter = TauriProgressReporter::new(app_handle);
        reporter.start();

        transcribe::transcribe_audio_with_reporter(&file_path, params, &reporter, Some(cancel_flag))
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

fn parse_whisper_model(s: &str) -> Option<WhisperModel> {
    match s.to_lowercase().as_str() {
        "tiny" => Some(WhisperModel::Tiny),
        "tiny.en" => Some(WhisperModel::TinyEn),
        "base" => Some(WhisperModel::Base),
        "base.en" => Some(WhisperModel::BaseEn),
        "small" => Some(WhisperModel::Small),
        "small.en" => Some(WhisperModel::SmallEn),
        "medium" => Some(WhisperModel::Medium),
        "medium.en" => Some(WhisperModel::MediumEn),
        "large" => Some(WhisperModel::Large),
        "large-v2" => Some(WhisperModel::LargeV2),
        "large-v3" => Some(WhisperModel::LargeV3),
        _ => None,
    }
}

/// Write text content to a file
#[tauri::command]
async fn write_text_file(path: String, content: String) -> std::result::Result<(), String> {
    tokio::task::spawn_blocking(move || {
        std::fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Get system capabilities (RAM, GPU, VRAM)
#[tauri::command]
async fn get_system_capabilities() -> std::result::Result<system_info::SystemCapabilities, String> {
    system_info::get_system_capabilities()
}

/// Model validation result for frontend
#[derive(serde::Serialize)]
pub struct ModelValidation {
    pub status: String,  // "ok" | "warning" | "error"
    pub messages: Vec<String>,
    pub recommended_model: Option<String>,
}

/// Validate model selection against system capabilities
#[tauri::command]
async fn validate_model_selection(
    model: String,
    force_cpu: bool,
) -> std::result::Result<ModelValidation, String> {
    tokio::task::spawn_blocking(move || {
        let model_enum = parse_whisper_model(&model)
            .ok_or_else(|| format!("Invalid model: {}", model))?;

        let validator = ModelValidator::new().map_err(|e| e.to_string())?;
        let result = validator.validate_model(model_enum, force_cpu);

        Ok(ModelValidation {
            status: match result {
                ValidationResult::Ok => "ok",
                ValidationResult::Warning(_) => "warning",
                ValidationResult::Error(_) => "error",
            }.to_string(),
            messages: match result {
                ValidationResult::Ok => vec![],
                ValidationResult::Warning(w) => w,
                ValidationResult::Error(e) => vec![e],
            },
            recommended_model: Some(validator.recommend_model().model_id().to_string()),
        })
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

// ============================================================================
// Text Translation Commands
// ============================================================================

/// Split plain text into translation-friendly chunks.
/// Splits on paragraph boundaries first, then sentences if paragraphs are too long.
fn split_text_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    for paragraph in text.split("\n\n") {
        let trimmed = paragraph.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.len() <= max_chars {
            chunks.push(trimmed.to_string());
        } else {
            // Split long paragraphs on sentence boundaries (". ", "! ", "? ")
            // We split by finding sentence-ending punctuation followed by a space,
            // which avoids breaking on ellipses, abbreviations, etc.
            let mut current = String::new();
            let mut rest = trimmed;
            while !rest.is_empty() {
                // Find the next sentence boundary: punctuation followed by a space
                let split_pos = rest
                    .char_indices()
                    .skip(1) // don't split at pos 0
                    .find(|&(i, c)| {
                        (c == ' ' || c == '\n')
                            && i > 0
                            && matches!(rest.as_bytes().get(i - 1), Some(b'.' | b'!' | b'?'))
                    })
                    .map(|(i, _)| i);

                let sentence = match split_pos {
                    Some(pos) => {
                        let (s, remainder) = rest.split_at(pos);
                        rest = remainder.trim_start();
                        s
                    }
                    None => {
                        let s = rest;
                        rest = "";
                        s
                    }
                };

                if !current.is_empty() && current.len() + sentence.len() + 1 > max_chars {
                    chunks.push(current.trim().to_string());
                    current = String::new();
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(sentence);
            }
            if !current.trim().is_empty() {
                chunks.push(current.trim().to_string());
            }
        }
    }

    chunks
}

/// Result of a text file translation
#[derive(serde::Serialize)]
pub struct TextTranslationResult {
    /// The translated text content
    pub translated_text: String,
    /// Original text (for comparison)
    pub original_text: String,
    /// Whether this was a subtitle file
    pub is_subtitle: bool,
    /// Source language used
    pub source_language: String,
    /// Target language used
    pub target_language: String,
    /// Model that was used
    pub model_used: String,
    /// Time taken for inference in seconds
    pub inference_time: f64,
    /// Number of segments translated (for SRT files)
    pub segments_translated: usize,
}

/// Translate a text file (.txt or .srt)
///
/// For .srt files, timestamps are preserved and only text content is translated.
/// For .txt files, the entire content is translated as a single block.
///
/// # Arguments
/// * `app_handle` - Tauri app handle for emitting progress events
/// * `file_path` - Path to the text or subtitle file
/// * `source_lang` - Source language code (e.g., "en")
/// * `target_lang` - Target language code (e.g., "es")
#[tauri::command]
async fn translate_text_file(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    file_path: String,
    source_lang: String,
    target_lang: String,
    allow_madlad: bool,
) -> std::result::Result<TextTranslationResult, String> {
    use tauri::Emitter;

    // Swap in a fresh cancel flag so any previous job's flag stays true
    let cancel_flag = {
        let mut guard = state.cancel_inference.lock().unwrap();
        let new_flag = Arc::new(AtomicBool::new(false));
        *guard = new_flag.clone();
        new_flag
    };

    tokio::task::spawn_blocking(move || {
        let start_time = std::time::Instant::now();

        // Read the file content
        let content = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Detect if this is an SRT file by extension
        let is_srt = file_path.to_lowercase().ends_with(".srt");

        // Emit loading model phase
        let _ = app_handle.emit("translation-progress", serde_json::json!({
            "phase": "loading_model",
            "current": null,
            "total": null,
            "message": "Loading translation model..."
        }));

        // Set up translation parameters
        let params = translate::TranslateParams {
            source_language: source_lang.clone(),
            target_language: target_lang.clone(),
            preferred_model: None,
            fallback_enabled: allow_madlad,
            force_cpu: false,
            use_quantized: false,
            max_length: Some(512),
            temperature: Some(0.0),
            top_p: None,
            repetition_penalty: Some(1.0),
        };

        let (translated_text, model_used, segments_count) = if is_srt {
            // Parse SRT file
            let srt_file = translate::SubtitleFile::parse(&content)
                .map_err(|e| format!("Failed to parse SRT file: {}", e))?;

            let total_segments = srt_file.len();
            let texts = srt_file.get_texts();

            // Create progress callback that emits Tauri events
            let app_handle_clone = app_handle.clone();
            let progress_callback: translate::BatchProgressCallback = Box::new(move |current, total, _text| {
                let _ = app_handle_clone.emit("translation-progress", serde_json::json!({
                    "phase": "translating",
                    "current": current,
                    "total": total,
                    "message": format!("Translating segment {} of {}", current, total)
                }));
            });

            // Translate all segments with single model load
            let (translated_texts, model_used, _inference_time) = translate::translate_texts_batch(
                &texts,
                params,
                Some(progress_callback),
                Some(cancel_flag),
            ).map_err(|e| format!("Translation failed: {}", e))?;

            // Reassemble SRT with translated text
            let translated_srt = srt_file.with_translated_text(translated_texts);
            (translated_srt.render(), model_used.display_name().to_string(), total_segments)
        } else {
            // Plain text file - split into chunks and batch translate
            // Keep chunks small (~125 tokens ≈ 500 chars) to match the scale
            // that Marian models handle well (similar to SRT subtitle segments).
            // Longer inputs cause quality degradation and hallucination.
            let chunks = split_text_into_chunks(&content, 500);
            let total_chunks = chunks.len();

            let app_handle_clone = app_handle.clone();
            let progress_callback: translate::BatchProgressCallback =
                Box::new(move |current, total, _text| {
                    let _ = app_handle_clone.emit(
                        "translation-progress",
                        serde_json::json!({
                            "phase": "translating",
                            "current": current,
                            "total": total,
                            "message": format!("Translating chunk {} of {}", current, total)
                        }),
                    );
                });

            let (translated_chunks, model_used, _time) = translate::translate_texts_batch(
                &chunks,
                params,
                Some(progress_callback),
                Some(cancel_flag),
            )
            .map_err(|e| format!("Translation failed: {}", e))?;

            let translated_text = translated_chunks.join("\n\n");
            (translated_text, model_used.display_name().to_string(), total_chunks)
        };

        let inference_time = start_time.elapsed().as_secs_f64();

        Ok(TextTranslationResult {
            translated_text,
            original_text: content,
            is_subtitle: is_srt,
            source_language: source_lang,
            target_language: target_lang,
            model_used,
            inference_time,
            segments_translated: segments_count,
        })
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}


// ============================================================================
// Model Download & Cache Management Commands
// ============================================================================

/// List all available models with their cache status
#[tauri::command]
async fn list_models() -> std::result::Result<Vec<download::ModelInfo>, String> {
    tokio::task::spawn_blocking(|| {
        download::list_all_models().map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Download a model by its HuggingFace model ID with progress events
#[tauri::command]
async fn download_model(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    model_id: String,
    model_name: String,
) -> std::result::Result<(), String> {
    // Prevent concurrent downloads
    if state.is_downloading.swap(true, Ordering::SeqCst) {
        return Err("A download is already in progress".to_string());
    }

    let cancel_flag = {
        let mut guard = state.cancel_download.lock().unwrap();
        let new_flag = Arc::new(AtomicBool::new(false));
        *guard = new_flag.clone();
        new_flag
    };

    let downloading_flag = state.is_downloading.clone();
    let result = tokio::task::spawn_blocking(move || {
        let res = download::download_model(&model_id, &model_name, &app_handle, &cancel_flag)
            .map_err(|e| e.to_string());
        downloading_flag.store(false, Ordering::SeqCst);
        res
    })
    .await
    .map_err(|e| {
        state.is_downloading.store(false, Ordering::SeqCst);
        format!("Task join error: {}", e)
    })?;

    result
}

/// Cancel a running model download
#[tauri::command]
fn cancel_download(state: tauri::State<AppState>) {
    let flag = state.cancel_download.lock().unwrap();
    flag.store(true, Ordering::SeqCst);
    tracing::info!("Download cancellation requested");
}

/// Delete a cached model
#[tauri::command]
async fn delete_model(model_id: String) -> std::result::Result<(), String> {
    tokio::task::spawn_blocking(move || {
        download::delete_model_cache(&model_id).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Get total cache size in bytes
#[tauri::command]
async fn get_cache_size() -> std::result::Result<u64, String> {
    tokio::task::spawn_blocking(download::total_cache_size)
        .await
        .map_err(|e| format!("Task join error: {}", e))
}

/// Check if a specific model is cached
#[tauri::command]
async fn is_model_cached(model_id: String) -> std::result::Result<bool, String> {
    tokio::task::spawn_blocking(move || download::check_model_cached(&model_id))
        .await
        .map_err(|e| format!("Task join error: {}", e))
}

/// Check if a Marian model exists for the given language pair
#[tauri::command]
fn check_marian_pair_supported(source_lang: String, target_lang: String) -> bool {
    translate::language::get_marian_for_pair(&source_lang, &target_lang).is_some()
}

// ============================================================================
// Audio Playback Commands
// ============================================================================

/// Start playing an audio file
#[tauri::command]
fn play_audio(file_path: String, state: tauri::State<AppState>) -> std::result::Result<(), String> {
    tracing::debug!("🎵 COMMAND: play_audio({})", file_path);
    let mut player = state.player.lock().map_err(|e| e.to_string())?;
    player.play_file(&file_path).map_err(|e| e.to_string())
}

/// Pause audio playback
#[tauri::command]
fn pause_audio(state: tauri::State<AppState>) -> std::result::Result<(), String> {
    tracing::debug!("⏸️  COMMAND: pause_audio");
    let player = state.player.lock().map_err(|e| e.to_string())?;
    player.pause();
    Ok(())
}

/// Resume audio playback
#[tauri::command]
fn resume_audio(state: tauri::State<AppState>) -> std::result::Result<(), String> {
    tracing::debug!("▶️  COMMAND: resume_audio");
    let player = state.player.lock().map_err(|e| e.to_string())?;
    player.resume();
    Ok(())
}

/// Toggle play/pause
#[tauri::command]
fn toggle_audio(state: tauri::State<AppState>) -> std::result::Result<(), String> {
    tracing::debug!("🔄 COMMAND: toggle_audio");
    let player = state.player.lock().map_err(|e| e.to_string())?;
    player.toggle();
    Ok(())
}

/// Seek to a specific time in seconds
#[tauri::command]
fn seek_audio(time_seconds: f64, state: tauri::State<AppState>) -> std::result::Result<(), String> {
    tracing::debug!("⏩ COMMAND: seek_audio({})", time_seconds);
    let player = state.player.lock().map_err(|e| e.to_string())?;
    player.seek(time_seconds);
    Ok(())
}

/// Stop audio playback
#[tauri::command]
fn stop_audio(state: tauri::State<AppState>) -> std::result::Result<(), String> {
    tracing::debug!("⏹️  COMMAND: stop_audio");
    let mut player = state.player.lock().map_err(|e| e.to_string())?;
    player.stop();
    Ok(())
}

/// Playback state returned to frontend
#[derive(serde::Serialize)]
pub struct PlaybackInfo {
    pub is_playing: bool,
    pub current_time: f64,
    pub duration: f64,
}

/// Get current playback state
#[tauri::command]
fn get_playback_state(state: tauri::State<AppState>) -> std::result::Result<PlaybackInfo, String> {
    let player = state.player.lock().map_err(|e| e.to_string())?;
    let (is_playing, current_time, duration) = player.get_state();
    Ok(PlaybackInfo {
        is_playing,
        current_time,
        duration,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    gpu::apply_optimizations();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            player: Mutex::new(AudioPlayer::new()),
            cancel_inference: Mutex::new(Arc::new(AtomicBool::new(false))),
            cancel_download: Mutex::new(Arc::new(AtomicBool::new(false))),
            is_downloading: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_waveform_peaks,
            trim_audio_file,
            transcribe_audio_file,
            write_text_file,
            get_system_capabilities,
            validate_model_selection,
            translate_text_file,
            check_marian_pair_supported,
            cancel_inference,
            list_models,
            download_model,
            cancel_download,
            delete_model,
            get_cache_size,
            is_model_cached,
            play_audio,
            pause_audio,
            resume_audio,
            toggle_audio,
            seek_audio,
            stop_audio,
            get_playback_state
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
