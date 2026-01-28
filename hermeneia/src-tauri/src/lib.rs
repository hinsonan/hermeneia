pub mod audio;
pub mod error;
pub mod gpu;
pub mod system_info;
pub mod transcribe;
pub mod translate;

use std::sync::Mutex;

// Re-export for convenience
pub use audio::*;
pub use error::{AudioError, Result};
pub use transcribe::*;



/// Global audio player state managed by Tauri
pub struct AppState {
    pub player: Mutex<AudioPlayer>,
}

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
    file_path: String,
    model: String,
    task: String,
    language: Option<String>,
    timestamps: bool,
) -> std::result::Result<TranscriptResult, String> {
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

        transcribe::transcribe_audio_with_reporter(&file_path, params, &reporter)
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
    file_path: String,
    source_lang: String,
    target_lang: String,
) -> std::result::Result<TextTranslationResult, String> {
    use tauri::Emitter;

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

        // Set up translation parameters (Marian-only, auto-select based on pair)
        let params = translate::TranslateParams {
            source_language: source_lang.clone(),
            target_language: target_lang.clone(),
            preferred_model: None, // Auto-select Marian model for the pair
            fallback_enabled: false, // Marian only, no MADLAD fallback
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

            // Translate each segment
            let mut translated_texts = Vec::with_capacity(texts.len());
            let mut model_name = String::new();

            for (i, text) in texts.iter().enumerate() {
                // Emit progress
                let _ = app_handle.emit("translation-progress", serde_json::json!({
                    "phase": "translating",
                    "current": i + 1,
                    "total": total_segments,
                    "message": format!("Translating segment {} of {}", i + 1, total_segments)
                }));

                // Translate this segment
                let result = translate::translate_text(text, params.clone())
                    .map_err(|e| format!("Translation failed at segment {}: {}", i + 1, e))?;

                translated_texts.push(result.translated_text);
                model_name = result.model_used.display_name().to_string();
            }

            // Reassemble SRT with translated text
            let translated_srt = srt_file.with_translated_text(translated_texts);
            (translated_srt.render(), model_name, total_segments)
        } else {
            // Plain text file - translate as a single block
            let _ = app_handle.emit("translation-progress", serde_json::json!({
                "phase": "translating",
                "current": 1,
                "total": 1,
                "message": "Translating text..."
            }));

            let result = translate::translate_text(&content, params)
                .map_err(|e| format!("Translation failed: {}", e))?;

            (result.translated_text, result.model_used.display_name().to_string(), 1)
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

/// Get supported Marian language pairs for the UI
#[tauri::command]
fn get_marian_language_pairs() -> Vec<(String, String, String)> {
    // Returns (source_code, target_code, display_name)
    vec![
        // Common pairs (English as source)
        ("en".to_string(), "es".to_string(), "English to Spanish".to_string()),
        ("en".to_string(), "fr".to_string(), "English to French".to_string()),
        ("en".to_string(), "de".to_string(), "English to German".to_string()),
        ("en".to_string(), "pt".to_string(), "English to Portuguese".to_string()),
        ("en".to_string(), "it".to_string(), "English to Italian".to_string()),
        ("en".to_string(), "ru".to_string(), "English to Russian".to_string()),
        ("en".to_string(), "zh".to_string(), "English to Chinese".to_string()),
        ("en".to_string(), "ja".to_string(), "English to Japanese".to_string()),
        ("en".to_string(), "ko".to_string(), "English to Korean".to_string()),
        ("en".to_string(), "ar".to_string(), "English to Arabic".to_string()),
        ("en".to_string(), "nl".to_string(), "English to Dutch".to_string()),
        ("en".to_string(), "pl".to_string(), "English to Polish".to_string()),
        ("en".to_string(), "tr".to_string(), "English to Turkish".to_string()),
        ("en".to_string(), "vi".to_string(), "English to Vietnamese".to_string()),
        ("en".to_string(), "th".to_string(), "English to Thai".to_string()),
        ("en".to_string(), "id".to_string(), "English to Indonesian".to_string()),
        ("en".to_string(), "hi".to_string(), "English to Hindi".to_string()),
        ("en".to_string(), "he".to_string(), "English to Hebrew".to_string()),
        ("en".to_string(), "el".to_string(), "English to Greek".to_string()),
        ("en".to_string(), "sv".to_string(), "English to Swedish".to_string()),
        // Reverse pairs (to English)
        ("es".to_string(), "en".to_string(), "Spanish to English".to_string()),
        ("fr".to_string(), "en".to_string(), "French to English".to_string()),
        ("de".to_string(), "en".to_string(), "German to English".to_string()),
        ("pt".to_string(), "en".to_string(), "Portuguese to English".to_string()),
        ("it".to_string(), "en".to_string(), "Italian to English".to_string()),
        ("ru".to_string(), "en".to_string(), "Russian to English".to_string()),
        ("zh".to_string(), "en".to_string(), "Chinese to English".to_string()),
        ("ja".to_string(), "en".to_string(), "Japanese to English".to_string()),
        ("ko".to_string(), "en".to_string(), "Korean to English".to_string()),
        ("ar".to_string(), "en".to_string(), "Arabic to English".to_string()),
    ]
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
            get_marian_language_pairs,
            check_marian_pair_supported,
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
