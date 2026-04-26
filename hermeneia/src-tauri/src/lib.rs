pub mod annotate;
pub mod audio;
pub mod cancel_registry;
pub mod download;
pub mod error;
pub mod gpu;
pub mod gpu_cleanup;
pub mod hf_cache;
pub mod runtime_cache;
pub mod runtime_pool;
pub mod speaker;
pub mod system_info;
pub mod transcribe;
pub mod translate;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

use cancel_registry::CancelRegistry;

// Re-export for convenience
pub use audio::*;
pub use error::{AudioError, Result};
pub use transcribe::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct UpdateInfo {
    pub available: bool,
    pub current_version: String,
    pub latest_version: String,
    pub release_url: String,
    pub release_notes: String,
}

const ANNOTATION_PROGRESS_EVENT: &str = "annotation-progress";
const DEFAULT_MAX_INFERENCE_CONCURRENCY: usize = 1;
const HARD_MAX_INFERENCE_CONCURRENCY: usize = 4;
const INFERENCE_RAM_HEADROOM_GB: f32 = 1.5;
const INFERENCE_VRAM_HEADROOM_GB: f32 = 0.8;

#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceRuntimeLimits {
    pub max_inference_concurrency: usize,
}

#[derive(Debug, Clone, Copy)]
struct InferenceMemoryRequirements {
    ram_gb: f32,
    vram_gb: f32,
}

struct TauriAnnotationProgressReporter {
    app_handle: tauri::AppHandle,
    job_id: String,
}

impl annotate::AnnotationProgressReporter for TauriAnnotationProgressReporter {
    fn report(&self, mut progress: annotate::AnnotationProgress) {
        use tauri::Emitter;

        progress.job_id = self.job_id.clone();

        if let Err(e) = self.app_handle.emit(ANNOTATION_PROGRESS_EVENT, progress) {
            tracing::warn!("Failed to emit annotation progress event: {}", e);
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SpeakerModelRequirement {
    pub key: String,
    pub display_name: String,
    pub approx_size_mb: f32,
    pub is_cached: bool,
    pub segmentation_model_id: String,
    pub segmentation_file: String,
    pub embedding_model_id: String,
    pub embedding_file: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranslationStrategy {
    Auto,
    FastOnly,
    Universal,
}

impl TranslationStrategy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::FastOnly => "fast_only",
            Self::Universal => "universal",
        }
    }
}

fn parse_translation_strategy(
    strategy: Option<String>,
    allow_madlad: Option<bool>,
) -> std::result::Result<TranslationStrategy, String> {
    if let Some(strategy) = strategy {
        let normalized = strategy.trim().to_lowercase();
        return match normalized.as_str() {
            "auto" => Ok(TranslationStrategy::Auto),
            "fast_only" | "fast-only" => Ok(TranslationStrategy::FastOnly),
            "universal" => Ok(TranslationStrategy::Universal),
            _ => Err(
                "Invalid translation strategy. Accepted values: auto, fast_only, universal"
                    .to_string(),
            ),
        };
    }

    Ok(match allow_madlad.unwrap_or(true) {
        true => TranslationStrategy::Auto,
        false => TranslationStrategy::FastOnly,
    })
}

fn build_translate_params_for_strategy(
    source_lang: String,
    target_lang: String,
    strategy: TranslationStrategy,
) -> std::result::Result<translate::TranslateParams, AudioError> {
    use translate::TranslationModel;

    let (preferred_model, fallback_enabled) = match strategy {
        TranslationStrategy::Auto => (None, true),
        TranslationStrategy::FastOnly => {
            let marian = translate::language::get_marian_for_pair(&source_lang, &target_lang)
                .ok_or_else(|| AudioError::ModelNotAvailable {
                    model: format!(
                        "Fast translation model for {} -> {}",
                        source_lang, target_lang
                    ),
                })?;
            (Some(marian), false)
        }
        TranslationStrategy::Universal => (Some(TranslationModel::Madlad3B), false),
    };

    Ok(translate::TranslateParams {
        source_language: source_lang,
        target_language: target_lang,
        preferred_model,
        fallback_enabled,
        force_cpu: false,
        use_quantized: false,
        max_length: Some(512),
        temperature: Some(0.0),
        top_p: None,
        repetition_penalty: Some(1.0),
    })
}

fn translation_inference_requirements(
    strategy: TranslationStrategy,
) -> InferenceMemoryRequirements {
    match strategy {
        TranslationStrategy::Universal => InferenceMemoryRequirements {
            // MADLAD 3B is large and can exceed 12GB VRAM once loaded.
            ram_gb: 14.0,
            vram_gb: 13.0,
        },
        // Auto may pick universal depending on language pair, so budget as universal.
        TranslationStrategy::Auto => InferenceMemoryRequirements {
            ram_gb: 14.0,
            vram_gb: 13.0,
        },
        TranslationStrategy::FastOnly => InferenceMemoryRequirements {
            // Marian models are much smaller (~298MB files) but runtime overhead still applies.
            ram_gb: 2.0,
            vram_gb: 1.5,
        },
    }
}

fn whisper_inference_requirements(model: transcribe::WhisperModel) -> InferenceMemoryRequirements {
    let reqs = model.requirements();
    InferenceMemoryRequirements {
        ram_gb: reqs.min_ram_gb,
        vram_gb: reqs.min_vram_gb,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TranslationModelResolution {
    pub strategy: String,
    pub model_id: String,
    pub model_name: String,
    pub engine_tier: String,
    pub label: String,
    pub message: String,
    pub speed_label: String,
    pub user_hint: String,
}

/// Global audio player state managed by Tauri
pub struct AppState {
    pub player: Mutex<AudioPlayer>,
    pub inference_cancel_registry: Arc<CancelRegistry>,
    pub inference_semaphore: Arc<Semaphore>,
    pub max_inference_concurrency: usize,
    pub cancel_download: Mutex<Arc<AtomicBool>>,
    pub is_downloading: Arc<AtomicBool>,
    pub runtime_cache: Arc<runtime_cache::RuntimeCacheManager>,
}

fn slots_for_memory_budget(available_gb: f32, headroom_gb: f32, per_job_gb: f32) -> usize {
    if !available_gb.is_finite() || !headroom_gb.is_finite() || !per_job_gb.is_finite() {
        return DEFAULT_MAX_INFERENCE_CONCURRENCY;
    }

    if available_gb <= headroom_gb || per_job_gb <= 0.0 {
        return DEFAULT_MAX_INFERENCE_CONCURRENCY;
    }

    let slots = ((available_gb - headroom_gb) / per_job_gb).floor() as isize;
    slots.max(1) as usize
}

fn max_requirements(
    a: InferenceMemoryRequirements,
    b: InferenceMemoryRequirements,
) -> InferenceMemoryRequirements {
    InferenceMemoryRequirements {
        ram_gb: a.ram_gb.max(b.ram_gb),
        vram_gb: a.vram_gb.max(b.vram_gb),
    }
}

fn compute_slots_from_requirements(
    caps: &system_info::SystemCapabilities,
    requirements: InferenceMemoryRequirements,
    force_cpu: bool,
) -> usize {
    let ram_slots = slots_for_memory_budget(
        caps.available_ram_gb,
        INFERENCE_RAM_HEADROOM_GB,
        requirements.ram_gb,
    );

    if force_cpu {
        return ram_slots;
    }

    let gpu_slots = caps.gpu_info.as_ref().map(|gpu| {
        if let Some(vram_available) = gpu.vram_available_gb {
            slots_for_memory_budget(
                vram_available,
                INFERENCE_VRAM_HEADROOM_GB,
                requirements.vram_gb,
            )
        } else {
            DEFAULT_MAX_INFERENCE_CONCURRENCY
        }
    });

    match gpu_slots {
        Some(gpu_slots) => ram_slots.min(gpu_slots),
        None => ram_slots,
    }
}

fn calculate_max_inference_concurrency_for_selection(
    caps: &system_info::SystemCapabilities,
    selected_whisper_model: Option<transcribe::WhisperModel>,
    whisper_force_cpu: bool,
    translation_strategy: Option<TranslationStrategy>,
    source_lang: Option<&str>,
    target_lang: Option<&str>,
) -> usize {
    let mut requirements = InferenceMemoryRequirements {
        ram_gb: 2.0,
        vram_gb: 1.5,
    };

    let mut force_cpu = whisper_force_cpu;

    if let Some(whisper_model) = selected_whisper_model {
        requirements =
            max_requirements(requirements, whisper_inference_requirements(whisper_model));
    }

    if let Some(strategy) = translation_strategy {
        let effective_strategy = match strategy {
            TranslationStrategy::Auto => {
                if let (Some(src), Some(tgt)) = (source_lang, target_lang) {
                    if translate::language::get_marian_for_pair(src, tgt).is_some() {
                        TranslationStrategy::FastOnly
                    } else {
                        TranslationStrategy::Universal
                    }
                } else {
                    TranslationStrategy::Universal
                }
            }
            other => other,
        };

        requirements = max_requirements(
            requirements,
            translation_inference_requirements(effective_strategy),
        );

        // Translation currently runs on auto device selection; keep GPU path enabled when available.
        force_cpu = false;
    }

    compute_slots_from_requirements(caps, requirements, force_cpu).clamp(
        DEFAULT_MAX_INFERENCE_CONCURRENCY,
        HARD_MAX_INFERENCE_CONCURRENCY,
    )
}

fn generate_job_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
    format!("{}-{}-{}", prefix, ts, seq)
}

async fn acquire_inference_permit(
    inference_semaphore: Arc<Semaphore>,
    cancel_flag: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
) -> std::result::Result<OwnedSemaphorePermit, String> {
    loop {
        if cancel_flag.load(Ordering::SeqCst) {
            return Err(AudioError::Cancelled.to_string());
        }

        tokio::select! {
            permit_result = inference_semaphore.clone().acquire_owned() => {
                let permit = permit_result.map_err(|_| "Inference queue is unavailable".to_string())?;
                if cancel_flag.load(Ordering::SeqCst) {
                    return Err(AudioError::Cancelled.to_string());
                }
                return Ok(permit);
            }
            _ = cancel_notify.notified() => {
                if cancel_flag.load(Ordering::SeqCst) {
                    return Err(AudioError::Cancelled.to_string());
                }
            }
        }
    }
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Cancel a running inference operation (transcription or translation)
#[tauri::command]
fn cancel_inference(state: tauri::State<AppState>) -> usize {
    let cancelled = state.inference_cancel_registry.cancel_all();
    tracing::info!(
        cancelled_jobs = cancelled,
        "Inference cancellation requested"
    );
    cancelled
}

#[tauri::command]
fn cancel_job(state: tauri::State<AppState>, job_id: String) -> bool {
    state.inference_cancel_registry.cancel_job(&job_id)
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
        audio::extract_waveform_peaks(&file_path, num_peaks).map_err(|e| e.to_string())
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
        let params = TrimParams::new(start_seconds, end_seconds).map_err(|e| e.to_string())?;

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
    job_id: Option<String>,
    batch_id: Option<String>,
) -> std::result::Result<TranscriptResult, String> {
    let job_id = job_id.unwrap_or_else(|| generate_job_id("transcribe"));
    let runtime_cache = state.runtime_cache.clone();
    let cancel_registry = state.inference_cancel_registry.clone();
    let registration = cancel_registry.register_job(job_id.clone(), batch_id);
    let cancel_flag = registration.cancel_flag();
    let cancel_notify = registration.cancel_notify();
    let inference_permit = acquire_inference_permit(
        state.inference_semaphore.clone(),
        cancel_flag.clone(),
        cancel_notify,
    )
    .await?;

    tokio::task::spawn_blocking(move || {
        let _inference_permit = inference_permit;
        let _registration = registration;
        let started = Instant::now();

        let model_enum = annotate::parse_whisper_model(&model).map_err(|e| e.to_string())?;

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

        // Create progress reporter
        let reporter = Arc::new(TauriProgressReporter::new(app_handle, job_id));

        // Stage 1: decode audio (with explicit decode progress events)
        reporter.emit_decoding_audio();
        let reporter_for_decode = reporter.clone();
        let cancel_for_decode = cancel_flag.clone();
        let decode_progress: DecodeProgressCallback = Box::new(move |current, total| {
            reporter_for_decode.emit_decoding_audio_progress(current, total);

            !cancel_for_decode.load(Ordering::SeqCst)
        });

        let audio_data = decode_audio_file_with_progress(&file_path, Some(decode_progress))
            .map_err(|e| e.to_string())?;

        if cancel_flag.load(Ordering::SeqCst) {
            return Err(AudioError::Cancelled.to_string());
        }

        // Stage 2: prepare mono 16kHz speech audio
        reporter.emit_preparing_audio();
        let speech_audio = prepare_speech_audio_owned(audio_data).map_err(|e| e.to_string())?;

        if cancel_flag.load(Ordering::SeqCst) {
            return Err(AudioError::Cancelled.to_string());
        }

        // Stage 3+: load model + transcribe
        reporter.start();
        let result = transcribe::transcribe_prepared_audio_with_reporter_cached(
            &speech_audio,
            params,
            reporter.clone(),
            Some(cancel_flag.clone()),
            Some(runtime_cache),
        )
        .map_err(|e| e.to_string());

        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            active_jobs = cancel_registry.active_jobs(),
            "Transcription job finished"
        );

        result
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Run full speaker annotation pipeline (diarize + transcribe + merge).
#[tauri::command]
async fn annotate_audio_file(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    file_path: String,
    transcribe_model: String,
    speaker_model: String,
    task: String,
    language: Option<String>,
    timestamps: bool,
    num_speakers: Option<i32>,
    threshold: f32,
    device: String,
    speaker_names: Option<std::collections::HashMap<i32, String>>,
    job_id: Option<String>,
    batch_id: Option<String>,
) -> std::result::Result<annotate::AnnotatedResult, String> {
    let job_id = job_id.unwrap_or_else(|| generate_job_id("annotate"));
    let runtime_cache = state.runtime_cache.clone();
    let cancel_registry = state.inference_cancel_registry.clone();
    let registration = cancel_registry.register_job(job_id.clone(), batch_id);
    let cancel_flag = registration.cancel_flag();
    let cancel_notify = registration.cancel_notify();
    let inference_permit = acquire_inference_permit(
        state.inference_semaphore.clone(),
        cancel_flag.clone(),
        cancel_notify,
    )
    .await?;

    tokio::task::spawn_blocking(move || {
        let _inference_permit = inference_permit;
        let _registration = registration;
        let started = Instant::now();

        let transcribe_model_enum =
            annotate::parse_whisper_model(&transcribe_model).map_err(|e| e.to_string())?;
        let speaker_model_enum =
            annotate::parse_speaker_model(&speaker_model).map_err(|e| e.to_string())?;
        let task_enum = annotate::parse_task(&task).map_err(|e| e.to_string())?;
        let device_enum = annotate::parse_speaker_device(&device).map_err(|e| e.to_string())?;

        let annotate_params = annotate::AnnotateParams {
            transcribe: TranscribeParams {
                model: transcribe_model_enum,
                task: task_enum,
                language,
                timestamps,
                force_cpu: false,
                use_quantized: false,
            },
            diarize: speaker::DiarizeParams {
                model: speaker_model_enum,
                num_speakers,
                threshold,
                device: device_enum,
            },
            speaker_names: speaker_names.unwrap_or_default(),
        };

        let reporter = Arc::new(TauriAnnotationProgressReporter {
            app_handle,
            job_id: job_id.clone(),
        });
        let result = annotate::annotate_audio_with_reporter_cached(
            &file_path,
            annotate_params,
            reporter,
            &job_id,
            Some(cancel_flag.clone()),
            Some(runtime_cache),
        )
        .map_err(|e| e.to_string());

        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            active_jobs = cancel_registry.active_jobs(),
            "Annotation job finished"
        );

        result
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Returns speaker diarization model bundles and required Hugging Face IDs.
#[tauri::command]
fn list_speaker_model_requirements() -> Vec<SpeakerModelRequirement> {
    let models = [
        speaker::SpeakerModel::English,
        speaker::SpeakerModel::Multilingual,
    ];
    models
        .iter()
        .map(|model| {
            let (seg_repo, seg_file) = model.segmentation_source();
            let (emb_repo, emb_file) = model.embedding_source();

            SpeakerModelRequirement {
                key: model.cli_key().to_string(),
                display_name: model.display_name().to_string(),
                approx_size_mb: model.approx_size_mb(),
                is_cached: speaker::SpeakerModelManager::is_cached(model),
                segmentation_model_id: seg_repo.to_string(),
                segmentation_file: seg_file.to_string(),
                embedding_model_id: emb_repo.to_string(),
                embedding_file: emb_file.to_string(),
            }
        })
        .collect()
}

/// Ensure speaker diarization model bundle is downloaded and cached.
#[tauri::command]
async fn ensure_speaker_model_downloaded(model: String) -> std::result::Result<(), String> {
    tokio::task::spawn_blocking(move || {
        let model_enum = annotate::parse_speaker_model(&model).map_err(|e| e.to_string())?;
        speaker::SpeakerModelManager::ensure_models(&model_enum)
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

#[tauri::command]
fn get_runtime_cache_stats(state: tauri::State<'_, AppState>) -> runtime_cache::RuntimeCacheStats {
    state.runtime_cache.stats()
}

#[tauri::command]
fn clear_runtime_cache(
    state: tauri::State<'_, AppState>,
    kind: String,
) -> std::result::Result<(), String> {
    match kind.as_str() {
        "whisper" => state.runtime_cache.clear_whisper().map(|_| ()),
        "speaker" => state.runtime_cache.clear_speaker().map(|_| ()),
        "all" => state.runtime_cache.clear_all().map(|_| ()),
        _ => {
            return Err(format!(
                "Invalid cache kind: {} (expected whisper|speaker|all)",
                kind
            ))
        }
    }
    .map_err(|e| e.to_string())
}

fn decode_percent_encoded(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = bytes[i + 1] as char;
            let h2 = bytes[i + 2] as char;
            let hex = format!("{}{}", h1, h2);
            if let Ok(value) = u8::from_str_radix(&hex, 16) {
                out.push(value);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn normalize_output_path(raw_path: &str) -> std::path::PathBuf {
    let trimmed = raw_path.trim();
    let without_scheme = trimmed.strip_prefix("file://").unwrap_or(trimmed);

    #[cfg(target_os = "windows")]
    let without_scheme = {
        if without_scheme.len() > 3
            && without_scheme.as_bytes()[0] == b'/'
            && without_scheme.as_bytes()[2] == b':'
        {
            &without_scheme[1..]
        } else {
            without_scheme
        }
    };

    std::path::PathBuf::from(decode_percent_encoded(without_scheme))
}

#[derive(Debug, serde::Deserialize)]
struct ZipArchiveEntry {
    path: String,
    content: String,
}

/// Write text content to a file
#[tauri::command]
async fn write_text_file(path: String, content: String) -> std::result::Result<(), String> {
    tokio::task::spawn_blocking(move || {
        let output_path = normalize_output_path(&path);
        if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent directory: {}", e))?;
        }
        std::fs::write(&output_path, content).map_err(|e| format!("Failed to write file: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Write a zip archive composed of multiple text entries
#[tauri::command]
async fn write_zip_archive(
    path: String,
    entries: Vec<ZipArchiveEntry>,
) -> std::result::Result<(), String> {
    tokio::task::spawn_blocking(move || {
        use std::fs::File;
        use std::io::Write;
        use std::path::PathBuf;
        use zip::write::SimpleFileOptions;

        let mut output_path = normalize_output_path(&path);
        if output_path.extension().is_none() {
            output_path.set_extension("zip");
        }
        if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create export directory: {}", e))?;
        }

        let temp_name = format!(
            ".{}.tmp-{}-{}.zip",
            output_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("export"),
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let temp_path = output_path
            .parent()
            .map(|p| p.join(&temp_name))
            .unwrap_or_else(|| PathBuf::from(&temp_name));

        let file = File::create(&temp_path)
            .map_err(|e| format!("Failed to create temporary zip file: {}", e))?;
        let mut zip = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        for entry in entries {
            let normalized = entry.path.trim().replace('\\', "/");
            if normalized.is_empty() {
                let _ = std::fs::remove_file(&temp_path);
                return Err("Archive entry path cannot be empty".to_string());
            }
            if normalized.starts_with('/') || normalized.split('/').any(|part| part == "..") {
                let _ = std::fs::remove_file(&temp_path);
                return Err(format!("Invalid archive entry path: {}", entry.path));
            }

            zip.start_file(&normalized, options).map_err(|e| {
                let _ = std::fs::remove_file(&temp_path);
                format!("Failed to add '{}' to zip: {}", normalized, e)
            })?;
            zip.write_all(entry.content.as_bytes()).map_err(|e| {
                let _ = std::fs::remove_file(&temp_path);
                format!("Failed to write '{}' in zip: {}", normalized, e)
            })?;
        }

        zip.finish().map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!("Failed to finalize zip archive: {}", e)
        })?;

        if output_path.exists() {
            std::fs::remove_file(&output_path)
                .map_err(|e| format!("Failed to replace existing zip archive: {}", e))?;
        }
        std::fs::rename(&temp_path, &output_path)
            .map_err(|e| format!("Failed to move zip archive into place: {}", e))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Get system capabilities (RAM, GPU, VRAM)
#[tauri::command]
async fn get_system_capabilities() -> std::result::Result<system_info::SystemCapabilities, String> {
    system_info::get_system_capabilities()
}

/// Get runtime limits used by the backend scheduler.
#[tauri::command]
async fn get_inference_runtime_limits(
    state: tauri::State<'_, AppState>,
) -> std::result::Result<InferenceRuntimeLimits, String> {
    Ok(InferenceRuntimeLimits {
        max_inference_concurrency: state.max_inference_concurrency,
    })
}

/// Model validation result for frontend
#[derive(serde::Serialize)]
pub struct ModelValidation {
    pub status: String, // "ok" | "warning" | "error"
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
        let model_enum = annotate::parse_whisper_model(&model).map_err(|e| e.to_string())?;

        let validator = ModelValidator::new().map_err(|e| e.to_string())?;
        let result = validator.validate_model(model_enum, force_cpu);

        Ok(ModelValidation {
            status: match result {
                ValidationResult::Ok => "ok",
                ValidationResult::Warning(_) => "warning",
                ValidationResult::Error(_) => "error",
            }
            .to_string(),
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
    strategy: Option<String>,
    allow_madlad: Option<bool>,
    job_id: Option<String>,
    batch_id: Option<String>,
) -> std::result::Result<TextTranslationResult, String> {
    use tauri::Emitter;

    let job_id = job_id.unwrap_or_else(|| generate_job_id("translate"));
    let strategy = parse_translation_strategy(strategy, allow_madlad)?;
    let cancel_registry = state.inference_cancel_registry.clone();
    let registration = cancel_registry.register_job(job_id.clone(), batch_id);
    let cancel_flag = registration.cancel_flag();
    let cancel_notify = registration.cancel_notify();
    let inference_permit = acquire_inference_permit(
        state.inference_semaphore.clone(),
        cancel_flag.clone(),
        cancel_notify,
    )
    .await?;

    tokio::task::spawn_blocking(move || {
        let _inference_permit = inference_permit;
        let _registration = registration;
        let start_time = std::time::Instant::now();

        // Read the file content
        let content = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Detect if this is an SRT file by extension
        let is_srt = file_path.to_lowercase().ends_with(".srt");

        // Emit loading model phase
        let _ = app_handle.emit(
            "translation-progress",
            serde_json::json!({
                "job_id": job_id.clone(),
                "phase": "loading_model",
                "current": null,
                "total": null,
                "message": "Loading translation model..."
            }),
        );

        // Set up translation parameters
        let params =
            build_translate_params_for_strategy(source_lang.clone(), target_lang.clone(), strategy)
                .map_err(|e| e.to_string())?;

        let (translated_text, model_used, segments_count) = if is_srt {
            // Parse SRT file
            let srt_file = translate::SubtitleFile::parse(&content)
                .map_err(|e| format!("Failed to parse SRT file: {}", e))?;

            let total_segments = srt_file.len();
            let texts = srt_file.get_texts_for_translation_ref();

            // Create progress callback that emits Tauri events
            let app_handle_clone = app_handle.clone();
            let progress_job_id = job_id.clone();
            let progress_callback: translate::BatchProgressCallback =
                Box::new(move |current, total, _text| {
                    let _ = app_handle_clone.emit(
                        "translation-progress",
                        serde_json::json!({
                            "job_id": progress_job_id.clone(),
                            "phase": "translating",
                            "current": current,
                            "total": total,
                            "message": format!("Translating segment {} of {}", current, total)
                        }),
                    );
                });

            // Translate all segments with single model load
            let (translated_texts, model_used, _inference_time) = translate::translate_texts_batch(
                &texts,
                params,
                Some(progress_callback),
                Some(cancel_flag),
            )
            .map_err(|e| format!("Translation failed: {}", e))?;

            // Reassemble SRT with translated text
            let translated_srt = srt_file.with_translated_text_preserving_labels(translated_texts);
            (
                translated_srt.render(),
                model_used.display_name().to_string(),
                total_segments,
            )
        } else {
            // Plain text file - split into chunks and batch translate
            // Keep chunks small (~125 tokens ≈ 500 chars) to match the scale
            // that Marian models handle well (similar to SRT subtitle segments).
            // Longer inputs cause quality degradation and hallucination.
            let chunks = split_text_into_chunks(&content, 500);
            let total_chunks = chunks.len();

            let app_handle_clone = app_handle.clone();
            let progress_job_id = job_id.clone();
            let progress_callback: translate::BatchProgressCallback =
                Box::new(move |current, total, _text| {
                    let _ = app_handle_clone.emit(
                        "translation-progress",
                        serde_json::json!({
                            "job_id": progress_job_id.clone(),
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
            (
                translated_text,
                model_used.display_name().to_string(),
                total_chunks,
            )
        };

        let inference_time = start_time.elapsed().as_secs_f64();
        tracing::info!(
            elapsed_ms = start_time.elapsed().as_millis(),
            active_jobs = cancel_registry.active_jobs(),
            "Translation job finished"
        );

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
    tokio::task::spawn_blocking(|| download::list_all_models().map_err(|e| e.to_string()))
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

#[tauri::command]
async fn recommend_inference_concurrency(
    whisper_model: Option<String>,
    whisper_force_cpu: Option<bool>,
    translation_strategy: Option<String>,
    translation_source_lang: Option<String>,
    translation_target_lang: Option<String>,
) -> std::result::Result<InferenceRuntimeLimits, String> {
    let caps = system_info::get_system_capabilities()?;

    let selected_whisper_model = whisper_model
        .as_deref()
        .map(annotate::parse_whisper_model)
        .transpose()
        .map_err(|e| e.to_string())?;

    let strategy = translation_strategy
        .map(|value| parse_translation_strategy(Some(value), None))
        .transpose()?;

    let max_inference_concurrency = calculate_max_inference_concurrency_for_selection(
        &caps,
        selected_whisper_model,
        whisper_force_cpu.unwrap_or(false),
        strategy,
        translation_source_lang.as_deref(),
        translation_target_lang.as_deref(),
    );

    Ok(InferenceRuntimeLimits {
        max_inference_concurrency,
    })
}

#[tauri::command]
async fn resolve_translation_model(
    source_lang: String,
    target_lang: String,
    strategy: Option<String>,
    allow_madlad: Option<bool>,
) -> std::result::Result<TranslationModelResolution, String> {
    tokio::task::spawn_blocking(move || {
        let strategy = parse_translation_strategy(strategy, allow_madlad)?;
        let params = build_translate_params_for_strategy(source_lang, target_lang, strategy)
            .map_err(|e| e.to_string())?;
        let mm = translate::model::ModelManager::new().map_err(|e| e.to_string())?;
        let model = mm.select_model(&params).map_err(|e| e.to_string())?;

        let (engine_tier, label, message) = if model.is_marian() {
            (
                "fast".to_string(),
                "Fast model".to_string(),
                "Using fast translation model for this language pair.".to_string(),
            )
        } else {
            (
                "universal".to_string(),
                "Universal model".to_string(),
                "Using a slower larger general model for broader language coverage.".to_string(),
            )
        };

        Ok(TranslationModelResolution {
            strategy: strategy.as_str().to_string(),
            model_id: model.model_id().to_string(),
            model_name: model.display_name().to_string(),
            engine_tier,
            label: label.clone(),
            message: message.clone(),
            speed_label: label,
            user_hint: message,
        })
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Check GitHub releases for a newer version of the app
#[tauri::command]
async fn check_for_updates(_force: Option<bool>) -> std::result::Result<UpdateInfo, String> {
    let current = env!("CARGO_PKG_VERSION");

    // In debug builds, `force: true` returns a fake update for UI testing
    #[cfg(debug_assertions)]
    if _force.unwrap_or(false) {
        return Ok(UpdateInfo {
            available: true,
            current_version: current.to_string(),
            latest_version: "99.99.99".to_string(),
            release_url: "https://github.com/hinsonan/hermeneia/releases".to_string(),
            release_notes: "[Test] Simulated update for development testing.".to_string(),
        });
    }

    let url = "https://api.github.com/repos/hinsonan/hermeneia/releases/latest";

    let client = reqwest::Client::builder()
        .user_agent("hermeneia-app")
        .build()
        .map_err(|e| e.to_string())?;

    let resp: serde_json::Value = client
        .get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;

    let tag = resp["tag_name"]
        .as_str()
        .unwrap_or("")
        .trim_start_matches('v');
    let release_url = resp["html_url"].as_str().unwrap_or("").to_string();
    let release_notes = resp["body"].as_str().unwrap_or("").to_string();

    let available = is_newer(tag, current);

    Ok(UpdateInfo {
        available,
        current_version: current.to_string(),
        latest_version: tag.to_string(),
        release_url,
        release_notes,
    })
}

/// Returns true if `remote` semver is greater than `local`.
fn is_newer(remote: &str, local: &str) -> bool {
    fn parse(v: &str) -> [u64; 3] {
        let mut parts = v.splitn(3, '.').map(|p| p.parse().unwrap_or(0));
        [
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
        ]
    }
    parse(remote) > parse(local)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    gpu::apply_optimizations();
    let max_inference_concurrency = match system_info::get_system_capabilities() {
        Ok(caps) => calculate_max_inference_concurrency_for_selection(
            &caps,
            None,
            false,
            None,
            None,
            None,
        ),
        Err(error) => {
            tracing::warn!(%error, "Failed to detect capabilities; falling back to default inference concurrency");
            DEFAULT_MAX_INFERENCE_CONCURRENCY
        }
    }
    .clamp(
        DEFAULT_MAX_INFERENCE_CONCURRENCY,
        HARD_MAX_INFERENCE_CONCURRENCY,
    );

    tracing::info!(
        max_inference_concurrency = max_inference_concurrency,
        "Configured inference concurrency"
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            player: Mutex::new(AudioPlayer::new()),
            inference_cancel_registry: Arc::new(CancelRegistry::new()),
            inference_semaphore: Arc::new(Semaphore::new(max_inference_concurrency)),
            max_inference_concurrency,
            cancel_download: Mutex::new(Arc::new(AtomicBool::new(false))),
            is_downloading: Arc::new(AtomicBool::new(false)),
            runtime_cache: runtime_cache::global_runtime_cache(),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_waveform_peaks,
            trim_audio_file,
            transcribe_audio_file,
            annotate_audio_file,
            write_text_file,
            write_zip_archive,
            get_system_capabilities,
            get_inference_runtime_limits,
            validate_model_selection,
            recommend_inference_concurrency,
            list_speaker_model_requirements,
            ensure_speaker_model_downloaded,
            get_runtime_cache_stats,
            clear_runtime_cache,
            translate_text_file,
            check_marian_pair_supported,
            resolve_translation_model,
            cancel_inference,
            cancel_job,
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
            get_playback_state,
            check_for_updates
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
