//! Model download manager with progress tracking, cache inspection, and deletion.
//!
//! Provides Tauri-facing functions for listing available models, downloading
//! them with progress events, cancelling downloads, and managing the cache.

use crate::error::{AudioError, Result};
use crate::hf_cache::hf_hub_cache_dir;
use crate::transcribe::WhisperModel;
use crate::translate::catalog::{load_model_catalog, CatalogModel, ModelFamily};
use hf_hub::api::Progress;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tauri::Emitter;

// ============================================================================
// Types
// ============================================================================

/// Progress event emitted during model downloads.
#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub model_name: String,
    pub file_name: String,
    pub file_index: usize,
    pub total_files: usize,
    pub bytes_downloaded: u64,
    pub bytes_total: Option<u64>,
    pub phase: String, // "downloading" | "complete" | "cancelled"
}

/// Information about a single model for the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub display_name: String,
    pub category: String, // "whisper" | "marian" | "madlad"
    pub size_mb: u64,
    pub is_cached: bool,
    pub description: String,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
}

// ============================================================================
// Cache helpers
// ============================================================================

/// Get the HuggingFace hub cache directory (cross-platform).
fn hf_cache_dir() -> PathBuf {
    hf_hub_cache_dir()
}

/// Convert a HuggingFace model id (e.g. "openai/whisper-tiny") to its cache
/// directory name (e.g. "models--openai--whisper-tiny").
fn model_cache_dir_name(model_id: &str) -> String {
    format!("models--{}", model_id.replace('/', "--"))
}

/// Get the full path to a model's cache directory.
fn model_cache_path(model_id: &str) -> PathBuf {
    hf_cache_dir().join(model_cache_dir_name(model_id))
}

fn has_nontrivial_cached_file(path: &std::path::Path, min_bytes: u64) -> bool {
    path.metadata()
        .map(|m| m.is_file() && m.len() >= min_bytes)
        .unwrap_or(false)
}

/// Check whether a model appears to be cached by looking for weight files in snapshots.
/// Uses metadata() to follow symlinks and verify the target file actually exists.
/// We check for weight files (not just config.json) to avoid treating partially-downloaded
/// models as cached — config.json is small and downloaded first, so an interrupted download
/// may leave only config.json behind.
fn is_model_cached_on_disk(model_id: &str) -> bool {
    let snapshots_dir = model_cache_path(model_id).join("snapshots");
    if !snapshots_dir.is_dir() {
        return false;
    }
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let dir = entry.path();
                // Check for weight files - the large files that actually matter
                let has_weights =
                    has_nontrivial_cached_file(&dir.join("model.safetensors"), 1_000_000)
                        || has_nontrivial_cached_file(&dir.join("pytorch_model.bin"), 1_000_000)
                        || has_nontrivial_cached_file(&dir.join("model-q8_0.gguf"), 1_000_000);
                if has_weights {
                    return true;
                }
            }
        }
    }
    false
}

/// Calculate the total size of a directory tree in bytes.
/// Uses symlink_metadata() to avoid double-counting symlinked files
/// (HF cache has snapshots/ symlinks pointing into blobs/).
fn dir_size(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            // Use file_type() from DirEntry which does NOT follow symlinks
            if let Ok(ft) = entry.file_type() {
                if ft.is_symlink() {
                    // Skip symlinks to avoid double-counting
                    // (HF cache snapshots/ contains symlinks into blobs/)
                } else if ft.is_dir() {
                    total += dir_size(&entry.path());
                } else if let Ok(meta) = entry.metadata() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

// ============================================================================
// Model listing
// ============================================================================

/// Build a list of all known models (Whisper + Translation) with cache status.
pub fn list_all_models() -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();

    // Whisper models
    let whisper_variants = [
        (
            WhisperModel::Tiny,
            "Tiny",
            "Fastest, least accurate (~150MB)",
        ),
        (
            WhisperModel::TinyEn,
            "Tiny (English)",
            "English-only, faster (~150MB)",
        ),
        (WhisperModel::Base, "Base", "Good balance (~290MB)"),
        (
            WhisperModel::BaseEn,
            "Base (English)",
            "English-only (~290MB)",
        ),
        (WhisperModel::Small, "Small", "Better accuracy (~970MB)"),
        (
            WhisperModel::SmallEn,
            "Small (English)",
            "English-only (~970MB)",
        ),
        (WhisperModel::Medium, "Medium", "High accuracy (~3.1GB)"),
        (
            WhisperModel::MediumEn,
            "Medium (English)",
            "English-only (~3.1GB)",
        ),
        (WhisperModel::Large, "Large", "Highest accuracy (~6.2GB)"),
        (
            WhisperModel::LargeV2,
            "Large v2",
            "Improved large model (~6.2GB)",
        ),
        (
            WhisperModel::LargeV3,
            "Large v3",
            "Latest large model (~6.2GB)",
        ),
    ];

    for (model, name, desc) in whisper_variants.iter() {
        let hf_id = model.model_id();
        let reqs = model.requirements();
        models.push(ModelInfo {
            model_id: hf_id.to_string(),
            display_name: name.to_string(),
            category: "whisper".to_string(),
            size_mb: (reqs.disk_size_gb * 1024.0) as u64,
            is_cached: is_model_cached_on_disk(hf_id),
            description: desc.to_string(),
            source_lang: None,
            target_lang: None,
        });
    }

    // Translation models from catalog
    let catalog = load_model_catalog()?;
    for entry in &catalog {
        let category = match entry.family {
            ModelFamily::Madlad => "madlad",
            ModelFamily::Marian => "marian",
        };
        models.push(ModelInfo {
            model_id: entry.model_id.clone(),
            display_name: entry
                .description
                .clone()
                .unwrap_or_else(|| entry.name.clone()),
            category: category.to_string(),
            size_mb: entry.size_mb,
            is_cached: is_model_cached_on_disk(&entry.model_id),
            description: entry.description.clone().unwrap_or_default(),
            source_lang: entry.source.clone(),
            target_lang: entry.target.clone(),
        });
    }

    Ok(models)
}

// ============================================================================
// Download
// ============================================================================

/// Look up the expected total size in bytes for a model.
fn expected_model_bytes(model_id: &str, catalog_entry: Option<&CatalogModel>) -> Option<u64> {
    // Check catalog first
    if let Some(entry) = catalog_entry {
        return Some(entry.size_mb * 1024 * 1024);
    }
    // Check Whisper models
    let whisper_variants = [
        WhisperModel::Tiny,
        WhisperModel::TinyEn,
        WhisperModel::Base,
        WhisperModel::BaseEn,
        WhisperModel::Small,
        WhisperModel::SmallEn,
        WhisperModel::Medium,
        WhisperModel::MediumEn,
        WhisperModel::Large,
        WhisperModel::LargeV2,
        WhisperModel::LargeV3,
    ];
    for model in &whisper_variants {
        if model.model_id() == model_id {
            return Some((model.requirements().disk_size_gb * 1024.0 * 1024.0 * 1024.0) as u64);
        }
    }
    None
}

/// Download a model by its HuggingFace model ID, emitting progress events.
///
/// The download runs in the current thread. Monitor a separate cancel flag
/// to request early termination.
pub fn download_model(
    model_id: &str,
    model_name: &str,
    app_handle: &tauri::AppHandle,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<()> {
    // Determine which files to download based on category
    let catalog = load_model_catalog().ok();
    let catalog_entry = catalog
        .as_ref()
        .and_then(|c| c.iter().find(|m| m.model_id == model_id));

    let total_bytes = expected_model_bytes(model_id, catalog_entry);
    let (files_to_download, revision) = determine_files(model_id, catalog_entry);

    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .map_err(|e| AudioError::ModelDownload {
            model: model_id.to_string(),
            details: format!("API init failed: {}", e),
        })?;

    let repo = if let Some(rev) = &revision {
        api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.repo(Repo::new(model_id.to_string(), RepoType::Model))
    };

    let total_files = files_to_download.len();
    let mut cumulative_downloaded = 0u64;

    for (idx, file_name) in files_to_download.iter().enumerate() {
        // Check cancellation before each file
        if cancel_flag.load(Ordering::SeqCst) {
            let _ = app_handle.emit(
                "download-progress",
                DownloadProgress {
                    model_id: model_id.to_string(),
                    model_name: model_name.to_string(),
                    file_name: file_name.to_string(),
                    file_index: idx,
                    total_files,
                    bytes_downloaded: cumulative_downloaded,
                    bytes_total: total_bytes,
                    phase: "cancelled".to_string(),
                },
            );
            return Err(AudioError::DownloadCancelled);
        }

        // Emit progress start for this file
        let _ = app_handle.emit(
            "download-progress",
            DownloadProgress {
                model_id: model_id.to_string(),
                model_name: model_name.to_string(),
                file_name: file_name.to_string(),
                file_index: idx,
                total_files,
                bytes_downloaded: cumulative_downloaded,
                bytes_total: total_bytes,
                phase: "downloading".to_string(),
            },
        );

        // For large weight files, download in a separate thread and monitor
        let is_large_file = file_name.ends_with(".safetensors")
            || file_name.ends_with(".bin")
            || file_name.ends_with(".gguf");

        if is_large_file {
            let downloaded_for_file = download_with_progress(
                model_id,
                revision.as_deref(),
                file_name,
                model_name,
                idx,
                total_files,
                cumulative_downloaded,
                total_bytes,
                app_handle,
                cancel_flag,
            )?;

            cumulative_downloaded = cumulative_downloaded.saturating_add(downloaded_for_file);
        } else {
            // Small files: download directly
            repo.get(file_name).map_err(|e| AudioError::ModelDownload {
                model: model_id.to_string(),
                details: format!("Failed to download {}: {}", file_name, e),
            })?;
        }
    }

    // Emit completion
    let _ = app_handle.emit(
        "download-progress",
        DownloadProgress {
            model_id: model_id.to_string(),
            model_name: model_name.to_string(),
            file_name: String::new(),
            file_index: total_files,
            total_files,
            bytes_downloaded: total_bytes.unwrap_or(cumulative_downloaded),
            bytes_total: total_bytes,
            phase: "complete".to_string(),
        },
    );

    Ok(())
}

/// Progress reporter used by hf-hub download callback.
#[derive(Debug)]
struct TauriDownloadProgressReporter {
    app_handle: tauri::AppHandle,
    model_id: String,
    model_name: String,
    file_name: String,
    file_index: usize,
    total_files: usize,
    cumulative_before_file: u64,
    total_bytes: Option<u64>,
    file_downloaded: Arc<AtomicU64>,
    file_total: Arc<AtomicU64>,
    cancel_flag: Arc<AtomicBool>,
    last_emit: std::time::Instant,
}

impl TauriDownloadProgressReporter {
    fn new(
        app_handle: tauri::AppHandle,
        model_id: &str,
        model_name: &str,
        file_name: &str,
        file_index: usize,
        total_files: usize,
        cumulative_before_file: u64,
        total_bytes: Option<u64>,
        file_downloaded: Arc<AtomicU64>,
        file_total: Arc<AtomicU64>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            app_handle,
            model_id: model_id.to_string(),
            model_name: model_name.to_string(),
            file_name: file_name.to_string(),
            file_index,
            total_files,
            cumulative_before_file,
            total_bytes,
            file_downloaded,
            file_total,
            cancel_flag,
            last_emit: std::time::Instant::now(),
        }
    }

    fn fallback_total_bytes(&self) -> Option<u64> {
        if let Some(total) = self.total_bytes {
            return Some(total);
        }

        let file_total = self.file_total.load(Ordering::Relaxed);
        if file_total > 0 {
            return Some(self.cumulative_before_file.saturating_add(file_total));
        }

        None
    }

    fn emit_downloading(&self, file_downloaded: u64) {
        if self.cancel_flag.load(Ordering::Relaxed) {
            return;
        }

        let overall_downloaded = self.cumulative_before_file.saturating_add(file_downloaded);
        let _ = self.app_handle.emit(
            "download-progress",
            DownloadProgress {
                model_id: self.model_id.clone(),
                model_name: self.model_name.clone(),
                file_name: self.file_name.clone(),
                file_index: self.file_index,
                total_files: self.total_files,
                bytes_downloaded: overall_downloaded,
                bytes_total: self.fallback_total_bytes(),
                phase: "downloading".to_string(),
            },
        );
    }
}

impl Progress for TauriDownloadProgressReporter {
    fn init(&mut self, size: usize, _filename: &str) {
        let total = size as u64;
        self.file_total.store(total, Ordering::Relaxed);
        self.file_downloaded.store(0, Ordering::Relaxed);
        self.last_emit = std::time::Instant::now();
        self.emit_downloading(0);
    }

    fn update(&mut self, size: usize) {
        let size = size as u64;
        let raw_downloaded = self
            .file_downloaded
            .fetch_add(size, Ordering::Relaxed)
            .saturating_add(size);

        let file_total = self.file_total.load(Ordering::Relaxed);
        let downloaded = if file_total > 0 {
            raw_downloaded.min(file_total)
        } else {
            raw_downloaded
        };

        let near_end = file_total > 0 && file_total.saturating_sub(downloaded) <= 64 * 1024;
        let should_emit = self.last_emit.elapsed() >= std::time::Duration::from_millis(150)
            || near_end
            || (file_total > 0 && downloaded == file_total);

        if should_emit {
            self.last_emit = std::time::Instant::now();
            self.emit_downloading(downloaded);
        }
    }

    fn finish(&mut self) {
        let file_total = self.file_total.load(Ordering::Relaxed);
        let downloaded = if file_total > 0 {
            self.file_downloaded.store(file_total, Ordering::Relaxed);
            file_total
        } else {
            self.file_downloaded.load(Ordering::Relaxed)
        };

        self.emit_downloading(downloaded);
    }
}

/// Download a large file using hf-hub progress callbacks.
fn download_with_progress(
    model_id: &str,
    revision: Option<&str>,
    file_name: &str,
    model_name: &str,
    file_index: usize,
    total_files: usize,
    cumulative_before_file: u64,
    total_bytes: Option<u64>,
    app_handle: &tauri::AppHandle,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<u64> {
    let file_downloaded = Arc::new(AtomicU64::new(0));
    let file_total = Arc::new(AtomicU64::new(0));

    // Create owned values for the download thread
    let file_name_owned = file_name.to_string();
    let model_id_owned = model_id.to_string();
    let model_name_owned = model_name.to_string();
    let revision_owned = revision.map(|s| s.to_string());
    let app_handle_owned = app_handle.clone();
    let file_downloaded_for_thread = file_downloaded.clone();
    let file_total_for_thread = file_total.clone();
    let cancel_flag_for_thread = cancel_flag.clone();

    // Spawn the actual download in a thread (build a fresh Api+Repo inside to avoid lifetime issues)
    let download_result = Arc::new(std::sync::Mutex::new(
        None::<std::result::Result<PathBuf, String>>,
    ));
    let result_clone = download_result.clone();

    let handle = std::thread::spawn(move || {
        let res = (|| -> std::result::Result<PathBuf, String> {
            let api = ApiBuilder::new()
                .with_progress(false)
                .build()
                .map_err(|e| format!("API init failed: {}", e))?;
            let repo = if let Some(rev) = &revision_owned {
                api.repo(Repo::with_revision(
                    model_id_owned.clone(),
                    RepoType::Model,
                    rev.clone(),
                ))
            } else {
                api.repo(Repo::new(model_id_owned.clone(), RepoType::Model))
            };

            let reporter = TauriDownloadProgressReporter::new(
                app_handle_owned,
                &model_id_owned,
                &model_name_owned,
                &file_name_owned,
                file_index,
                total_files,
                cumulative_before_file,
                total_bytes,
                file_downloaded_for_thread,
                file_total_for_thread,
                cancel_flag_for_thread,
            );

            repo.download_with_progress(&file_name_owned, reporter)
                .map_err(|e| format!("Failed to download {}: {}", file_name_owned, e))
        })();
        *result_clone.lock().unwrap() = Some(res);
    });

    // Monitor loop: keep cancellation responsive while download thread runs.
    loop {
        // Check cancellation
        if cancel_flag.load(Ordering::SeqCst) {
            let downloaded = file_downloaded.load(Ordering::Relaxed);
            let model_total = total_bytes.or_else(|| {
                let file_total = file_total.load(Ordering::Relaxed);
                if file_total > 0 {
                    Some(cumulative_before_file.saturating_add(file_total))
                } else {
                    None
                }
            });

            let _ = app_handle.emit(
                "download-progress",
                DownloadProgress {
                    model_id: model_id.to_string(),
                    model_name: model_name.to_string(),
                    file_name: file_name.to_string(),
                    file_index,
                    total_files,
                    bytes_downloaded: cumulative_before_file.saturating_add(downloaded),
                    bytes_total: model_total,
                    phase: "cancelled".to_string(),
                },
            );
            return Err(AudioError::DownloadCancelled);
        }

        // Check if download thread finished
        if handle.is_finished() {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Collect the result
    if handle.join().is_err() {
        return Err(AudioError::ModelDownload {
            model: model_id.to_string(),
            details: "Download thread panicked".to_string(),
        });
    }
    let result = download_result.lock().unwrap().take();
    match result {
        Some(Ok(_)) => {
            let downloaded = file_downloaded.load(Ordering::Relaxed);
            Ok(downloaded)
        }
        Some(Err(e)) => Err(AudioError::ModelDownload {
            model: model_id.to_string(),
            details: e,
        }),
        None => Err(AudioError::ModelDownload {
            model: model_id.to_string(),
            details: "Download thread completed without result".to_string(),
        }),
    }
}

/// Determine which files need downloading for a given model.
fn determine_files(
    model_id: &str,
    catalog_entry: Option<&CatalogModel>,
) -> (Vec<&'static str>, Option<String>) {
    // Whisper models
    if model_id.starts_with("openai/whisper") {
        return (
            vec!["config.json", "tokenizer.json", "model.safetensors"],
            None,
        );
    }

    // MADLAD models
    if model_id.contains("madlad") {
        return (
            vec!["config.json", "tokenizer.json", "model.safetensors"],
            None,
        );
    }

    // MarianMT models
    if let Some(entry) = catalog_entry {
        let revision = entry.revision.clone();
        if entry.has_safetensors {
            return (
                vec![
                    "config.json",
                    "vocab.json",
                    "source.spm",
                    "model.safetensors",
                ],
                revision,
            );
        } else {
            return (
                vec![
                    "config.json",
                    "vocab.json",
                    "source.spm",
                    "pytorch_model.bin",
                ],
                revision,
            );
        }
    }

    // Default for Helsinki-NLP MarianMT
    if model_id.starts_with("Helsinki-NLP/") {
        let is_tc_big = model_id.contains("tc-big");
        let revision = if is_tc_big {
            None
        } else {
            Some("refs/pr/4".to_string())
        };
        return (
            vec![
                "config.json",
                "vocab.json",
                "source.spm",
                "model.safetensors",
            ],
            revision,
        );
    }

    // Fallback
    (vec!["config.json", "model.safetensors"], None)
}

// ============================================================================
// Cache management
// ============================================================================

/// Check if a single model is cached.
pub fn check_model_cached(model_id: &str) -> bool {
    is_model_cached_on_disk(model_id)
}

/// Delete a model's cache directory.
pub fn delete_model_cache(model_id: &str) -> Result<()> {
    let cache_path = model_cache_path(model_id);
    if !cache_path.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(&cache_path).map_err(|e| AudioError::ModelDelete {
        model: model_id.to_string(),
        details: e.to_string(),
    })?;
    Ok(())
}

/// Get total size of all cached HuggingFace models in bytes.
pub fn total_cache_size() -> u64 {
    dir_size(&hf_cache_dir())
}
