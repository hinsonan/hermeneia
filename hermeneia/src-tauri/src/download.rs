//! Model download manager with progress tracking, cache inspection, and deletion.
//!
//! Provides Tauri-facing functions for listing available models, downloading
//! them with progress events, cancelling downloads, and managing the cache.

use crate::error::{AudioError, Result};
use crate::transcribe::WhisperModel;
use crate::translate::catalog::{load_model_catalog, CatalogModel, ModelFamily};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("huggingface")
        .join("hub")
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

/// Check whether a model appears to be cached by looking for config.json in snapshots.
/// Uses metadata() to follow symlinks and verify the target file actually exists.
fn is_model_cached_on_disk(model_id: &str) -> bool {
    let snapshots_dir = model_cache_path(model_id).join("snapshots");
    if !snapshots_dir.is_dir() {
        return false;
    }
    // Look for a snapshot directory containing config.json (present in all models)
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let config_path = entry.path().join("config.json");
                // metadata() follows symlinks - returns Err if target doesn't exist
                if config_path.metadata().map(|m| m.is_file()).unwrap_or(false) {
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
        (WhisperModel::Tiny, "Tiny", "Fastest, least accurate (~150MB)"),
        (WhisperModel::TinyEn, "Tiny (English)", "English-only, faster (~150MB)"),
        (WhisperModel::Base, "Base", "Good balance (~290MB)"),
        (WhisperModel::BaseEn, "Base (English)", "English-only (~290MB)"),
        (WhisperModel::Small, "Small", "Better accuracy (~970MB)"),
        (WhisperModel::SmallEn, "Small (English)", "English-only (~970MB)"),
        (WhisperModel::Medium, "Medium", "High accuracy (~3.1GB)"),
        (WhisperModel::MediumEn, "Medium (English)", "English-only (~3.1GB)"),
        (WhisperModel::Large, "Large", "Highest accuracy (~6.2GB)"),
        (WhisperModel::LargeV2, "Large v2", "Improved large model (~6.2GB)"),
        (WhisperModel::LargeV3, "Large v3", "Latest large model (~6.2GB)"),
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
            display_name: entry.description.clone().unwrap_or_else(|| entry.name.clone()),
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
        WhisperModel::Tiny, WhisperModel::TinyEn, WhisperModel::Base, WhisperModel::BaseEn,
        WhisperModel::Small, WhisperModel::SmallEn, WhisperModel::Medium, WhisperModel::MediumEn,
        WhisperModel::Large, WhisperModel::LargeV2, WhisperModel::LargeV3,
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
                    bytes_downloaded: 0,
                    bytes_total: None,
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
                bytes_downloaded: 0,
                bytes_total: None,
                phase: "downloading".to_string(),
            },
        );

        // For large weight files, download in a separate thread and monitor
        let is_large_file = file_name.ends_with(".safetensors")
            || file_name.ends_with(".bin")
            || file_name.ends_with(".gguf");

        if is_large_file {
            download_with_progress(
                model_id,
                revision.as_deref(),
                file_name,
                model_name,
                idx,
                total_files,
                total_bytes,
                app_handle,
                cancel_flag,
            )?;
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
            bytes_downloaded: 0,
            bytes_total: None,
            phase: "complete".to_string(),
        },
    );

    Ok(())
}

/// Download a large file with blob-size monitoring for progress.
fn download_with_progress(
    model_id: &str,
    revision: Option<&str>,
    file_name: &str,
    model_name: &str,
    file_index: usize,
    total_files: usize,
    total_bytes: Option<u64>,
    app_handle: &tauri::AppHandle,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<()> {
    // Get the blobs directory to monitor size growth
    let blobs_dir = model_cache_path(model_id).join("blobs");

    // Measure starting size of blobs dir
    let start_size = dir_size(&blobs_dir);

    // Create owned values for the download thread
    let file_name_owned = file_name.to_string();
    let model_id_owned = model_id.to_string();
    let revision_owned = revision.map(|s| s.to_string());

    // Spawn the actual download in a thread (build a fresh Api+Repo inside to avoid lifetime issues)
    let download_result = Arc::new(std::sync::Mutex::new(None::<std::result::Result<PathBuf, String>>));
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
            repo.get(&file_name_owned)
                .map_err(|e| format!("Failed to download {}: {}", file_name_owned, e))
        })();
        *result_clone.lock().unwrap() = Some(res);
    });

    // Monitor loop: poll blob size growth and emit progress
    loop {
        // Check cancellation
        if cancel_flag.load(Ordering::SeqCst) {
            // We can't easily kill the download thread, but we signal cancellation
            // and return error. The thread will finish in background harmlessly.
            let _ = app_handle.emit(
                "download-progress",
                DownloadProgress {
                    model_id: model_id.to_string(),
                    model_name: model_name.to_string(),
                    file_name: file_name.to_string(),
                    file_index,
                    total_files,
                    bytes_downloaded: 0,
                    bytes_total: total_bytes,
                    phase: "cancelled".to_string(),
                },
            );
            return Err(AudioError::DownloadCancelled);
        }

        // Check if download thread finished
        if handle.is_finished() {
            break;
        }

        // Measure current blob size
        let current_size = dir_size(&blobs_dir);
        let downloaded = current_size.saturating_sub(start_size);

        let _ = app_handle.emit(
            "download-progress",
            DownloadProgress {
                model_id: model_id.to_string(),
                model_name: model_name.to_string(),
                file_name: file_name.to_string(),
                file_index,
                total_files,
                bytes_downloaded: downloaded,
                bytes_total: total_bytes,
                phase: "downloading".to_string(),
            },
        );

        // Poll every 2 seconds to reduce I/O overhead from dir_size traversal
        std::thread::sleep(std::time::Duration::from_millis(2000));
    }

    // Collect the result
    let _ = handle.join();
    let result = download_result.lock().unwrap().take();
    match result {
        Some(Ok(_)) => Ok(()),
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
fn determine_files(model_id: &str, catalog_entry: Option<&CatalogModel>) -> (Vec<&'static str>, Option<String>) {
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
                vec!["config.json", "vocab.json", "source.spm", "model.safetensors"],
                revision,
            );
        } else {
            return (
                vec!["config.json", "vocab.json", "source.spm", "pytorch_model.bin"],
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
            vec!["config.json", "vocab.json", "source.spm", "model.safetensors"],
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
