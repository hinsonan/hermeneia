use crate::error::{AudioError, Result};
use crate::hf_cache::hf_hub_cache_dir;
use crate::translate::catalog::{load_model_catalog, CatalogModel, CatalogModelStatus};
use crate::translate::language::get_marian_for_pair;
use crate::translate::types::TranslateParams;
use crate::translate::types::TranslationModel;
use candle_core::Device;
use hf_hub::api::sync::ApiRepo;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::{Path, PathBuf};

/// Paths to translation model files
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub spm_model: Option<PathBuf>, // SentencePiece model for MarianMT
    pub weights: Vec<PathBuf>,      // One path, or all shards for sharded checkpoints
}

/// Model manager for downloading and caching translation models
pub struct ModelManager {
    api: hf_hub::api::sync::Api,
}

impl ModelManager {
    /// Create new model manager using HuggingFace default cache
    pub fn new() -> Result<Self> {
        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .map_err(|e| AudioError::ModelDownload {
                model: "N/A".to_string(),
                details: format!("API init failed: {}", e),
            })?;
        Ok(Self { api })
    }

    /// Select best available model for the given language pair
    ///
    /// Selection logic:
    /// 1. If user specified preferred model and it's available or downloadable, use it
    /// 2. If preferred unavailable and fallback disabled, error
    /// 3. Try language-specific MarianMT (if cached)
    /// 4. Fall back to MADLAD-400 3B (default multilingual model)
    pub fn select_model(&self, params: &TranslateParams) -> Result<TranslationModel> {
        // 1. If user specified preferred model
        if let Some(preferred) = params.preferred_model {
            // Check if it supports the language pair
            if !preferred.supports_pair(&params.source_language, &params.target_language) {
                return Err(AudioError::UnsupportedLanguagePair {
                    source_lang: params.source_language.clone(),
                    target_lang: params.target_language.clone(),
                });
            }

            // If fallback disabled, always use preferred (will download if needed)
            if !params.fallback_enabled {
                return Ok(preferred);
            }

            // If fallback enabled and model is cached, use it
            if self.is_model_cached(preferred)? {
                return Ok(preferred);
            }

            // Otherwise, continue to auto-selection logic
        }

        // 2. Try language-specific MarianMT
        if let Some(marian) = get_marian_for_pair(&params.source_language, &params.target_language)
        {
            // Use MarianMT if available (fine-tuned for specific pair)
            tracing::info!(
                "Selected MarianMT model for {} -> {} (specialized for this pair)",
                params.source_language,
                params.target_language
            );
            return Ok(marian);
        }

        if !params.fallback_enabled {
            return Err(AudioError::ModelNotAvailable {
                model: "MarianMT".to_string(),
            });
        }

        // 3. Fall back to MADLAD-400 3B (multilingual, supports 450+ languages)
        tracing::info!(
            "No specialized MarianMT model for {} -> {}, falling back to MADLAD-400 3B",
            params.source_language,
            params.target_language
        );
        Ok(TranslationModel::Madlad3B)
    }

    /// Build a repository handle for a model at the given revision.
    ///
    /// `None` or `"main"` selects the repository's default branch.
    fn repo_for(&self, repo_id: &str, revision: Option<&str>) -> ApiRepo {
        match revision {
            Some(rev) if rev != "main" => self.api.repo(Repo::with_revision(
                repo_id.to_string(),
                RepoType::Model,
                rev.to_string(),
            )),
            _ => self
                .api
                .repo(Repo::new(repo_id.to_string(), RepoType::Model)),
        }
    }

    fn config_cached_at(&self, repo_id: &str, revision: Option<&str>) -> Result<bool> {
        let repo = self.repo_for(repo_id, revision);
        match repo.get("config.json") {
            Ok(path) => Ok(path.exists()),
            Err(_) => Ok(false),
        }
    }

    /// Check if a model is already cached locally
    pub fn is_model_cached(&self, model: TranslationModel) -> Result<bool> {
        let repo_id = model.model_id();
        let revision = model.catalog_revision();

        if self.config_cached_at(repo_id, revision.as_deref())? {
            return Ok(true);
        }

        // The catalog revision may have been removed; try the default branch.
        if revision.as_deref().is_some_and(|rev| rev != "main") {
            return self.config_cached_at(repo_id, None);
        }

        Ok(false)
    }

    /// Check if a catalog model entry is cached locally
    pub fn is_catalog_model_cached(&self, model: &CatalogModel) -> Result<bool> {
        let revision = model.revision.as_deref();

        if self.config_cached_at(&model.model_id, revision)? {
            return Ok(true);
        }

        if revision.is_some_and(|rev| rev != "main") {
            return self.config_cached_at(&model.model_id, None);
        }

        Ok(false)
    }

    /// Download model if not cached, return paths to required files
    ///
    /// The revision is driven by the catalog. When a catalog revision is no
    /// longer available, the resolver retries the default branch (main), which
    /// keeps models reachable even if a PR ref is deleted.
    pub fn ensure_model(&self, model: TranslationModel) -> Result<ModelFiles> {
        let repo_id = model.model_id();
        let revision = model.catalog_revision();

        tracing::info!("Loading model: {} ({})", model.display_name(), repo_id);

        match self.ensure_model_at_revision(model, revision.as_deref()) {
            Ok(files) => Ok(files),
            Err(primary_err) => {
                if revision.as_deref().is_some_and(|rev| rev != "main") {
                    tracing::warn!(
                        repo = repo_id,
                        revision = revision.as_deref().unwrap_or("main"),
                        error = %primary_err,
                        "Catalog revision unavailable; retrying on the default branch"
                    );
                    self.ensure_model_at_revision(model, None)
                } else {
                    Err(primary_err)
                }
            }
        }
    }

    fn ensure_model_at_revision(
        &self,
        model: TranslationModel,
        revision: Option<&str>,
    ) -> Result<ModelFiles> {
        let repo_id = model.model_id();
        let repo = self.repo_for(repo_id, revision);

        let config = get_required(&repo, repo_id, "config.json")?;

        // Different model families have different tokenizer file names
        let (tokenizer, spm_model) = if model.is_marian() {
            let vocab = get_required(&repo, repo_id, "vocab.json")?;
            let spm = get_required(&repo, repo_id, "source.spm")?;
            (vocab, Some(spm))
        } else {
            let tokenizer = get_required(&repo, repo_id, "tokenizer.json")?;
            (tokenizer, None)
        };

        let weights = download_weights(&repo, repo_id)?;

        tracing::info!("Model files resolved successfully");

        Ok(ModelFiles {
            config,
            tokenizer,
            spm_model,
            weights,
        })
    }

    /// List models from the catalog with cache status
    pub fn list_catalog_models(&self) -> Result<Vec<CatalogModelStatus>> {
        let catalog = load_model_catalog()?;
        let mut models = Vec::with_capacity(catalog.len());

        for model in catalog {
            let cached = self.is_catalog_model_cached(&model)?;
            models.push(CatalogModelStatus { model, cached });
        }

        Ok(models)
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> PathBuf {
        hf_hub_cache_dir()
    }
}

fn get_required(repo: &ApiRepo, repo_id: &str, file: &str) -> Result<PathBuf> {
    repo.get(file).map_err(|e| AudioError::ModelDownload {
        model: repo_id.to_string(),
        details: format!("Failed to download {}: {}", file, e),
    })
}

/// Download model weights.
///
/// Safetensors is the only supported checkpoint format. Handles sharded
/// checkpoints: when a monolithic `model.safetensors` is not published, the
/// `model.safetensors.index.json` weight map is resolved and every shard is
/// downloaded.
fn download_weights(repo: &ApiRepo, repo_id: &str) -> Result<Vec<PathBuf>> {
    if let Ok(path) = repo.get("model.safetensors") {
        tracing::info!("Using safetensors format");
        return Ok(vec![path]);
    }

    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let shards = shard_names_from_index(&index_path, repo_id)?;
        tracing::info!(shards = shards.len(), "Using sharded safetensors format");
        let mut paths = Vec::with_capacity(shards.len());
        for shard in shards {
            paths.push(repo.get(&shard).map_err(|e| AudioError::ModelDownload {
                model: repo_id.to_string(),
                details: format!("Failed to download shard {}: {}", shard, e),
            })?);
        }
        return Ok(paths);
    }

    Err(AudioError::ModelDownload {
        model: repo_id.to_string(),
        details: "No safetensors weights found (tried model.safetensors and model.safetensors.index.json)".to_string(),
    })
}

/// Read a safetensors index file and return its unique shard file names.
pub(crate) fn shard_names_from_index(index_path: &Path, repo_id: &str) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(index_path).map_err(|e| AudioError::ModelLoad {
        model: repo_id.to_string(),
        details: format!("Failed to read safetensors index: {}", e),
    })?;

    let index: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| AudioError::ModelLoad {
            model: repo_id.to_string(),
            details: format!("Failed to parse safetensors index: {}", e),
        })?;

    let weight_map = index
        .get("weight_map")
        .and_then(|value| value.as_object())
        .ok_or_else(|| AudioError::ModelLoad {
            model: repo_id.to_string(),
            details: "safetensors index is missing a weight_map object".to_string(),
        })?;

    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|value| value.as_str())
        .map(|name| name.to_string())
        .collect();
    shards.sort();
    shards.dedup();

    if shards.is_empty() {
        return Err(AudioError::ModelLoad {
            model: repo_id.to_string(),
            details: "safetensors index contains no shard files".to_string(),
        });
    }

    Ok(shards)
}

/// Get compute device (CPU, CUDA, or Metal)
pub fn get_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    {
        return Device::cuda_if_available(0).map_err(|e| AudioError::GpuError(e.to_string()));
    }

    #[cfg(feature = "metal")]
    {
        return Device::new_metal(0).map_err(|e| AudioError::GpuError(e.to_string()));
    }

    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_manager_new() {
        let manager = ModelManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_cache_dir() {
        let manager = ModelManager::new().unwrap();
        let cache = manager.cache_dir();
        assert!(cache.to_string_lossy().contains("huggingface"));
    }

    #[test]
    fn test_select_model_with_preferred() {
        let manager = ModelManager::new().unwrap();
        let params = TranslateParams {
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            preferred_model: Some(TranslationModel::Madlad3B),
            fallback_enabled: false,
            ..Default::default()
        };

        let selected = manager.select_model(&params).unwrap();
        assert_eq!(selected, TranslationModel::Madlad3B);
    }

    #[test]
    fn test_select_model_auto() {
        let manager = ModelManager::new().unwrap();
        let params = TranslateParams {
            source_language: "en".to_string(),
            target_language: "fr".to_string(),
            preferred_model: None,
            ..Default::default()
        };

        let selected = manager.select_model(&params).unwrap();
        // en->fr has no safetensors Marian model, so it falls back to MADLAD
        assert_eq!(selected, TranslationModel::Madlad3B);
    }

    #[test]
    fn test_select_model_unsupported_pair() {
        let manager = ModelManager::new().unwrap();
        let params = TranslateParams {
            source_language: "es".to_string(),
            target_language: "fr".to_string(),
            preferred_model: Some(TranslationModel::MarianEnEs), // Only supports en->es
            fallback_enabled: false,
            ..Default::default()
        };

        let result = manager.select_model(&params);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AudioError::UnsupportedLanguagePair { .. }
        ));
    }

    fn write_index(content: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "hermeneia-shard-index-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, content).unwrap();
        (dir, index_path)
    }

    #[test]
    fn test_shard_names_from_index_dedups_and_sorts() {
        let (dir, index_path) = write_index(
            r#"{"metadata":{},"weight_map":{
                "a":"model-00002-of-00002.safetensors",
                "b":"model-00001-of-00002.safetensors",
                "c":"model-00001-of-00002.safetensors"
            }}"#,
        );

        let shards = shard_names_from_index(&index_path, "test/model").unwrap();
        assert_eq!(
            shards,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string()
            ]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_shard_names_from_index_rejects_missing_weight_map() {
        let (dir, index_path) = write_index(r#"{"metadata":{}}"#);

        let result = shard_names_from_index(&index_path, "test/model");
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_shard_names_from_index_rejects_empty_weight_map() {
        let (dir, index_path) = write_index(r#"{"weight_map":{}}"#);

        let result = shard_names_from_index(&index_path, "test/model");
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
