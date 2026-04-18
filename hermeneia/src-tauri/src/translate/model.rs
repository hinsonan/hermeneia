use crate::error::{AudioError, Result};
use crate::hf_cache::hf_hub_cache_dir;
use crate::translate::catalog::{load_model_catalog, CatalogModel, CatalogModelStatus};
use crate::translate::language::get_marian_for_pair;
use crate::translate::types::TranslateParams;
use crate::translate::types::TranslationModel;
use candle_core::Device;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;

/// Paths to translation model files
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub spm_model: Option<PathBuf>, // SentencePiece model for MarianMT
    pub weights: PathBuf,
    pub is_quantized: bool,
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

    /// Check if a model is already cached locally
    pub fn is_model_cached(&self, model: TranslationModel) -> Result<bool> {
        let repo_id = model.model_id();

        // Different model families use different revisions for safetensors
        let repo = if let Some(revision) = model.safetensors_revision() {
            self.api.repo(Repo::with_revision(
                repo_id.to_string(),
                RepoType::Model,
                revision.to_string(),
            ))
        } else {
            self.api
                .repo(Repo::new(repo_id.to_string(), RepoType::Model))
        };

        // Try to get the config file - if it's cached, the model is available
        match repo.get("config.json") {
            Ok(path) => Ok(path.exists()),
            Err(_) => Ok(false),
        }
    }

    /// Check if a catalog model entry is cached locally
    pub fn is_catalog_model_cached(&self, model: &CatalogModel) -> Result<bool> {
        let repo = if let Some(revision) = model.revision.as_deref() {
            self.api.repo(Repo::with_revision(
                model.model_id.to_string(),
                RepoType::Model,
                revision.to_string(),
            ))
        } else {
            self.api
                .repo(Repo::new(model.model_id.to_string(), RepoType::Model))
        };

        match repo.get("config.json") {
            Ok(path) => Ok(path.exists()),
            Err(_) => Ok(false),
        }
    }

    /// Download model if not cached, return paths to required files
    pub fn ensure_model(&self, model: TranslationModel, quantized: bool) -> Result<ModelFiles> {
        let repo_id = model.model_id();

        // Different model families use different revisions for safetensors
        let repo = if let Some(revision) = model.safetensors_revision() {
            tracing::info!("Using revision '{}' for safetensors support", revision);
            self.api.repo(Repo::with_revision(
                repo_id.to_string(),
                RepoType::Model,
                revision.to_string(),
            ))
        } else {
            self.api
                .repo(Repo::new(repo_id.to_string(), RepoType::Model))
        };

        tracing::info!("Loading model: {} ({})", model.display_name(), repo_id);

        // Download required files
        let config = repo
            .get("config.json")
            .map_err(|e| AudioError::ModelDownload {
                model: repo_id.to_string(),
                details: e.to_string(),
            })?;

        // Different model families have different tokenizer file names
        let (tokenizer, spm_model) = if model.is_marian() {
            // MarianMT uses vocab.json + source.spm
            let vocab = repo
                .get("vocab.json")
                .map_err(|e| AudioError::ModelDownload {
                    model: repo_id.to_string(),
                    details: format!("Failed to download vocab.json: {}", e),
                })?;
            let spm = repo
                .get("source.spm")
                .map_err(|e| AudioError::ModelDownload {
                    model: repo_id.to_string(),
                    details: format!("Failed to download source.spm: {}", e),
                })?;
            (vocab, Some(spm))
        } else {
            // MADLAD models use tokenizer.json from main repo
            let tok = repo
                .get("tokenizer.json")
                .map_err(|e| AudioError::ModelDownload {
                    model: repo_id.to_string(),
                    details: format!("Failed to download tokenizer.json: {}", e),
                })?;
            (tok, None)
        };

        // Download model weights
        let (weights, is_quantized) = if quantized {
            // Try quantized first, fall back to normal
            match repo.get("model-q8_0.gguf") {
                Ok(w) => (w, true),
                Err(_) => {
                    tracing::warn!("Quantized model not available, using full precision");
                    let w =
                        repo.get("model.safetensors")
                            .map_err(|e| AudioError::ModelDownload {
                                model: repo_id.to_string(),
                                details: e.to_string(),
                            })?;
                    (w, false)
                }
            }
        } else {
            // For non-quantized models, prefer safetensors
            match repo.get("model.safetensors") {
                Ok(w) => {
                    tracing::info!("Using safetensors format");
                    (w, false)
                }
                Err(_) => {
                    // Try pytorch_model.bin as fallback, but warn that it may not work
                    tracing::warn!(
                        "SafeTensors not available, trying PyTorch bin (may fail for some models)"
                    );
                    match repo.get("pytorch_model.bin") {
                        Ok(w) => (w, false),
                        Err(e) => {
                            return Err(AudioError::ModelDownload {
                                model: repo_id.to_string(),
                                details: format!("No compatible model format found. Tried: model.safetensors, pytorch_model.bin. Last error: {}", e),
                            });
                        }
                    }
                }
            }
        };

        tracing::info!("Model loaded successfully (quantized: {})", is_quantized);

        Ok(ModelFiles {
            config,
            tokenizer,
            spm_model,
            weights,
            is_quantized,
        })
    }

    /// List all cached translation models
    pub fn list_cached_models(&self) -> Result<Vec<(TranslationModel, u64)>> {
        let mut cached = Vec::new();

        // Check all known models
        let all_models = [
            TranslationModel::Madlad3B,
            TranslationModel::Madlad7B,
            TranslationModel::Madlad10B,
            TranslationModel::MarianEnEs,
            TranslationModel::MarianEsEn,
            TranslationModel::MarianEnFr,
            TranslationModel::MarianFrEn,
            TranslationModel::MarianEnDe,
            TranslationModel::MarianDeEn,
            TranslationModel::MarianEnPt,
            TranslationModel::MarianPtEn,
            TranslationModel::MarianEnIt,
            TranslationModel::MarianItEn,
            TranslationModel::MarianEnRo,
            TranslationModel::MarianRoEn,
            TranslationModel::MarianEnNl,
            TranslationModel::MarianNlEn,
            TranslationModel::MarianEnSv,
            TranslationModel::MarianSvEn,
            TranslationModel::MarianEnDa,
            TranslationModel::MarianDaEn,
            TranslationModel::MarianEnNo,
            TranslationModel::MarianNoEn,
            TranslationModel::MarianEnRu,
            TranslationModel::MarianRuEn,
            TranslationModel::MarianEnPl,
            TranslationModel::MarianPlEn,
            TranslationModel::MarianEnCs,
            TranslationModel::MarianCsEn,
            TranslationModel::MarianEnUk,
            TranslationModel::MarianUkEn,
            TranslationModel::MarianEnZh,
            TranslationModel::MarianZhEn,
            TranslationModel::MarianEnJa,
            TranslationModel::MarianJaEn,
            TranslationModel::MarianEnKo,
            TranslationModel::MarianKoEn,
            TranslationModel::MarianEnVi,
            TranslationModel::MarianViEn,
            TranslationModel::MarianEnTh,
            TranslationModel::MarianThEn,
            TranslationModel::MarianEnId,
            TranslationModel::MarianIdEn,
            TranslationModel::MarianEnAr,
            TranslationModel::MarianArEn,
            TranslationModel::MarianEnHe,
            TranslationModel::MarianHeEn,
            TranslationModel::MarianEnFa,
            TranslationModel::MarianFaEn,
            TranslationModel::MarianEnTr,
            TranslationModel::MarianTrEn,
            TranslationModel::MarianEnHi,
            TranslationModel::MarianHiEn,
            TranslationModel::MarianEnBn,
            TranslationModel::MarianBnEn,
            TranslationModel::MarianEnUr,
            TranslationModel::MarianUrEn,
            TranslationModel::MarianEnHu,
            TranslationModel::MarianHuEn,
            TranslationModel::MarianEnFi,
            TranslationModel::MarianFiEn,
            TranslationModel::MarianEnEl,
            TranslationModel::MarianElEn,
            TranslationModel::MarianEnSw,
            TranslationModel::MarianSwEn,
        ];

        for model in all_models {
            if self.is_model_cached(model)? {
                cached.push((model, model.approx_size_mb()));
            }
        }

        Ok(cached)
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
        // Should select either MarianEnFr (if specialized) or Madlad3B as fallback
        assert!(
            selected == TranslationModel::MarianEnFr || selected == TranslationModel::Madlad3B,
            "Expected MarianEnFr or Madlad3B, got {:?}",
            selected
        );
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
}
