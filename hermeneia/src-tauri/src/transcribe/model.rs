use crate::error::{AudioError, Result};
use crate::hf_cache::hf_hub_cache_dir;
use crate::transcribe::types::{ModelFiles, WhisperModel};
use candle_core::Device;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

/// Model manager for downloading and caching Whisper models
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

    /// Download model if not cached, return paths to required files
    pub fn ensure_model(&self, model: WhisperModel, quantized: bool) -> Result<ModelFiles> {
        let repo_id = model.model_id();
        let repo = self
            .api
            .repo(Repo::new(repo_id.to_string(), RepoType::Model));

        // Download required files
        let config = repo
            .get("config.json")
            .map_err(|e| AudioError::ModelDownload {
                model: repo_id.to_string(),
                details: e.to_string(),
            })?;

        let tokenizer = repo
            .get("tokenizer.json")
            .map_err(|e| AudioError::ModelDownload {
                model: repo_id.to_string(),
                details: e.to_string(),
            })?;

        let (weights, is_quantized) = if quantized {
            // Try quantized first, fall back to normal
            match repo.get("model-q8_0.gguf") {
                Ok(w) => (w, true),
                Err(_) => {
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
            let w = repo
                .get("model.safetensors")
                .map_err(|e| AudioError::ModelDownload {
                    model: repo_id.to_string(),
                    details: e.to_string(),
                })?;
            (w, false)
        };

        Ok(ModelFiles {
            config,
            tokenizer,
            weights,
            is_quantized,
        })
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> std::path::PathBuf {
        hf_hub_cache_dir()
    }
}

// Prevent enabling both CUDA and Metal simultaneously
#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("Features `cuda` and `metal` are mutually exclusive. Enable only one.");

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
    fn test_get_device_cpu() {
        let device = get_device(true);
        assert!(device.is_ok());
    }
}
