use crate::error::{AudioError, Result};
use crate::hf_cache::hf_hub_cache_dir;
use crate::speaker::types::SpeakerModel;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;

/// Manages download and caching of speaker diarization ONNX models.
///
/// Uses hf-hub to download from HuggingFace. Files are stored in the
/// hf-hub default cache (~/.cache/huggingface/hub/) and symlinked paths
/// are returned for sherpa-rs to use directly.
pub struct SpeakerModelManager;

impl SpeakerModelManager {
    /// Ensure both ONNX model files are present locally.
    /// Downloads from HuggingFace on first use via hf-hub.
    /// Returns (segmentation_path, embedding_path).
    pub fn ensure_models(model: &SpeakerModel) -> Result<(PathBuf, PathBuf)> {
        let api = ApiBuilder::new().with_progress(true).build().map_err(|e| {
            AudioError::ModelDownload {
                model: "speaker-diarization".to_string(),
                details: format!("HF API init failed: {}", e),
            }
        })?;

        let (seg_repo_id, seg_file) = model.segmentation_source();
        let (emb_repo_id, emb_file) = model.embedding_source();

        tracing::info!("Ensuring segmentation model: {}/{}", seg_repo_id, seg_file);
        let seg_repo = api.repo(Repo::new(seg_repo_id.to_string(), RepoType::Model));
        let seg_path = seg_repo
            .get(seg_file)
            .map_err(|e| AudioError::ModelDownload {
                model: format!("{}/{}", seg_repo_id, seg_file),
                details: e.to_string(),
            })?;

        tracing::info!("Ensuring embedding model: {}/{}", emb_repo_id, emb_file);
        let emb_repo = api.repo(Repo::new(emb_repo_id.to_string(), RepoType::Model));
        let emb_path = emb_repo
            .get(emb_file)
            .map_err(|e| AudioError::ModelDownload {
                model: format!("{}/{}", emb_repo_id, emb_file),
                details: e.to_string(),
            })?;

        Ok((seg_path, emb_path))
    }

    /// True if both model repos have been fetched to the hf-hub cache.
    /// Checks for the blobs directory as a proxy for downloaded content.
    pub fn is_cached(model: &SpeakerModel) -> bool {
        let base = hf_hub_cache_dir();

        let (seg_repo, _) = model.segmentation_source();
        let (emb_repo, _) = model.embedding_source();

        let seg_cache = base.join(format!("models--{}", seg_repo.replace('/', "--")));
        let emb_cache = base.join(format!("models--{}", emb_repo.replace('/', "--")));

        seg_cache.join("blobs").is_dir() && emb_cache.join("blobs").is_dir()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cached_returns_bool() {
        let model = SpeakerModel::English;
        // Should not panic, just return true or false depending on cache state
        let _ = SpeakerModelManager::is_cached(&model);
    }
}
